import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange
from utils.helpers import *


class SinusoidalPositionEmbeddings(nn.Module):
    """Convert t of noise level to sinusoidal vector (batch, 1) -> (batch, dim)"""

    def __init__(self, dim):
        """
        :param dim: output dim
        :type dim: int
        """

        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = embeddings.type_as(time)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class ConvBlock(nn.Module):
    """One convolution block"""

    def __init__(self, in_ch, out_ch, groups):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, t=None):
        x = self.conv(x)
        x = self.norm(x)

        # x: (b, c, h, w)
        # t: (b, 2c)

        # AdaGN conditioned with t
        if exists(t):
            c = t.shape[1]
            t = rearrange(t, "b c -> b c 1 1")
            x = (t[:, : c // 2] * x) + t[:, c // 2 :]

        x = self.act(x)

        return x


class ResBlock(nn.Module):
    """One residual convolution block"""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8, rate=0):
        super().__init__()

        if exists(time_emb_dim):
            self.time_emb = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, 2 * out_ch))

        self.block1 = ConvBlock(in_ch, out_ch, groups)
        self.block2 = ConvBlock(out_ch, out_ch, groups)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.dropout = nn.Dropout(rate)

    def forward(self, x, t=None):
        if exists(t):
            t = self.time_emb(t)

        h = self.block1(x, t)
        h = self.block2(h, t)
        x = self.res_conv(x) + h
        x = self.dropout(x)

        return x


class Up(nn.Module):
    """Upscaling then convolution"""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, rate=0):
        super().__init__()

        # to reduce the number of channels
        self.block = ResBlock(in_ch, out_ch, time_emb_dim, rate=rate)

    def forward(self, x1, x2, t=None):
        x1 = F.interpolate(x1, scale_factor=2)
        x = torch.cat([x2, x1], dim=1)

        return self.block(x, t)


class Down(nn.Module):
    """Downscaling with maxpool then convolution"""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, rate=0, stride=2):
        super().__init__()

        self.maxpool = nn.MaxPool2d(stride)
        self.block = ResBlock(in_ch, out_ch, time_emb_dim, rate=rate)

    def forward(self, x, t=None):
        return self.block(self.maxpool(x), t)


class RelativeAttention(nn.Module):
    """Compute attention matrix with relative position"""

    def __init__(self, d_model, head, max_seq=2048):
        super().__init__()

        self.head = head
        self.max_seq = max_seq
        self.d_model = d_model
        self.dh_model = d_model // head

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.E = torch.randn([max_seq, self.dh_model], requires_grad=False)

    def forward(self, inputs, mask):
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.shape[0], q.shape[1], self.head, -1))
        q = q.permute(0, 2, 1, 3)  # (batch, h, seq, dh)

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.shape[0], k.shape[1], self.head, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.shape[0], v.shape[1], self.head, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]
        self.E = self.E.type_as(q)

        E = self._get_left_embedding(self.len_q)
        QE = torch.einsum("bhld,md->bhlm", [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / np.sqrt(self.dh_model)

        if mask is not None:
            logits += mask * -1e9

        attn_weights = F.softmax(logits, -1)
        attn = torch.matmul(attn_weights, v)

        out = attn.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.shape[0], -1, self.d_model))
        out = self.fc(out)

        return out, attn_weights

    def _get_left_embedding(self, len_q):
        start_idx = max(0, self.max_seq - len_q)
        e = self.E[start_idx:, :]

        return e

    def _skewing(self, tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(
            padded, shape=[padded.shape[0], padded.shape[1], padded.shape[-1], padded.shape[-2]]
        )
        Srel = reshaped[:, :, 1:, :]

        return Srel

    def _qe_masking(self, qe):
        len_idx = torch.arange(self.len_q - 1, -1, -1)
        mask = ~self.sequence_mask(len_idx, self.len_q)
        mask = mask.type_as(qe)

        return mask * qe

    def sequence_mask(self, length, max_length):
        x = torch.arange(max_length)

        return x.unsqueeze(0) < length.unsqueeze(1)


class EncoderLayer(nn.Module):
    """An encoder layer of transformer"""

    def __init__(self, d_model, head, rate=0, max_seq=2048):
        super().__init__()

        self.d_model = d_model
        self.rga = RelativeAttention(d_model, head, max_seq=max_seq)
        self.rga2 = RelativeAttention(d_model, head, max_seq=max_seq)

        self.FFN_pre = nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, mask):
        # masked multi-head attention
        attn_out1, attn_weights1 = self.rga([x, x, x], mask=mask)
        attn_out1 = self.dropout1(attn_out1)
        out1 = self.layernorm1(x + attn_out1)

        # multi-head attention
        attn_out2, attn_weights2 = self.rga2([out1, out1, out1], mask=mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1 + attn_out2)

        ffn_out = nn.GELU()(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2 + ffn_out)

        return out


class Transformer(nn.Module):
    """Full transformer"""

    def __init__(self, num_layers, d_model, vocab_size, rate=0, max_seq=2048):
        super().__init__()

        self.head = d_model // 64
        self.d_model = d_model
        self.num_layers = num_layers

        self.in_linear = nn.Linear(vocab_size, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, vocab_size)

        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, self.head, rate=rate, max_seq=max_seq)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = self.in_linear(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        x = self.out_linear(x)

        return x
