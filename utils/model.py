import time
import torch
import torch.optim as optim
import pytorch_lightning as pl

from utils.loss import *
from utils.model import *
from utils.layers import *
from utils.helpers import *

from torch.optim.lr_scheduler import ReduceLROnPlateau


class TransUNet(nn.Module):
    """The overall structre of TransUNet"""

    def __init__(
        self,
        in_ch,
        out_ch,
        num_layers,
        d_model,
        latent_dim,
        time_emb_dim,
        rate,
        max_seq,
    ):
        super().__init__()

        # time embedding
        time_out_dim = time_emb_dim * 4
        self.time_expand = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_out_dim),
            nn.GELU(),
            nn.Linear(time_out_dim, time_out_dim),
        )

        # input and output layer
        self.in_layer = nn.Conv2d(in_ch, latent_dim // 4, kernel_size=7, padding=3)
        self.out_layer = nn.Conv2d(latent_dim // 4, out_ch, kernel_size=1, padding=0)

        # down operation
        self.down1 = Down(latent_dim // 4, latent_dim // 2, time_out_dim, rate=rate)
        self.down2 = Down(latent_dim // 2, latent_dim, time_out_dim, rate=rate)
        self.down3 = Down(latent_dim, latent_dim, time_out_dim, rate=rate)

        # up operation
        self.up1 = Up(latent_dim * 2, latent_dim // 2, time_out_dim, rate=rate)
        self.up2 = Up(latent_dim, latent_dim // 4, time_out_dim, rate=rate)
        self.up3 = Up(latent_dim // 2, latent_dim // 4, time_out_dim, rate=rate)

        # transformer
        self.transformer = Transformer(num_layers, d_model, latent_dim, rate, max_seq)

    def forward(self, x, t):
        x1 = self.in_layer(x)

        # time embedding
        if t is not None:
            t = t.type_as(x)
            t = self.time_expand(t)

        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        # reshape
        x_flatten = x4.flatten(2)
        x_flatten = x_flatten.transpose(-1, -2)

        # fed into transformer
        x_flatten = self.transformer(x_flatten)
        x4 = x_flatten.reshape(x4.shape)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        logits = self.out_layer(x)

        return logits


class TransUNet_Lightning(pl.LightningModule):
    """Training or inference a model with pytorch_lightning"""

    def __init__(
        self,
        in_ch,
        out_ch,
        num_layers,
        d_model,
        latent_dim,
        time_emb_dim,
        time_steps,
        rate=0,
        max_seq=2048,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.automatic_optimization = False

        self.betas = cosine_beta_schedule(time_steps=time_steps)

        (
            self.alphas_cumprod,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.sqrt_recip_alphas,
            self.posterior_variance,
        ) = get_nice_property(self.betas)

        self.model = TransUNet(
            in_ch,
            out_ch,
            num_layers,
            d_model,
            latent_dim,
            time_emb_dim,
            rate,
            max_seq,
        )

    def diffusion(self, x, mixture):
        batch_size = x.shape[0]

        noise = torch.randn_like(x)
        t = torch.randint(0, self.time_steps, (batch_size,)).long()

        # get noisy sample
        x = q_sample(
            x,
            t,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            noise=noise,
        )

        # guide
        x = x * mixture
        noise = noise * mixture

        # forward model
        x_hat = self.model(x, t)
        loss = p_losses(noise, x_hat, loss_type="huber")

        return loss

    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        betas_t = betas_t.type_as(x)

        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.type_as(x)

        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.type_as(x)

        # algorithm 2 p(x(t-1)|x(t))
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        posterior_variance_t = posterior_variance_t.type_as(x)

        noise = torch.randn_like(x)
        noise = noise.type_as(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    def ddim_sample(self, x, t, t_index, mixture, eta=0):
        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_t = alphas_cumprod_t.type_as(x)

        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.type_as(x)

        pred = self.model(x, t)
        f_t = (x - (sqrt_one_minus_alphas_cumprod_t * pred)) / alphas_cumprod_t.sqrt()

        alphas_cumprod_t_1 = extract(self.alphas_cumprod, t - 1, x.shape)
        alphas_cumprod_t_1 = alphas_cumprod_t_1.type_as(x)

        sigma_t = ((1 - alphas_cumprod_t_1) / (1 - alphas_cumprod_t)).sqrt()
        sigma_t = sigma_t * (1 - (alphas_cumprod_t / alphas_cumprod_t_1)).sqrt()
        sigma_t_eta = eta * sigma_t

        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.type_as(x)

        sqrt_alphas_cumprod_t_1 = extract(self.sqrt_alphas_cumprod, t - 1, x.shape)
        sqrt_alphas_cumprod_t_1 = sqrt_alphas_cumprod_t_1.type_as(x)

        # first term
        pred_x0 = sqrt_alphas_cumprod_t_1 * f_t

        # second term
        pointing = (1 - alphas_cumprod_t_1 - (sigma_t_eta ** 2)).sqrt()
        pointing = pointing * pred

        # third term
        noise = sigma_t_eta * torch.randn_like(x) * mixture

        # the sum of three term
        x = pred_x0 + pointing + noise

        return x

    def forward(self, mixture, shape, device, eta=None, mode="ddpm"):
        x = torch.randn(shape, device=device) * mixture

        start_time = time.time()
        for t in reversed(range(1, self.time_steps)):
            if mode == "ddpm":
                x = self.p_sample(x, torch.full((shape[0],), t), t)
            elif mode == "ddim":
                x = self.ddim_sample(x, torch.full((shape[0],), t), t, mixture, eta)

        print("It takes %0.3f sec" % (time.time() - start_time))

        return x

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=1e-3)
        sch = ReduceLROnPlateau(opt, "min", factor=0.95, patience=0, min_lr=1e-5, verbose=True)

        return [opt], [sch]

    def training_step(self, train_batch, batch_idx):
        melody, mixture, music, track = train_batch

        opt = self.optimizers()
        sch = self.lr_schedulers()

        loss = self.diffusion(music, mixture)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        melody, mixture, music, track = val_batch

        loss = self.diffusion(music, mixture)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        # update scheduler
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])


class SimpleEncoder(nn.Module):
    """Encoder for evaluation"""

    def __init__(self, in_ch, out_ch, latent_dim):
        super().__init__()

        self.in_layer = nn.Conv2d(in_ch, latent_dim, kernel_size=7, padding=3, stride=2)
        self.out_layer = nn.Conv2d(latent_dim // 4, out_ch, kernel_size=3, padding=1)
        self.layer_1 = nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, padding=1, stride=2)
        self.layer_2 = nn.Conv2d(
            latent_dim // 2, latent_dim // 4, kernel_size=3, padding=1, stride=2
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.out_layer(x)
        x = torch.reshape(x, (x.shape[0], -1))

        return x


class SimpleDecoder(nn.Module):
    """Decoder for decoding x1 to x0"""

    def __init__(self, in_ch, out_ch, latent_dim):
        super().__init__()

        self.in_layer = nn.Conv2d(in_ch, latent_dim // 4, kernel_size=7, padding=3)
        self.out_layer = nn.Conv2d(latent_dim // 4, out_ch, kernel_size=3, padding=1)
        self.layer_1 = ResBlock(latent_dim // 4, latent_dim // 4)
        self.layer_2 = ResBlock(latent_dim // 4, latent_dim // 4)

    def forward(self, x, t=None):
        x = self.in_layer(x)
        x = self.layer_1(x, t)
        x = self.layer_2(x, t)
        x = self.out_layer(x)

        return x


class SimpleDecoder_Lightning(pl.LightningModule):
    def __init__(self, in_ch, out_ch, diffusion_model, latent_dim, rate=0):
        super().__init__()

        self.automatic_optimization = False
        self.decoder = SimpleDecoder(in_ch, out_ch, latent_dim)

        self.time_steps = diffusion_model.time_steps
        self.sqrt_alphas_cumprod = diffusion_model.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = diffusion_model.sqrt_one_minus_alphas_cumprod

    def get_x1(self, x):
        batch_size = x.shape[0]

        noise = torch.randn_like(x)
        t = torch.zeros(batch_size).long()

        # get noisy sample
        x1 = q_sample(
            x,
            t,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            noise=noise,
        )

        return x1

    def forward(self, x):
        return self.decoder(x)

    def configure_optimizers(self):
        opt = optim.AdamW(self.decoder.parameters(), lr=1e-3)
        sch = ReduceLROnPlateau(opt, "min", factor=0.9, patience=0, min_lr=1e-5, verbose=True)

        return [opt], [sch]

    def training_step(self, train_batch, batch_idx):
        melody, mixture, music, track = train_batch

        opt = self.optimizers()
        sch = self.lr_schedulers()

        x1 = self.get_x1(music)
        x1 = x1 * mixture  # guide

        x0_hat = self.decoder(x1)
        loss = bce_loss(music, x0_hat)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        melody, mixture, music, track = val_batch

        x1 = self.get_x1(music)
        x1 = x1 * mixture  # guide

        x0_hat = self.decoder(x1)
        loss = bce_loss(music, x0_hat)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        # update scheduler
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])
