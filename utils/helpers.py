import torch
import torch.nn.functional as F


def exists(x):
    return x is not None


def cosine_beta_schedule(time_steps, s=0.008):
    beta_start = 0.0001
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, time_steps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)

    # matching dimension with input x
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def get_nice_property(betas):
    # define alphas
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return (
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
        posterior_variance,
    )


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """forward diffusion q(x(t+1)|x(t))"""

    if noise is None:
        noise = torch.randn_like(x_start)

    noise = noise.type_as(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.type_as(x_start)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.type_as(x_start)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
