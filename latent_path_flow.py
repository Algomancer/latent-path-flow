from __future__ import annotations
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import trange

device = torch.device("cuda")


def gauss_logprob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-12)
    return -0.5 * ((x - mean) ** 2 / var + var.log() + math.log(2.0 * math.pi)).sum(dim=-1)

def visualise_data(xs: Tensor, filename: str = "figure.jpg"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for seq in xs:
        ax.plot(seq[:, 0].detach().cpu(), seq[:, 1].detach().cpu(), seq[:, 2].detach().cpu())
    for a in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels): a([])
    ax.set_xlabel('$z_1$', fontsize=14); ax.set_ylabel('$z_2$', fontsize=14); ax.set_zlabel('$z_3$', fontsize=14)
    plt.savefig(filename, format='jpg', dpi=300); plt.close(fig)


class StochasticLorenzSDE(nn.Module):
    def __init__(self, a=(10., 28., 8/3), b=(.15, .15, .15)):
        super().__init__()
        self.a = a; self.b = b

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        x1, x2, x3 = x.split([1, 1, 1], dim=1)
        a1, a2, a3 = self.a
        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def vol(self, x: Tensor, t: Tensor) -> Tensor:
        x1, x2, x3 = x.split([1, 1, 1], dim=1)
        b1, b2, b3 = self.b
        return torch.cat([x1 * b1, x2 * b2, x3 * b3], dim=1)

@torch.no_grad()
def solve_sde(
    sde: StochasticLorenzSDE,
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5
    path = [z]
    for t in tt:
        f, g = sde.drift(z, t), sde.vol(z, t)
        z = z + f * dt + torch.randn_like(z) * g * dt_2
        path.append(z)
    return torch.stack(path)  # [T, B, D]

@torch.no_grad()
def gen_data(
    batch_size: int,
    ts: float,
    tf: float,
    n_steps: int,
    noise_std: float,
    n_inner_steps: int = 100
) -> Tuple[Tensor, Tensor]:
    sde = StochasticLorenzSDE().to(device)
    z0 = torch.randn(batch_size, 3, device=device)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)
    zs = zs[::n_inner_steps]             # downsample to observation grid
    zs = zs.permute(1, 0, 2)             # [B,T,3]

    mean = zs.mean(dim=(0, 1), keepdim=True)
    std  = zs.std (dim=(0, 1), keepdim=True).clamp_min(1e-6)
    xs = (zs - mean) / std + noise_std * torch.randn_like(zs)

    tgrid = torch.linspace(ts, tf, n_steps + 1, device=device)[None, :, None].repeat(batch_size, 1, 1)
    return xs, tgrid


# ------------------------------------------------------------
# OU path prior with exact path density & sampling (diagonal)
# ------------------------------------------------------------
class DiagOUSDE(nn.Module):
    def __init__(self, D: int, init_mu: float = 0.0, init_logk: float = -0.7, init_logs: float = -0.7):
        super().__init__()
        self.mu        = nn.Parameter(torch.full((D,), init_mu))
        self.log_kappa = nn.Parameter(torch.full((D,), init_logk))
        self.log_sigma = nn.Parameter(torch.full((D,), init_logs))

    def _params(self):
        kappa = F.softplus(self.log_kappa) + 1e-6
        sigma = F.softplus(self.log_sigma) + 1e-6
        mu    = self.mu
        return mu, kappa, sigma

    @staticmethod
    def _A_Q(kappa: Tensor, sigma: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        A = torch.exp(-kappa * dt)
        two_k_dt = 2.0 * kappa * dt
        small = (two_k_dt < 1e-6).to(dt.dtype)
        Q_exact  = (sigma**2) * (1.0 - torch.exp(-two_k_dt)) / (2.0 * kappa).clamp_min(1e-12)
        Q_taylor = (sigma**2) * dt * (1.0 - kappa * dt + (two_k_dt**2)/6.0)
        Q = small * Q_taylor + (1.0 - small) * Q_exact
        return A, Q

    @torch.no_grad()
    def sample_path(self, ts: Tensor, batch_size: int) -> Tensor:
        ts = ts.view(-1)
        mu, kappa, sigma = self._params()
        D = mu.numel()
        T = ts.numel()
        y = torch.empty(batch_size, T, D, device=ts.device)
        var0 = (sigma**2) / (2.0 * kappa)
        y[:, 0, :] = mu + torch.randn(batch_size, D, device=ts.device) * var0.sqrt()
        for k in range(T - 1):
            dt = (ts[k + 1] - ts[k]).clamp_min(1e-6)
            A, Q = self._A_Q(kappa, sigma, dt)
            mean = mu + A * (y[:, k, :] - mu)
            y[:, k + 1, :] = mean + torch.randn_like(mean) * Q.sqrt()
        return y

    def path_log_prob(self, y: Tensor, ts: Tensor) -> Tensor:
        B, T, D = y.shape
        if ts.dim() == 1: ts = ts[None, :, None].expand(B, -1, 1)
        if ts.dim() == 2: ts = ts[None, :, :].expand(B, -1, -1)
        mu, kappa, sigma = self._params()
        mu = mu.view(1,1,D); kappa = kappa.view(1,1,D); sigma = sigma.view(1,1,D)
        var0 = (sigma**2) / (2.0 * kappa)
        lp0 = gauss_logprob(y[:, 0, :], mu[:, 0, :], var0[:, 0, :])
        dt = (ts[:, 1:, :] - ts[:, :-1, :]).clamp_min(1e-6)               # [B,T-1,1]
        A, Q = self._A_Q(kappa, sigma, dt)                                 # [B,T-1,D]
        mean = mu + A * (y[:, :-1, :] - mu)                                # [B,T-1,D]
        lp_trans = gauss_logprob(y[:, 1:, :], mean, Q).sum(dim=1)          # [B]
        return (lp0 + lp_trans).float()


# ------------------------------------------------------------
# Time features
# ------------------------------------------------------------
class TimeEmbed(nn.Module):
    def __init__(self, model_dim: int = 128, max_freq: float = 16.0):
        super().__init__()
        assert model_dim % 2 == 0
        freqs = torch.exp(torch.linspace(0., math.log(max_freq), model_dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        t = t.to(self.freqs.dtype)                    # [B,T,1]
        ang = t * self.freqs[None, None, :]          # [B,T,D/2]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, zero_last: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        if zero_last:
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def bounded_exp_scale(raw_scale: Tensor, scale_factor: float) -> Tuple[Tensor, Tensor]:
    log_s = torch.tanh(raw_scale) * scale_factor
    return torch.exp(log_s), log_s


# ------------------------------------------------------------
# Invertible blocks (self-contained)
# ------------------------------------------------------------
class ReversePermute(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.register_buffer("perm", torch.arange(D - 1, -1, -1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x.index_select(dim=-1, index=self.perm), torch.zeros(x.size(0), device=x.device)

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        inv = torch.argsort(self.perm)
        return z.index_select(dim=-1, index=inv), torch.zeros(z.size(0), device=z.device)


class AffineCoupling(nn.Module):
    """
    Per-time affine coupling conditioned on:
      - x_a (frozen split),
      - sinusoidal time features,
      - latent path c_t (OU VAE latent).
    """
    def __init__(self, D: int, cond_dim: int, model_dim: int, scale_factor: float = 3.0):
        super().__init__()
        assert D >= 2
        self.D = D
        self.d_a = D // 2
        self.d_b = D - self.d_a
        self.scale_factor = scale_factor

        self.x_proj   = nn.Linear(self.d_a, model_dim)
        self.t_embed  = TimeEmbed(model_dim)
        self.t_mlp    = MLP(model_dim, model_dim, model_dim, zero_last=False)
        self.c_proj   = nn.Linear(cond_dim, model_dim) if cond_dim > 0 else None
        in_feats      = 3 * model_dim if cond_dim > 0 else 2 * model_dim
        self.out_mlp  = MLP(in_feats, model_dim, 2 * self.d_b, zero_last=True)

        nn.init.trunc_normal_(self.x_proj.weight, std=0.02); nn.init.zeros_(self.x_proj.bias)

    def _features(self, x_a: Tensor, t: Tensor, c: Optional[Tensor]) -> Tensor:
        f_x = self.x_proj(x_a)                 # [B,T,mdim]
        f_t = self.t_mlp(self.t_embed(t))      # [B,T,mdim]
        if self.c_proj is not None and c is not None:
            f_c = self.c_proj(c)               # [B,T,mdim]
            return torch.cat([f_x, f_t, f_c], dim=-1)
        else:
            return torch.cat([f_x, f_t], dim=-1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        x_a, x_b = x.split([self.d_a, self.d_b], dim=-1)
        feat = self._features(x_a, t, c)
        params = self.out_mlp(feat)                            # [B,T,2*d_b]
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        z_b = (x_b + bias) * scale
        z = torch.cat([x_a, z_b], dim=-1)
        ldj = log_s.sum(dim=-1).sum(dim=1)                     # sum over d_b and time
        return z, ldj

    def inverse(self, z: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        z_a, z_b = z.split([self.d_a, self.d_b], dim=-1)
        feat = self._features(z_a, t, c)
        params = self.out_mlp(feat)
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        x_b = (z_b / scale) - bias
        x = torch.cat([z_a, x_b], dim=-1)
        ldj = -log_s.sum(dim=-1).sum(dim=1)
        return x, ldj


class FlowSequence(nn.Module):
    def __init__(self, D: int, cond_dim: int, model_dim: int, depth: int, scale_factor: float = 3.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(AffineCoupling(D, cond_dim, model_dim, scale_factor=scale_factor))
            blocks.append(ReversePermute(D))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(x.size(0), device=x.device)
        for b in self.blocks:
            if isinstance(b, AffineCoupling):
                x, ldj_i = b(x, t, c)
            else:
                x, ldj_i = b(x)
            ldj = ldj + ldj_i
        return x, ldj

    def inverse(self, z: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(z.size(0), device=z.device)
        for b in reversed(self.blocks):
            if isinstance(b, AffineCoupling):
                z, ldj_i = b.inverse(z, t, c)
            else:
                z, ldj_i = b.inverse(z)
            ldj = ldj + ldj_i
        return z, ldj

class SoftTimeAttention(nn.Module):
    """
    Given GRU outputs h_seq[B,T,H] and query times t[B,T,1],
        w_{k,j} ∝ exp( - (L * (t_k - t_j))^2 / τ^2 )
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, h_seq: Tensor, t: Tensor) -> Tensor:
        _, T, _ = h_seq.shape
        # Pairwise time differences: [B,T,T]
        delta = (t.unsqueeze(2) - t.unsqueeze(1)).squeeze(-1)
        logits = - (T * delta)**2 / (self.temperature.clamp_min(1e-6) ** 2)
        w = self.sm(logits)   
        return w @ h_seq                       # [B,T,T] @ [B,T,H] -> [B,T,H]


# OU-VAE posterior q_phi(c|x) 
class PosteriorOUEncoder(nn.Module):
    """
    tldr: 

    OU posterior q_phi(c | x) using soft time-local attention:
      - GRU -> h_seq[B,T,H], h_global[B,H]
      - compute attn_k = Σ_j w_{k,j} * h_seq_j around each t_k
      - build ctx_k = concat(h_global + attn_k, t_k)
      - produce per-step OU params (mu_k, kappa_k, sigma_k) from ctx_k
      - q(c_0 | x): from h_global + attn_0
    """
    def __init__(self, x_dim: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.cond_dim = cond_dim
        self.gru = nn.GRU(input_size=x_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.attn = SoftTimeAttention()

        # Initial (c0) params from global + local(t0)
        self.init_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * cond_dim)    # m0, logs0
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3 * cond_dim)    # mu_k, logk_k, logs_k
        )

        nn.init.zeros_(self.init_head[-1].weight); nn.init.zeros_(self.init_head[-1].bias)
        nn.init.zeros_(self.step_head[-1].weight); nn.init.zeros_(self.step_head[-1].bias)

    @staticmethod
    def _A_Q(kappa: Tensor, sigma: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        A = torch.exp(-kappa * dt)
        two_k_dt = 2.0 * kappa * dt
        small = (two_k_dt < 1e-6).to(dt.dtype)
        Q_exact  = (sigma**2) * (1.0 - torch.exp(-two_k_dt)) / (2.0 * kappa).clamp_min(1e-12)
        Q_taylor = (sigma**2) * dt * (1.0 - kappa * dt + (two_k_dt**2)/6.0)
        Q = small * Q_taylor + (1.0 - small) * Q_exact
        return A, Q

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample c ~ q(c|x) and compute log q(c|x).
        x: [B,T,X], t: [B,T,1]
        returns: (c: [B,T,C], log_q: [B])
        """
        h_seq, h_last = self.gru(x)                    # [B,T,H], [1,B,H]
        h_global = h_last[0]                           # [B,H]

        # Time-local pooled features around each query time t_k
        attn = self.attn(h_seq, t)                     # [B,T,H]

        h0_ctx = h_global + attn[:, 0, :]              # [B,H]
        m0, logs0 = self.init_head(h0_ctx).chunk(2, dim=-1)
        s0 = F.softplus(logs0) + 1e-6

        # Per-step OU parameters from ctx_k = concat(h_global + attn_k, t_k)
        ctx = torch.cat([h_global.unsqueeze(1) + attn, t], dim=-1)    # [B,T,H+1]
        params = self.step_head(ctx[:, :-1, :])                        # [B,T-1,3C]
        mu_k, logk_k, logs_k = params.chunk(3, dim=-1)
        kappa_k = F.softplus(logk_k) + 1e-6
        sigma_k = F.softplus(logs_k) + 1e-6

        dt = (t[:, 1:, :] - t[:, :-1, :]).clamp_min(1e-6)             # [B,T-1,1]
        A_k, Q_k = self._A_Q(kappa_k, sigma_k, dt)                     # [B,T-1,C]
        c, log_q = self.ou_posterior(m0, s0, mu_k, A_k, Q_k)

        return c, log_q


    def ou_posterior(
        self,
        m0: torch.Tensor,           # [B,C]
        s0: torch.Tensor,           # [B,C]  (std)
        mu_k: torch.Tensor,         # [B,T-1,C]
        A_k: torch.Tensor,          # [B,T-1,C]
        Q_k: torch.Tensor,          # [B,T-1,C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns c: [B,T,C], log_q: [B]
        Reparameterized, fully vectorized over time.
        """
        B, Tm1, C = mu_k.shape
        T = Tm1 + 1
        device = mu_k.device
        dtype  = mu_k.dtype

        # Sample c0 ~ N(m0, s0^2) 
        eps0 = torch.randn(B, C, device=device, dtype=dtype)
        c0   = m0 + s0 * eps0
        log_q0 = -0.5 * (((c0 - m0) / s0.clamp_min(1e-12))**2 + (s0.clamp_min(1e-12)).log()*2 + math.log(2*math.pi)).sum(-1)

        #  innovations e_k ~ N(0, Q_k)
        eps = torch.randn(B, Tm1, C, device=device, dtype=dtype)
        noise = eps * Q_k.clamp_min(1e-12).sqrt()                    # [B,T-1,C]

        # b_k = (1 - A_k) * mu_k + noise
        b = (1.0 - A_k) * mu_k + noise                               # [B,T-1,C]

        # Phi_full: [B,T,C], Phi_0=1, Phi_k = prod_{i=0}^{k-1} A_i
        one = torch.ones(B, 1, C, device=device, dtype=dtype)
        Phi_prefix = torch.cumprod(A_k, dim=1)                       # [B,T-1,C]
        Phi_full = torch.cat([one, Phi_prefix], dim=1)               # [B,T,C]

        # S_full: S_0=0, S_k = sum_{j=0}^{k-1} b_j / Phi_{j+1}
        denom = Phi_full[:, 1:, :].clamp_min(1e-20)                  # [B,T-1,C]
        s = b / denom                                                # [B,T-1,C]
        zero = torch.zeros(B, 1, C, device=device, dtype=dtype)
        S_prefix = torch.cumsum(s, dim=1)                            # [B,T-1,C]
        S_full = torch.cat([zero, S_prefix], dim=1)                  # [B,T,C]

        # c_k = Phi_k * (c0 + S_k)
        c = Phi_full * (c0[:, None, :] + S_full)                     # [B,T,C]

        # transition log-prob
        mean_trans = mu_k + A_k * (c[:, :-1, :] - mu_k)              # [B,T-1,C]
        var_trans  = Q_k.clamp_min(1e-12)                            # [B,T-1,C]
        log_q_trans = -0.5 * (
            ((c[:, 1:, :] - mean_trans)**2 / var_trans)
            + var_trans.log()
            + math.log(2.0 * math.pi)
        ).sum(-1).sum(-1)                                            # [B]

        log_q = log_q0 + log_q_trans                                  # [B]
        return c, log_q


class PathFlowOUVAE(nn.Module):
    """
    Generative model:
      c ~ OU_c                      (latent conditioning path)
      y ~ OU_y                      (latent base path)
      x = g^{-1}( y ; t, c )
    wrt conditional likelihood: log p(x | c) = log p(y) + sum log|det J|.
    ELBO: E_q(c|x)[ log p(x|c) + log p(c) - log q(c|x) ].
    """
    def __init__(self, x_dim: int, cond_dim: int, model_dim: int, depth: int):
        super().__init__()

        self.temporal_flow = FlowSequence(D=x_dim, cond_dim=cond_dim, model_dim=model_dim, depth=depth, scale_factor=3.0)

        self.prior_y = DiagOUSDE(D=x_dim)
        self.prior_c = DiagOUSDE(D=cond_dim)

        self.posterior_c = PosteriorOUEncoder(x_dim=x_dim, cond_dim=cond_dim, hidden=model_dim)

    def log_p_x_given_c(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        y,  ldj_tmp = self.temporal_flow(x, t, c)    # z1 -> y
        log_p_y = self.prior_y.path_log_prob(y, t)    # OU path density
        return log_p_y + ldj_tmp            # [B]

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        c, log_q = self.posterior_c(x, t)             # [B,T,C], [B]
        log_px_c = self.log_p_x_given_c(x, t, c)      # [B]
        log_pc   = self.prior_c.path_log_prob(c, t)   # [B]
        elbo = log_px_c + log_pc - log_q
        return -elbo.mean()

    @torch.no_grad()
    def sample(self, n_samples: int, steps: int = 40) -> Tensor:
        ts = torch.linspace(0, 1, steps=steps, device=device)     # [T]
        tB = ts[None, :, None].repeat(n_samples, 1, 1)            # [B,T,1]
        c  = self.prior_c.sample_path(ts, n_samples)              # [B,T,C]
        y  = self.prior_y.sample_path(ts, n_samples)              # [B,T,D]
        x, _ = self.temporal_flow.inverse(y, tB, c)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2 ** 10
    ts_ = 0.0
    tf_ = 1.0
    n_steps = 40
    noise_std = 0.01

    xs, ts_arr = gen_data(batch_size, ts_, tf_, n_steps, noise_std)  # xs:[B,T,3], ts_arr:[B,T,1]
    xs = xs.to(device); ts_arr = ts_arr.to(device)

    x_dim    = 3
    cond_dim = 16         # latent conditioning path dimensionality
    model_dim= 128
    depth    = 6

    model = PathFlowOUVAE(x_dim=x_dim, cond_dim=cond_dim, model_dim=model_dim, depth=depth).to(device)

    # Train
    iters = 10000
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    pbar = trange(iters)
    for step in pbar:
        loss = model(xs, ts_arr)
        pbar.set_description(f"{loss.item():.4f}")
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        if step % 100 == 0:
            with torch.no_grad():
                samples = model.sample(6, steps=n_steps + 1).cpu()
                visualise_data(samples, 'x_t.jpg')

    # Final samples
    with torch.no_grad():
        samples = model.sample(6, steps=n_steps + 1).cpu()
        visualise_data(samples, 'x_t_final.jpg')
