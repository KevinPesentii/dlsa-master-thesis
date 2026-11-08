"""LongConv trading policy: residual window -> residual-space portfolio weight.

Appendix A of Epstein et al. (2025) and Fu et al. (2023).  For one asset with hidden
state u in R^{d x L} (L = residual lookback) the layer computes

    y = Kbar * u + D (.) u,      Kbar = sign(K) (.) max(|K| - lambda_squash, 0),

with K in R^{d x L} initialised as  K[h, t] = x exp(-t/L (d/2)^{h/d}),  x ~ N(0, 1)
(Table 4's "Geom Decay"; kernel_init "random" leaves the decay out, K[h, t] = x).
With one layer the weight is read at the last position of the window, so the causal
convolution reduces to a weighted sum over the window per hidden channel; no FFT is needed
at L = 30 (the FFT form in the paper is an implementation of the same operator, not a
different one). A deeper stack (Table 4's "Nr layers") runs every layer but the last at all
L positions, through a (d, L, L) Toeplitz matrix, and reads the last layer at the last
position only.

Block structure as in Fu et al.'s LongConv layer: GELU, dropout, then a GLU output
projection, with a residual connection; a linear encoder (1 -> d) in front and a linear
decoder (d -> 1) behind.  There is no separate allocation head: the decoder output is
the residual-space weight w_port.  Nothing here looks across assets or across dates:
the same map is applied to every (date, asset) window, so the module is a pure function
of eps_{i, t-L+1..t}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

KERNEL_INITS = ("geom_decay", "random")
CHUNK = 1024      # windows per piece of a deeper stack (fastest on CPU, 2026-11-08)
KEEP = 2 ** 27    # hidden values (windows x L x d x extra layers) above which a training step recomputes
                  # each piece in backward instead of keeping it: ~4 GB kept at most, +35-50% time
GPU_CHUNK = 8192   # on a GPU: tiny pieces leave it idle between kernel launches
GPU_KEEP = 2 ** 26  # and half the CPU's memory kept, since several workers share one GPU (neither measured yet)


def pieces(device: torch.device) -> tuple[int, int]:
    """(windows per piece, hidden values kept) of a deeper stack on `device`. Neither changes a number:
    the pieces are independent windows, and a recomputed piece replays its dropout masks."""
    return (CHUNK, KEEP) if device.type == "cpu" else (GPU_CHUNK, GPU_KEEP)


def geometric_decay_init(d: int, L: int) -> torch.Tensor:
    h = torch.arange(d, dtype=torch.float32)[:, None]
    t = torch.arange(L, dtype=torch.float32)[None, :]
    decay = torch.exp(-t / L * (d / 2) ** (h / d))
    return torch.randn(d, L) * decay


class LongConvLayer(nn.Module):
    def __init__(self, d: int, L: int, dropout: float, lambda_squash: float, init: str = "geom_decay"):
        super().__init__()
        if init not in KERNEL_INITS:
            raise ValueError(f"policy.kernel_init {init!r}: one of {KERNEL_INITS}")
        k = geometric_decay_init(d, L) if init == "geom_decay" else torch.randn(d, L)
        self.kernel = nn.Parameter(k)                              # K[h, t]: weight on lag t
        self.skip = nn.Parameter(torch.randn(d))                   # D
        self.out = nn.Linear(d, 2 * d)                             # GLU projection
        self.drop = nn.Dropout(dropout)
        self.lam = lambda_squash

    def squashed_kernel(self) -> torch.Tensor:
        return torch.sign(self.kernel) * torch.clamp(self.kernel.abs() - self.lam, min=0.0)

    def forward(self, x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L) windows, oldest first; a, c: the affine encoder u[b, l, h] = a_h x[b, l] + c_h.
        With the encoder affine, the causal convolution read at the last position is
        sum_l Kbar[h, L-1-l] u[b, l, h] = a_h (x Kbar_flipped')[b, h] + c_h sum_l Kbar[h, l],
        so the (B, L, d) hidden state never has to be materialised."""
        k = self.squashed_kernel()
        u_last = a * x[:, -1:] + c                                 # (B, d)
        y = (x @ k.flip(-1).T) * a + c * k.sum(-1) + self.skip * u_last
        y = self.drop(F.gelu(y))
        y = F.glu(self.out(y), dim=-1)
        return y + u_last                                          # residual connection

    def forward_seq(self, u: torch.Tensor) -> torch.Tensor:
        """u: (d, B, L) hidden states, channels first, oldest day last -> the layer's output at every
        position t: y[h, b, t] = sum_{s <= t} Kbar[h, t - s] u[h, b, s] + D_h u[h, b, t] (nothing
        before the window's first day), then GELU, dropout, GLU and the residual as in forward.
        Channels first, the convolution is one batched matmul over channels and the GLU projection
        one matmul, with no layout copies."""
        d, L = u.shape[0], u.shape[-1]
        steps = torch.arange(L, device=u.device)
        lag = steps[None, :] - steps[:, None]                                 # [s, t] = t - s
        toeplitz = self.squashed_kernel()[:, lag.clamp(min=0)] * (lag >= 0)   # (d, L, L): [h, s, t]
        y = self.drop(F.gelu(torch.baddbmm(self.skip[:, None, None] * u, u, toeplitz)))
        z = torch.addmm(self.out.bias[:, None], self.out.weight, y.reshape(d, -1))   # (2d, B * L)
        return F.glu(z, dim=0).reshape(u.shape) + u

    def forward_last(self, u: torch.Tensor) -> torch.Tensor:
        """forward_seq(u)[..., -1].T without the other positions: (d, B, L) -> (B, d)."""
        k = self.squashed_kernel().flip(-1)[:, :, None]                       # (d, L, 1)
        y = (torch.bmm(u, k).squeeze(-1) + self.skip[:, None] * u[..., -1]).T
        return F.glu(self.out(self.drop(F.gelu(y))), dim=-1) + u[..., -1].T


class LongConvPolicy(nn.Module):
    """Table 4: one layer, hidden 32, dropout 0.1, lambda_squash 0.001, lookback 30, geometric
    decay init. One layer is read at the last position only (forward, exact). With `layers` > 1
    the first layers run over the whole window (forward_seq) and the last one is read at the
    last position (forward_last). One layer builds the same modules from the same random
    numbers as before `layers` and `kernel_init` existed: earlier runs and saved models hold."""

    def __init__(self, hidden: int = 32, lookback: int = 30, dropout: float = 0.1,
                 lambda_squash: float = 1e-3, layers: int = 1, kernel_init: str = "geom_decay"):
        super().__init__()
        if layers < 1:
            raise ValueError(f"policy.layers {layers}: at least one")
        self.encoder = nn.Linear(1, hidden)
        self.layer = LongConvLayer(hidden, lookback, dropout, lambda_squash, kernel_init)
        self.deeper = nn.ModuleList(LongConvLayer(hidden, lookback, dropout, lambda_squash, kernel_init)
                                    for _ in range(layers - 1))
        self.decoder = nn.Linear(hidden, 1)
        self.lookback = lookback

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (..., L) residual windows, oldest first.  Returns (...) weights."""
        shape = windows.shape[:-1]
        x = windows.reshape(-1, self.lookback)
        a, c = self.encoder.weight[:, 0], self.encoder.bias
        chunk, keep = pieces(x.device)
        if len(self.deeper) == 0:
            y = self.layer(x, a, c)
        elif torch.is_grad_enabled() and x.numel() * a.shape[0] * len(self.deeper) > keep:
            # recomputed in backward: the same numbers and dropout masks (the RNG state is replayed)
            y = torch.cat([checkpoint(self.stack, xc, a, c, use_reentrant=False) for xc in x.split(chunk)])
        else:
            y = torch.cat([self.stack(xc, a, c) for xc in x.split(chunk)])
        return self.decoder(y).reshape(shape)

    def stack(self, x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """A deeper stack on windows x (B, L): the first layers at every position, the last one at
        the last position. -> (B, d)."""
        u = a[:, None, None] * x + c[:, None, None]                    # (d, B, L)
        for lay in [self.layer, *self.deeper[:-1]]:
            u = lay.forward_seq(u)
        return self.deeper[-1].forward_last(u)
