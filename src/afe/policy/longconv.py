"""LongConv trading policy: residual window -> residual-space portfolio weight.

Appendix A of Epstein et al. (2025) and Fu et al. (2023).  For one asset with hidden
state u in R^{d x L} (L = residual lookback) the layer computes

    y = Kbar * u + D (.) u,      Kbar = sign(K) (.) max(|K| - lambda_squash, 0),

with K in R^{d x L} initialised as  K[h, t] = x exp(-t/L (d/2)^{h/d}),  x ~ N(0, 1).
The weight is read at the last position of the window, so the causal convolution reduces
to a weighted sum over the window per hidden channel; no FFT is needed at L = 30 (the
FFT form in the paper is an implementation of the same operator, not a different one).

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


def geometric_decay_init(d: int, L: int) -> torch.Tensor:
    h = torch.arange(d, dtype=torch.float32)[:, None]
    t = torch.arange(L, dtype=torch.float32)[None, :]
    decay = torch.exp(-t / L * (d / 2) ** (h / d))
    return torch.randn(d, L) * decay


class LongConvLayer(nn.Module):
    def __init__(self, d: int, L: int, dropout: float, lambda_squash: float):
        super().__init__()
        self.kernel = nn.Parameter(geometric_decay_init(d, L))   # K[h, t]: weight on lag t
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


class LongConvPolicy(nn.Module):
    """Table 4: one layer, hidden 32, dropout 0.1, lambda_squash 0.001, lookback 30.
    Reading the layer at the last position only is exact for one layer; a deeper stack
    would need the full sequence through each layer, so it is not offered here."""

    def __init__(self, hidden: int = 32, lookback: int = 30, dropout: float = 0.1,
                 lambda_squash: float = 1e-3, layers: int = 1):
        super().__init__()
        if layers != 1:
            raise NotImplementedError("LongConvPolicy reads the last position only; layers must be 1")
        self.encoder = nn.Linear(1, hidden)
        self.layer = LongConvLayer(hidden, lookback, dropout, lambda_squash)
        self.decoder = nn.Linear(hidden, 1)
        self.lookback = lookback

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (..., L) residual windows, oldest first.  Returns (...) weights."""
        shape = windows.shape[:-1]
        x = windows.reshape(-1, self.lookback)
        y = self.layer(x, self.encoder.weight[:, 0], self.encoder.bias)
        return self.decoder(y).reshape(shape)
