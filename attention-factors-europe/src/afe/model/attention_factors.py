"""Attention factors (Epstein, Wang, Choi & Pelger 2025, Section 3.2), universe-masked.

Equation (1) of the paper, in the slot layout of model/pca_factors.py and
policy/trading.py: on each date the universe fills slots 0..n_t-1 of a fixed-width
(T, S) array and the rest is padding.

    Xtilde_t = X_t W_K                                   (T, S, d)
    scores   = Q Xtilde_t' / sqrt(d)                     (T, K, S)
    w_F      = Softmax(scores, over assets)              (T, K, S), rows sum to 1
    beta'    = w_F' (w_F w_F' + lambda_ridge I_K)^-1     (T, S, K)
    eps_t    = R_t - beta' (w_F R_t)                     low rank, no (S, S) matrix

Why the mask matters. The softmax normalises across assets, so a padded slot left in
the sum takes a share of the row's unit budget away from the names that are there.
Zeroing w_F after the softmax is not equivalent: it leaves the surviving weights too
small and no longer summing to one. Scores on padded slots are therefore set to the
smallest representable value before the softmax, so each row renormalises over the
tradable names only.

`torch.finfo(dtype).min` rather than -inf: a date with nothing tradable (warm-up, a
calendar gap) would give a row of all -inf, whose softmax is NaN and whose gradient is
NaN. With a finite floor such a row comes back uniform and the multiplication by the
mask zeroes it, so the date simply does not trade.

Two things this module does NOT do. Characteristics that are missing for a name that is
in the universe must be imputed upstream; padding is zero-filled here, but a NaN on a
tradable slot propagates through the embedding into every score of that date. And the
lag is the caller's: w_F for date t must be built from characteristics known at t-1, so
pass an X that is already shifted.

The forward pass follows the prototype in scripts/bench_epstein.py; what is added here
is the mask, the slot layout and the NaN guard.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionFactors(nn.Module):
    """Characteristics -> factor portfolio weights and loadings, masked to the universe.

    Args:
        n_features: M, characteristics per asset.
        n_factors: K, number of attention factors (30 in the paper's headline).
        embedding_dim: d, embedding width (32 in Table 4).
        lambda_ridge: ridge penalty in the loading solve.
    """

    def __init__(self, n_features: int, n_factors: int = 30, embedding_dim: int = 32,
                 lambda_ridge: float = 1e-4):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(n_features, embedding_dim) / math.sqrt(n_features))
        self.Q = nn.Parameter(torch.randn(n_factors, embedding_dim) / math.sqrt(embedding_dim))
        self.d = embedding_dim
        self.K = n_factors
        self.lambda_ridge = lambda_ridge

    def factor_weights(self, X: torch.Tensor, tradable: torch.Tensor) -> torch.Tensor:
        """X: (T, S, M), tradable: (T, S) bool. Returns w_F: (T, K, S), zero on padding."""
        if X.shape[:2] != tradable.shape:
            raise ValueError(f"X {tuple(X.shape)} and tradable {tuple(tradable.shape)} disagree")
        m = tradable.unsqueeze(-1)
        Xz = torch.where(m, X, torch.zeros_like(X))           # padding may hold NaN
        scores = torch.einsum("kd,tsd->tks", self.Q, Xz @ self.W_K) / math.sqrt(self.d)
        keep = tradable.unsqueeze(1)                          # (T, 1, S)
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1) * keep.to(scores.dtype)

    def loadings(self, w_F: torch.Tensor) -> torch.Tensor:
        """beta': (T, S, K) from w_F: (T, K, S), with the ridge penalty of Section 3.2."""
        gram = w_F @ w_F.transpose(1, 2)
        eye = torch.eye(self.K, dtype=gram.dtype, device=gram.device)
        return torch.linalg.solve(gram + self.lambda_ridge * eye, w_F).transpose(1, 2)

    def forward(self, X: torch.Tensor, tradable: torch.Tensor, R: torch.Tensor):
        """Returns (w_F, betaT, eps) of shapes (T, K, S), (T, S, K), (T, S)."""
        w_F = self.factor_weights(X, tradable)
        betaT = self.loadings(w_F)
        Rz = torch.where(tradable, R, torch.zeros_like(R))
        factors = torch.einsum("tks,ts->tk", w_F, Rz)
        eps = Rz - torch.einsum("tsk,tk->ts", betaT, factors)
        return w_F, betaT, eps * tradable.to(eps.dtype)


def compose(w_port: torch.Tensor, w_F: torch.Tensor, betaT: torch.Tensor) -> torch.Tensor:
    """Residual-space weights to asset-space weights, w = w_eps' w_port, low rank.

    Not L1-normalised: policy/trading.py owns the normalisation and the costs.
    """
    bw = torch.einsum("tsk,ts->tk", betaT, w_port)
    return w_port - torch.einsum("tks,tk->ts", w_F, bw)
