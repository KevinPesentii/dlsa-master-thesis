"""Attention factors (Epstein, Wang, Choi & Pelger 2025, Section 3.2), universe-masked.

Equation (1) of the paper, with a scalar score temperature (see calibrate_temperature),
in the slot layout of model/pca_factors.py and
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

Why X is centred on each date's tradable names. A column that is the same for every
name on a date (the cross-sectional medians med_*, rf) adds the same amount to every
score of a factor, which the softmax over names ignores, so centring changes nothing in
exact arithmetic; the same holds for the mean of any column. In float32 it matters: the
medians are raw levels (med_Vol up to 3.4e7 in 2008-2011), so X @ W_K carried an offset
of 1e6-1e7 whose rounding step (0.06-4) swamped the +-0.5 rank signal and made w_F jitter
from day to day with the level. Measured on the US panel (2026-09-23): October 2008 with
sharpened weights, raw vs zeroed medians moved w_F by 0.71 in L1 per factor-day; in
trained models by 0.11-0.13, with 80% more day-to-day factor-weight change and daily
turnover 0.79 against 0.66 (2008). Under this linear embedding those columns carry no
information anyway; an embedding meant to use the levels would need them rescaled.

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
                 lambda_ridge: float = 1e-4, level_hidden: int = 0):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(n_features, embedding_dim) / math.sqrt(n_features))
        self.Q = nn.Parameter(torch.randn(n_factors, embedding_dim) / math.sqrt(embedding_dim))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.d = embedding_dim
        self.K = n_factors
        self.lambda_ridge = lambda_ridge
        self.level = None
        if level_hidden:
            # Level conditioning, NOT in the paper. The paper adds the cross-sectional medians
            # and the risk-free rate to X "in order to keep the level information", but a column
            # that is the same for every asset adds the same constant to every entry of a score
            # row, and the row-wise softmax drops constants: measured here, zeroing those 40
            # columns moves the factor weights by 1e-9 and their rows of W_K get a gradient
            # 2e5 times smaller than the rank columns. Level information cannot reach the model
            # through the keys.
            #
            # It can through the QUERIES. A score is <q_k, key_i>, so shifting q_k by the same
            # vector changes each asset's score differently, because the keys differ: the shift
            # survives the softmax. So q_k(t) = q_k + g(z_t), with z_t the cross-sectional mean
            # of X on that date (the level of every characteristic, which is exactly what the
            # medians carry; the rank columns average to about zero and contribute little) and g
            # shared across factors. The last layer starts at zero, so the model starts out
            # identical to Equation (1) and learns the deviation.
            self.level = nn.Sequential(
                nn.Linear(n_features, level_hidden), nn.GELU(),
                nn.Linear(level_hidden, embedding_dim))
            nn.init.zeros_(self.level[2].weight)
            nn.init.zeros_(self.level[2].bias)

    def calibrate_temperature(self, X: torch.Tensor, tradable: torch.Tensor,
                              target_std: float = 1.0, max_dates: int = 200) -> float:
        """Set the temperature so the scores start with a cross-sectional spread of target_std.

        Softmax over 500 names needs score differences of order one to concentrate weight; at
        initialisation the spread here is around 0.03, because the characteristics are rank
        quantiles in [-0.5, 0.5] and the scores are divided by sqrt(d). The softmax is then
        flat, every factor is the equal-weighted universe, and the gradient that would sharpen
        it has to grow K*d + M*d parameters coherently from an almost flat surface.

        tau multiplies the scores, so it is exactly equivalent to rescaling Q: the model class
        of Equation (1) is unchanged. What changes is the parameterisation, and a single scalar
        collects the sharpening signal from every factor and every asset at once.

        MUST be called on training dates only, never on the dates being traded.
        """
        with torch.no_grad():
            step = max(1, X.shape[0] // max_dates)
            Xs, ms = X[::step], tradable[::step]
            m = ms.unsqueeze(-1)
            Xz = torch.where(m, Xs, torch.zeros_like(Xs))
            n = m.sum(dim=1, keepdim=True).clamp_min(1).to(Xz.dtype)
            lv = Xz.sum(dim=1) / n.squeeze(1)
            Xz = torch.where(m, Xz - lv.unsqueeze(1), torch.zeros_like(Xz))
            raw = torch.einsum("kd,tsd->tks", self.Q, Xz @ self.W_K) / math.sqrt(self.d)
            keep = ms.unsqueeze(1).expand_as(raw)
            spread = raw[keep].std()
            self.log_tau.fill_(math.log(target_std / max(float(spread), 1e-12)))
        return float(self.log_tau.detach().exp())

    def factor_weights(self, X: torch.Tensor, tradable: torch.Tensor) -> torch.Tensor:
        """X: (T, S, M), tradable: (T, S) bool. Returns w_F: (T, K, S), zero on padding."""
        if X.shape[:2] != tradable.shape:
            raise ValueError(f"X {tuple(X.shape)} and tradable {tuple(tradable.shape)} disagree")
        m = tradable.unsqueeze(-1)
        Xz = torch.where(m, X, torch.zeros_like(X))           # padding may hold NaN
        n = m.sum(dim=1, keepdim=True).clamp_min(1).to(Xz.dtype)
        levels = Xz.sum(dim=1) / n.squeeze(1)                                  # (T, M) date levels
        Xz = torch.where(m, Xz - levels.unsqueeze(1), torch.zeros_like(Xz))  # centre per date
        keys = Xz @ self.W_K
        if self.level is None:
            scores = torch.einsum("kd,tsd->tks", self.Q, keys) / math.sqrt(self.d)
        else:
            q = self.Q.unsqueeze(0) + self.level(levels).unsqueeze(1)          # (T, K, d)
            scores = torch.einsum("tkd,tsd->tks", q, keys) / math.sqrt(self.d)
        scores = scores * self.log_tau.exp()
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
