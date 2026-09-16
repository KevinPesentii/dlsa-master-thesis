# PCA + LongConv: the two-step benchmark of Table 2

Reproduction attempt of the "PCA Factors (Two-Step Approach)" block of Table 2 in Epstein,
Wang, Choi & Pelger (2025): PCA residuals fixed first, LongConv trading policy trained on
the net-Sharpe objective second. Paper reference numbers (K = 1 ... 100): gross SR
2.26-2.79, net SR 1.19-1.57, beta 0.07-0.10.

Status: first build 2026-09-15. Numbers live in `runs/*_pca_longconv_K*/metrics.json`;
nothing in this file is a result.

Build:

    python scripts/build_pca_residuals.py --config configs/us_pca_longconv.yaml   # ~12 min, 5.9 GB
    python scripts/run_pca_longconv.py    --config configs/us_pca_longconv.yaml   # ~20 min per K on 4 CPU threads

Code: `src/afe/model/pca_factors.py` (step one), `src/afe/policy/longconv.py`,
`src/afe/policy/trading.py`, `src/afe/evaluation/metrics.py` (step two).

## What the paper specifies, and what it does not

Specified: PCA on the past 252 trading days; the same LongConv policy as the attention
model (1 layer, hidden 32, dropout 0.1, lambda_squash 0.001, lookback 30); the Sharpe
objective with the 5bp / 1bp cost term; AdamW lr 0.003, weight decay 0.05, 30 epochs;
8-year rolling window refit annually; out-of-sample Jan 1998 - Dec 2021; "essentially a
benchmark in the spirit of [19]" (Guijarro-Ordonez, Pelger & Zanotti).

Not specified, settled here as follows. Each is a config key so it can be varied.

| choice | taken | why |
|---|---|---|
| loadings | OLS of returns on the PCA factors over the same 252-day window, no intercept (`factors.loading_window: 252`) | the paper says only "PCA using the past 252 trading days". GPZ regress on the last 60 days (`60`, built in `data/us/pca`); the PCA projection `B = D V` needs no regression (`0`, built in `data/us/pca_proj`). See the variant grid below |
| K = 100 with a 60-day regression | if `loading_window: 60` is used: numpy `lstsq` returns the minimum-norm solution, as sklearn's `LinearRegression` in the GPZ code does | under-determined (perfect in-sample fit, noisy out of sample) |
| PCA set | universe members with a full 252-day return history and non-zero volatility | GPZ requires the full window; recent IPOs are excluded until they have it (463-496 of 500 names per day) |
| residuals for non-members | computed for every pool name with a full history, using the members' factors | a name entering the universe then already has the 30-day residual lookback; nothing about the factors uses non-members |
| tradable set | PCA set with 30 finite residuals | the policy needs the full window |
| policy input | cumulative residual path over the 30-day window, x 100 (`policy.input: cumulative`, `input_scale`) | GPZ's `preprocess_cumsum`, i.e. the two-step benchmark's own convention; the paper's formula LongConv(eps_{t-s..t-1}) is `policy.input: returns`. A price-level signal changes more slowly than a return signal, which is the whole cost story (grid below). Percent units put the GELU in a non-degenerate regime; the L1-normalised output is invariant to the scale |
| LongConv block | encoder 1->32, kernel + skip, GELU, dropout, GLU projection, residual, decoder 32->1 | Fu et al.'s layer; no layer norm (not mentioned) |
| batching | contiguous 125-day blocks, one Sharpe per block, block order shuffled per epoch (`training.batch_days`; null = whole window) | GPZ's `batch_size: 125`; 30 full-window steps would barely move AdamW at lr 0.003 |
| objective | -(mean(r_net - rf) / sd(r_net)); no explained-variance term | the term is constant once the factors are fixed |
| turnover at refits | charged: weights of consecutive years are concatenated in pool space before costs | the seam is a real trade |
| reported SR | mu / sigma, no rf | Table 2's own rows satisfy this (market 8.61 / 20.37 = 0.42) |
| beta | OLS slope on the daily equal-weighted universe return | the table's market row is the equal-weighted portfolio |

First training window: the paper's data start in Jan 1990 and PCA needs 252 prior days.
The build prepends 1989 from the raw daily files, so the Jan 1990 - Dec 1997 window is
complete. Names that switch primary share class change permno and lose their history
(rare, ~3% of members have more than one class).

## Variant grid that fixed the two defaults (K = 30, OOS years 1998, 2005, 2015 pooled)

Paper hyperparameters throughout; only the two unspecified choices vary. Scratch run of
2026-09-15, seed 0, not a run directory; it is here to show why the defaults are what
they are, and that they were chosen on three out-of-sample years.

| loadings | input | SR | mu % | sigma % | SR net | turnover/day | break-even bp |
|---|---|---|---|---|---|---|---|
| 60-day OLS (GPZ) | returns | 2.06 | 4.5 | 2.2 | -2.06 | 0.62 | 2.1 |
| 60-day OLS (GPZ) | cumulative | 1.89 | 4.4 | 2.3 | -1.30 | 0.49 | 2.5 |
| 252-day OLS | returns | 4.34 | 11.4 | 2.6 | -0.08 | 0.82 | 4.9 |
| 252-day OLS | cumulative | 4.20 | 11.0 | 2.6 | 1.13 | 0.54 | 7.2 |
| projection | returns | 3.97 | 10.4 | 2.6 | -0.34 | 0.80 | 4.6 |
| projection | cumulative | 4.18 | 10.7 | 2.6 | 0.69 | 0.61 | 4.6 |

Two things are visible. With GPZ's 60-day regression at K = 30 the hedge leg
`D^-1 V B' w_port` is 1.3-1.5 x the L1 size of the signal leg (0.35-0.4 x with 252-day
loadings), so after `||w||_1 = 1` most of the book is hedge and the gross mean halves.
And the turnover is set by the policy input, not by the factor model: the composition
alone moves 0.02-0.2 of the book per day, a return-window signal 0.6-0.8, a price-level
signal 0.5-0.6.

## The volatility gap

Table 2's PCA rows have sigma = 5.4-5.9 %; ours are 2.2-2.6 % at K = 30 for the same
`||w||_1 = 1`. The K = 30 book holds ~490 names with an effective number
`1 / sum(w^2)` of about 290. A 5.4 % annual volatility with ~1.5 % daily residual
volatility per name needs an effective number near 20, i.e. a far more concentrated
policy output than a smooth per-asset filter gives after 30 epochs. Since the cost
term is charged on the L1-normalised book, a 2 x more concentrated book with the same
signal roughly doubles the gross mean at the same turnover, which is most of the gap
between our net Sharpe and the paper's. Whether the authors' LongConv output is that
concentrated, or their tradable set is smaller, is not recoverable from the paper.

## Identities checked on the built data

`eps_t = (I - B V' D^-1) R_t` holds to 1e-9 on the PCA set (`pca_factors.composition_check`),
and `R_t' w = eps_t' w_port` for the composed weights (scratch check, not a repo test:
per CLAUDE.md the invariant tests are to be written by hand).

## Time-series explained variance, 1990-1991 (diagnostic, not a result)

K = 3 leaves 74% of daily variance in the residual of a typical member; K = 100 with the
60-day regression leaves 73%, i.e. the extra 97 factors buy almost nothing out of
sample because the loadings are fit on 60 observations. With a 252-day loading window
K = 100 explains far more.

## Open

- The concentration / volatility gap above.
- The input scale (100) and the absence of any input normalisation in the policy.
- Whether the authors exclude non-tradable days differently at the start of a window.
- Seeds: Table 3 reports 0.12-0.17 SR standard deviation across seeds for the attention
  model; run `--seed 1 2 ...` before quoting a single number.
