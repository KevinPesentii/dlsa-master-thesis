# PCA + OU threshold: the parametric benchmark of Table 2

Reproduction attempt of the "Parametric Benchmark, PCA + OU Thresh" block of Table 2 in
Epstein, Wang, Choi & Pelger (2025): the PCA residuals of the two-step benchmark, traded
with the Ornstein-Uhlenbeck threshold rule of Avellaneda & Lee (2010), "the same
implementation as in [19]" (Guijarro-Ordonez, Pelger & Zanotti). Lookback 30 days, PCA on
252 days. Paper reference numbers (K = 1 ... 100): gross SR -0.35 to 1.26, net SR -2.54 to
-7.05, beta 0.00-0.02.

Status: built and run 2026-09-28 (results below). Numbers live in
`runs/*_pca_ou_threshold_K*/metrics.json`; nothing in this file is a result unless it names
its run directory.

    python scripts/build_pca_residuals.py --config configs/us_pca_longconv.yaml   # stage one, shared with PCA + LongConv, ~25 min
    python scripts/run_pca_longconv.py    --config configs/us_pca_ou.yaml         # all 8 K in ~1.5 min, nothing trained
    python scripts/summarize_runs.py --glob "runs/*_pca_ou_threshold_K*" --paper

Code: `src/afe/policy/ou_threshold.py` (signal and rule), the `ou_threshold` branch of
`scripts/run_pca_longconv.py`; stage one and the accounting are the two-step benchmark's
(`src/afe/model/pca_factors.py`, `src/afe/policy/trading.py`, `src/afe/evaluation/metrics.py`).

## The rule

Epstein et al. give no formula; GPZ do (Section III.C, p. 14; Appendix B, p. 54-55). For the
cumulative path x_1..x_L of a residual's last L = 30 daily returns, fit the AR(1)
`x_{l+1} = a + b x_l + e` by OLS. For 0 < b < 1 (a mean-reverting OU process)

    mu = a / (1 - b),   sigma = sqrt(var(e) / (1 - b^2)),   s = (x_L - mu) / sigma,   R2 = corr(x_l, x_{l+1})^2

    w_eps = -1 if s > 1.25 and R2 > 0.25,   +1 if s < -1.25 and R2 > 0.25,   0 otherwise.

GPZ's `preprocess_ou` (`preprocess.py` at the clone root) computes this signal, but feeds it
to their OU + FFN model (`models/OUFFN.py`); the threshold rule itself is not in their public
code. Reusing `preprocess_ou` and OUFFN unchanged would give a different Table 2 row.

| choice | taken | why |
|---|---|---|
| thresholds | c_thresh = 1.25, c_crit = 0.25, fixed | GPZ's benchmark values, "the optimal values in Avellaneda and Lee (2010) and Yeo and Papanicolaou (2017)". GPZ validate over {1, 1.25, 1.5} x {0.25, 0.5, 0.75}; `--c-thresh` / `--c-crit` rerun that grid, not done here |
| memory | none: a position is held only while abs(s) > c_thresh | GPZ's allocation function has no closing band. Avellaneda & Lee open at 1.25 and close at 0.75 / 0.50; Epstein et al. cite them but use GPZ's implementation |
| c_crit | cutoff on the R2 of the AR(1) | GPZ, after Yeo & Papanicolaou; not Avellaneda & Lee's mean-reversion-speed filter |
| sign | short when s > c_thresh (the path is above its mean), long below | GPZ's text: the process "reverts back to its long term mean"; their commented-out signal `(mus - Ys[:,-1])/sigmas` in `preprocess_ou` has the same sign |
| path | cumulative sum of the 30 residuals of the window, ending the day before the trade | `preprocess_ou`; `trading.make_windows(cumulative=True)` builds the same path |
| moments | population moments, 1e-6 added to 1 - b and 1 - b^2 | GPZ's code, kept so the signal is theirs to float precision; the guard only matters for b close to 1 |
| tradable | PCA set with 30 finite residuals and 0 < b < 1 | GPZ: no missing residual in the window and 0 < b < 1 |
| residuals | the two-step benchmark's stage one: 252-day correlation PCA, 252-day OLS loadings | both PCA rows of Table 2 share one factor model; the loading window is argued in `docs/pca_longconv_benchmark.md` |
| portfolio | residual weights composed to asset weights, `abs(w)` summed to 1, 5bp / 1bp costs, turnover charged across year boundaries | the accounting of every other row (`trading.py`) |
| training | none; the 8-year window and the annual refit do not apply | the rule has no parameters to fit |

`tests/test_ou_threshold.py` checks the signal against GPZ's `preprocess_ou` (s and R2 to
1e-5, the 0 < b < 1 mask exactly, every position of the rule), the mapping to mu and sigma
on a simulated AR(1) with known parameters, and the cases of the rule.

## Results (2026-09-28, commit a6ea300)

Out of sample 1998-01-02 to 2021-12-31 (6040 days), c_thresh 1.25, c_crit 0.25, stage one
rebuilt the same day on the paper-style universe (25 min); the rule itself takes seconds per
K. Run directories `runs/20260928T1427*_pca_ou_threshold_K*_s0` and `...T1428*...`. The
paper's value is in brackets: published, not ours.

| K | SR | mu % | sigma % | SR net | mu net % | beta | turnover/day | break-even bp |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.45 (0.40) | 2.24 (1.85) | 4.97 (4.57) | -2.28 (-2.54) | -11.34 (-11.62) | 0.03 (0.02) | 0.98 | 0.4 |
| 3 | 1.52 (1.26) | 5.50 (4.18) | 3.63 (3.33) | -2.20 (-2.72) | -7.99 (-9.04) | 0.01 (0.01) | 0.97 | 1.7 |
| 5 | 1.45 (0.99) | 4.77 (2.91) | 3.29 (2.93) | -2.64 (-3.44) | -8.68 (-10.11) | 0.01 (0.00) | 0.97 | 1.4 |
| 8 | 1.73 (0.78) | 5.13 (2.04) | 2.96 (2.61) | -2.80 (-4.15) | -8.28 (-10.83) | 0.01 (0.00) | 0.96 | 1.6 |
| 10 | 1.75 (0.80) | 4.99 (1.99) | 2.85 (2.50) | -2.94 (-4.32) | -8.38 (-10.80) | 0.01 (0.00) | 0.96 | 1.5 |
| 15 | 1.76 (0.51) | 4.61 (1.12) | 2.62 (2.20) | -3.34 (-5.24) | -8.75 (-11.53) | 0.00 (0.00) | 0.96 | 1.4 |
| 30 | 1.86 (0.18) | 4.26 (0.40) | 2.30 (2.24) | -3.96 (-6.45) | -9.09 (-14.74) | 0.01 (-0.00) | 0.96 | 1.2 |
| 100 | 1.32 (-0.35) | 2.27 (-0.66) | 1.72 (1.87) | -6.49 (-7.05) | -11.15 (-13.23) | 0.00 (-0.00) | 0.97 | 0.4 |

What reproduces: net Sharpe is negative at every K, for the paper's reason. Without memory
the rule turns over 0.96-0.98 of the book a day, so gross minus net is 13.4-13.6 points a
year (5bp on turnover, 1bp on a short side of 0.50); the paper's rows imply 12.6-15.1.
Sigma and beta match, sigma falling with K in both.

What does not: our gross mean hardly falls with K (5.5 % at K = 3, 4.3 % at K = 30), the
paper's falls from 4.2 % to 0.4 %; only K = 1 agrees within half a point on every column.
Untested candidate: the loading window. GPZ's factor model regresses on 60 days, and at
K = 30 that makes the hedge leg larger than the signal and halves the LongConv row's gross
mean (docs/pca_longconv_benchmark.md); "the same implementation as in [19]" may include
GPZ's residuals. A stage one with `--loading-window 60` and a rerun would test it.

## Open

- The K pattern of the gross mean above (60-day loadings, universe, or something else).
- GPZ's validation grid for the two thresholds is not rerun; the paper's row uses 1.25 / 0.25.
- `raw/crsp_daily/1989.parquet` predates the paper universe (no receipt or unit lines), so
  those names enter the PCA set a year late in 1990. The out-of-sample rule starts in 1998
  and needs no training window, so this row is unaffected; the LongConv row's first
  training window is.
