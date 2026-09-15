# From Guijarro-Ordonez/Pelger/Zanotti to Epstein/Wang/Choi/Pelger

Assessment of `Code/dlsa-public` against both papers. Change list only, no code written.

Sources: `Papers/GuijarroOrdonez_Pelger_Zanotti_DeepLearningStatArb.pdf` (Sept 2022 draft),
arXiv:2510.11616v1 full text incl. Appendices A and B, and the cloned repo at
`Code/dlsa-public` (commit as checked out 2026-09-09).

Status of the clone: it is the *second stage only* of a two-stage pipeline, and the first
stage it depends on (the residual and composition-matrix arrays under `residuals/`) is
exactly the stage Epstein et al. delete. Reuse is smaller than the repo size suggests.

---

## 1. What actually differs between the papers

| | GPZ (Zanotti) | Epstein et al. |
|---|---|---|
| Estimation | Two step. Factors fixed before trading. | One step. Factors and policy share one objective. |
| Factor model | FF5 / PCA (252d cov, 60d regression) / IPCA, all maximising explained variance | Attention factors: `w_F = Softmax(Q Xtilde^T / sqrt(d))`, `Xtilde = X W_K` |
| Loadings | From the factor estimation | Implied: `beta^T = w_F^T (w_F w_F^T + lambda_ridge I)^-1` |
| Residual map | `Phi_{t-1}` precomputed offline, stored as a T x N x N array on disk | `w_eps = I - beta^T w_F`, recomputed each forward pass, differentiable |
| Inputs to factors | none (residuals arrive as data) | 79 features: 39 characteristics + prev day/week return and vol + cross-sectional medians + rf |
| Sequence model | 2-layer CNN (8 filters, size 2) + 1-layer Transformer, 4 heads | LongConv, 1 layer, hidden 32, FFT convolution, Squash regulariser, geometric-decay init |
| Allocation | separate FFN on the signal | LongConv output *is* the residual weight, no separate allocation head |
| Objective | Sharpe (costs optional, off by default) | net Sharpe (rf subtracted) + `lambda_Var` * mean explained variance, `lambda_Var = 100`, costs always on |
| Training window | 1000 days rolling, refit every 125 days | 8 years rolling, refit annually |
| Universe | market cap > 0.01% of total, ~550 names, unbalanced | 500 largest by prior-month cap |
| OOS | Jan 2002 to Dec 2016 | Jan 1998 to Dec 2021 |
| Costs | 5bp turnover + 1bp short (same formula) | identical formula |

**The one structural change everything else follows from.** In GPZ the residual composition
matrix is *data*: `run_factor_model.py` writes it, `run_train_test.py:153-158` loads it,
`train_test.py:105-107` multiplies it in as a constant `torch.tensor`. In Epstein it is
*model*: a function of the learned `Q` and `W_K`, and the gradient of the net-Sharpe loss
has to flow back through it into the characteristics embedding. Nothing in the clone carries
a gradient into the factor side, so this is not a parameter change, it is a rewrite of the
seam between the two halves of the pipeline.

**Why the net Sharpe improves, mechanically.** The softmax makes every factor portfolio
long-only with weights summing to one, and the cost term sits inside the objective the
factor weights are optimised against. PCA factors have neither property, which is the
turnover and shorting story the paper tells but never quantifies (no turnover or
short-exposure number appears anywhere in it). Producing those numbers is a contribution
available to us for free, since the clone already computes both.

**Convenient identity worth keeping.** `eps^T w_port = R^T (w_eps)^T w_port`, so the clone's
trick of computing returns in residual space while normalising and charging costs in asset
space (`train_test.py:100-116`) stays valid unchanged. That block is the most reusable code
in the repo.

---

## 2. Keep

- `preprocess.py:63-109` (`preprocess_ou`) and `models/OUFFN.py`. Epstein's Table 2 uses
  PCA+OU-Threshold "implemented as in (Guijarro-Ordonez et al., 2025)", so keeping this
  verbatim is the *correct* choice for comparability. Do not rewrite it.
- `train_test.py:100-116` and `158-169`: residual-to-asset mapping, L1 normalisation, cost
  formula. Port the logic, change what it multiplies by.
- `factor_models/pca.py` as the specification for the PCA + LongConv benchmark row. The
  implementation needs replacing (see D3) but the 252-day covariance / 60-day regression
  construction is what the benchmark must match.
- `models/CNNTransformer.py` as a robustness check. The paper says "we expect alternative
  sequence models such as Transformers to perform similarly", and we already have it, so
  a swap-in comparison is a cheap add-on.
- The rolling train/test skeleton in `train_test.py:451-515`, with a different schedule.

Everything else is either dead weight (twilio SMS, `gpuinfo` GPU auto-selection, petname
run tags) or gets replaced.

---

## 3. Action items

Ordered by what blocks what. A-items block everything.

### A. Architecture (the actual migration)

**A1. Make the factor stage a differentiable module.**
New `src/afe/model/attention_factors.py` implementing `w_F = Softmax(Q Xtilde^T / sqrt(d))`,
`beta^T = w_F^T (w_F w_F^T + lambda_ridge I)^-1`, `w_eps = I_N - beta^T w_F`, returning
`w_eps` per date as a tensor with grad. Learned parameters: `Q` (K x d), `W_K` (M x d).
Consequence: `run_factor_model.py` is no longer part of the attention path, and the
`residuals/*.npy.gz` files (roughly 830MB in the clone) are benchmark inputs only.

**A2. Residuals become an intermediate tensor, not a file.**
Delete the file-path construction in `run_train_test.py:92-121` and the disk load at
`:140-158`. Residuals are `eps_t = w_eps R_t` inside the forward pass. Keep a debug hook
to dump `residuals.parquet` per `docs/schemas.md`, but nothing downstream reads it for the
attention model.

**A3. Replace CNN+Transformer with LongConv.**
New `src/afe/policy/longconv.py`. From Appendix A: `y = K * u + D (elementwise) u`,
convolution via FFT, Squash `Kbar = sign(K) * max(|K| - lambda_squash, 0)` with
`lambda_squash = 0.001`, kernel init `K_t^(h) = x exp(-t/T (d/2)^(h/d))`, `x ~ N(0,1)`.
1 layer, hidden 32. Note there is no separate FFN allocation head: LongConv output is the
residual weight directly.

**A4. Add the characteristics tensor to the training loop.**
The clone's `model(...)` call at `train_test.py:99` takes only residual windows. The signature
becomes `(X_{t-1}, R_t)` with `X_{t-1}` of shape (T_batch, N, 79). This changes every function
in `train_test.py`.

**A5. Batch over the whole training window, not 125-day slices.**
GPZ splits the 1000-day window into 125-day batches and computes a Sharpe per batch
(`train_test.py:83-180`, and the paper's Appendix B.C confirms this is deliberate). Epstein's
objective is one Sharpe over the whole training window. With N=500 and an 8-year window
(~2000 days), the full forward pass is feasible on one A6000-class GPU. Doing so also removes
GPZ's previous-allocation approximation (paper p.56): with all weights computed in one pass,
`w_t - w_{t-1}` is exactly differentiable and no stale-epoch approximation is needed.

**A6. Fix the turnover discontinuity at batch boundaries.**
`train_test.py:160-169` prepends a zero to each batch's weight difference, so the first day of
every batch is charged no turnover, and `:193-194` back-fills the last turnover value from
the previous day. Both are artefacts of the 125-day batching. With A5 they disappear, but if
any batching survives, turnover must carry across the seam.

**A7. Change the universe rule.**
`train_test.py:42, 331, 428` derive the tradable set from `np.count_nonzero(...) >= lookback`
on the residual array. Epstein fixes the 500 largest by prior-month cap. Per `docs/schemas.md`
this is `universe.parquet` and no module re-derives it, so the count-nonzero filters get
deleted rather than reimplemented.

### B. Objective and training protocol

**B1. Add the explained-variance term.**
`loss = -net_sharpe - lambda_Var * mean_i(1 - Var(e_i)/Var(R_i))` with `lambda_Var = 100`.
The paper states this term is *necessary for identification*, not a nice-to-have: without it
the factor rotation is unpinned. Replaces `train_test.py:173-180` entirely.

**B2. Subtract the risk-free rate in the Sharpe numerator.**
The clone uses raw `mean_ret/std` (`train_test.py:171-174`). Epstein's numerator is
`Rbar_net - R_f`. Small in the US, not negligible for a 1998-2021 European sample.

**B3. Turn costs on by default.**
`configs/cnntransformer-full.yaml:49-51` ships `trans_cost: 0`, `hold_cost: 0`. In Epstein
the cost term is inside the objective the factors are trained against. Our
`configs/us_replication.yaml` already has 0.0005 / 0.0001; the code must stop treating them
as optional.

**B4. Retrain schedule 1000/125 to 8 years/annual.**
`train_test.py:451-460` and `configs/*.yaml:42-45`. Already recorded as a hard invariant in
`CLAUDE.md`. Also adopt the paper's hyperparameters (Table 4): d=32, d_x=32, dropout 0.1,
30 epochs, lr 0.003, AdamW weight decay 0.05. Note the clone's 100 epochs and lr 0.001 are
GPZ's, not Epstein's.

### C. Data layer (new build, nothing to port)

**C1. Feature builder.** 39 characteristics, rank-quantile normalised, plus prev-day and
prev-week return and volatility, plus cross-sectional medians, plus rf, to 79 columns.
Missing values: last observed, else cross-sectional median. Nothing in the clone touches
characteristics, so this is greenfield against `features.parquet`.

**C2. Trading calendar and a tradability mask.** This is the largest Europe-specific item and
it has no counterpart in either paper. Fifteen markets means dates where a name is listed and
in the universe but its market is shut. A plain softmax over a padded asset dimension would
put factor weight on names that cannot trade. Needs a masked softmax (`-inf` on absent names)
that also propagates into the L1 normalisation and the cost term. Get this wrong and the
Sharpe inflates, which is the failure mode `CLAUDE.md` already flags.

**C3. FX before residuals.** Returns must be in one numeraire before `w_eps` is formed,
otherwise the softmax mixes units. EUR base, USD as the robustness panel, per `schemas.md`.

**C4. Replace the Ken French date index.** `run_train_test.py:124-131` downloads US FF5 daily
purely to get a trading-date index, hard-coded to 1998-2016. Europe needs a real calendar and
FF international factors for the beta/alpha regressions.

### D. Bugs in the clone that will bite in Europe regardless

**D1. Zero return is used as the missing-data sentinel.** `preprocess.py:27, 47, 72` and
`train_test.py:42, 331, 428`. In a 500-name US sample a genuine zero daily return is rare; in
European mid caps and on partial-holiday sessions it is not. This silently drops tradable
names and, worse, does so in a way that varies by country. Replace with an explicit mask
column throughout.

**D2. `np.bool` at `train_test.py:445` and `:584`.** Removed in numpy >= 1.24. The pinned
`requirements.txt` (numpy 1.21.2, torch 1.9.0, sklearn 0.24.2) will not install cleanly on
current CUDA either. Decide now whether the benchmark path runs in a pinned legacy env or
gets ported; do not discover this the week the benchmarks are due.

**D3. `factor_models/pca.py` is not runnable as shipped.** Hard-coded paths at `:12-14`
(`../data/DailyReturns-RFadjusted-old.npz`), and `:33` uses a literal `1998` where it should
use `initialOOSYear`, so any other start year silently misaligns the monthly and daily
indices. `OOSRollingWindowPermnos` also only saves the residual array for `factor == 20`
(`:117-118`), which looks like leftover debugging.

**D4. Universe masks are selected by string matching on the model tag.**
`train_test.py:437-448` and `:576-587` branch on `'IPCA' in model_tag`, `'FamaFrench' in
model_tag`, and load `residuals/superMask.npy` and
`residuals/famafrench-universe/assets-to-consider.npy`. The second file is not in the clone.
Universe belongs in `universe.parquet`, not in a tag string.

**D4b. The benchmark path cannot run as cloned: the composition matrices are missing.**
`run_train_test.py:92-121` builds paths to `AvPCA_OOSMatrixresiduals_*`,
`IPCA_DailyMatrixOOSresiduals_*` and `DailyFamaFrench_OOSMatrixresiduals_*`, and the README
(`:50`) states `use_residual_weights: True` is required to reproduce the paper. The clone
ships only the 21 residual-return arrays; not one matrix file is present. So every benchmark
row has to be regenerated locally, through the `OOSRollingWindowPermnos` branch of
`pca.py` that carries the D3 bugs (the vectorised variant does not build matrices at all).
Budget real time for this: it is the step that makes the PCA + LongConv comparison possible,
and it is not a download.

**D5. `torch.autograd.set_detect_anomaly(True)` at import** in `train_test.py:13`, `data.py:5`
and `run_train_test.py:27`. Large slowdown for no benefit outside debugging.

**D6. `nn.DataParallel` (`train_test.py:54`) plus the GPU-count heuristic at
`run_train_test.py:184-195`.** Deprecated; if we use more than one GPU it should be DDP, and
the heuristic hard-codes GPZ's model sizes.

### E. Evaluation gaps

**E1. Persist turnover and short exposure as headline results.** The clone computes both
(`train_test.py:193-195`, `388-389`) and then buries them in a pickle. Epstein reports
neither. Per `CLAUDE.md` these belong in the metrics JSON of every run directory.

**E2. Break-even cost curve.** Not in either paper. Needs the cost coefficients to be a swept
config value, and the run harness to loop the grid already declared in
`configs/us_replication.yaml:41`.

**E3. Market beta and alpha.** Table 2 reports beta; the clone has no factor regression at all
(the `tools/` directory referenced in its README is not in the repo).

**E4. Seed variance.** Table 3 reports SD across seeds (0.12-0.17 on the Sharpe). Multiple
seeds per configuration must be a first-class part of the run harness, not a manual rerun.

---

## 4. Open questions to settle before writing code

1. **Cumulative or raw residuals into the sequence model?** GPZ is explicit: the input is
   `x = Int(eps)`, the cumulative residual "price" (paper Section II.D, and
   `preprocess.py:14-36`). Epstein writes `LongConv(eps_{i,(t-s,t-1)})`, i.e. the residuals
   themselves. GPZ's CNN also applies InstanceNorm to the input; LongConv as specified does
   not, and raw daily residuals are order 1e-2. This changes the input scale by roughly an
   order of magnitude and is not a detail. Check the authors' code or ask.
2. **`lambda_ridge` is never given a value.** It appears in the loadings formula but not in
   Table 4 and nowhere in the text. Needs a sensitivity check, and it should be recorded as
   a CONFIRM item in `docs/schemas.md` alongside the existing three.
3. **Batching and validation split.** The paper says only "the last two years of the first
   training window" for tuning. Batch size, shuffling, and whether the Sharpe is computed
   over the full window or in chunks are unspecified. A5 assumes full window; flag it as our
   choice in the thesis rather than as replication.
4. **Are the previous-day/week return and volatility features rank-normalised too,** or left
   in levels? "In addition to the most important characteristics ... we also include the
   previous day and week return and volatility" sits after the normalisation sentence and is
   ambiguous.
5. **Does `Ret_D1` / `Ret_W1` / `STD_W1` in Table 1 duplicate item 4?** Table 1 lists them
   under Past Returns as part of the 39, but the text adds them again when arriving at 79.
   Reconcile the arithmetic: 39 characteristics + 39 medians = 78, + rf = 79, which implies
   the day/week return and vol are already inside the 39 and the text is double-counting.
   Worth confirming, because it fixes the exact shape of `features.parquet`.

---

## 5. Suggested sequencing

1. C1 to C3 (data layer, Henrik) and A1 plus A3 (attention factors and LongConv, Kevin) in
   parallel against `src/afe/data/synthetic.py`.
2. A2, A4, A5, B1 to B4: wire the one-step loop. This is the commit where the clone stops
   being the reference and our code takes over.
3. US replication run. Acceptance criterion is net Sharpe near 2.28, as `CLAUDE.md` states.
4. Benchmarks: PCA + LongConv and PCA + OU-Threshold, reusing D3-fixed `pca.py` and the
   untouched OU code. Start D4b early, it is the long pole here.
5. E1 to E4, then Europe.

Items D1, D2 and D4b are cheap to start and block step 4, so begin them whenever the
benchmark path is first run rather than at the end.

## 6. Confidence

Everything above is read off the two papers and the checked-out code. Two things I could not
verify: `factor_models/ipca.py` (59KB) was not read line by line, and the composition-matrix
claim in D4b rests on the file listing rather than on a failed run, so confirm it by
launching one benchmark config before planning around it.
