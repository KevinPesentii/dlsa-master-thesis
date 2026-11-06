# Project constitution

Read this before writing or changing code. It holds the decisions, not the code.
If a convention here is wrong, change it in a commit so both of us pick it up.
Owners: Henrik Reiss, Kevin Pesenti. MSc thesis, SSE, due December 2026.

## What the thesis is

Out-of-sample test in European equities of Epstein, Wang, Choi & Pelger (2025),
"Attention Factors for Statistical Arbitrage", ICAIF '25, arXiv:2510.11616.

One research question: does the attention factor result, and the cost advantage that
produces it, survive in European equities?

Two legs:
1. US replication, Jan 1998 - Dec 2021, to validate the pipeline against the paper's
   net Sharpe of 2.28.
2. Europe, SAME window 1998-2021. Weights re-estimated on European data, with
   hyperparameters validated on European data by the same pre-registered searches
   (the headline, decided 2026-10-12) and with the US headline values (comparison).

## Hard invariants

Violating any of these invalidates the result. Tests exist for most of them; do not
weaken a test to make code pass.

- POINT IN TIME. Any feature or universe decision at date t uses only information
  available at t. Universe membership for month m is set by market cap in month m-1.
- Prefix invariance is the operational form: rebuilding features from data truncated
  at t must leave the values at t unchanged. `tests/test_no_lookahead.py` enforces it.
- Europe: one line per company. No secondary listings, no depositary receipts, no
  secondary share classes. The US replication follows the paper instead (decided
  2026-09-23): its Figures 3-4 name depositary receipts (Mizuho, NatWest, Ecopetrol),
  Canadian listings, MLP units, REITs and Alphabet twice, so the US universe ranks every
  listed common-equity line, receipts at company-level cap (docs/us_characteristics.md).
- Weights refit ANNUALLY, on rolling 8-year windows. Not the 125/1000-day schedule of
  Guijarro-Ordonez, Pelger & Zanotti.
- Hyperparameters come only from a pre-registered search on a market's first training
  window, never from out-of-sample years. Europe: the US search's values, or (decided
  2026-10-12) the same searches on Europe's own first window (`configs/europe_search_*.yaml`:
  fit 1993-1995, net Sharpe on 1996-1997; only the data differ from the US searches).
- A Sharpe above 10 is a diagnostic of a broken universe, not a result. Long & Xiao
  (2024) was withdrawn for exactly this.
- US CRSP data comes from the CIZ tables (`crsp.dsf_v2`), never legacy `crsp.dsf`.
  Legacy stopped updating after 2024 and silently truncates the sample. CIZ folds the
  delisting return into `dlyret` on the delisting day, so no Shumway (1997) merge is
  needed, BUT that row fails the common-share flags (they are blank on it): it is
  pulled separately (`crsp_delisting.parquet`) and merged in stage 2. The first build
  (2026-09-15) lacked it; fixed 2026-09-22.

## Paper parameters (verified against arXiv 2510.11616, 2026-09-06)

- Universe: 500 largest by prior-month market cap.
- 39 characteristics in six themes, built per Chen, Pelger & Zhu (2024),
  rank-quantile normalised; 79 features after adding cross-sectional medians and rf.
- Embedding dim 32. K queries with softmax factor weights. K=30 is the headline.
- LongConv time-series filter: 1 layer, 32 hidden. Residual lookback 30 days.
- lambda_Var = 100.
- Cost term in the objective: 0.0005 * ||w_t - w_{t-1}||_1 + 0.0001 * ||max(-w_t, 0)||_1.
- Reference results (Table 2): attention K=30 gross SR 3.97 (mu 16.66%, sigma 4.20%),
  net SR 2.28 (mu 9.52%), market beta 0.05. Costs mu - mu_net = 7.14% a year, ~2.8bp
  a day, imply turnover ~0.47/day at 5bp/1bp with a short leg of 0.5 (||w||_1 = 1).
  PCA + LongConv, K = 1-100: gross SR 2.3-2.8 (sigma 5.3-5.9%) falls to 1.2-1.6 net.
  PCA + OU goes negative net for every K.

## Headline specification (decided 2026-10-10)

- Configs: `configs/us_headline.yaml`, `configs/europe_headline.yaml`. Until 2026-10-12
  one set of hyperparameters for both markets, selected by the pre-registered US search
  (`configs/us_search.yaml`, fit 1990-1995, net Sharpe on 1996-1997; run
  20261010T110947Z_search_K30_val1996-1997): sobol-036 = lambda_var 17.6, input_scale
  87.9, score temperature 0.342, lambda_ridge 2.63e-5, batch 125, cumulative input /
  window_vol, level_hidden 0; the rest is Table 4. Kevin's sobol-009 placed 4th.
- Every reported return uses validated values (2026-10-10): the PCA benchmarks and the
  rows that change how the model trades (raw input, lag 1) get their own searches on the
  same window (`searches:` in `configs/thesis_matrix.yaml`); rows that change only the
  data keep the headline values.
- Europe: europe17 (all 17 exchange countries), top 200 by prior-month cap, ECU to 1998 /
  EUR after, training from 1993. europe12 is a robustness row.
- Europe holidays: gaps of <= 3 pooled days filled with a zero local return (stage 2b,
  `scripts/fill_europe_holidays.py` -> `data/<market>/holfill/`) and positions held on
  every non-traded close (`execution.stale_when_closed`). `data/<market>/shared/` stays
  the unfilled build, for the robustness row.
- Execution lag 0 (the paper's timing); lag 1 is a robustness row.
- Europe headline = Europe-validated values (decided 2026-10-12, before any of those
  runs existed): `configs/thesis_matrix_europe.yaml` reruns every Europe row of the
  thesis matrix with the winners of the six searches repeated on Europe's first window
  (each carries its US winner into the finalists); results in `runs/_matrix/thesis_eu`.
  The Europe rows of `runs/_matrix/thesis` (US values) are the comparison, not the
  headline, whatever the two show.
- Paper-gap diagnostics (2026-10-16): `configs/gap_matrix.yaml` (with the 800-name build
  `configs/us_data_top800.yaml` -> `data/us/top800`) tests the explanations of the gap to the
  paper in `Thesis/diagnostics/2026-10-16_paper_gap`. Diagnostic rows only: they never
  replace a headline value and select nothing; results in `runs/_matrix/gap`. Likewise
  `configs/gap2_matrix.yaml` (2026-10-19; size vs breadth, with the ranks 301-800 build
  `configs/us_data_rank301_800.yaml` -> `data/us/rank301_800`): results in `runs/_matrix/gap2`.
- Fixed-window experiment (2026-11-01), outside the thesis rows: `configs/exp_fixedwin_matrix.yaml`
  compares `policy.residual_window: fixed` (the 30-day window re-hedged with the trade date's
  model, `configs/us_headline_fixedwin.yaml`) with the headline's rolling window. Results in
  `runs/_matrix/exp_fixedwin`; never a headline value. The full sequential version
  (`configs/exp_fixedwin_full_matrix.yaml`: the thesis `us_attention` row twice, K 1-100, seeds
  0-4) gives the fixed row its own search, `configs/us_search_fixedwin.yaml` (= us_search.yaml
  with residual_window held fixed); results in `runs/_matrix/exp_fixedwin_full`.
- Ensemble experiment (2026-11-02), outside the thesis rows: averaging the five seeds' weights into one
  book raised K = 30 net Sharpe out of sample (`runs/_matrix/exp_fixedwin_full/seed_ensemble.py`), an
  idea found after seeing those runs. So the validation window decides first:
  `scripts/ensemble_validation.py` (`configs/us_ensemble_val*.yaml`, K = 30) compares the search winner on
  five seeds with the search's top five configurations, and `configs/exp_ensemble_k30_matrix.yaml` runs
  the other four candidates out of sample. Never a headline value until a decision says so.
- Receipts experiment (2026-11-04): removing depositary receipts from the saved
  K = 30 books lowered net Sharpe by about 0.55, all of it before 2010, an idea found after seeing those runs
  (`runs/_matrix/exp_ensemble_k30/traded/rebook_no_receipts.py`). `configs/exp_no_receipts_matrix.yaml` retrains
  the headline on two universes, rows that change only the data and so keep the headline values:
  `configs/us_data_no_receipts.yaml` (the 500 largest non-receipts) and `configs/us_data_drop_receipts.yaml`
  (the default members without their receipts, not replaced). Results in `runs/_matrix/exp_no_receipts`; they
  bear on the US-Europe comparison, since Europe has no receipts. Retrained, the row with empty slots also loses
  after 2010, so "all of it before 2010" describes the re-booking only. Decided 2026-11-06: `us_no_receipts` is a
  robustness row of Table 11, read from that matrix on seeds 0-2 like the other rows (`scripts/thesis_tables.py`);
  `us_drop_receipts` stays a diagnostic. Never a headline value.
- Model v2 (decided 2026-11-06): `configs/us_headline_v2.yaml`, `configs/europe_headline_v2.yaml`, the headline
  with four changes, all config keys, so every earlier row still reproduces. (1) X is the 39 rank
  characteristics: the 39 medians and rf are the same for every name on a date, so they cannot move a softmax
  over names, whatever the paper's 79 features intend; `level_hidden` has nothing left to read and stays 0.
  (2) `features.timing: asof` reads `features_asof.parquet` (stage 2c, `scripts/build_asof_tables.py`): monthly
  and annual characteristics reach X on the first trading day of a month instead of the second, ranks are over
  that day's universe and an entrant has real values. Same numbers as before on every other day. (3)
  `policy.residual_window: fixed` with `data.returns_history`: a name trades from its first day in the universe
  on returns from before it entered, residualised with the trade date's factor model, once it has 30 of them.
  (4) The policy input is cumulative, not searched, in both markets (Europe's 2026-10 search had picked raw).
  `configs/thesis_v2_matrix.yaml` reruns every attention row but the raw-input ones with new searches
  (`configs/*_search_v2*.yaml`, same windows, budget and rule), plus a ladder of K = 30 rows on the old values,
  one change at a time; results in `runs/_matrix/thesis_v2`. The fixed window was tested out of sample before
  it was adopted (`runs/_matrix/exp_fixedwin_full`), and the thesis has to say so. The rows above stay the
  headline until the results are in and a decision says otherwise.
- PCA rows under model v2 (2026-11-06): `configs/thesis_v2_pca_matrix.yaml`, six searches
  (`configs/*_search_pca_v2*.yaml`, `*_search_ou_v2.yaml`) and every PCA row, with a ladder; results in
  `runs/_matrix/thesis_v2_pca`. Two of the four changes reach PCA: `policy.residual_window: fixed`
  (`trading.make_fixed_windows`) and cumulative input. PCA reads no characteristics, and a PCA name already has 252
  days of returns, so as-of features and `returns_history` do nothing; stage one is unchanged. The PCA book
  holds 486 of 498 names (Europe 186 of 199) against attention's 497 (198.5). Not the same experiment as attention's
  fixed window: B_t is fitted on a span containing the window days, so the re-hedged path is in sample for the
  hedge, and its sd falls 14% (US K = 30), 18% (Europe K = 30), 42% (US K = 100) against rolling. Read
  the ladder before calling the two rows like for like. The rows of `runs/_matrix/thesis` stay until a decision.

## Trading costs: limits, not replication

The thesis does NOT reproduce European execution costs one for one. That needs intraday
quote and venue data and a calibrated impact model, none of which we are simulating.

- Headline statistic is the BREAK-EVEN COST: the per-trade cost at which each strategy
  stops earning.
- Performance is reported as a FUNCTION of assumed cost, not at a single point. The
  paper's 5bp/1bp is one point on that curve, kept for comparability.
- Statutory transaction taxes are a computable floor, not a cost model:
  UK stamp duty 50bp on purchases, French FTT 30bp, Italian FTT 10bp.
- Spreads, market impact, venue fragmentation and borrow costs are named as unmodelled.

## Data

- US characteristics: CRSP CIZ + Compustat, cross-checked against Open Source Asset
  Pricing (Chen & Zimmermann 2022).
- Europe returns and market cap: Compustat Global Security Daily via WRDS.
- Europe characteristics: Jensen, Kelly & Pedersen Global Factor Data
  (`contrib.global_factor` on WRDS). Forward-fill monthly to daily as in the US build.
- Europe evaluation factors: Fama-French international five-factor; JKP as cross-check.
- European universe: 200 largest pooled across the 17 exchange countries of europe17
  (GB DE FR CH NL IT ES SE BE DK FI NO IE PT AT LU GR), point in time monthly, returns
  in ECU/EUR with USD as robustness. (The proposal's 500 was replaced on 2026-10-10:
  EU rank ~125-193 has the cap of the US 500th.)

OPEN, to settle before the feature builder is written: JKP's European characteristics do
not map one to one onto the 39 US Chen-Pelger-Zhu definitions. Current plan is the
INTERSECTION so both markets share one feature definition, at the cost of dropping
characteristics. Alternative is JKP's fuller European set at the cost of comparability.

## Working rules

- NO DATA IN THE REPO. CRSP, Compustat and JKP extracts are licensed. `data/` and
  `*.parquet` are gitignored. WRDS credentials live in `.env`, never in code.
- Notebooks are not source. The pipeline lives in `src/afe/` with YAML configs in
  `configs/`. Notebooks are for looking at results and have outputs stripped.
- Never report a number that did not come out of a run directory. If you do not have
  the number, say so. Do not estimate, interpolate or recall it from the paper and
  present it as ours.
- Every run writes a directory under `runs/` containing the resolved config, git commit
  hash, seed, package versions and a metrics JSON. See `src/afe/runs.py`.
- Numerical invariant tests are written by hand from the paper, not generated. A test
  written by the same model that wrote the code mostly asserts that the code does what
  it does.
- Pull requests stay under roughly 300 lines. Anything larger does not get read
  properly, and unread generated code is how a lookahead bug reaches the thesis.
- Neither of us merges code we generated without the other reading the diff.
- Interfaces before implementation. `docs/schemas.md` is the contract between the data
  layer and everything downstream. Change it in its own commit, before the code.

## Ordering

Europe does not start until the US replication reproduces net Sharpe near 2.28.
Until then, downstream work runs against the synthetic fixture in
`src/afe/data/synthetic.py`, which conforms to the same schema.

## Ownership

- Henrik: data and universe layer. WRDS extraction, universe construction, feature
  build, both markets to one schema.
- Kevin: attention factor model, LongConv policy, evaluation, break-even cost curve.
- Both review each other. Swap halves later in the project so both can defend all of it.
