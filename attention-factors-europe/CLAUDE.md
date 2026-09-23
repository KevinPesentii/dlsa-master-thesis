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
2. Europe, SAME window 1998-2021, Potentially SAME hyperparameters. Reestimation of CNN.

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
- Europe uses the US-validated config. No European hyperparameter search.
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
- Reference results: attention K=30 gross SR 4.20, net 2.28, market beta 0.05.
  PCA + LongConv gross 5.3-5.9 falls to 1.2-1.6 net. PCA + OU goes negative net.

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
- European universe: 500 largest pooled across AT BE DK FI FR DE IE IT NL NO PT ES SE
  CH GB, point in time monthly, returns in EUR with USD as robustness.

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
