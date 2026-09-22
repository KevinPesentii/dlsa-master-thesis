# Interface contract

Everything downstream of the data layer is written against these tables and nothing
else. This file changes before the code that depends on it, in its own commit.

Status: DRAFT of 2026-09-08. Three items marked CONFIRM are not settled yet.

Conventions common to all tables:

- Parquet, one file per market and stage, under `data/<market>/shared/<stage>.parquet`.
  `data/<market>/shared/` holds exactly what runs downstream of the data layer (these
  tables, `build_report.txt`, and for the US the rf series `raw/ff_daily.parquet`, the
  PCA history year `raw/crsp_daily/1989.parquet` and the stage-one PCA output
  `pca_l252/`); it is the directory that gets copied to the shared cloud box.
  `data/<market>/private/` holds the raw WRDS pulls (`raw/`), the step-1 Compustat cap
  tables and every build intermediate; only the build scripts read it, and only the
  machine that talks to WRDS needs it. Configs: `output.dir` and `data.dir` point at
  `shared`, `raw.dir` and `output.inspect_dir` at `private`.
- `market` is `us` or `eu`. It is a config key, never a code fork. European handling
  lives in adapters that produce these same tables.
- `date` is a trading date, dtype `date32`. `sec_id` is a string, stable through time,
  one per company (not per listing).
- Sorted by (`date`, `sec_id`). The pair is unique. No NaN in any column marked
  required.
- Float columns are `float32`. Returns are simple, not log, and not in percent.

## returns.parquet

| column      | dtype   | required | notes                                        |
|-------------|---------|----------|----------------------------------------------|
| date        | date32  | yes      |                                              |
| sec_id      | string  | yes      | permno on US, gvkey+iid mapped to one company on EU |
| ret         | float32 | yes      | simple daily return, delisting-adjusted      |
| mktcap_lag  | float32 | yes      | market cap as of the prior month end         |
| country     | string  | EU only  | ISO-2                                        |
| currency    | string  | EU only  | native currency before conversion            |

Returns are in EUR for the European leg, USD for the US leg. A USD European panel is
built as a robustness check and lives at `data/eu_usd/`.

## universe.parquet

| column   | dtype  | required | notes                                          |
|----------|--------|----------|------------------------------------------------|
| month    | date32 | yes      | first trading day of the month membership applies to |
| sec_id   | string | yes      |                                                |
| cap_rank | int16  | yes      | 1 to 500, by prior-month market cap            |

Exactly 500 rows per month. Membership for month m uses only information available at
the end of month m-1. This table is the single source of truth for who is in; no other
module re-derives it.

## features.parquet

| column     | dtype   | required | notes                                        |
|------------|---------|----------|----------------------------------------------|
| date       | date32  | yes      |                                              |
| sec_id     | string  | yes      |                                              |
| char_*     | float32 | yes      | 39 characteristics, rank-quantile normalised |
| med_*      | float32 | yes      | cross-sectional median of each characteristic, broadcast |
| rf         | float32 | yes      | daily risk-free rate                         |

79 columns of features in total. Rows are restricted to the universe on that date.

CONFIRM 1: the normalisation range. Chen, Pelger & Zhu map ranks to [-0.5, 0.5]; the
attention paper says "rank-quantile normalised" without giving the range. Check the
authors' code before the feature build is written, and record the answer here.

CONFIRM 2: the characteristic list. See the OPEN item in `CLAUDE.md`. If the
intersection with JKP's European coverage drops characteristics, the US leg uses the
same reduced set, and the count above changes for both markets.

CONFIRM 3: whether `med_*` is the median over the universe or over all listed names.

## residuals.parquet

| column | dtype   | required | notes                                         |
|--------|---------|----------|-----------------------------------------------|
| date   | date32  | yes      |                                               |
| sec_id | string  | yes      |                                               |
| resid  | float32 | yes      | return net of the estimated factor exposure   |

Produced by the factor model, consumed by the policy. This is the seam between the two
halves of the project.

## weights.parquet

| column | dtype   | required | notes                                            |
|--------|---------|----------|--------------------------------------------------|
| date   | date32  | yes      |                                                  |
| sec_id | string  | yes      |                                                  |
| w      | float32 | yes      | portfolio weight, signed, before cost accounting |

Weights are on the underlying assets, not on residuals. A residual is a portfolio that
has to be traded, which is the paper's whole point, so turnover and short exposure are
measured here and nowhere else.
