# European raw data: what was pulled, from where, in which currency, saved where

Status: first pull 2026-09-21; stage 2a (EUR market data, section at the end) built the
same day. Universe = pooled top-500 by market cap across the eleven
1999 euro founders (AT BE DE ES FI FR IE IT LU NL PT), month ends 1999-01 to 2025-12,
from the step-1 cap table (`docs/eu_mktcap_compustat_global.md`). Raw tables cover every
company that was EVER a member (1,543 gvkeys, 1,725 listings), daily from 1994 so the
five-year Beta window exists at the first ranked month.

Build: `python scripts/fetch_eu_raw.py --config configs/eu_data.yaml` (resumable per
year; `--universe-only` rewrites the universe files without WRDS). Code:
`src/afe/data/wrds_eu.py`. Everything lands under `data/eu/raw/` (gitignored, licensed);
`manifest.json` there records row counts and the pull time.

Nothing in `raw/` is converted or computed: prices stay in the quotation currency,
fundamentals in the reporting currency, and the FX table is what converts them
downstream. Returns are NOT stored; they are one formula on the daily file (below).

## Files

| file | rows (2026-09-21) | one row per | source |
|---|---|---|---|
| `universe_top500_monthly.parquet` / `.csv` | 162,000 | month end x rank | step-1 cap table |
| `universe_top500_gvkey_wide.csv` | 324 | month end (columns = ranks 1..500) | same |
| `listings.csv` | 1,725 | (gvkey, iid) pulled daily | same |
| `secd_daily/<year>.parquet` | 7,065,748 (6,249,504 from 1999) | listing x trading day, 1994-2025, 32 files, 119 MB | `comp.g_secd` |
| `g_funda.parquet` | 30,399 (INDL 23,775 + FS 6,624), 1,509 companies | company x fiscal year, 1991-2025 | `comp.g_funda` |
| `fx_daily.parquet` | 241,041, 22 currencies | currency x day, 1990-2025 | `comp.g_exrt_dly` |
| `jkp/<year>.parquet` | 287,325 | security x month end, 1998-2025, 28 files, 527 MB | `contrib.global_factor` |
| `contrib_factors_daily.parquet` | 117,031 | country x day, 1989-07 to 2018-02 | `contrib.factors_daily` |
| `ff_daily.parquet`, `ff_monthly.parquet` | 26,296 / 1,201 | day / month | `ff.factors_daily`, `ff.factors_monthly` |
| `manifest.json` | | row counts, pull time, listing and company counts | |

Headers (`security_header.parquet`, `company_header.parquet`) and the cap table
(`company_month_mktcap.parquet`) sit one level up in `data/eu/`.

## Variables

### universe_top500_monthly (step-1 cap table, EUR)

| variable | meaning | currency / unit |
|---|---|---|
| datadate | calendar month end m; the ranking is AS-OF m (universe for m+1 downstream) | |
| cap_rank | 1 = largest | |
| gvkey | Compustat company id, zero-padded string | |
| mktcap | company cap = sum of common share classes on the home exchange | EUR mn |
| country | exchange country of the most active listing (ISO-3) | |
| iid | the listing that priced the main class = the line to trade (`sec_id` = gvkey+iid) | |
| exchg, isin, conm, fic, sic | Compustat header fields (current values) | |
| n_classes, n_listings | share classes summed / listings seen that month | |
| turnover | trailing 6-month median of month-end-day volume / shares of the home line | fraction/day |
| curcdd, prccd, cshoc, price_date | quotation currency, close, shares of the main class, actual trade date | local |

### secd_daily (Compustat Global Security Daily, `comp.g_secd`)

All in the quotation currency `curcdd` of the listing (EUR for nearly all rows from
1999; legacy DEM/FRF/ITL/... before 1999; a few USD/GBP lines).

| variable | meaning | unit |
|---|---|---|
| gvkey, iid | company, listing | |
| datadate | trading day | |
| prccd, prchd, prcld, prcod | close, high, low, open | local ccy per share |
| ajexdi | cumulative adjustment factor for splits and stock dividends (divide prices by it) | |
| trfd | total return factor: cumulative reinvested cash dividends (multiply adjusted price by it) | |
| cshoc | shares outstanding of the issue | shares |
| cshtrd | volume | shares |
| qunit | quotation unit (1 everywhere in the euro-11 sample) | |
| div, divd, divsp | cash dividend (total, ordinary, special) on the ex-date | local ccy per share |
| cheqv | cash-equivalent distribution | local ccy per share |
| curcddv | currency of the dividend | |
| prcstd | 10 = traded close, 5 = price carried from the last trade | |
| exchg | exchange code of the listing | |
| monthend | 1 on the listing's last row of the month | |

Total return in local currency, day t (JKP definition; Global has no return field):
`ret_t = (prccd_t / ajexdi_t * trfd_t) / (prccd_{t-1} / ajexdi_{t-1} * trfd_{t-1}) - 1`.
EUR return: convert each day's price with that day's rate first, then take the ratio.
Compustat Global carries NO delisting return; a company's last row is its last trade.

### g_funda (Compustat Global Fundamentals Annual, `comp.g_funda`)

`datafmt HIST_STD, popsrc I, consol C`, and BOTH `indfmt` INDL and FS: unlike North
America, Global files banks and insurers under the financial-services format only
(Santander, BBVA, ABN Amro, AIB, Generali; 346 of the ever-members). No company-year
carries both formats. FS rows have `revt, xint, ib, at, seq, ceq, oiadp, dp, cshoi` but
no `sale, cogs, xsga, capx`, so sales-based characteristics need `revt` for financials.
34 ever-members have no fundamentals at all. Amounts in `curcd`, the reporting currency
(EUR 78%; FRF/DEM/ITL/ESP/NLG/FIM before 1999; USD 777 rows), in millions; shares in
millions. Keys: `gvkey, datadate` (fiscal year end), `fyear`, `fyr` (fiscal year-end
month), `curcd`, `indfmt`.

| group | items | notes |
|---|---|---|
| balance sheet | at, lt, seq, teq, ceq, pstk, txditc, txdb, che, ivao, act, lct, dlc, dltt, mib, txp, wcap, wcapch, ppegt, ppent, invt, rect, ap, intan, gdwl | as in `wrds_us.FUNDA_ITEMS` where Global has the item |
| income statement | sale, revt, cogs, xsga, xopr, xrd, xint, oiadp, oibdp, ebit, ebitda, pi, txt, ib, ibc, nicon, spi, xido, dp, emp | `nicon` = net income (Global's `ni`); gross profit = revt - cogs (`gp` absent) |
| cash flow / payout | capx, oancf, fincf, ivncf, aqc, sstk, prstkc, dltis, dltr, dlcch, chech, dvc, dvt, dv | for JKP-style items if the intersection needs them |
| shares / adjustment | cshoi (shares outstanding, issued), cshpria, ajexi | Global's `csho` / `ajex`; no `prcc_f`, use the daily close at `datadate` |

Absent in Global versus the US list: `pstkl`, `pstkrv` (BEME falls back to `pstk`),
`ni` (use `nicon`), `gp`, `xad` (treated as zero in the US build), `csho`, `ajex`,
`prcc_f`.

### fx_daily (`comp.g_exrt_dly`)

| variable | meaning |
|---|---|
| datadate | day |
| tocurd | currency code |
| exratd | units of `tocurd` per 1 GBP |

EUR per unit of local = `exratd(EUR) / exratd(local)` on the same day. EUR exists from
1985-12-31 (synthetic before 1999-01-01), legacy currencies to 2018-06-04. Covers every
currency seen in `secd_daily.curcdd` and `g_funda.curcd`, plus USD and GBP; no day is
missing for any currency inside its range. The derived table with the EUR and DEM legs
is `data/eu/fx_to_eur_daily.parquet` (stage 2a below).

### jkp (Jensen-Kelly-Pedersen Global Factor Data, `contrib.global_factor`)

Monthly, one row per security (`gvkey`, `iid`, `excntry`, `eom`), all 444 columns.
Identifiers and market fields:

| variable | meaning | currency |
|---|---|---|
| id, gvkey, iid, excntry, eom, date | keys; `excntry` = exchange country | |
| primary_sec, obs_main, common, exch_main | JKP's own screens (all 1 for the euro-11 rows sampled) | |
| curcd, fx | quotation currency and its USD rate | |
| me, me_company | market equity of the security / the company | USD mn |
| prc, prc_local | month-end price | USD / local |
| ret, ret_local, ret_exc | monthly total return, in USD / local / USD minus US T-bill | |
| market_equity, size_grp | JKP size variables | USD |

Characteristics (the remaining ~400 columns): JKP's standardised names, e.g.
`be_me, at_me, ni_me, ocf_me, sale_me, ret_12_1, ret_1_0, ret_36_12, rvol_21d,
ivol_ff3_21d, beta_60m, turnover_126d, dolvol_126d, prc_highprc_252d, bidaskhl_21d,
noa_at, at_gr1, oaccruals_at, gp_at, ope_be, cop_at, ...`. Ratios are unit-free; level
variables (`assets, sales, book_equity, net_income, enterprise_value`) are USD mn.
Point in time as of `eom`. The map from these to the 39 US characteristics is the open
"intersection" decision in CLAUDE.md; this pull keeps all of them so it can be made
offline.

### contrib_factors_daily (`contrib.factors_daily`) and ff_daily / ff_monthly

| file | variables | currency | note |
|---|---|---|---|
| contrib_factors_daily | date, country (Germany, France, Italy, Netherlands, Spain, UK, Switzerland, Sweden, Denmark, Norway, USA, ...), vwret, ewret, smb, hml, rmw, cma, rf | local | 1989-07 to **2018-02-14 only**: cross-check, not the build's factor series |
| ff_daily, ff_monthly | date, mktrf, smb, hml, umd, rf | USD | US factors; `rf` is the US 1-month T-bill that JKP use for every country |

## Checks run on 2026-09-21

- Daily rows exist in the following month for 99.4% of the 162,000 universe
  member-months (99.05% with >= 15 days). The rest: the 500 members of 2025-12 (their
  trading month 2026-01 is outside the pull) and ~15-30 delistings a year whose last
  member month has no successor (Royal Dutch 2005-07 into Shell, ...).
- The daily total-return formula above, compounded to calendar months and compared
  with JKP's `ret_local` for 2005-2015: 115,078 security-months matched, correlation
  0.99994, 99.9th percentile of |difference| = 0. The 8 months that differ by more than
  2pp are JKP's imputed delisting return of -30% on a security's last month; Compustat
  Global itself has none. Whether to adopt that convention is a stage-2 decision.
- 97.1% of daily rows from 1999 are traded closes (`prcstd 10`), 2.9% carried prices;
  `trfd` is never missing, `ajexdi`/`prccd` missing on 0.01%. Quotation currency EUR on
  99.7% of rows from 1999 (USD 14,291 rows, GBP 539, GRD 500).

## Not pulled, and why

- **Euro risk-free rate and European market factor: not on WRDS**, so the market is
  built from the data (stage 2a) and rf comes from the Bundesbank (below).
- **Closing bid/ask.** Compustat Global has none, so the US `Spread` characteristic has
  no European counterpart; JKP's `bidaskhl_21d` (Corwin-Schultz high-low estimator) is
  the substitute.
- **Quarterly fundamentals** (`comp.g_fundq`): the US build uses annual only.
- **Preferred shares' daily data**: prefs are not in the cap and not traded (open
  decision in the cap doc).
- Non-euro markets (GB, CH, SE, DK, NO) and the later euro entrants: not in this run.

## Stage 2a: EUR market data (`scripts/build_eu_returns.py`, `src/afe/data/eu_panel.py`)

Built 2026-09-21 from the raw tables above. Every value is in EUR; conventions are config
keys in `configs/eu_data.yaml`:

- **Numeraire**: EUR from 1994-01-01, Compustat's EUR series (synthetic, ECU-based,
  before 1999); the Deutsche Mark at the irrevocable 1.95583 DEM/EUR before 1994, so the
  unit is continuous. DEM is carried as a second leg: quoted to 2018-06-04, EUR x 1.95583
  after. The only visible seam is 1999-01-04, the first EUR quotation day: Compustat's
  synthetic EUR of 1998-12-31 sits 0.8% off the fixed conversion rates, so every
  legacy-currency line shows an EUR return 0.8% below the fixed-rate return on that one
  day (measured in the build report, uniform across currencies, cross-sectionally
  almost neutral).
- **Returns**: total return on the EUR price, `(p_eur/ajexdi*trfd)_t / (...)_{t-1} - 1`
  with `p_eur = prccd/qunit x eur_per_unit(curcdd, t)`. Lines quoted in USD/GBP/GRD
  (13,508 rows, 0.22%; 3,934 universe member-days) carry the currency move. A line whose
  quotation switched DEM -> EUR has no break. `ret_local` (in the detail table) is
  blanked at a quotation-currency change.
- **One line per company**: the company's return in month M is that of the listing that
  priced its cap at the end of M-1 (`iid` in the cap table). No splicing across venues.
- **Calendar**: days with >= 100 traded closes (`prcstd 10`) across the pool: 8,195 days
  1994-2025. Compustat carries prices for ~20 lines on Good Friday, Easter Monday,
  Christmas and New Year; those days are out. Partial days (Whit Monday: Milan and
  Madrid open, Frankfurt and Paris closed) are in, with a zero local return for the
  closed lines (`traded` flag in the detail table).
- **Universe**: month M = top-500 of the cap table at the end of M-1; `month` = first
  pooled trading day of M; `sec_id` = gvkey. 323 months, 1999-02 .. 2025-12.
- **Market**: value-weighted EUR return of the month's universe, weights = the cap that
  set membership; equal-weighted alongside. No external index. Correlation with the
  country value-weighted markets of `contrib.factors_daily` (to 2018): FR 0.98, NL 0.93,
  IT 0.92, ES 0.90, DE 0.89.
- **Delisting: OPEN, flagged.** None applied (`returns.delisting_return: null`); 762 of
  1,540 companies' series end before 2025-12. Compustat Global has no delisting return.
  JKP's blanket -30% is one config key away, but the US evidence (593 member
  delistings, median +0.1%, mostly mergers; see `docs/us_characteristics.md`) says a
  large-cap universe's delistings are acquisitions, not failures. Proposal to settle:
  0 for mergers, a negative return only for bankruptcies/liquidations, using
  `g_security.dlrsni` (delisting reason).
- **Risk-free**: 1-month EURIBOR from 1999-01-01, 1-month FIBOR (Deutsche Mark) before,
  both daily quotations, act/360, percent p.a., from the Bundesbank time-series API
  (`api.statistiken.bundesbank.de`, dataflow BBIG1; the ECB portal carries Euribor only
  as monthly averages). Files in `data/eu/raw/external/`: `bbk_ST0310_euribor_1m_daily.csv`
  (1998-12-30 ..), `bbk_ST0262_fibor_1m_daily.csv` (1990-07-02 .. 1998-12-30),
  `bbk_ST0104_frankfurt_1m_daily.csv` (Frankfurt banks' 1-month funds to 2012-05, a
  cross-check: correlation 1.000 with both, 2-5 bp mean gap). `scripts/build_eu_rf.py`
  splices them into `data/eu/raw/rf_euro.csv` (`date, rate_pct_pa, series`); the seam
  is 3.23% FIBOR vs 3.26% EURIBOR on 1998-12-30. Stage 2a converts to a daily simple
  rate (rate/100/360) and forward-fills onto the trading calendar -> `rf_daily.parquet`
  (8,195 days, no gaps; negative 2015-2021). Note the Bundesbank's licence text: daily
  Euribor is EMMI data, free for non-commercial use, which this is.

| file (in `data/eu/`) | rows | one row per | currency |
|---|---|---|---|
| `fx_to_eur_daily.parquet` | 233,472 | (date, currency): eur_per_unit, dem_per_unit, eur_source, dem_source | EUR, DEM per unit |
| `returns_detail_daily.parquet` | 7,065,748 | (gvkey, iid, date): prc_eur, fx_eur, ret (EUR), ret_local, prccd, ajexdi, trfd, cshoc, cshtrd, prchd, prcld, prcstd, traded | EUR / local |
| `returns.parquet` | 5,975,845 | (date, sec_id): ret, mktcap_lag, country (ISO-2), currency (docs/schemas.md) | EUR |
| `universe.parquet` | 161,500 | (month, sec_id): cap_rank (docs/schemas.md) | |
| `market_daily.parquet` / `.csv` | 6,892 | date: mkt_vw, mkt_ew, n | EUR |
| `rf_daily.parquet` | 8,195 | date: rf (daily simple, act/360), series | EUR (DEM before 1999) |
| `returns_build_report.txt` | | coverage by year, the 1999 seam, foreign lines, market by year | |

Checks: 99.6-99.9% of universe member-months have returns (462 of 161,500 have none:
delisted inside the month); a rebuild from inputs truncated at 2010-09-15
(`--cutoff`) reproduces the universe on all 140 common months and the returns and
market on all 2.7M common rows exactly.
