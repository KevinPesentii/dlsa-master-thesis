# US data build: the 39 characteristics, as implemented

Status: first build 2026-09-15. Source of the definitions: `Data Collection/
DataCollectionDetails.docx`, which transcribes Table IA.XVIII of Chen, Pelger & Zhu (2024)
and adds four daily inputs from the attention paper. Code: `src/afe/data/characteristics.py`
(one function per characteristic, this file mirrors it), `us_panel.py` (who is in, when a
number is known), `build_us.py` (assembly). Config: `configs/us_data.yaml`.

Two stages, run separately:

```
python scripts/fetch_us_raw.py        # once: WRDS -> data/us/private/raw/*.parquet (~2 h, resumable)
python scripts/build_us_dataset.py    # offline: raw -> returns / universe / features
```

## Sources (WRDS)

| table | used for |
|---|---|
| `crsp.dsf_v2` (CIZ daily) | returns, prices, volume, closing bid/ask, daily high, cap, adjustment factor |
| `crsp.dsf_v2`, `dlydelflg = 'Y'` rows | the delisting-day row of every security: its `dlyret` is the delisting return. Pulled separately into `crsp_delisting.parquet` because CRSP blanks the share/security/trading flags on it and the common-share filter drops it; merged into the pool's daily rows in `us_panel.load_daily`. Lehman: -60% on 2008-09-18 after -56.7% on 09-17. The 2026-09-15 build had none of these rows (1,205 for the pool, 593 while a universe member; median +0.1%, mean +0.8%, min -60%). The monthly file needs no fix: its delisting-month row keeps normal flags and `mthret` already spans to the delisting price. |
| `crsp.msf_v2` (CIZ monthly) | monthly returns, prices, volume, shares, cap, eligibility flags |
| `crsp.ccmxpf_lnkhist` | permno <-> gvkey with date ranges (LU/LC links; P, C, J) |
| `comp.funda` | annual fundamentals (items listed in `wrds_us.FUNDA_ITEMS`) |
| `ff.factors_daily` | mktrf, smb, hml, rf for Resid_Var, Beta and the `rf` feature |

Compustat is NOT used for prices in this build (that was step 1, the Compustat-only
universe, kept as a cross-check). CLAUDE.md fixes CRSP CIZ for the US. It supplies the
share base of depositary receipts (below) and, via `curcd`, the currency of the
fundamentals: Canadian filers report in CAD and are converted to USD at the fiscal year
end (`comp_funda_currency.parquet`, `fx_daily.parquet`; every other row is USD).

## Who is in

Since 2026-09-23 the paper's universe, whose Figures 3-4 name depositary receipts
(Toyota, Mizuho, NatWest, Ecopetrol), Canadian listings, MLP units, REITs and Alphabet
twice. Eligible pool, from the monthly CIZ flags (historical per row, so safe to filter
on): `sharetype NS/AD/UG/SB/CE, securitytype EQTY, securitysubtype COM, usincflg Y/N,
issuertype CORP/ACOR/REIT, primaryexch N/A/Q, conditionaltype RW, tradingstatusflg A`,
cap > 0 (config `eligibility`). ACOR marks corporations CRSP later records as acquired
(Kraft, RCA); securitysubtype COM keeps out ETFs and closed-end funds. The 2026-09-22
build (share codes 10/11: NS, Y, CORP, one line per company) is in
`data/us/private/prev_20260922`.

Each line (permno) is ranked on its own cap (config `universe.unit: line`), so a company
with two listed classes can hold two slots. A depositary receipt's cap is its company-level
cap, not CRSP's, which counts receipts outstanding only (Mizuho 2020-12: $0.2bn against
$32bn): underlying shares of its class over the receipt ratio x its close, from
`comp.secm` (cshom / adrrm) from 1998-04, before that `comp.funda` csho (receipt
equivalents) of a fiscal year at least six months old, never below CRSP's cap
(`us_panel.receipt_caps`). Since 2026-10-07 that number is checked against the issuer's
home-market cap in Compustat Global the same month (`us_panel.home_caps`, raw
`global_home/` from `fetch_us_raw.py --global-home`; largest active exchange country,
preferred shares included): more than 1.5x the whole company (`universe.receipt_home_tol`)
means a share base in the wrong unit and the home cap replaces it (funda csho in ordinary
shares: Fiat 1990-2000 ~5x, rank 1; secm receipt ratios: CNOOC 2004, Turkcell 2003; CRSP
counting underlying shares: TIM Participacoes 2011-20), unless the gap is over 100x
(`receipt_home_max_ratio`: then Global is the broken side, De Beers 1995-2001). A receipt
cap below its home company is left alone (one class of several; raising it would import
Global's errors). Company = permco; company cap = sum of its eligible lines' caps,
which is `LME` and the December ME of the value ratios. Universe for month M = the 500
largest lines by cap at the end of M-1; `month` in `universe.parquet` is the first trading
day of M; `mktcap_lag` in `returns.parquet` is the line's cap. A diagnostic build can start
the window further down (`sample.universe_first_rank`, default 1): `configs/us_data_rank301_800.yaml`
keeps the lines ranked 301-800, re-ranked 1-500 in `universe.parquet` (`cap_rank_all` in
`universe_asof.parquet` keeps the rank among all lines).

Screens added after the input audit of 2026-11-17 (`Thesis/diagnostics/2026-11-17_input_data_audit`,
private; config `universe`). They decide only who may be RANKED; a screened line keeps its cap
for LME and December ME (`us_panel.rankable`):
- `exclude_permnos`: three receipts whose CRSP or secm share count makes a micro cap a top-500
  company and that no home-market cap can check (OneSmart Education 2018-20, Genetic
  Technologies 2021, Crosswave 2002). They are dropped from the eligible pool altogether.
- `receipt_min_traded_share: 0.9` over `receipt_activity_months: 6`: a receipt whose line traded
  (volume > 0, `crsp_daily_extra`) on fewer than 90% of the days of the last six months is not
  ranked. CRSP prices a no-trade day at the quote midpoint, so dormant 1990s programmes
  (Daiei, Kubota, Mitsui, JAL) carried stale and quote-driven returns at ranks 40-100.
- `min_history_months: 1`: a line needs a CRSP monthly row in the month before the ranking
  month, so a listing is not ranked on its first month's price (Addentax 2022-08: $21bn on
  its IPO day, -95% the next). Spin-offs and new merger permnos wait a month too.
- `listed_within_days: 10` (Kevin's data check, same day): a line must still be listed at the end of
  the ranking month, its last daily return at most 10 days before the month end and no delisting row
  on or before it (`us_panel.listed_at_month_end`). A line that stopped mid-month was ranked on its
  monthly row and held a slot for the whole next month without a return (339 member-months).
- `receipt_carry_links: true`: a receipt whose CCM link has ended keeps that last link, so its share
  base and the home-market check still apply (AmBev's receipt had no link 1998-04..2013-11 and sat at
  CRSP's receipts-only cap, out of the universe, for 15 years).
The receipt guard (a funda count above 20,000x CRSP's receipt cap is dropped) applies only where
the issuer has no home-market cap that month; where it has one, the home-market check above
judges the count (the guard had thrown away Mitsubishi Bank's correct $43bn in Dec 1990 because
only 103 thousand receipts were outstanding).

Pool for characteristics = every permno that is ever a member (config `pool`). Ranks and
medians are over the 500 members of the day (`median_over`; CONFIRM 3 in schemas.md).

## When a number is known

- Monthly characteristics dated at month end m are used on every trading day of m+1.
- Daily characteristics dated d use data through d.
- Annual fundamentals: fiscal years ending in calendar year y are usable from the end
  of June y+1 (Fama-French; config `fundamentals.availability: june`). Alternative:
  `lag_months: 6` after each firm's own fiscal year end. A record older than
  `max_age_months` (30) is dropped rather than carried on.
- `me_dec` (denominator of A2ME, BEME, CF2P, Q, E2P) is the company cap at the December
  end of the fiscal year's calendar year, via the CCM link valid on that date. Since
  2026-11-17 it is used in month m only while the company cap of m is within 8x of `me_dec`
  grown by the stock's return since that December (`fundamentals.me_dec_max_ratio`,
  `us_panel.validate_dec_me`); otherwise those five characteristics are missing that month.
  It catches a December cap from CRSP's receipts outstanding, a CRSP share count in the wrong
  unit (EnCana Dec 2001: 255 thousand shares) and real recapitalisations. Point in time.
- Risk-free rate (`risk_free.source: ff_monthly`, since 2026-11-17): the daily file states rf in whole
  basis points (four values since 1990), so day t earns the month's 1-month T-bill rate for the
  calendar days since the previous trading day, (1 + rf_m) ** (days / days in month) - 1, with mktrf
  moved by the same amount (`build_us.rf_from_monthly`); written to `rf_daily.parquet` too. It reaches
  Beta, Resid_Var and the features' rf column; the PCA runner still reads `raw/ff_daily` unless its
  config sets `data.rf_file`.
- Missing items treated as zero: `txditc txdb pstk mib ivao xrd xad txp dlc`
  (config `fundamentals.fill_zero`). Everything else missing propagates to NaN.
- A missing characteristic takes the stock's last observed value if it has one, else,
  after ranking, the cross-sectional median (0 in rank units): Epstein et al. (2025),
  Section 4.1 (config `normalisation.missing: last_observed`, since 2026-09-24). The carry
  only looks back, so it is point in time. A daily characteristic carries at most 5
  trading days, a monthly one 12 months (`carry_max`); an annual one comes from the
  latest fiscal year in which it is not missing, within `max_age_months` of that year's
  end. `missing: median` is the median-only rule of Chen-Pelger-Zhu, used by every build
  before 2026-09-24 (kept in `data/us/private/prev_20260923_medianfill`). The build report
  shows coverage after the carry, before the median fill.

## Definitions

Notation: `x_lag` = same item one fiscal year earlier; `BE` = book equity; `ME_dec` as
above; `m` = month end; `t = m+1` = prediction month. Windows are in the config.

### Past returns
| name | definition |
|---|---|
| r2_1 | return of month m |
| ST_Rev | return of month m (identical to r2_1; both kept because the paper lists both) |
| r12_2 | compounded return over months m-11..m-1 (t-12..t-2); needs a price at m-12, a return at m-1, >= 8 of 11 months |
| r12_7 | compounded return over m-11..m-6 (t-12..t-7); >= 4 of 6 months |
| r36_13 | compounded return over m-35..m-12 (t-36..t-13); >= 16 of 24 months |
| Ret_D1 | return on day d |
| Ret_W1 | compounded return over the 5 trading days ending on d |
| STD_W1 | standard deviation of the 5 daily returns ending on d |

### Value (annual)
| name | definition |
|---|---|
| A2ME | AT / ME_dec |
| BEME | BE / ME_dec; BE = SH + TXDITC - PS, SH = SEQ, else CEQ + PS, else AT - LT; PS = PSTKRV, else PSTKL, else PSTK |
| C | CHE / AT |
| CF | (NI + DP - dWC - CAPX) / BE; dWC = WCAPCH, else WCAP - WCAP_lag (WCAPCH is empty for most firms after the 1980s) |
| CF2P | (IB + DP + TXDB) / ME_dec |
| Q | (AT + ME_dec - CEQ - TXDB) / AT (CEQ is common equity; the IA's "cash" label is a typo) |
| Lev | (DLTT + DLC) / (DLTT + DLC + SEQ) |
| E2P | IB / ME_dec |

### Investment (annual)
| name | definition |
|---|---|
| Investment | (AT - AT_lag) / AT_lag |
| NOA | ((AT - CHE - IVAO) - (AT - DLC - DLTT - MIB - PSTK - CEQ)) / AT_lag |
| DPI2A | ((PPEGT - PPEGT_lag) + (INVT - INVT_lag)) / AT_lag |

### Trading frictions
| name | definition |
|---|---|
| AT | total assets, USD mn (annual) |
| LME | company cap at m, USD mn |
| LTurnover | monthly volume / (shrout x 1000); Nasdaq volume divided by 2.0 / 1.8 / 1.6 before 2001-02 / 2002 / 2004 (Gao-Ritter, `volume.nasdaq_divisor`, since 2026-11-17) |
| Rel2High | (prc_m / cumfacpr_m) / max over the past 252 days of (high / cumfacpr); >= 120 days. Ratio form, so a later split cannot change a past value |
| Resid_Var | residual variance of daily excess returns on FF3 + intercept over the 2 calendar months ending at m; >= 20 days |
| Spread | mean over month m of the daily Corwin-Schultz (2012) high-low estimate (`hl_spread`), two consecutive traded days, prices over cumfacpr; >= 10 days (`spread.source: corwin_schultz`, since 2026-11-17, as in Europe). Before: (ask - bid) / midpoint of CRSP closing quotes (`quotes`), whose NYSE quotes are several times too wide until a break in 2003 |
| SUV | daily volume regressed on 1, r+, r- over months m-6..m-1 (>= 60 days); mean residual in month m under those coefficients, divided by the estimation residual s.d. |
| Variance | sample variance of daily returns over the 2 calendar months ending at m; >= 20 days |
| Vol | value traded (shares x close, USD) over the 5 trading days ending on d (since 2026-11-17; shares traded before, which ranked by price level); Nasdaq shares as for LTurnover |
| Beta | corr(3-day log excess returns, market; 1260 days, >= 750) x sd(daily log excess; 252 days, >= 120) / same for the market. No shrinkage |

### Profitability (annual)
| name | definition |
|---|---|
| PROF | GP / BE |
| CTO | SALE / AT_lag |
| FC2Y | (XSGA + XRD + XAD) / SALE |
| OP | (REVT - COGS - XINT - XSGA) / BE ("TIE" in the spec is Compustat XINT) |
| PM | OIADP / SALE |
| RNA | OIADP / NOA_level_lag, NOA_level = operating assets - operating liabilities |
| D2A | DP / AT |

### Intangibles (annual)
| name | definition |
|---|---|
| OA | ((NWC - NWC_lag) - DP) / AT_lag; NWC = ACT - CHE - LCT - DLC - TXP, literally as in the IA (Sloan nets DLC and TXP out of LCT; one-line switch in `noncash_working_capital`) |
| OL | (COGS + XSGA) / AT |
| PCM | (SALE - COGS) / SALE |

## Deviations and judgement calls (to revisit)

1. r2_1 and ST_Rev are the same column. Downstream may want to drop one.
2. SUV's estimation window (6 months before m) is our reading of Garfinkel (2009); the
   IA only says "previous month" for the evaluation. Config `suv_estimation_months`.
3. Momentum minimum-observation rules (8/4/16) are ours; Fama-French require all months
   for r12_2 except that some may be missing-coded. Config `momentum_min_obs`.
4. Beta without the 0.6/0.4 shrinkage of Frazzini-Pedersen, as the IA does not mention it.
5. CF's working-capital fallback (WCAP change) is ours; WCAPCH is empty for Apple in
   every recent year.
6. Rank range is (-0.5, 0.5] via `rank(pct=True) - 0.5`, same as `synthetic.py`.
   CONFIRM 1 in schemas.md still stands.
7. Foreign-incorporated companies (`usincflg N`) and REITs are out, mirroring CRSP share
   codes 10/11. Step 1's Compustat universe kept REITs; the build report shows the
   overlap between the two universes by year.
8. Volume-based characteristics (LTurnover, Vol, SUV) divide Nasdaq volume by the Gao and
   Ritter (2010) factors since 2026-11-17 (Nasdaq double-counts dealer trades relative to
   NYSE, Anderson & Dyl 2005; Jensen-Kelly-Pedersen and Chen-Zimmermann adjust, Chen-Pelger-Zhu
   do not). Before the adjustment Nasdaq was 13% of members but 34-42% of the top turnover
   quintile before 2004.
10. Daily returns (since 2026-11-17, config `returns.broken_quote_spread: 0.10`): a no-trade
   day whose closing quote is wider than 10% of its midpoint keeps the previous close with a
   zero return, and CRSP's return of that day is compounded into the next row, so two-day
   returns are unchanged (`us_panel.fold_broken_quotes`; ABN AMRO's receipt 2007-10-04: bid
   53, ask 141, close 97, +82.6% then -45.4% at rank 52). `corrections.drop_delisting` drops
   delisting rows that are not a holder's return (ABB's ADR -94.6% in 2023).
9. Point in time: `scripts/build_us_dataset.py --cutoff DATE` rebuilds from truncated
   inputs. On the 2010 smoke build, features and universe at the cutoff were identical to
   the full build and returns identical on common rows. `returns.parquet` carries the
   pre-entry history of every stock that is EVER a member, so its row set (not its
   values) depends on the sample end; a prefix test on it must compare common rows.
