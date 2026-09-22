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

Compustat is NOT used for prices or shares in this build (that was step 1, the
Compustat-only universe, kept as a cross-check). CLAUDE.md fixes CRSP CIZ for the US.

## Who is in

Eligible pool, from the monthly CIZ flags (historical per row, so safe to filter on):
`sharetype NS, securitytype EQTY, securitysubtype COM, usincflg Y, issuertype CORP,
primaryexch N/A/Q, conditionaltype RW, tradingstatusflg A`, cap > 0. This is the CIZ
spelling of share code 10/11 + exchange 1/2/3. REITs are `issuertype REIT` and are out
by default (config `eligibility.issuertype`).

Company = permco. Company cap = sum of the caps of its eligible share classes (Alphabet
GOOGL + GOOG, Berkshire A + B). The class with the largest cap is the company's `sec_id`
(its permno, as a string) for that month and supplies the return / volume / quote series.
Universe for month M = the 500 largest companies by cap at the end of M-1; `month` in
`universe.parquet` is the first trading day of M. The same cap is `LME` and `mktcap_lag`.

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
  end of the fiscal year's calendar year, via the CCM link valid on that date.
- Missing items treated as zero: `txditc txdb pstk mib ivao xrd xad txp dlc`
  (config `fundamentals.fill_zero`). Everything else missing propagates to NaN.
- After ranking, a missing characteristic is set to the cross-sectional median (0 in
  rank units), as in Chen-Pelger-Zhu. The build report shows coverage before the fill.

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
| LTurnover | monthly volume / (shrout x 1000) |
| Rel2High | (prc_m / cumfacpr_m) / max over the past 252 days of (high / cumfacpr); >= 120 days. Ratio form, so a later split cannot change a past value |
| Resid_Var | residual variance of daily excess returns on FF3 + intercept over the 2 calendar months ending at m; >= 20 days |
| Spread | mean over month m of (ask - bid) / midpoint from closing quotes, ask > bid > 0; >= 10 days |
| SUV | daily volume regressed on 1, r+, r- over months m-6..m-1 (>= 60 days); mean residual in month m under those coefficients, divided by the estimation residual s.d. |
| Variance | sample variance of daily returns over the 2 calendar months ending at m; >= 20 days |
| Vol | shares traded over the 5 trading days ending on d |
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
8. Volume-based characteristics (LTurnover, Vol, SUV) use CRSP volume as reported.
   Nasdaq volume before 2001 double-counts dealer trades relative to NYSE (Anderson &
   Dyl 2005); Chen-Pelger-Zhu do not adjust and neither do we. Cross-sectional ranking
   within a large-cap universe limits the effect.
9. Point in time: `scripts/build_us_dataset.py --cutoff DATE` rebuilds from truncated
   inputs. On the 2010 smoke build, features and universe at the cutoff were identical to
   the full build and returns identical on common rows. `returns.parquet` carries the
   pre-entry history of every stock that is EVER a member, so its row set (not its
   values) depends on the sample end; a prefix test on it must compare common rows.
