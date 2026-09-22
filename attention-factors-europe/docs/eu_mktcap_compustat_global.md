# European company market caps from Compustat Global

Status: first build 2026-09-17, eleven 1999 euro founders (AT BE DE ES FI FR IE IT LU NL
PT), 1999-01 to 2025-12, 324 month ends. Decisions below were checked against
`comp.g_secd`, `g_security`, `g_company`, `g_exrt_dly` on WRDS that day; every number
comes from `data/eu/mktcap_build_report.txt`. NO universe size is fixed yet: this build
exists to choose one.

Build: `python scripts/build_eu_mktcap.py` with `configs/universe_eu_compustat.yaml`
(`--offline` reruns transform + report from the extracts on disk, ~1 min;
`--offline --refresh-headers` refetches only the two header tables). Full pull ~12 min.
Code: `src/afe/data/compustat_global.py`. Output under `data/eu/` (gitignored).
Tests: `tests/test_compustat_global.py` (cross-listing collapse, eligibility, FX,
turnover screen).

## Definition

Market cap of company g at month end t, in EUR millions, = sum over its common share
classes (`tpci = '0'`) listed in its home exchange country of
`prccd / qunit * fx(curcdd -> EUR) * cshoc`, taken from the `monthend = 1` rows of the
daily security file. Same idea as the US permco sum (Shell A + B add up), plus the
collapse of cross-listings in point 2.

A company-month is ELIGIBLE (counted in the cutoffs) when
(a) the header links it to the market: `fic`, `loc` or the exchange country of `prirow`
    is in the country set, and
(b) its most active listing in the set is alive: trailing 6-month median of
    month-end-day volume / shares >= 3e-5.
`company_month_mktcap.parquet` keeps ALL companies with a cap and the flags
`header_link`, `active`, `eligible`, `turnover`; the CSV holds eligible rows only.

The ranking is AS-OF: row m ranks by cap at the end of m. The universe for trading month
m+1 is row m. The lag lives downstream, as in the US build.

## What the data forced

1. **No monthly share count in Global.** `comp.g_secm` carries prices only, so the
   month-end rows of `comp.g_secd` (`cshoc`) are the source. 93-99% of priced
   common-share rows carry `cshoc` (lowest 2016-18). `monthend = 1` marks each listing's
   own last row of the month, so dates differ inside a month (12-30 vs 12-31); rows are
   keyed on the calendar month end, the trade date kept as `price_date`.

2. **Cross-listings share one `cshoc`.** One share class often trades on several
   exchanges, each a separate `iid` (Fiat: Milan 01W, Paris 04W, Xetra 05W; every DAX
   name on Xetra, Frankfurt, Stuttgart, Munich, ...), all carrying the SAME `cshoc`.
   Summing over `iid` as `compustat_us` does would multiply caps by the number of
   listings. Listings are collapsed to classes inside the home country: rows of one
   company-month with the same ISIN, or an identical `cshoc`, are one class, priced by
   the most active listing. Distinct classes still add up.

3. **Header primary-issue tags cannot identify the home market.** Frankfurt alone lists
   ~11,000 gvkeys, mostly dormant secondary lines of foreign companies. The first build
   used `g_company.prirow` (primary issue, rest of world) and got Accenture as the
   largest German company (EUR 243bn, its `prirow` IS its Frankfurt line) while
   excluding Linde plc, Fiat and Alcatel (a NYSE `priusa` next to their real
   Frankfurt/Milan/Paris listing). No combination of `prirow`/`priusa`/`fic` separates
   the two cases. Trading activity does: real home listings turn over 0.1-1.2% of
   shares on the month-end day (Siemens 0.33%, Total 0.31%, CRH 0.23%, Fiat Milan
   1.2%), dormant lines 0.0001% or less (Accenture, Medtronic, Spotify, Tencent, and
   Total's own Brussels/Frankfurt lines). Hence rule (b). The 3e-5 threshold sits
   below the largest German names in 1999-2000 (Deutsche Telekom 8e-5), when g_secd
   still reports Frankfurt FLOOR volume; from 2001 those jump 30x.
   Side effect, wanted: (b) also drops listed subsidiaries with a sub-3% float and
   Frankfurt shells. Largest removed: Elf Aquitaine 2000-07 (EUR 86bn on 279mn shares,
   >95% Total), Audi (255 months), EnBW (319), CIC, Ergo, Allianz Leben, Hoechst
   1999-2004, Wella after P&G, Kabel Deutschland after Vodafone, AIB 2011-17 (99.8%
   state), and "EPG Engineered Nanoproducts", a Frankfurt shell at EUR 31bn. In all
   3.8% of the pooled top-500 rows, mostly 1999-2004. Dassault Aviation (float ~10%
   before 2016) goes too; defensible for a stat-arb universe.
   Volume is never zero in g_secd, only missing. A blank counts as no trading when the
   country's DOMESTIC listings (fic == exchange country) have >= 50% volume coverage
   that month; otherwise it is skipped and a listing with only blanks is kept as
   indeterminate. That is Ireland before 2000-06 (0% coverage, then 85%+), Luxembourg
   (17-64%), Austria Q1 1999, plus single IPO months. Exchange-level coverage does not
   work: on venue 154 foreign lines are 75% of rows and vote the coverage down to 10%.

4. **Country is the exchange country of the most active listing, not incorporation.**
   Airbus (NLD, Paris), ArcelorMittal (LUX, Amsterdam), Stellantis and Ferrari (NLD,
   Milan), Aegon (BMU, Amsterdam) sit where they trade. 10,921 of 162,000 top-500 rows
   have fic != home country. Shell (fic GBR, primary London) and Unilever PLC are OUT
   despite liquid Amsterdam lines: no header link to a euro country. They come back
   through London when GB is added. Unilever NV's 1999-2020 history is under the PLC
   gvkey, so the Netherlands lacks it in this run.

5. **Currency.** Prices are in the exchange's quotation currency, EUR for 1.39mn of
   1.40mn rows; the rest USD/GBP/JPY. Conversion uses Compustat's GBP cross rates
   (`g_exrt_dly`, units per GBP) at the trade date, else the latest within 7 days.

6. **Preferred shares (`tpci = '1'`) are pulled but not counted.** VW prefs EUR 49bn
   next to EUR 140bn ords; Porsche AG (2022 IPO) and Porsche SE have ONLY prefs listed,
   so they are absent from a common-only universe; Sartorius, Henkel, BMW, Fuchs have
   both. Which class to count and to trade is open.

7. **Price status.** `prcstd` is 10 (traded close, 84% of rows) or 5 (carried price, no
   volume ever, 16%). Nothing is filtered on it; the turnover screen absorbs most of it.

## Numbers that matter for sizing (EUR bn, yearly median of the 12 month ends)

Pooled cap at rank k, eligible companies per month in brackets:

| year | n | 100 | 200 | 300 | 400 | 500 | 750 | 1000 |
|------|------|------|------|------|------|------|------|------|
| 1999 | 1749 | 8.4 | 3.1 | 1.7 | 1.05 | 0.69 | 0.30 | 0.15 |
| 2005 | 2215 | 9.4 | 4.1 | 2.2 | 1.41 | 0.90 | 0.34 | 0.17 |
| 2011 | 2097 | 9.5 | 3.8 | 2.1 | 1.24 | 0.80 | 0.30 | 0.13 |
| 2015 | 1907 | 14.2 | 5.9 | 3.1 | 1.89 | 1.18 | 0.40 | 0.16 |
| 2021 | 1986 | 19.7 | 8.5 | 4.4 | 2.68 | 1.58 | 0.60 | 0.26 |
| 2025 | 1853 | 21.7 | 8.6 | 3.9 | 2.17 | 1.32 | 0.42 | 0.16 |

Per-country cap at rank 20 / 30 / 40 / 50 in 2005 and 2021, and eligible companies:

| | DEU | FRA | ITA | ESP | NLD | BEL | FIN | AUT | IRL | PRT | LUX |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2005 rank 20 | 9.1 | 14.4 | 7.6 | 5.7 | 4.7 | 1.4 | 1.6 | 1.1 | 0.7 | 0.2 | - |
| 2005 rank 30 | 6.1 | 10.8 | 4.0 | 3.3 | 1.6 | 0.8 | 0.7 | 0.4 | 0.1 | 0.07 | - |
| 2005 rank 40 | 3.7 | 5.8 | 2.8 | 2.4 | 1.0 | 0.4 | 0.4 | 0.1 | 0.02 | 0.01 | - |
| 2005 rank 50 | 2.3 | 4.8 | 2.4 | 1.7 | 0.8 | 0.3 | 0.25 | 0.05 | - | - | - |
| 2005 companies | 612 | 632 | 256 | 127 | 152 | 140 | 131 | 69 | 44 | 44 | 9 |
| 2021 rank 20 | 26.7 | 36.7 | 9.1 | 6.1 | 11.1 | 3.8 | 2.7 | 2.3 | 0.3 | 0.09 | - |
| 2021 rank 30 | 18.8 | 20.5 | 5.2 | 3.1 | 5.1 | 1.8 | 1.5 | 0.5 | - | 0.0 | - |
| 2021 rank 40 | 13.4 | 12.7 | 3.8 | 2.2 | 2.6 | 1.1 | 0.9 | 0.15 | - | - | - |
| 2021 rank 50 | 10.0 | 9.7 | 2.4 | 1.3 | 1.0 | 0.7 | 0.6 | 0.0 | - | - | - |
| 2021 companies | 464 | 542 | 364 | 138 | 100 | 110 | 162 | 48 | 24 | 28 | 9 |

Composition of the pooled top-500 (Dec 2021): DEU 130, FRA 120, ITA 63, NLD 45, ESP 44,
BEL 31, FIN 27, AUT 22, IRL 9, PRT 9, LUX 0. In 1999: FRA 129, DEU 94, ITA 78, NLD 59.
Full tables: `data/eu/cutoffs_*.csv`, `composition_pooled.csv`.

Reading: the 500th euro-area company is EUR 0.5-1.6bn depending on the year. A pooled
500 is 50% DE+FR (Dec 2021) and takes Austria to its 22nd company (its 20th is EUR
2.2bn, its 30th 0.5bn), Ireland and Portugal to their 9th. Per-country quotas of 40-50
only exist for DE, FR, IT, ES, NL; BE and FI have ~1bn at rank 30-40; AT, IE, PT run
out of companies above EUR 0.2bn by rank 30-40.

## Open

- Universe size and shape: pooled top-N vs per-country quotas vs a cap floor, given the
  numbers above; and whether the non-euro markets (GB, CH, SE, DK, NO) come in before
  the choice, since they change the pooled cutoffs materially.
- Preferred shares in the cap (6) and which line to trade for dual-class names.
- Turnover threshold 3e-5 and window 6 are set from the observed gap, not optimised;
  a monthly volume (`g_secm.cshtrm`) would be a cleaner activity measure than the
  month-end day.
- Unilever NV history (4); Greece 2001 and later euro entrants.
