# European builds `europe17` and `europe12`: top 200, 1990-2025, ECU/EUR

Status: first build 2026-09-30. Two pooled top-200 universes from Compustat Global with
the rules of the euro-11 build (docs/eu_mktcap_compustat_global.md), the US
characteristic definitions (docs/us_characteristics.md) and one numeraire. The
availability report up to 1997 is generated from the outputs:
`data/europe/availability_report.md` (+ `availability_tables/*.csv`).

| version | exchange countries (g_security.excntry) | config |
|---|---|---|
| europe17 | GBR DEU FRA CHE NLD ITA ESP SWE BEL DNK FIN NOR IRL PRT AUT LUX GRC | configs/europe17_data.yaml |
| europe12 | the first twelve: London .. Oslo (no Dublin, Lisbon, Vienna, Luxembourg, Athens) | configs/europe12_data.yaml |

```
python scripts/fetch_europe_extract.py                                        # WRDS: month ends, headers, FX (shared)
python scripts/build_europe_universe.py --config configs/europe17_data.yaml   # cap table, ranking, lines (and europe12)
python scripts/fetch_europe_raw.py --configs configs/europe17_data.yaml configs/europe12_data.yaml --tables funda daily fx
python scripts/fetch_europe_raw.py --configs configs/europe17_data.yaml configs/europe12_data.yaml --tables jkp   # in parallel
python scripts/build_europe_dataset.py --config configs/europe17_data.yaml    # stage 2 (and europe12)
python scripts/fill_europe_holidays.py --config configs/europe17_data.yaml    # stage 2b: holfill/ (and europe12)
python scripts/report_europe_availability.py --configs configs/europe17_data.yaml configs/europe12_data.yaml
```
WRDS scripts run in the `py3.12-4338` env through `conda run` (its python.exe alone exits
127: the MKL DLLs need the activation); everything offline in `afe`. `.env` lives in the
top-level clone only, so pass `WRDS_USERNAME` in the environment from a worktree.

Data: `data/europe/private/` = the shared WRDS store (month-end extract, headers, FX,
`raw/` = daily, g_funda, JKP for the union of both versions' lines, `raw/external/` =
Eurostat's ECU check file and Ken French's `Europe_3_Factors_Daily_CSV.zip` as downloaded);
`data/europe17|12/{shared,private}` = each version's tables.

Stage 2b (`src/afe/data/europe_holidays.py`, the headline convention since 2026-10-10):
before ~2010 Global has no row on a line's own exchange holiday, so the 30-day slot
lookback drops the whole country for 30 days after each one (members tradable 41-80% of
days 1993-2009 against 96% after). `data/<version>/holfill/` holds the shared tables with
gaps of <= 3 pooled days filled by a zero local return (the day's currency move, the
next day's return divided by it, so the gap compounds unchanged) and `returns.traded`
False there and on Global's carried closes; runs hold positions on those days
(`execution.stale_when_closed`). europe17: 25,185 filled member-days, 94% of them
before 2010. Identical to the 2026-10-07 experiment copy
`private/exp_holfill_traded`.

## Decisions

1. **Universe.** Month M = the 200 largest eligible companies by cap at the end of M-1,
   1990-01 .. 2025-12 (ranked from 1985-12 for the pre-sample market). Eligibility is the
   euro-11 rule unchanged: header link (fic, loc or prirow's exchange country in the set)
   AND trailing 6-month median of month-end-day volume / shares >= 3e-5, common shares,
   cross-listings collapsed to classes WITHIN the home country, home = the most active
   listing's exchange country. Summing classes across countries was tried and rejected:
   foreign cross-listings often carry their own ISIN or a stale share count, so it double
   counts (Total, Sanofi, ING); the price is that Shell A/B count only their home-country
   class (Unilever NV/PLC, one gvkey, is summed by an explicit correction, below). The line
   that carries a company (returns, characteristics)
   is its most active class, unless that class holds under 5% of the cap
   (`filters.min_class_share`), then its largest class; lines that are not shares
   (`compustat_global.non_equity`: VVPR strips, subscription / bonus / offer rights,
   stock-dividend rights, nil-paid) are dropped before anything else (2026-10-03, see
   Checks).
   **Cap corrections (2026-10-07)**, found by `scripts/check_caps.py` (dividend yield of the
   price feed vs the accounts; cap change vs price change): (a) Chi-X / Cboe Europe quotes
   (g_secd exchg 349, London, from 2012-05, no ISIN) are not listings: their share count
   differed slightly from the London line's in about half the months, so they were summed
   as a second class and doubled most large UK caps 2012-19 (14-17% of the top 200's total
   cap at the worst month of each year; Rotork's Chi-X line carried 10x the shares and put
   a EUR 3bn company in the top 100, 2012-15). (b) Buyback, second-trading, tendered
   ("ASD") and redemption-share lines are not classes (`non_equity`; Adecco 2015-24 was 5x).
   (c) A class within 2% of another in share count AND price is the same class
   (`filters.dup_tol`). (d) Participation certificates count as classes (tpci '8': Swiss PS,
   French certificats d'investissement), and Genussscheine on SIX (tpci 'Q',
   `filters.genussschein_countries`: Roche's, ~80% of its equity; Roche was sized on its
   bearer shares alone, rank 50-100 instead of top 10, 1989-2011); the extract got these
   rows with `fetch_europe_extract.py --add-issue-types 8 Q`. (e) Dated corrections in the
   config (`corrections`): Eurocommercial's share count in shares against a price per
   depositary receipt of ten (to 2005-04-26), Unilever NV + PLC summed as one company to
   the 2020 unification with NV's duplicate lines excluded, Paribas' frozen FRF100 line,
   one-month share-count switches (Atos 2024-11, Vetropack 2020-04, Bonheur to 1990-05),
   and second lines the rules miss (Lindt, Logitech, UBS AG after 2014, MTU young shares).
   Effect on europe17: 1,578 of 86,400 universe member-months swap (1,122 of them
   2012-19), the old top 200's total cap was 14-18% too large in 2012-18 and 10% in
   2023-25; 825 -> 795 companies ever in the as-of ranking. Previous outputs:
   data/europe1{7,2}/private/prev_20261007_cap_fixes.
2. **Numeraire.** ECU to 1998-12-31 = the official basket (three compositions, 1979, 1984,
   1989) valued at Compustat's GBP cross rates; EUR from 1999-01-01 = Compustat's quote;
   one to one at the switch (Reg. 1103/97). Pseudo currency `XEU` in the FX table, so the
   existing conversions run unchanged. Checks in every universe report: the basket
   reproduces the 1998 conversion rates within 1.5 bp, jumps < 2 bp at the revisions, and
   tracks Eurostat's official daily ECU within 0.07% a year. Compustat's own `XEU` (off by
   1.1-1.3% in 1995-96) and its pre-1999 synthetic EUR (off by up to 4.2% in 1991) are not
   used. Returns carry the FX move (price converted day by day, then the ratio).
3. **Characteristics** = the 39 of the US build through characteristics.py, fed with
   company-level series (`europe_panel.py`): the company's line in month M = the line that
   priced its cap at the end of M-1. Substitutes for what Global lacks, documented in
   `europe_panel.py` and `europe_accounts.py` (fundamentals, JKP): Spread = Corwin-Schultz
   high-low (local prices, overnight adjustment,
   >= 10 days); market, SMB and HML = Ken French's Europe three factors (value-weighted,
   all sizes; 16 countries = europe17's less Luxembourg), every column of which is in USD:
   the market's USD total return (Mkt-RF + RF) converted to the numeraire, SMB and HML
   (dollar long-short spreads) times the day's change of the numeraire per dollar, the
   same on both legs. From 1990-07-02, FF's first day; before it the top 200's own, which
   Beta's five-year window needs: its value-weighted market and a 2x3 sort (cap median,
   BEME 30/70; config `factors`). Resid_Var only where both months of
   its window have all three factors; ni = nicon else ib + xido; mib else mibt;
   sale = revt for FS (banks, insurers); pstkrv/pstkl absent; xad zero. Fundamentals in
   the numeraire at the fiscal year end, usable from June of the next year (US rule).
4. **JKP.** `contrib.global_factor`, all columns, every line of the ever-members, eom
   1985-2025, pulled in 7-year chunks (each query scans the table, ~4 min). The table of a
   version = members' rows as of the end of M-1 (home line, else JKP's primary
   security), USD amounts converted at eom, `ret` re-expressed in the numeraire,
   `ret_exc` = that minus our rf, `ret_exc_lead1m` dropped (a future return). JKP's
   return-based characteristics stay as JKP computed them, from USD returns.
5. **Risk-free**: Bundesbank 1-month rates, Frankfurt banks' funds before 1990-07,
   FIBOR to 1998, EURIBOR from 1999: a Mark rate, not an ECU rate, before 1999 (open).
   Trading day t earns the previous trading day's quote over the calendar days since
   (act/360). The first build gave each trading day one day's interest, so a year added
   up to only ~72% of the quoted rate (fixed 2026-09-30). FF's RF, the US 1-month T-bill,
   is a dollar rate and is used only to rebuild FF's dollar market from its Mkt-RF.

## Checks (2026-09-30 build)

- Size: europe17 824 ever-ranked companies (1,068 lines since 2026-10-03, 1,131 before),
  757 universe members, returns 4.29M rows; europe12 793 (1,017 lines; 1,079). Union raw
  pull: 1,147 lines, 833 companies, 5.34M daily rows 1984-2025, 19,100 g_funda records,
  224k JKP rows. Stage 2 ~3 min each.
- Line selection (rebuild of 2026-10-03; previous outputs in
  `data/europe1{7,2}/private/prev_20261003_line_selection`): the cap table had let the most
  active line price a company even when it was not a share or a sliver of the cap.
  Electrabel 2006-02 .. 2007-08 was carried by its VVPR strip (EUR 0.01, daily returns of
  +100% / -50%; in the K=30 attention run December 2006 alone was 56% of the squared
  returns), Petrofina 1999-2001 by its strips, KBC, AXA, Repsol, Ferrovial briefly by
  strips or rights, Land Securities 2002-03 by the B shares of its return of capital, and
  SEB, Handelsbanken, GUS, Soc. Gen. de Belgique, Munich Re, Sanofi, Vinci (new-share
  lines) by classes under 5% of the cap. Now 383 europe17 member-months (33 companies) take
  another line; with the strip gone Electrabel 2006-05 .. 2007-07 and Petrofina 2000-05 ..
  08 fail the turnover screen like Elf (13 member-months, 13 others enter). Two lines not
  pulled before (Sandvik 01W, Sydkraft 05W, restricted A shares 1984-92) were added with
  `fetch_europe_raw.py --add-lines`. Member-days with |return| > 40%: 119 -> 24 in europe17
  (117 -> 22 in europe12); what is left are events (VW 2008, Steinhoff, Bankia, Atos).
  Features: 9% of cells moved, 1.2% by more than one rank step, 0.09% of the other
  companies' cells by more than 0.05.
- Prefix invariance: `--cutoff 1996-09-15` rebuild of europe17 reproduces the universe
  (16,200 rows), features (344,400 rows, all 79 columns, max difference 0) and returns
  (751,826 rows) of the full build exactly, with factors, rf and market identical too;
  re-run after the FF factors and rf change and after the line selection change.
- Factors (rebuild of 2026-09-30): FF Europe in the numeraire against the top 200's own,
  1990-07 .. 2025: market daily correlation 0.93, monthly 0.98, monthly tracking error
  2.7% a year, the top 200 returning 1.1 points a year more (9.7% against 8.6% a year in
  1991-2025, 3.7 points a year in 1991-97); SMB daily correlation 0.43 (a size split
  inside the 200 largest is not a size factor), HML 0.83. On the days both calendars
  share, the factors equal FF's USD file converted as in Decisions 3 to 1e-16. Only Beta,
  Resid_Var (ranks correlating 0.996 and 0.973 with the first build's) and the rf column
  moved in features; JKP `ret_exc` fell 0.07 pp a month (0.15 in 1990-98) with the rf
  accrual.
- Numeraire: see Decisions 2. `tests/test_europe_fx.py`: basket by date, ECU -> EUR
  switch. `tests/test_europe_panel.py`: Corwin-Schultz on a pure bid-ask bounce (recovers
  (H-L)/mid exactly) and the overnight gap, the line carrying a company, the 2x3 sort.
  `tests/test_europe_panel.py` also: the rf accrual (Friday to Monday = three days) and
  the FF factors in the numeraire (the market's FX move, SMB times the dollar's change, a
  file day off the calendar, the T-bill spread). `tests/test_compustat_global.py`: strips
  and rights neither price nor count, a class under 5% of the cap does not price. 88 tests
  pass with the suite.

Code map: `europe_fx.py` (numeraire) -> `fetch_europe_extract.py`, `build_europe_universe.py`,
`fetch_europe_raw.py` (stage 1) -> `europe_panel.py` (company series, spread, market and
factors, characteristic inputs), `europe_accounts.py` (fundamentals, JKP table),
`europe_coverage.py` (coverage tables, build report) -> `build_europe.py` +
`build_europe_dataset.py` (stage 2) -> `report_europe_availability.py`.

## Open

- Sample start: volume (turnover screen, LTurnover, SUV, Vol, Spread inputs) does not
  exist before 1992 and is patchy to 1996; see the availability report.
- The screen is blind before 1993: eligibility is the header link alone there.
- An ECU money-market rate before 1999; delisting returns (none in Global); preferred
  shares; the known leaks of US-listed firms (Linde plc 2023-25, CNH 2024-25).
- 2024-25 month-end volume gaps let a London line win the home-country choice for a few
  continental companies (Mercedes-Benz, UMG in some months; Roche and the Chi-X cases are
  fixed, 2026-10-07).
- Left after the 2026-10-07 cap corrections (`check_caps.py` lists them): one-month class
  gaps where a class has no month-end row (11 member-months, mostly Swedish A/B before
  1999), pricing-line flips between Swedish restricted / free lines before 1993 (ABB AB
  1991), Belgian AFV series 1990-94 counted as classes.
