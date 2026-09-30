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
python scripts/report_europe_availability.py --configs configs/europe17_data.yaml configs/europe12_data.yaml
```
WRDS scripts run in the `py3.12-4338` env through `conda run` (its python.exe alone exits
127: the MKL DLLs need the activation); everything offline in `afe`. `.env` lives in the
top-level clone only, so pass `WRDS_USERNAME` in the environment from a worktree.

Data: `data/europe/private/` = the shared WRDS store (month-end extract, headers, FX,
`raw/` = daily, g_funda, JKP for the union of both versions' lines, `raw/external/` =
Eurostat's ECU check file); `data/europe17|12/{shared,private}` = each version's tables.

## Decisions

1. **Universe.** Month M = the 200 largest eligible companies by cap at the end of M-1,
   1990-01 .. 2025-12 (ranked from 1985-12 for the pre-sample market). Eligibility is the
   euro-11 rule unchanged: header link (fic, loc or prirow's exchange country in the set)
   AND trailing 6-month median of month-end-day volume / shares >= 3e-5, common shares,
   cross-listings collapsed to classes WITHIN the home country, home = the most active
   listing's exchange country. Summing classes across countries was tried and rejected:
   foreign cross-listings often carry their own ISIN or a stale share count, so it double
   counts (Total, Sanofi, ING); the price is that Unilever NV/PLC and Shell A/B count only
   their home-country class.
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
   >= 10 days); market = value-weighted numeraire return of the month's top 200; SMB/HML =
   2x3 sort of the same 200 (cap median, BEME 30/70), Resid_Var only where both months of
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

## Checks (2026-09-30 build)

- Size: europe17 824 ever-ranked companies (1,131 lines), 757 universe members, returns
  4.29M rows; europe12 793 (1,079 lines). Union raw pull: 1,145 lines, 833 companies,
  5.34M daily rows 1984-2025, 19,100 g_funda records, 224k JKP rows. Stage 2 ~3 min each.
- Prefix invariance: `--cutoff 1996-09-15` rebuild of europe17 reproduces the universe
  (16,200 rows), features (344,400 rows, all 79 columns, max difference 0) and returns
  (750,879 rows) of the full build exactly.
- Numeraire: see Decisions 2. `tests/test_europe_fx.py`: basket by date, ECU -> EUR
  switch. `tests/test_europe_panel.py`: Corwin-Schultz on a pure bid-ask bounce (recovers
  (H-L)/mid exactly) and the overnight gap, the line carrying a company, the 2x3 sort.
  70 tests pass with the suite.

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
  continental companies (Roche, Mercedes-Benz, UMG in some months).
