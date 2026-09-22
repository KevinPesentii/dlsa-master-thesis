# US top-500 universe from Compustat Security Monthly

Status: first build 2026-09-15. Decisions below were checked against `comp.secm` on WRDS
that day; the numbers come from the probes, not from memory.

Build: `python scripts/build_us_universe.py` with `configs/universe_us_compustat.yaml`.
Code: `src/afe/data/compustat_us.py`. Output under `data/us/private/` (gitignored; step-1 tables are a cross-check for the build, not shipped downstream).

## Definition

Market cap of company g at month end t = sum over its common-stock issues (`tpci = '0'`,
`curcdm = 'USD'`) of `prccm * cshom`. Listed share classes add up (Berkshire A + B,
Alphabet GOOGL + GOOG); unlisted classes do not count. This is what a CRSP permco-level
sum gives, so it lines up with the CRSP-based US literature.

The ranking is AS-OF: row m ranks by cap at the end of m. The universe for trading month
m+1 is row m. The lag lives downstream (`universe.parquet` in `docs/schemas.md`).

## What the data forced

1. `cshom` (monthly issue-level shares) starts **1998-04-30**. Nothing earlier, for any
   company; the daily file's `cshoc` is empty before then too (3-11 issues a year).
   For 1990-01 to 1998-03 the build uses the company-level share count from the
   fundamentals files, `fundq.cshoq` (else `funda.csho`, millions), forward-filled from
   the latest fiscal period end (cap 12 months) and multiplied by the price of the
   PRIMARY issue at t, split-adjusted with that issue's `ajexm`:
   cap_t = prccm_t(primary) * shares_q * ajexm_q / ajexm_t.
   Why the primary issue: the company count is stated in units of the primary issue at
   the fiscal date (Berkshire: 1.15M A shares in 1990). `primiss` is historical, so the
   primary issue at t is known. The copy of `cshoq` inside secm is NOT used: it is
   attached to an arbitrary issue row (Berkshire's B row, priced 30x lower than the A
   count it carries, giving a bogus $1.3bn) and is missing for many firms before 1996;
   the first build with it lost Berkshire for all of 1990-1997.
   `shares_source` on every row says which tier was used. Both measures are kept for
   every month; the build report checks, on 1998-04 onward where both exist, how much
   the two top-500 sets overlap (96-99% in the 2026-09-15 build), and back-casts the
   priced companies that have no share count at all to see whether any would have
   ranked (393 company-months, mostly IPO and spin-off months before the first filing).

2. `exchg`, `fic`, `loc`, `conm`, `tic` are **header fields** (`primiss` is not): the current
   value is stamped on every historical row. Enron, Lehman, Washington Mutual, Sears,
   Kodak (old issue), Circuit City and Fannie Mae all carry `exchg = 19` (Other-OTC)
   for their whole history. An NYSE/AMEX/NASDAQ filter would remove them from the years
   they were top-500 names. So: no exchange filter. Market-cap ranking already keeps
   small OTC names out.

3. The country filter is `fic = 'USA'` (incorporation), the Compustat analogue of CRSP
   share codes 10/11. Because it is a header, firms that redomiciled (Medtronic and
   Tyco to IRL, Chubb to CHE, ...) are dropped for their entire history, including the
   years they were US-incorporated. The build report lists the largest names removed so
   the effect is visible. Switch to `loc` in the config to filter on headquarters instead.

4. `conm` / `tic` are current names (GE AEROSPACE, EXXONMOBIL HOLDINGS CORP, ...). They
   are identification only; `gvkey` is the key.

5. REITs are `tpci = '0'` in Compustat and stay in. CRSP-based studies that keep share
   codes 10/11 exclude them (share code 18). `sic = 6798` is carried on every row so
   they can be removed downstream; the report counts them per year.

## Open

- Point in time of the pre-1998 fallback: the quarter-end share count is treated as
  known at month end. The count is observable in real time, but the reported `cshoq`
  appears with the 10-Q, up to ~45 days later. Acceptable for ranking; record it in the
  data appendix.
- Whether the paper's CRSP universe excludes REITs and foreign-incorporated firms.
  Settle by reading the authors' code before the US replication is compared to 2.28.
