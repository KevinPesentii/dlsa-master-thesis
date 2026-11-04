# Analysis scripts

One-off diagnostics of the side experiments, copied unchanged from the run folders they were written
in. They are not thesis rows and have no tests.

Run them from `attention-factors-europe/`:

    python scripts/analysis/exp_ensemble_k30/oos_ensemble.py

They read `runs/` and `data/`, and write their outputs into `runs/_matrix/<experiment>/`. Neither folder
is in git, so the run matrices have to exist first (`configs/exp_ensemble_k30_matrix.yaml`,
`configs/exp_fixedwin_full_matrix.yaml`).

| Folder | What it holds |
| --- | --- |
| `exp_fixedwin_full/` | Fixed vs rolling residual window, and the seed ensemble of those runs |
| `exp_ensemble_k30/` | Out-of-sample ensemble books, then betas, netting, periods and correlations from them |
| `exp_ensemble_k30/traded/` | What the K = 30 books trade, and the re-booking without receipts (`_common.py` is the shared loader) |

Run `oos_ensemble.py` before the rest of `exp_ensemble_k30/`: `oos_periods.py` and
`traded/rebook_no_receipts.py` read its outputs. `traded/check_no_receipts_universe.py` also needs the
receipt-free builds of the `exp/no-receipts` work. The FF12 sector ranges in `traded/profile.py` were
typed from memory and have not been checked against the Siccodes12 file.
