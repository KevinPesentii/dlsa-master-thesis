"""Five-seed books of the K = 30 holdout, 2022-2025 (review follow-ups, tier 1 (c), 2026-11-26).

    python scripts/holdout_books.py [--run-root runs ...] [--matrix-root runs/_matrix ...] [--models attention pca_longconv]
                                    [--out runs/_matrix/review_t1/books]

No training: for each market, the five seeds' saved target weights (weights.parquet) of
    attention        thesis_v3   <m>_attention_t4q_K30_2022_2025
    PCA + LongConv   review_t1   <m>_pca_longconv_perk_K30_2022_2025
are averaged and rescaled to ||w||_1 = 1 on every day (`avg_l1`, ensemble_validation.average, the book of
thesis_v3's ensemble rule), then booked with the runner's own accounting: run_attention_us.daily_book on
the run's slot panel, or run_pca_longconv.execute on the PCA return panel (held positions on closed
markets, execution lag). Every member is booked the same way first and must reproduce its own
oos_daily.csv (gross, turnover, short) within --tol, else the script stops. Metrics come from
metrics.performance at the run's evaluation costs (5bp / 1bp), the fields of a run's metrics.json.

Reads runs/_matrix/<matrix>/summary.csv and configs/<spec>.yaml (what ran) and the run directories
(by name, under each --run-root; default runs/), and the data/ tables of this repo. Writes
<out>/holdout_books.csv (every member and each book) and <out>/holdout_books_daily.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ensemble_validation as ev  # noqa: E402  (average)
import run_attention_us as rau  # noqa: E402  (window_start, daily_book)
import run_pca_longconv as rp  # noqa: E402  (load_panel, execute)
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402

BOOKS = {("us", "attention"): ("thesis_v3", "us_attention_t4q_K30_2022_2025"),
         ("eu", "attention"): ("thesis_v3", "eu_attention_t4q_K30_2022_2025"),
         ("us", "pca_longconv"): ("review_t1", "us_pca_longconv_perk_K30_2022_2025"),
         ("eu", "pca_longconv"): ("review_t1", "eu_pca_longconv_perk_K30_2022_2025")}
SEEDS = [0, 1, 2, 3, 4]


def find_run(name: str, roots: list[Path]) -> Path:
    for r in roots:
        if (r / name / "weights.parquet").exists():
            return r / name
    raise SystemExit(f"run directory {name} with weights.parquet not found under {[str(r) for r in roots]}")


class Booker:
    """Target pool weights -> the runner's daily series (date, gross, turnover, short, mkt_ew, rf)."""

    def __init__(self, runner: str, cfg: dict):
        self.runner, self.cfg = runner, cfg
        start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
        if runner == "attention":
            d = Path(cfg["data"]["dir"])
            self.p = slots.load_config(cfg, d if d.is_absolute() else ROOT / d, f"{rau.window_start(cfg, start.year):%Y-%m-%d}",
                                       f"{end.year}-12-31", max_rank=cfg["sample"].get("universe_size"))
            dates, pool = self.p.dates, self.p.sec_ids[:self.p.n_pool]
        else:
            self.p = rp.load_panel(cfg, str(cfg["sample"]["end"]))
            dates, pool = self.p.dates, self.p.sec_ids
            self.R_pool = np.nan_to_num(self.p.R).astype(np.float32)
            with np.errstate(all="ignore"):
                self.mkt_ew = np.nanmean(np.where(self.p.member, self.p.R, np.nan), axis=1)
        self.t_idx = np.arange(dates.searchsorted(start), dates.searchsorted(end + pd.Timedelta(days=1)))
        self.dates = dates[self.t_idx]
        self.col = pd.Series(np.arange(len(pool)), index=pool)
        assert self.col.index.is_unique

    def weights(self, run: Path) -> np.ndarray:
        w = pd.read_parquet(run / "weights.parquet")
        r, c = self.dates.get_indexer(pd.to_datetime(w["date"])), self.col.reindex(w["sec_id"]).to_numpy()
        if (r < 0).any() or np.isnan(c).any():
            raise SystemExit(f"{run.name}: weights outside the panel's out-of-sample days or pool")
        wp = np.zeros((len(self.t_idx), len(self.col)), dtype=np.float32)
        np.add.at(wp, (r, c.astype(int)), w["w"].to_numpy(np.float32))
        return wp

    def book(self, wp: np.ndarray) -> pd.DataFrame:
        if self.runner == "attention":
            return rau.daily_book(self.p, wp, self.t_idx, self.cfg)
        h = rp.execute(wp, self.p, self.t_idx, self.cfg)
        return pd.DataFrame({"date": self.dates, "gross": (h * self.R_pool[self.t_idx]).sum(axis=1),
                             "turnover": np.abs(np.diff(h, axis=0, prepend=np.zeros((1, h.shape[1]), dtype=h.dtype))).sum(axis=1),
                             "short": np.clip(-h, 0, None).sum(axis=1), "mkt_ew": self.mkt_ew[self.t_idx],
                             "rf": self.p.rf[self.t_idx]})


def row(market: str, model: str, book: str, d: pd.DataFrame, cfg: dict) -> dict:
    m = metrics.performance(d, *metrics.eval_costs(cfg), cfg["evaluation"]["cost_grid_bps"])
    return {"market": market, "model": model, "book": book, "first": m["first"], "last": m["last"],
            "SR": m["gross"]["SR"], "mu_pct": m["gross"]["mu_pct"], "sigma_pct": m["gross"]["sigma_pct"],
            "SR_net": m["net"]["SR"], "mu_net_pct": m["net"]["mu_pct"], "beta": m["beta_ew_market"],
            "turnover": m["turnover_daily"], "short": m["short_exposure"],
            "break_even_bps": m["break_even_turnover_cost_bps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, nargs="+", default=[ROOT / "runs"], help="where the run directories are")
    ap.add_argument("--matrix-root", type=Path, nargs="+", default=[ROOT / "runs" / "_matrix"],
                    help="directories holding thesis_v3/ and review_t1/ (the first with <matrix>/summary.csv is used)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "_matrix" / "review_t1" / "books")
    ap.add_argument("--models", nargs="+", choices=["attention", "pca_longconv"], default=["attention", "pca_longconv"])
    ap.add_argument("--tol", type=float, default=1e-6, help="largest member re-booking error allowed")
    args = ap.parse_args()
    rows, frames = [], []
    for (market, model), (matrix, spec) in BOOKS.items():
        if model not in args.models:
            continue
        mdir = next((r / matrix for r in args.matrix_root if (r / matrix / "summary.csv").exists()), None)
        if mdir is None:
            raise SystemExit(f"{matrix}/summary.csv not found under {[str(r) for r in args.matrix_root]}")
        cfg = yaml.safe_load((mdir / "configs" / f"{spec}.yaml").read_text())
        s = pd.read_csv(mdir / "summary.csv")
        s = s[s["spec"] == spec].sort_values("seed")
        if sorted(s["seed"]) != SEEDS:
            raise SystemExit(f"{matrix}/{spec}: seeds {sorted(s['seed'])} in summary.csv, need {SEEDS}")
        bk = Booker("pca" if (cfg.get("factors") or {}).get("kind") == "pca" else "attention", cfg)
        W, worst = [], 0.0
        for r in s.itertuples():
            run = find_run(Path(str(r.run_dir).replace("\\", "/")).name, args.run_root)
            wp = bk.weights(run)
            mine = bk.book(wp)
            ref = pd.read_csv(run / "oos_daily.csv", parse_dates=["date"])
            assert (mine["date"].to_numpy() == ref["date"].to_numpy()).all(), run.name
            err = max(float(np.abs(mine[k].to_numpy(float) - ref[k].to_numpy(float)).max()) for k in ("gross", "turnover", "short"))
            worst = max(worst, err)
            if err > args.tol:
                raise SystemExit(f"{run.name}: re-booked series differ from oos_daily.csv by {err:.2e} > {args.tol}")
            rows.append({**row(market, model, f"seed {int(r.seed)}", mine, cfg), "SR_net_summary": r.SR_net, "run_dir": run.name})
            W.append(wp)
        d = bk.book(ev.average(W, True))
        rows.append(row(market, model, "five-seed avg_l1", d, cfg))
        frames.append(d.assign(market=market, model=model))
        print(f"{market} {model}: 5 members re-booked (largest error {worst:.1e}); five-seed book net SR "
              f"{rows[-1]['SR_net']:.2f}, mean single {np.mean([x['SR_net'] for x in rows[-6:-1]]):.2f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "holdout_books.csv", index=False)
    pd.concat(frames).to_parquet(args.out / "holdout_books_daily.parquet", index=False)
    print(f"-> {args.out / 'holdout_books.csv'}, {args.out / 'holdout_books_daily.parquet'}")


if __name__ == "__main__":
    main()
