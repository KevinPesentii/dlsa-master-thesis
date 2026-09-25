"""Merge run directories that each hold some out-of-sample years into one run directory.

scripts/run_years_parallel.sh runs every year as its own process. Each of those runs is
complete except at its first date: a standalone run starts from no position, so it
charges the whole book (||w||_1 = 1) as turnover there. A sequential run instead charges
the trade from the previous year's last weights, in pool space (run_attention_us.py). This
script puts the years back together, recomputes turnover on those seam dates from
weights.parquet, and recomputes every metric with evaluation.metrics.performance, so the
merged metrics.json is what the sequential runner would have written.

Runs are grouped by (name, seed, git commit, config without "years"); each group becomes
one new run directory runs/<stamp>_<name>_merged/ whose manifest lists its sources.

    python scripts/merge_year_runs.py runs/2026..._attention_val_K30_s0 runs/... [--dry-run]

run_attention_us_val.py seeds each year from (seed, year), so a merged val run reproduces a
sequential one. run_attention_us.py seeds once per process, so every standalone year starts
from the same draw: a valid run, but not the same numbers as a sequential one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe import runs  # noqa: E402
from afe.evaluation import metrics  # noqa: E402

# metrics.json keys that are per-year dictionaries: merged by union
BY_YEAR = ("lambda_by_year", "validation_net_SR_by_year")


def load(d: Path) -> dict:
    man = json.loads((d / "manifest.json").read_text())
    if not (d / "metrics.json").exists():
        raise SystemExit(f"{d}: no metrics.json, the run did not finish")
    daily = pd.read_csv(d / "oos_daily.csv", parse_dates=["date"])
    w = pd.read_parquet(d / "weights.parquet")
    w["date"] = pd.to_datetime(w["date"])
    return {"dir": d, "manifest": man, "metrics": json.loads((d / "metrics.json").read_text()),
            "daily": daily, "w": w}


def group_key(r: dict) -> str:
    m = r["manifest"]
    cfg = {k: v for k, v in m["config"].items() if k != "years"}
    return json.dumps([m["name"], m["seed"], m["git_commit"], cfg], sort_keys=True, default=str)


def book(w: pd.DataFrame, date) -> pd.Series:
    return w[w["date"] == date].groupby("sec_id")["w"].sum().astype(np.float64)


def merge_group(rs: list[dict], dry_run: bool) -> None:
    rs = sorted(rs, key=lambda r: r["daily"]["date"].min())
    name, man0 = rs[0]["manifest"]["name"], rs[0]["manifest"]
    daily = pd.concat([r["daily"] for r in rs], ignore_index=True)
    if daily["date"].duplicated().any():
        raise SystemExit(f"{name}: the runs overlap in dates; pass each year once")
    years = sorted({y for r in rs for y in r["manifest"]["config"]["years"]})
    gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
    if gaps:
        print(f"  WARNING {name}: years missing from the merge: {gaps}")

    for prev, cur in zip(rs[:-1], rs[1:]):
        d_prev, d_cur = prev["daily"]["date"].max(), cur["daily"]["date"].min()
        a, b = book(prev["w"], d_prev), book(cur["w"], d_cur)
        standalone = float(cur["daily"].loc[cur["daily"]["date"] == d_cur, "turnover"].iloc[0])
        if abs(standalone - b.abs().sum()) > 1e-4:
            raise SystemExit(f"{cur['dir']}: turnover on its first date is {standalone:.4f}, not the "
                             f"book size {b.abs().sum():.4f}; is it a standalone run?")
        seam = float(b.sub(a, fill_value=0.0).abs().sum())
        daily.loc[daily["date"] == d_cur, "turnover"] = seam
        print(f"  seam {d_prev:%Y-%m-%d} -> {d_cur:%Y-%m-%d}: turnover {standalone:.3f} -> {seam:.3f}")

    ob, ev = man0["config"]["objective"], man0["config"]["evaluation"]
    m = metrics.performance(daily, ob["turnover_cost"], ob["short_cost"], ev["cost_grid_bps"])
    extra = {k: v for k, v in rs[0]["metrics"].items() if k not in m and k not in BY_YEAR}
    m.update(extra)
    for key in BY_YEAR:
        parts = [r["metrics"][key] for r in rs if key in r["metrics"]]
        if parts:
            m[key] = {k: v for p in parts for k, v in p.items()}
    if m.get("lambda_selection") == "once":
        picks = {json.dumps(v) for v in m.get("lambda_by_year", {}).values()}
        if len(picks) > 1:
            print(f"  WARNING {name}: selection 'once' chose different values across processes: {picks}")
    m["merged_from"] = [r["dir"].name for r in rs]

    print(f"  {name}: {len(rs)} runs, {years[0]}-{years[-1]}")
    print(f"  K={m.get('K', '?'):>3}  {metrics.table_row(m)}   turnover {m['turnover_daily']:.3f}  "
          f"short {m['short_exposure']:.3f}  break-even {m['break_even_turnover_cost_bps']:.1f}bp")
    if dry_run:
        return
    out = runs.create_run(f"{name}_merged", {**man0["config"], "years": years,
                                             "merged_from": m["merged_from"]},
                          man0["seed"], root=rs[0]["dir"].parent)
    if runs.git_commit(ROOT) != man0["git_commit"]:
        print(f"  WARNING: this checkout is at {runs.git_commit(ROOT)}, the runs at {man0['git_commit']}")
    runs.write_metrics(out, m)
    daily.to_csv(out / "oos_daily.csv", index=False)
    pd.concat([r["w"] for r in rs], ignore_index=True).to_parquet(out / "weights.parquet", index=False)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the merged metrics, write nothing")
    args = ap.parse_args()
    groups: dict[str, list[dict]] = {}
    for d in args.run_dirs:
        r = load(d)
        groups.setdefault(group_key(r), []).append(r)
    for rs in groups.values():
        merge_group(rs, args.dry_run)


if __name__ == "__main__":
    main()
