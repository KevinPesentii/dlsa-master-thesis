"""Check that every job of a matrix left the files the thesis analyses read (2026-11-17).

    python scripts/check_matrix_outputs.py --matrix configs/thesis_v3_matrix.yaml [--smoke]
    python scripts/check_matrix_outputs.py --matrix ... --list-run-dirs     # for tar -T

Per job, from the matrix's ledger (runs/_matrix/<name>/ledger.jsonl):
- runs: manifest.json, metrics.json, oos_daily.csv covering every out-of-sample year of the config;
  weights.parquet where evaluation.save_weights; models/<year>.pt for every year where
  evaluation.save_models; the six diagnostics files where evaluation.diagnostics (attention only).
- searches: manifest.json, metrics.json (with its finalists), trials.jsonl, selected.yaml.
- ensemble validations: metrics.json (with its choice), books.csv, trials.jsonl and one weights file
  per (candidate, seed).
Prints one line per job with a problem and a count at the end; exit status 1 if anything is
missing or a job has not finished. --list-run-dirs prints the run directories of the finished runs,
relative to the repo's attention-factors-europe/, one per line, and checks nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402

DIAGNOSTICS = ("diag_daily.parquet", "factor_returns.parquet", "factor_top.parquet", "factor_chars.parquet",
               "name_year.parquet", "train_log.csv")


def run_problems(d: Path) -> list[str]:
    miss = [f for f in ("manifest.json", "metrics.json", "oos_daily.csv") if not (d / f).exists()]
    if miss:
        return [f"missing {', '.join(miss)}"]
    cfg = json.loads((d / "manifest.json").read_text())["config"]
    ev, years = cfg.get("evaluation") or {}, [int(y) for y in cfg.get("years", [])]
    out = []
    seen = set(pd.to_datetime(pd.read_csv(d / "oos_daily.csv", usecols=["date"])["date"]).dt.year)
    if years and set(years) - seen:
        out.append(f"oos_daily.csv lacks years {sorted(set(years) - seen)}")
    if ev.get("save_weights") and not (d / "weights.parquet").exists():
        out.append("missing weights.parquet")
    if ev.get("save_models"):
        gone = [y for y in years if not (d / "models" / f"{y}.pt").exists()]
        if gone:
            out.append(f"missing models for {gone}")
    if ev.get("diagnostics"):                # only the attention runner writes them
        gone = [f for f in DIAGNOSTICS if not (d / f).exists()]
        if gone:
            out.append(f"missing {', '.join(gone)}")
    return out


def search_problems(d: Path) -> list[str]:
    miss = [f for f in ("manifest.json", "metrics.json", "trials.jsonl", "selected.yaml") if not (d / f).exists()]
    if miss:
        return [f"missing {', '.join(miss)}"]
    return [] if json.loads((d / "metrics.json").read_text()).get("finalists") else ["metrics.json has no finalists"]


def ensemble_problems(d: Path) -> list[str]:
    miss = [f for f in ("manifest.json", "metrics.json", "books.csv", "trials.jsonl") if not (d / f).exists()]
    if miss:
        return [f"missing {', '.join(miss)}"]
    E = json.loads((d / "manifest.json").read_text())["config"]
    n_c, n_s = len(E["ensemble"]["candidates"]), len(E["ensemble"]["seeds"])
    want = min(2, n_c) * min(2, n_s) if E.get("smoke") else n_c * n_s
    have = len(list((d / "weights").glob("*.npy")))
    out = [] if have >= want else [f"{have} of {want} validation weight files"]
    return out + ([] if "choice" in json.loads((d / "metrics.json").read_text()) else ["metrics.json has no choice"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list-run-dirs", action="store_true")
    ap.add_argument("--groups", nargs="*", help="only these groups, as run_matrix.py --groups")
    args, _ = ap.parse_known_args()          # run_matrix arguments (--jobs ...) are ignored
    matrix = yaml.safe_load((ROOT / args.matrix).read_text() if not Path(args.matrix).is_absolute()
                            else Path(args.matrix).read_text())
    out = ROOT / "runs" / "_matrix" / (matrix["name"] + ("_smoke" if args.smoke else ""))
    state = rm.ledger_state(out / "ledger.jsonl")
    jobs = rm.expand(matrix, args.groups, out, 2, args.smoke)
    if args.list_run_dirs:
        for j in jobs:
            if j.spec is not None and j.year is None and rm.finished(state.get(j.id)):
                print(rm.rel(ROOT / state[j.id]["run_dir"]))      # the ledger may hold absolute paths
        return
    bad, n = 0, 0
    for j in jobs:
        if j.year is not None:
            continue
        n += 1
        if j.spec is not None:
            e = state.get(j.id)
            probs = run_problems(ROOT / e["run_dir"]) if rm.finished(e) else \
                [f"not finished ({(e or {}).get('status', 'never started')})"]
        elif j.id.startswith("search_"):
            probs = search_problems(j.done_file.parent)
        elif j.id.startswith("ensval_"):
            probs = ensemble_problems(j.done_file.parent)
        else:                                 # prep: its done file
            probs = [] if j.done_file.exists() else ["not done"]
        if probs:
            bad += 1
            print(f"{j.id:48s} {'; '.join(probs)}")
    print(f"{n} jobs checked, {bad} with problems ({out.relative_to(ROOT).as_posix()})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
