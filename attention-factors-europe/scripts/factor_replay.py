"""Full factor weights of a finished run on any dates, from its saved models and the panel.

    python scripts/factor_replay.py runs/<run dir> --dates 2008-10-01 2008-10-02 [--out file.parquet]

A run keeps the top names of each factor once a month (evaluation/diagnostics.py), not w_F:
that would be T x K x 500 numbers. models/<year>.pt (evaluation.save_models) and the panel give
back every one of them, here as (date, factor, sec_id, w_F) with w_F > 0 and beta, the loading
of the name on the factor. Reads the run's manifest for the data, the timing and K, so it
needs the same tables the run read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import slots  # noqa: E402
from afe.model.attention_factors import AttentionFactors  # noqa: E402


def replay(run_dir: Path, day: pd.Timestamp, data_dir: Path | None = None) -> pd.DataFrame:
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    data_dir = data_dir or Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    p = slots.load_config(cfg, data_dir, f"{day - pd.DateOffset(days=10):%Y-%m-%d}", f"{day:%Y-%m-%d}",
                          max_rank=cfg["sample"].get("universe_size"))
    if cfg["features"].get("drop_groups"):
        p = slots.drop_groups(p, cfg["features"]["drop_groups"])
    if p.dates[-1] != day:
        raise SystemExit(f"{day:%Y-%m-%d} is not a trading day of the panel")
    state = torch.load(run_dir / "models" / f"{day.year}.pt", map_location="cpu")
    mc = cfg["model"]
    f = AttentionFactors(len(p.features), cfg["K"], mc["embedding_dim"], mc["lambda_ridge"], mc.get("level_hidden", 0))
    f.load_state_dict({k.removeprefix("factors."): v for k, v in state.items() if k.startswith("factors.")})
    f.eval()
    with torch.no_grad():
        w_F = f.factor_weights(p.X[-1:], p.in_universe[-1:])
        beta = f.loadings(w_F)[0]                                    # (S, K)
    k, s = torch.nonzero(w_F[0] > 0, as_tuple=True)
    return pd.DataFrame({"date": day, "factor": k.numpy().astype("int16"), "sec_id": p.sec_ids[p.idx[-1][s].numpy()],
                         "w_F": w_F[0][k, s].numpy(), "beta": beta[s, k].numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, help="default: the data.dir of the run's config")
    ap.add_argument("--out", type=Path, help="default: <run dir>/factor_weights_<first date>.parquet")
    args = ap.parse_args()
    days = sorted(pd.Timestamp(d) for d in args.dates)
    out = pd.concat([replay(args.run_dir, d, args.data_dir) for d in days], ignore_index=True)
    path = args.out or args.run_dir / f"factor_weights_{days[0]:%Y%m%d}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"{len(out):,} rows, {len(days)} dates, {out['factor'].nunique()} factors -> {path}")


if __name__ == "__main__":
    main()
