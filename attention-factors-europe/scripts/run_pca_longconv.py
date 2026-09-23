"""Step two of the two-step benchmark: LongConv policy on fixed PCA residuals.

    python scripts/run_pca_longconv.py [--config configs/us_pca_longconv.yaml]
                                       [--K 30] [--seed 0] [--years 1998 1999] [--epochs 2]
                                       [--pca-dir data/us/shared/pca_l252] [--input cumulative]

For each K: rolling 8-year training windows refit every January, out-of-sample
Jan 1998 - Dec 2021, one run directory runs/<stamp>_pca_longconv_K<K>_s<seed>/ with the
manifest, a daily out-of-sample series, the asset-space weights and metrics.json in the
units of Table 2.  Needs the stage-one output of scripts/build_pca_residuals.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe import runs  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.model import pca_factors  # noqa: E402
from afe.policy import trading  # noqa: E402
from afe.policy.longconv import LongConvPolicy  # noqa: E402


class Stage1:
    """Memory-mapped stage-one output for one K, plus the return panel in slot layout."""

    def __init__(self, pca_dir: Path, K: int, panel: pca_factors.Panel):
        d = pca_dir / f"K{K}"
        self.dates = pd.DatetimeIndex(np.load(pca_dir / "dates.npy"))
        assert (self.dates == panel.dates).all(), "stage-one dates differ from the return panel"
        self.resid = np.load(d / "resid.npy", mmap_mode="r")
        self.V = np.load(d / "V.npy", mmap_mode="r")
        self.B = np.load(d / "B.npy", mmap_mode="r")
        self.invvol = np.load(d / "invvol.npy")
        self.idx = np.load(d / "idx.npy").astype(np.int64)
        self.n = np.load(d / "n.npy")
        self.n_pool = panel.R.shape[1]
        R_ext = np.concatenate([np.nan_to_num(panel.R), np.zeros((len(panel.dates), 1))], axis=1)
        self.R_slot = np.take_along_axis(R_ext, self.idx, axis=1).astype(np.float32)
        self.R_pool = np.nan_to_num(panel.R).astype(np.float32)
        self.rf = panel.rf.astype(np.float32)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # days before the universe starts
            self.mkt_ew = np.nanmean(np.where(panel.member, panel.R, np.nan), axis=1)

    def batch(self, t0: int, t1: int, L: int, scale: float, cumulative: bool) -> trading.SlotBatch:
        X, tradable = trading.make_windows(self.resid, self.idx, self.n, t0, t1, L, scale, cumulative)
        f = lambda a: torch.from_numpy(np.array(a[t0:t1]))  # noqa: E731  (copies out of the memmap)
        return trading.SlotBatch(torch.from_numpy(X), torch.from_numpy(tradable), f(self.V), f(self.B),
                                 f(self.invvol), f(self.idx), f(self.R_slot), f(self.rf), self.n_pool)


def sub(b: trading.SlotBatch, s: int, e: int) -> trading.SlotBatch:
    return trading.SlotBatch(b.windows[s:e], b.tradable[s:e], b.V[s:e], b.B[s:e], b.invvol[s:e],
                             b.idx[s:e], b.R[s:e], b.rf[s:e], b.n_pool)


def forward(model, b: trading.SlotBatch) -> torch.Tensor:
    w_port = model(b.windows) * b.tradable
    return trading.compose(w_port, b)


def train_one_window(model, b: trading.SlotBatch, cfg: dict, rng: np.random.Generator, log) -> None:
    tr, ob = cfg["training"], cfg["objective"]
    tc, sc = ob["turnover_cost"], ob["short_cost"]
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    T = b.windows.shape[0]
    bd = tr["batch_days"] or T
    starts = list(range(0, T, bd))
    has_asset = b.tradable.any(dim=1)
    model.train()
    for epoch in range(tr["epochs"]):
        rng.shuffle(starts)
        losses = []
        for s in starts:
            e = min(s + bd, T)
            s0 = max(s - 1, 0)                        # one extra day so turnover on day s is charged
            blk = sub(b, s0, e)
            w = forward(model, blk)
            _, _, net, _, _, _ = trading.net_returns(w, blk, tc, sc)
            valid = has_asset[s0:e].clone()
            valid[0] = valid[0] and s0 == s           # the extra day is not scored
            loss = trading.sharpe_loss(net, blk.rf, valid, ob["subtract_rf"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            log(f"      epoch {epoch + 1:2d}: block Sharpe (daily) {-np.mean(losses):.3f}")


@torch.no_grad()
def evaluate_window(model, b: trading.SlotBatch):
    model.eval()
    return forward(model, b)


def run_K(K: int, cfg: dict, seed: int, panel: pca_factors.Panel, years: list[int], log) -> dict:
    pc, ev, ob = cfg["policy"], cfg["evaluation"], cfg["objective"]
    s1 = Stage1(ROOT / cfg["factors"]["out_dir"], K, panel)
    meta = json.loads((ROOT / cfg["factors"]["out_dir"] / "meta.json").read_text())
    cfg = {**cfg, "factors": {**cfg["factors"], "loading_window": meta["loading_window"], "cov_window": meta["cov_window"]}}
    run_dir = runs.create_run(f"pca_longconv_K{K}_s{seed}", {**cfg, "K": K, "years": years}, seed)
    log(f"K={K}: run dir {run_dir}")
    L, scale, cum = pc["residual_lookback"], pc["input_scale"], pc["input"] == "cumulative"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dates = s1.dates
    w_pool_all, w_slot_all, date_all = [], [], []
    for year in years:
        t_tr0 = dates.searchsorted(pd.Timestamp(year - cfg["training"]["window_years"], 1, 1))
        t_te0 = dates.searchsorted(pd.Timestamp(year, 1, 1))
        t_te1 = dates.searchsorted(pd.Timestamp(year + 1, 1, 1))
        t0 = time.time()
        train_b = s1.batch(t_tr0, t_te0, L, scale, cum)
        log(f"  {year}: train {dates[t_tr0]:%Y-%m-%d}..{dates[t_te0 - 1]:%Y-%m-%d} "
            f"({t_te0 - t_tr0} days, {int(train_b.tradable.sum(1).float().mean())} tradable/day), "
            f"test {t_te1 - t_te0} days")
        model = LongConvPolicy(pc["hidden"], L, pc["dropout"], pc["lambda_squash"], pc["layers"])
        train_one_window(model, train_b, cfg, rng, log)
        del train_b
        test_b = s1.batch(t_te0, t_te1, L, scale, cum)
        w = evaluate_window(model, test_b)
        w_slot_all.append(w.numpy())
        w_pool_all.append(trading.to_pool(w, test_b).numpy())
        date_all.append(np.arange(t_te0, t_te1))
        log(f"      done in {time.time() - t0:.0f}s")

    t_idx = np.concatenate(date_all)
    wp = np.concatenate(w_pool_all)
    gross = (wp * s1.R_pool[t_idx]).sum(axis=1)
    turnover = np.abs(np.diff(wp, axis=0, prepend=np.zeros((1, wp.shape[1]), dtype=wp.dtype))).sum(axis=1)
    short = np.clip(-wp, 0, None).sum(axis=1)
    daily = pd.DataFrame({"date": dates[t_idx], "gross": gross, "turnover": turnover, "short": short,
                          "n_traded": (wp != 0).sum(axis=1), "mkt_ew": s1.mkt_ew[t_idx], "rf": s1.rf[t_idx]})
    m = metrics.performance(daily, ob["turnover_cost"], ob["short_cost"], ev["cost_grid_bps"])
    m.update({"K": K, "seed": seed, "policy_input": pc["input"],
              "factor_model": f"pca_cov{meta['cov_window']}_loadings{meta['loading_window'] or 'projection'}"})
    runs.write_metrics(run_dir, m)
    daily.to_csv(run_dir / "oos_daily.csv", index=False)
    if ev.get("save_weights"):
        ws = np.concatenate(w_slot_all)
        rows = ws != 0
        tt, ss = np.nonzero(rows)
        pd.DataFrame({"date": dates[t_idx[tt]], "sec_id": panel.sec_ids[s1.idx[t_idx[tt], ss]],
                      "w": ws[rows].astype(np.float32)}).to_parquet(run_dir / "weights.parquet", index=False)
    log(f"K={K:3d}  {metrics.table_row(m)}   turnover {m['turnover_daily']:.3f}  "
        f"short {m['short_exposure']:.3f}  break-even {m['break_even_turnover_cost_bps']:.1f}bp")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_pca_longconv.yaml")
    ap.add_argument("--K", type=int, nargs="*")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--years", type=int, nargs="*", help="out-of-sample years to run (default: all)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--pca-dir", help="override factors.out_dir (a stage-one output directory)")
    ap.add_argument("--input", choices=["returns", "cumulative"], help="override policy.input")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.pca_dir:
        cfg["factors"]["out_dir"] = args.pca_dir
    if args.input:
        cfg["policy"]["input"] = args.input
    if args.threads:
        torch.set_num_threads(args.threads)
    seed = cfg["seed"] if args.seed is None else args.seed
    Ks = args.K or cfg["factors"]["n_factors"]
    y0, y1 = pd.Timestamp(cfg["sample"]["start"]).year, pd.Timestamp(cfg["sample"]["end"]).year
    years = args.years or list(range(y0, y1 + 1))

    log = lambda s: print(s, flush=True)  # noqa: E731
    panel = pca_factors.load_panel(ROOT / cfg["data"]["dir"], str(cfg["sample"]["end"]),
                                   str(cfg["data"]["history_start"]), ROOT / cfg["data"]["raw_daily_dir"])
    log(f"panel {panel.R.shape}, torch threads {torch.get_num_threads()}")
    log("      K    SR     mu   sigma   SRnet  munet signet   beta")
    for K in Ks:
        run_K(K, cfg, seed, panel, years, log)


if __name__ == "__main__":
    main()
