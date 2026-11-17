"""Shared loader for the what-is-traded diagnostics of the K = 30 books, 2026-11-04. Run the scripts from the repo root.

The 25 models = 5 candidates x seeds 0-4 of the ensemble validation: sobol-036 from runs/_matrix/exp_fixedwin_full,
the other four from runs/_matrix/exp_ensemble_k30. load() re-books every run and checks it against its oos_daily.csv.
Diagnostics, not thesis rows.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from afe.data import slots
import run_attention_us as rau

BASE, FULL = ROOT / "runs/_matrix/exp_ensemble_k30", ROOT / "runs/_matrix/exp_fixedwin_full"
OUT = BASE / "traded"
ANN, TC, SC = 252, 5e-4, 1e-4
WINDOWS = {"roll": ("us_ensemble_val", "us_attention", "us_k30_"),
           "fixed": ("us_ensemble_val_fixedwin", "us_attention_fixedwin", "us_k30_fixedwin_")}
SAMPLES = {"1998-2021": (None, None), "1998-2009": (None, "2009"), "2010-2021": ("2010", None)}


def load(w: str) -> dict:
    """Panel plus the (25, T, n_pool) float32 stack of target pool weights for window `w` (roll | fixed)."""
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs/us_headline.yaml").read_text())
    assert cfg["execution"]["lag"] == 0 and cfg["objective"]["turnover_cost"] == TC and cfg["objective"]["short_cost"] == SC
    p = slots.load_slots(ROOT / cfg["data"]["dir"], f"{rau.window_start(cfg, 1998):%Y-%m-%d}", "2021-12-31",
                         max_rank=cfg["sample"]["universe_size"])
    assert p.closed is None
    t_idx = np.arange(p.dates.searchsorted(pd.Timestamp("1998-01-01")), len(p.dates))
    dates = p.dates[t_idx]
    R = np.asarray(p.R_pool[t_idx], dtype=np.float32)
    ids = p.sec_ids[:p.n_pool]
    col = pd.Series(np.arange(p.n_pool), index=ids)
    val_name, spec036, prefix = WINDOWS[w]
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    cands, seeds = list(E["candidates"]), list(E["seeds"])
    full, new = pd.read_csv(FULL / "summary.csv"), pd.read_csv(BASE / "summary.csv")
    keys, W = [], np.zeros((len(cands) * len(seeds), len(t_idx), p.n_pool), dtype=np.float32)
    for c in cands:
        g = full[(full.spec == spec036) & (full.K == 30)] if c == cands[0] else new[new.spec == prefix + c.replace("-", "")]
        assert sorted(g.seed) == seeds, (w, c)
        for _, r in g.sort_values("seed").iterrows():
            d = ROOT / "runs" / Path(r.run_dir).name
            x = pd.read_parquet(d / "weights.parquet")
            rr, cc = dates.get_indexer(x["date"]), col.reindex(x["sec_id"]).to_numpy()
            assert (rr >= 0).all() and not np.isnan(cc).any()
            i = len(keys)
            np.add.at(W[i], (rr, cc.astype(int)), x["w"].to_numpy())
            ref = pd.read_csv(d / "oos_daily.csv", usecols=["gross", "turnover"])
            err = np.abs((W[i].astype(np.float64) * R).sum(1) - ref["gross"].to_numpy()).max()
            assert err < 1e-5, (d.name, err)
            keys.append((c, int(r.seed)))
    print(f"{w}: {len(keys)} runs, {len(t_idx)} days, {p.n_pool} names, every gross series = oos_daily.csv, {time.time() - t0:.0f}s", flush=True)
    return {"p": p, "cfg": cfg, "t_idx": t_idx, "dates": dates, "R": R, "ids": ids, "keys": keys, "W": W, "cands": cands, "seeds": seeds}


def pca_resid(K: int, dates: pd.DatetimeIndex, ids: np.ndarray) -> np.ndarray:
    """(T, n_pool) residual of the PCA-K model on these days: V and B come from the 252 days before each date,
    so day t is out of sample. NaN where the name lacks the 252-day history."""
    d = ROOT / "data/us/shared/pca_l252"
    pdates = pd.DatetimeIndex(np.load(d / "dates.npy", allow_pickle=True))
    pids = np.load(d / "sec_ids.npy", allow_pickle=True).astype(str)
    ti = pdates.get_indexer(dates); cj = pd.Series(np.arange(len(pids)), index=pids).reindex(ids.astype(str)).to_numpy()
    assert (ti >= 0).all()
    r = np.load(d / f"K{K}" / "resid.npy", mmap_mode="r")
    known = ~np.isnan(cj)                            # pool names outside the PCA pool stay NaN; none is ever held
    out = np.full((len(ti), len(ids)), np.nan, dtype=np.float32)
    out[:, known] = np.asarray(r[ti][:, cj[known].astype(int)], dtype=np.float32)
    return out


def listing_types(ids: np.ndarray):
    """Type of every pool name from its whole history: a delisted name's last secinfo row is "(LAST KNOWN)" with
    blank flags. Receipts by share type or security name, foreign by Compustat headquarters."""
    raw = ROOT / "data/us/private/raw"
    st = pd.read_parquet(raw / "crsp_secinfo.parquet", columns=["permno", "sharetype", "issuertype", "securitynm"])
    st["sec_id"] = st["permno"].astype("int64").astype(str)
    adr = set(st.loc[st["sharetype"].eq("AD").fillna(False) | st["securitynm"].str.contains(" ADR", na=False), "sec_id"])
    reit = set(st.loc[st["issuertype"].eq("REIT").fillna(False) | st["sharetype"].eq("UG").fillna(False), "sec_id"])
    lk = pd.read_parquet(raw / "ccm_link.parquet", columns=["gvkey", "permno", "linkdt"]).dropna(subset=["permno"]).sort_values("linkdt").groupby("permno").tail(1)
    lk = lk.merge(pd.read_parquet(ROOT / "data/us/private/company_header.parquet", columns=["gvkey", "loc"]), on="gvkey", how="left")
    hq = pd.Series(lk["loc"].to_numpy(), index=lk["permno"].astype("int64").astype(str)).reindex(ids)
    names = ["US common", "depositary receipt", "foreign ordinary", "REIT or unit"]
    return names, np.select([np.isin(ids, list(adr)), np.isin(ids, list(reit)), (hq.notna() & hq.ne("USA")).to_numpy(bool)], [1, 3, 2], 0)


def sel(x, s: str):
    a, b = SAMPLES[s]
    return x.loc[a:b]
