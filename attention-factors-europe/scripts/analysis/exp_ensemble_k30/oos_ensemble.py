"""Out of sample, the books the K = 30 ensemble validation chose between (scripts/ensemble_validation.py).

The validation window (fit 1990-95, score 1996-97) compared two averages of the 25 models
(5 candidates x seeds 0-4): rule "seeds" = the search winner sobol-036 on five seeds, rule "configs" =
the five candidates at one seed, its Sharpe the mean over the five seeds. Here every one of those
books on 1998-2021 from the full sequential runs: sobol-036 from runs/_matrix/exp_fixedwin_full
(us_attention / us_attention_fixedwin, K = 30), the other four from runs/_matrix/exp_ensemble_k30.
Weights re-booked with the runner's own daily_book (checked against every run's oos_daily.csv) and
averaged with ensemble_validation.ensembles / average, as on the validation window. A diagnostic,
not a thesis row. Run from the repo root.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from afe.data import slots
from afe.evaluation import metrics
import run_attention_us as rau
import ensemble_validation as ev

OUT, FULL = ROOT / "runs/_matrix/exp_ensemble_k30", ROOT / "runs/_matrix/exp_fixedwin_full"
ANN, TC, SC = 252, 5e-4, 1e-4
# window -> (validation config, sobol-036 spec in exp_fixedwin_full, spec prefix in exp_ensemble_k30)
WINDOWS = {"roll": ("us_ensemble_val", "us_attention", "us_k30_"),
           "fixed": ("us_ensemble_val_fixedwin", "us_attention_fixedwin", "us_k30_fixedwin_")}
cfg = yaml.safe_load((ROOT / "configs/us_headline.yaml").read_text())
assert cfg["execution"]["lag"] == 0 and cfg["objective"]["turnover_cost"] == TC and cfg["objective"]["short_cost"] == SC
t0 = time.time()
p = slots.load_slots(ROOT / cfg["data"]["dir"], f"{rau.window_start(cfg, 1998):%Y-%m-%d}", "2021-12-31",
                     max_rank=cfg["sample"]["universe_size"])
t_idx = np.arange(p.dates.searchsorted(pd.Timestamp("1998-01-01")), len(p.dates))
oos_dates = p.dates[t_idx]
col = pd.Series(np.arange(p.n_pool), index=p.sec_ids[:p.n_pool])
assert col.index.is_unique
print(f"panel {p.n_pool} pool names, {len(t_idx)} oos days, {time.time() - t0:.0f}s", flush=True)

def pool_weights(d: Path) -> np.ndarray:
    w = pd.read_parquet(d / "weights.parquet")
    r = oos_dates.get_indexer(w["date"]); c = col.reindex(w["sec_id"]).to_numpy()
    assert (r >= 0).all() and not np.isnan(c).any()
    wp = np.zeros((len(t_idx), p.n_pool), dtype=np.float32)
    np.add.at(wp, (r, c.astype(int)), w["w"].to_numpy())
    return wp

def sr(x): return x.mean() / x.std(ddof=1) * np.sqrt(ANN)
def net_of(d: pd.DataFrame) -> pd.Series: return d["gross"] - TC * d["turnover"] - SC * d["short"]
def stats(d: pd.DataFrame) -> dict:
    net = net_of(d)
    return {"net_SR": sr(net), "gross_SR": sr(d["gross"]), "mu_pct": 100 * ANN * d["gross"].mean(),
            "mu_net_pct": 100 * ANN * net.mean(), "sigma_pct": 100 * np.sqrt(ANN) * d["gross"].std(ddof=1),
            "costs_pct": 100 * ANN * (d["gross"] - net).mean(),
            "BE_bp": metrics.break_even_turnover_cost(d["gross"], d["turnover"], d["short"], SC),
            "turnover": d["turnover"].mean(), "short": d["short"].mean(),
            "net_98_09": sr(net.loc[:"2009"]), "net_10_21": sr(net.loc["2010":])}

full, new = pd.read_csv(FULL / "summary.csv"), pd.read_csv(OUT / "summary.csv")
rows, nets, rules, frames, checked = [], {}, [], [], 0
for w, (val_name, spec036, prefix) in WINDOWS.items():
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    cands, seeds = list(E["candidates"]), list(E["seeds"])
    assert cands[0] == "sobol-036"
    W = {}
    for c in cands:
        g = full[(full.spec == spec036) & (full.K == 30)] if c == cands[0] else new[new.spec == prefix + c.replace("-", "")]
        assert sorted(g.seed) == seeds, (w, c)
        for _, r in g.iterrows():
            d = ROOT / "runs" / Path(r.run_dir).name
            wp = pool_weights(d)
            mine = rau.daily_book(p, wp, t_idx, cfg)  # re-booking must reproduce the run's own series
            ref = pd.read_csv(d / "oos_daily.csv", parse_dates=["date"])
            for k in ("gross", "turnover", "short"):
                err = np.abs(mine[k].to_numpy() - ref[k].to_numpy()).max()
                assert err < 1e-6, (d.name, k, err)
            mine = mine.set_index("date")
            assert abs(sr(net_of(mine)) - r.SR_net) < 1e-6, (d.name, sr(net_of(mine)), r.SR_net)
            assert abs(stats(mine)["BE_bp"] - r.break_even_bps) < 1e-3, (d.name, r.break_even_bps)
            checked += 1
            W[(c, int(r.seed))] = wp
            nets[(w, f"single:{c}:s{int(r.seed)}", "single")] = net_of(mine).to_numpy()
            frames.append(mine[["gross", "turnover", "short"]].assign(w=w, book=f"single:{c}:s{int(r.seed)}", kind="single"))
            rows.append({"w": w, "book": f"single:{c}:s{int(r.seed)}", "kind": "single", "models": 1, **stats(mine)})
    for name, members in ev.ensembles(cands, seeds).items():
        mean = sum(W[m].astype(np.float64) for m in members) / len(members)
        for kind in ("avg", "avg_l1"):
            d = rau.daily_book(p, ev.average([mean], kind == "avg_l1"), t_idx, cfg).set_index("date")
            nets[(w, name, kind)] = net_of(d).to_numpy()
            frames.append(d[["gross", "turnover", "short"]].assign(w=w, book=name, kind=kind))
            rows.append({"w": w, "book": name, "kind": kind, "models": len(members),
                         "l1": float(np.abs(ev.average([mean], kind == "avg_l1")).sum(1).mean()), **stats(d)})
    print(f"{w:5s} {len(W)} runs booked, {time.time() - t0:.0f}s", flush=True)
    del W
print(f"re-booking checked on {checked} runs: gross, turnover, short within 1e-6 of oos_daily.csv, net SR = summary.csv")
df = pd.DataFrame(rows)
df.to_csv(OUT / "oos_books.csv", index=False)
pd.concat(frames).reset_index().to_parquet(OUT / "oos_daily_books.parquet", index=False)  # every book's daily series

# the two rules, validation against out of sample
for w, (val_name, _, _) in WINDOWS.items():
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    cands, seeds, win = list(E["candidates"]), list(E["seeds"]), list(E["candidates"])[0]
    vb = pd.read_csv(ROOT / "runs/_ensval" / val_name / "books.csv")
    vm = json.loads((ROOT / "runs/_ensval" / val_name / "metrics.json").read_text())
    for kind in ("avg_l1", "avg"):
        for src, b in (("validation", vb.assign(w=w)), ("oos", df[df.w == w])):
            s = b[b.kind == kind].set_index("book")["net_SR"]; one = b[b.kind == "single"].set_index("book")["net_SR"]
            rules.append({"w": w, "kind": kind, "sample": src,
                          "single_winner": one[[f"single:{win}:s{x}" for x in seeds]].mean(),
                          "single_all": one.mean(),
                          "rule_seeds": s[f"seeds:{win}"], "rule_configs": s[[f"configs:s{x}" for x in seeds]].mean(),
                          "latin": s[[f"configs:latin{r}" for r in range(len(seeds))]].mean(), "all": s["all"],
                          "choice": vm["choice"] if src == "validation" and kind == vm["book"] else ""})
R = pd.DataFrame(rules)
R["seeds_minus_configs"] = R.rule_seeds - R.rule_configs
R.to_csv(OUT / "oos_rules.csv", index=False)

# per candidate: validation and out-of-sample single-run means, and its seed book
cand_rows = []
for w, (val_name, _, _) in WINDOWS.items():
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    vb = pd.read_csv(ROOT / "runs/_ensval" / val_name / "books.csv").set_index(["book", "kind"])["net_SR"]
    ob = df[df.w == w].set_index(["book", "kind"])["net_SR"]
    for c in E["candidates"]:
        singles = [f"single:{c}:s{x}" for x in E["seeds"]]
        cand_rows.append({"w": w, "cand": c,
                          "val_single": np.mean([vb[(b, "single")] for b in singles]), "val_seeds_book": vb[(f"seeds:{c}", "avg_l1")],
                          "oos_single": np.mean([ob[(b, "single")] for b in singles]),
                          "oos_single_sd": np.std([ob[(b, "single")] for b in singles], ddof=1),
                          "oos_seeds_book": ob[(f"seeds:{c}", "avg_l1")]})
C = pd.DataFrame(cand_rows)
C.to_csv(OUT / "oos_candidates.csv", index=False)

# paired differences, moving-block bootstrap over dates (21 days, 2000 reps, as seed_ensemble.py)
n, B, reps = len(t_idx), 21, 2000
rng = np.random.default_rng(0)
starts = rng.integers(0, n - B, size=(reps, n // B + 1))
ix = (starts[:, :, None] + np.arange(B)).reshape(reps, -1)[:, :n]
def boot_sr(x): xb = x[ix]; return xb.mean(1) / xb.std(1, ddof=1) * np.sqrt(ANN)
def boot_mean(keys): return np.mean([boot_sr(nets[k]) for k in keys], axis=0), np.mean([sr(nets[k]) for k in keys])
pair = []
for w, (val_name, _, _) in WINDOWS.items():
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    cands, seeds, win = list(E["candidates"]), list(E["seeds"]), list(E["candidates"])[0]
    single = [(w, f"single:{win}:s{x}", "single") for x in seeds]
    for kind in ("avg_l1", "avg"):
        sets = {"seeds": [(w, f"seeds:{win}", kind)], "configs": [(w, f"configs:s{x}", kind) for x in seeds],
                "latin": [(w, f"configs:latin{r}", kind) for r in range(len(seeds))], "all": [(w, "all", kind)],
                "single_winner": single}
        for a, b in (("seeds", "configs"), ("all", "seeds"), ("all", "configs"), ("latin", "configs"),
                     ("seeds", "single_winner"), ("configs", "single_winner")):
            (ba, pa), (bb, pb) = boot_mean(sets[a]), boot_mean(sets[b])
            pair.append({"w": w, "kind": kind, "a": a, "b": b, "SR_a": pa, "SR_b": pb, "d_net": pa - pb,
                         "boot_se": (ba - bb).std(), "p_le0": float(np.mean(ba - bb <= 0))})
P = pd.DataFrame(pair)
P.to_csv(OUT / "oos_paired.csv", index=False)

pd.set_option("display.width", 250)
print("\nrules, net SR (validation 1996-97 / out of sample 1998-2021)")
print(R.round(3).to_string(index=False))
print("\ncandidates (net SR; books avg_l1)")
print(C.round(3).to_string(index=False))
print("\npaired, net SR a - b out of sample (block bootstrap 21 days, 2000 reps)")
print(P.round(3).to_string(index=False))
print("\nbooks out of sample")
print(df[df.kind != "single"].sort_values(["w", "kind", "book"]).round(3).to_string(index=False))
