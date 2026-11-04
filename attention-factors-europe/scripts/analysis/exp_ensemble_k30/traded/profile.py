"""Who are the positions the 25 models agree on, and who are the contested ones? 2026-11-04. Run from the repo root.

Every held stock-day gets a group by how many of the 25 models sit on w_bar's side: contested < 20, mixed 20-22,
strong 23-24, unanimous 25; and a side, long or short w_bar. Characteristics are row t of the panel, known at the close
of t-1, like the weights. Residuals are PCA-30 (loadings from the 252 days before each date). Written to traded/:
  profile_group    names, hit rate, position size, position age, residual vol, size of the 5-day residual move
  profile_tilts    mean rank of each characteristic by group and side (ranks run -0.5..0.5, equal weight)
  profile_corr     daily cross-sectional correlation of w_bar with characteristics and lagged residuals (signed),
                   of the agreement share with characteristics and |lagged residuals|
  profile_slopes   the same variables together in one daily regression (Fama-MacBeth)
  profile_r2       adjusted R2 of a member, w_bar, member minus w_bar on characteristics, lags, both
  profile_quint    exposure, P&L, costs by size, volatility, liquidity quintile
  profile_sector   the same by Fama-French 12 industry (SIC ranges of Siccodes12), net exposure, share contested
  profile_split    book return = market (net exposure x EW return) + sector + within sector, exact, variance and mean
  profile_paths    mean PCA-30 residual on days -30..+20 around a position, by group and side
  profile_churn    turnover of contested stock-days: consensus nearby (exit or entry leg) or not
  profile_type     by listing type: US common, depositary receipt, foreign ordinary, REIT or unit
  profile_names    largest names by P&L and exposure per group
Returns per exposure are raw and, as ret_dm, demeaned across held names each day: a long book earns the market on raw
returns, so long against short is only comparable demeaned.
"""
import argparse, time
import numpy as np, pandas as pd
from _common import ROOT, OUT, ANN, TC, SC, load, pca_resid, listing_types

ap = argparse.ArgumentParser(); ap.add_argument("--w", default="roll"); a = ap.parse_args()
D = load(a.w); W, R, dates, keys, ids, p, t_idx = D["W"], D["R"], D["dates"], D["keys"], D["ids"], D["p"], D["t_idx"]
T, n = R.shape; M = len(keys); t0 = time.time()
GROUPS, EDGES = ["contested", "mixed", "strong", "unanimous"], [20, 23, 25]
SIDES = ["short", "long"]
LAGS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30)]; LAG_N = [f"lag{x}-{y}" for x, y in LAGS]
QCH = ["LME", "Variance", "Resid_Var", "Beta", "Spread", "LTurnover"]
HOR = np.arange(-30, 21)
late = dates >= pd.Timestamp("2010-01-01")
PERIODS = {"1998-2021": np.ones(T, bool), "1998-2009": ~late, "2010-2021": late}

ci = [j for j, c in enumerate(p.features) if c.startswith("char_")]; CN = [p.features[j][5:] for j in ci]
assert len(ci) == 39
VARS = CN + LAG_N

FF12 = {"NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
        "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792),
                  (3900, 3939), (3990, 3999)],
        "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713),
                  (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
        "Enrgy": [(1200, 1399), (2900, 2999)], "Chems": [(2800, 2829), (2840, 2899)],
        "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)], "Telcm": [(4800, 4899)],
        "Utils": [(4900, 4949)], "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
        "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)], "Money": [(6000, 6999)]}
SEC_N = list(FF12) + ["Other"]; NS = len(SEC_N)
look = np.full(10000, NS - 1, dtype=np.int8)
for s, ranges in enumerate(FF12.values()):
    for x, y in ranges:
        look[x:y + 1] = s


def sectors() -> np.ndarray:
    raw = ROOT / "data/us/private/raw"
    m = pd.concat([pd.read_parquet(raw / f, columns=["permno", "date", "siccd"]) for f in ("crsp_monthly.parquet", "crsp_monthly_extra.parquet")])
    m["sec_id"] = m["permno"].astype("int64").astype(str)
    m = m[m["sec_id"].isin(ids)]
    m["ym"] = pd.to_datetime(m["date"]).dt.to_period("M")
    s = m.drop_duplicates(["ym", "sec_id"]).pivot(index="ym", columns="sec_id", values="siccd").astype("float64")
    s = s.reindex(columns=ids).replace({0: np.nan, 9999: np.nan}).sort_index().ffill().bfill()
    print(f"names without a SIC code: {int(s.isna().all().sum())} of {len(ids)}", flush=True)
    s = s.reindex(dates.to_period("M"), method="ffill").fillna(9999).to_numpy().astype(int)
    return look[s]


SEC = sectors()
raw = ROOT / "data/us/private/raw"
si = pd.read_parquet(raw / "crsp_secinfo.parquet", columns=["permno", "secinfostartdt", "ticker", "securitynm"])
si = si[(si["secinfostartdt"] <= pd.Timestamp("2021-12-31")) & si["ticker"].notna()].sort_values("secinfostartdt").groupby("permno").tail(1)
si["sec_id"] = si["permno"].astype("int64").astype(str); si = si.set_index("sec_id")
TYPE_N, TYPE = listing_types(ids)
print("listing types in the pool:", dict(zip(TYPE_N, np.bincount(TYPE, minlength=4).tolist())), flush=True)

E = pca_resid(30, p.dates[t_idx[0] - 30:], ids); E = np.nan_to_num(E)            # E[t + 30] = day t
assert p.dates[t_idx[0] - 30:][30] == dates[0] and len(E) == T + 30
C = np.zeros((T + 31, n)); C[1:] = np.cumsum(E, axis=0, dtype=np.float64)
C2 = np.zeros((T + 31, n)); C2[1:] = np.cumsum(E.astype(np.float64) ** 2, axis=0)
print(f"residuals ready, {time.time() - t0:.0f}s", flush=True)


def zs(x):
    x = x - x.mean(0); s = x.std(0)
    return x / np.where(s == 0, 1.0, s)


def adj_r2(Xs, Y, h):
    Q, _ = np.linalg.qr(Xs); k = Xs.shape[1]
    r2 = ((Q.T @ Y) ** 2).sum(0) / np.maximum((Y ** 2).sum(0), 1e-30)
    return 1 - (1 - r2) * (h - 1) / (h - k - 1)


NG, NV = len(GROUPS), len(VARS)
K_Q = ["names", "expo", "expo_contested", "expo_unanimous", "pnl", "turn", "short", "pnl_dm"]
K_S = ["names", "expo", "expo_contested", "expo_unanimous", "pnl", "turn", "short", "net", "abs_net", "pnl_dm"]
K_G = ["names", "expo", "bar_abs", "pnl", "hit", "age", "resvol", "move5", "turn", "short", "pnl_dm", "hit_dm"]
K_T = ["names", "expo", "pnl", "pnl_dm", "turn", "short"]
tilt = np.full((T, 2 * NG, 39), np.nan, dtype=np.float32)
corr = np.full((T, NV, 2), np.nan); beta = np.full((T, NV, 2), np.nan); r2 = np.full((T, 3, 3), np.nan)
quint = np.zeros((T, len(QCH), 5, len(K_Q)))
sect = np.zeros((T, NS, len(K_S)))
grp = np.zeros((T, 2 * NG, len(K_G))); typ = np.zeros((T, 4 * NG, len(K_T)))
split = np.zeros((T, M, 3)); l1 = np.zeros(T)
num_e, num_w = np.zeros((2, 2 * NG, len(HOR))), np.zeros((2, 2 * NG, len(HOR)))
den_e, den_w = np.zeros((2, 2 * NG, len(HOR))), np.zeros((2, 2 * NG, len(HOR)))
nm_pnl, nm_exp = np.zeros((3, NG, n)), np.zeros((3, NG, n))
S = np.full((T, n), NG, dtype=np.int8); BAR = np.zeros((T, n), dtype=np.float32)
TB, TM = np.zeros((T, n), dtype=np.float32), np.zeros((T, n), dtype=np.float32)
prev, prev_bar = np.zeros((M, n)), np.zeros(n); age = np.zeros(n); psign = np.zeros(n)

for t in range(T):
    w = W[:, t].astype(np.float64); bar = w.mean(0); mabs = np.abs(w).mean(0)
    dw = np.abs(w - prev).mean(0); dbar = np.abs(bar - prev_bar); short = np.clip(-w, 0, None).mean(0)
    TB[t] = dbar; TM[t] = dw; BAR[t] = bar; l1[t] = np.abs(bar).sum()
    sgn = np.sign(bar); age = np.where((sgn == psign) & (sgn != 0), age + 1, (sgn != 0) * 1.0); psign = sgn
    A = np.flatnonzero(mabs > 0); h = len(A)
    if h == 0:
        prev, prev_bar = w, bar
        continue
    slot = p.idx[t_idx[t]].numpy(); pos = np.full(n + 1, -1); pos[slot] = np.arange(len(slot))
    assert (pos[A] >= 0).all()
    Xh = p.X[t_idx[t]].numpy()[pos[A]][:, ci].astype(np.float64)
    same = (np.sign(w[:, A]) == sgn[A]).sum(0); g = np.digitize(same, EDGES); S[t, A] = g
    side = (bar[A] > 0).astype(int); gs = g * 2 + side; G = np.eye(2 * NG)[gs]
    r = R[t, A].astype(np.float64); ab = np.abs(bar[A]); m = mabs[A]; pnl = bar[A] * r; d = dw[A]; sh = short[A]
    rm = r.mean(); pdm = bar[A] * (r - rm)
    # residual lags: window t-30..t-1 is E[t .. t+29]
    i = t + 30
    mu = (C[i, A] - C[i - 30, A]) / 30; sd = np.sqrt(np.maximum((C2[i, A] - C2[i - 30, A]) / 30 - mu ** 2, 0.0))
    sd = np.where(sd > 0, sd, np.nan)
    Z = np.column_stack([(C[i - x + 1, A] - C[i - y, A]) / (sd * np.sqrt(y - x + 1)) for x, y in LAGS])
    Z = np.nan_to_num(Z, posinf=0.0, neginf=0.0)
    mv5 = np.nan_to_num((C[i, A] - C[i - 5, A]) / (sd * np.sqrt(5)))
    # by group and side
    cnt = G.sum(0); tilt[t][cnt > 0] = ((G.T @ Xh)[cnt > 0] / cnt[cnt > 0, None]).astype(np.float32)
    for k, v in (("names", np.ones(h)), ("expo", m), ("bar_abs", ab), ("pnl", pnl), ("hit", (pnl > 0) * 1.0), ("age", age[A]),
                 ("resvol", np.nan_to_num(sd)), ("move5", np.abs(mv5)), ("turn", d), ("short", sh), ("pnl_dm", pdm), ("hit_dm", (pdm > 0) * 1.0)):
        grp[t, :, K_G.index(k)] = np.bincount(gs, weights=v, minlength=2 * NG)
    tg = TYPE[A] * NG + g
    for k, v in (("names", np.ones(h)), ("expo", m), ("pnl", pnl), ("pnl_dm", pdm), ("turn", d), ("short", sh)):
        typ[t, :, K_T.index(k)] = np.bincount(tg, weights=v, minlength=4 * NG)
    # cross-sectional correlations, slopes, explained shares
    Xz, Zz, Za = zs(Xh), zs(Z), zs(np.abs(Z))
    yb, ya = zs(bar[A]), zs(same / M)
    F1, F2 = np.column_stack([Xz, Zz]), np.column_stack([Xz, Za])
    corr[t, :, 0] = F1.T @ yb / h; corr[t, :, 1] = F2.T @ ya / h
    beta[t, :, 0] = np.linalg.lstsq(F1, yb, rcond=None)[0]; beta[t, :, 1] = np.linalg.lstsq(F2, ya, rcond=None)[0]
    Y = np.column_stack([bar[A], w[:, A].T, (w[:, A] - bar[A]).T]); Y = Y - Y.mean(0)
    for j, Fx in enumerate((Xz[:, Xz.std(0) > 0], Zz, F1[:, F1.std(0) > 0])):
        q = adj_r2(Fx, Y, h)
        r2[t, j] = q[0], q[1:M + 1].mean(), q[M + 1:].mean()
    # size, volatility, liquidity quintiles
    for j, c in enumerate(QCH):
        rk = np.argsort(np.argsort(Xh[:, CN.index(c)], kind="stable"), kind="stable") * 5 // h
        for k, v in (("names", np.ones(h)), ("expo", m), ("expo_contested", m * (g == 0)), ("expo_unanimous", m * (g == NG - 1)),
                     ("pnl", pnl), ("turn", d), ("short", sh), ("pnl_dm", pdm)):
            quint[t, j, :, K_Q.index(k)] = np.bincount(rk, weights=v, minlength=5)
    # sectors, and market + sector + within sector for every model
    sc = SEC[t, A]; Gs = np.eye(NS)[sc]; ns = Gs.sum(0)
    rs = np.where(ns > 0, (Gs.T @ r) / np.maximum(ns, 1), rm)
    wA = w[:, A]; net = wA @ Gs
    split[t, :, 0] = wA.sum(1) * rm; split[t, :, 1] = (net * (rs - rm)).sum(1); split[t, :, 2] = (wA * (r - rs[sc])).sum(1)
    netb = net.mean(0)
    for k, v in (("names", ns), ("expo", Gs.T @ m), ("expo_contested", Gs.T @ (m * (g == 0))), ("expo_unanimous", Gs.T @ (m * (g == NG - 1))),
                 ("pnl", Gs.T @ pnl), ("turn", Gs.T @ d), ("short", Gs.T @ sh), ("net", netb), ("abs_net", np.abs(netb)), ("pnl_dm", Gs.T @ pdm)):
        sect[t, :, K_S.index(k)] = v
    # residual path around the position
    pc = int(late[t]); ok = (i + HOR >= 0) & (i + HOR < len(E)); P_ = E[i + HOR[ok]][:, A].astype(np.float64)
    num_e[pc][:, ok] += G.T @ P_.T; den_e[pc][:, ok] += cnt[:, None]
    num_w[pc][:, ok] += (G * ab[:, None]).T @ P_.T; den_w[pc][:, ok] += (G * ab[:, None]).sum(0)[:, None]
    # names
    for per in (0, 1 + pc):
        np.add.at(nm_pnl[per], (g, A), pnl); np.add.at(nm_exp[per], (g, A), m)
    prev, prev_bar = w, bar
    if t % 1000 == 0:
        print(f"day {t}, {time.time() - t0:.0f}s", flush=True)
print(f"days done, {time.time() - t0:.0f}s", flush=True)

# totals must equal the books
gross = np.stack([(W[i].astype(np.float64) * R).sum(1) for i in range(M)])
assert np.abs(split.sum(2).T - gross).max() < 1e-9
assert abs(grp[:, :, K_G.index("pnl")].sum() - gross.mean(0).sum()) < 1e-6
assert abs(sect[:, :, K_S.index("expo")].sum(1).mean() - 1) < 1e-6
assert abs(quint[:, 0, :, K_Q.index("pnl")].sum() - gross.mean(0).sum()) < 1e-6
print(f"checks ok: market + sector + within = gross for {M} runs, mu {100 * ANN * gross.mean():.3f}%", flush=True)


def monthly_t(x, mask):
    """Mean of a daily series and its t from monthly means (x may hold NaN)."""
    s = pd.Series(x[mask], index=dates[mask]); mm = s.groupby(s.index.to_period("M")).mean().dropna()
    return s.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))


rows = []
for per, mask in PERIODS.items():
    tot = grp[mask].sum((0, 1))
    for b in range(2 * NG):
        x = grp[mask, b].sum(0); e, nm, pn = x[K_G.index("expo")], x[K_G.index("names")], x[K_G.index("pnl")]
        c = TC * x[K_G.index("turn")] + SC * x[K_G.index("short")]
        rows.append({"period": per, "group": GROUPS[b // 2], "side": SIDES[b % 2], "names_day": nm / mask.sum(), "share_expo": e / tot[K_G.index("expo")],
                     "share_pnl": pn / tot[K_G.index("pnl")], "ret_per_expo_pct": 100 * ANN * pn / e, "ret_dm_per_expo_pct": 100 * ANN * x[K_G.index("pnl_dm")] / e,
                     "share_pnl_dm": x[K_G.index("pnl_dm")] / tot[K_G.index("pnl_dm")], "cost_per_expo_pct": 100 * ANN * c / e,
                     "hit": x[K_G.index("hit")] / nm, "hit_dm": x[K_G.index("hit_dm")] / nm, "size_bp": 1e4 * e / nm, "bar_bp": 1e4 * x[K_G.index("bar_abs")] / nm,
                     "age_days": x[K_G.index("age")] / nm, "res_vol_pct": 100 * np.sqrt(ANN) * x[K_G.index("resvol")] / nm, "move5_z": x[K_G.index("move5")] / nm})
pd.DataFrame(rows).to_csv(OUT / f"profile_group_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    mu_t = np.nanmean(tilt[mask], 0)
    for j, c in enumerate(CN):
        d = {"period": per, "char": c}
        for b in range(2 * NG):
            d[f"{GROUPS[b // 2]}_{SIDES[b % 2]}"] = mu_t[b, j]
        rows.append(d)
pd.DataFrame(rows).to_csv(OUT / f"profile_tilts_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    for j, v in enumerate(VARS):
        c0, t0_ = monthly_t(corr[:, j, 0], mask); c1, t1_ = monthly_t(corr[:, j, 1], mask)
        b0, bt0 = monthly_t(beta[:, j, 0], mask); b1, bt1 = monthly_t(beta[:, j, 1], mask)
        rows.append({"period": per, "var": v, "corr_w": c0, "t_corr_w": t0_, "corr_agree": c1, "t_corr_agree": t1_,
                     "slope_w": b0, "t_slope_w": bt0, "slope_agree": b1, "t_slope_agree": bt1})
pd.DataFrame(rows).to_csv(OUT / f"profile_corr_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    for j, s in enumerate(("characteristics", "lagged residuals", "both")):
        for k, c in enumerate(("average", "member", "member minus average")):
            rows.append({"period": per, "regressors": s, "target": c, "adj_r2": np.nanmean(r2[mask, j, k])})
pd.DataFrame(rows).to_csv(OUT / f"profile_r2_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    for j, c in enumerate(QCH):
        tq = quint[mask, j].sum((0, 1))
        for q in range(5):
            x = quint[mask, j, q].sum(0); e, pn = x[K_Q.index("expo")], x[K_Q.index("pnl")]
            cs = TC * x[K_Q.index("turn")] + SC * x[K_Q.index("short")]
            rows.append({"period": per, "char": c, "quintile": q + 1, "share_expo": e / tq[K_Q.index("expo")], "share_pnl": pn / tq[K_Q.index("pnl")],
                         "share_cost": cs / (TC * tq[K_Q.index("turn")] + SC * tq[K_Q.index("short")]), "ret_per_expo_pct": 100 * ANN * pn / e,
                         "ret_dm_per_expo_pct": 100 * ANN * x[K_Q.index("pnl_dm")] / e,
                         "cost_per_expo_pct": 100 * ANN * cs / e, "net_per_expo_pct": 100 * ANN * (pn - cs) / e,
                         "contested_of_expo": x[K_Q.index("expo_contested")] / e, "unanimous_of_expo": x[K_Q.index("expo_unanimous")] / e})
pd.DataFrame(rows).to_csv(OUT / f"profile_quint_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    ts = sect[mask].sum((0, 1))
    for s, nm in enumerate(SEC_N):
        x = sect[mask, s].sum(0); e, pn = x[K_S.index("expo")], x[K_S.index("pnl")]
        cs = TC * x[K_S.index("turn")] + SC * x[K_S.index("short")]
        rows.append({"period": per, "sector": nm, "share_names": x[K_S.index("names")] / ts[K_S.index("names")], "share_expo": e / ts[K_S.index("expo")],
                     "share_pnl": pn / ts[K_S.index("pnl")], "ret_per_expo_pct": 100 * ANN * pn / e if e else np.nan,
                     "ret_dm_per_expo_pct": 100 * ANN * x[K_S.index("pnl_dm")] / e if e else np.nan, "cost_per_expo_pct": 100 * ANN * cs / e if e else np.nan,
                     "net_per_expo_pct": 100 * ANN * (pn - cs) / e if e else np.nan, "contested_of_expo": x[K_S.index("expo_contested")] / e if e else np.nan,
                     "unanimous_of_expo": x[K_S.index("expo_unanimous")] / e if e else np.nan,
                     "mean_net": sect[mask, s, K_S.index("net")].mean(), "mean_abs_net": sect[mask, s, K_S.index("abs_net")].mean()})
pd.DataFrame(rows).to_csv(OUT / f"profile_sector_{a.w}.csv", index=False)


def pieces(s, mask):
    s = s[mask]; t_ = s.sum(1); v = t_.var(ddof=1); c = np.cov(s.T)
    return {"sigma_pct": 100 * np.sqrt(ANN * v), "share_market": c[0, 0] / v, "share_sector": c[1, 1] / v, "share_within": c[2, 2] / v,
            "share_cov": 1 - (c[0, 0] + c[1, 1] + c[2, 2]) / v, "mu_market_pct": 100 * ANN * s[:, 0].mean(), "mu_sector_pct": 100 * ANN * s[:, 1].mean(),
            "mu_within_pct": 100 * ANN * s[:, 2].mean(), "SR_within": s[:, 2].mean() / s[:, 2].std(ddof=1) * np.sqrt(ANN)}


rows = []
for name, idx in {"all 25": list(range(M)), "seeds sobol-036": [i for i, k in enumerate(keys) if k[0] == "sobol-036"]}.items():
    bar_s = split[:, idx].mean(1); wb = sum(W[i].astype(np.float64) for i in idx) / len(idx)
    sc_ = np.divide(1.0, np.abs(wb).sum(1), out=np.zeros(T), where=np.abs(wb).sum(1) > 0)
    for per, mask in PERIODS.items():
        rows.append({"group": name, "period": per, "piece": "member (mean)", **pd.DataFrame([pieces(split[:, i], mask) for i in idx]).mean().to_dict()})
        rows.append({"group": name, "period": per, "piece": "average", **pieces(bar_s, mask)})
        rows.append({"group": name, "period": per, "piece": "average rescaled", **pieces(bar_s * sc_[:, None], mask)})
        rows.append({"group": name, "period": per, "piece": "disagreement (mean)", **pd.DataFrame([pieces(split[:, i] - bar_s, mask) for i in idx]).mean().to_dict()})
pd.DataFrame(rows).to_csv(OUT / f"profile_split_{a.w}.csv", index=False)

rows = []
for per, mask in PERIODS.items():
    a_ = typ[mask].sum(0); tt = a_.sum(0)
    for ty, nm in enumerate(TYPE_N):
        x = a_[ty * NG:(ty + 1) * NG]; al = x.sum(0); e, pn, pd_ = al[K_T.index("expo")], al[K_T.index("pnl")], al[K_T.index("pnl_dm")]
        cs = TC * al[K_T.index("turn")] + SC * al[K_T.index("short")]
        rows.append({"period": per, "type": nm, "names_day": al[K_T.index("names")] / mask.sum(), "share_expo": e / tt[K_T.index("expo")], "share_pnl": pn / tt[K_T.index("pnl")],
                     "share_pnl_dm": pd_ / tt[K_T.index("pnl_dm")], "share_cost": cs / (TC * tt[K_T.index("turn")] + SC * tt[K_T.index("short")]), "ret_per_expo_pct": 100 * ANN * pn / e,
                     "ret_dm_per_expo_pct": 100 * ANN * pd_ / e, "cost_per_expo_pct": 100 * ANN * cs / e, "contested_of_expo": x[0, K_T.index("expo")] / e, "unanimous_of_expo": x[3, K_T.index("expo")] / e,
                     "unanimous_share_pnl_dm": x[3, K_T.index("pnl_dm")] / tt[K_T.index("pnl_dm")], "unanimous_ret_dm_pct": 100 * ANN * x[3, K_T.index("pnl_dm")] / x[3, K_T.index("expo")],
                     "contested_ret_dm_pct": 100 * ANN * x[0, K_T.index("pnl_dm")] / x[0, K_T.index("expo")]})
pd.DataFrame(rows).to_csv(OUT / f"profile_type_{a.w}.csv", index=False)

rows = []
for pc, per in enumerate(("1998-2009", "2010-2021")):
    for b in range(2 * NG):
        for j, hh in enumerate(HOR):
            rows.append({"period": per, "group": GROUPS[b // 2], "side": SIDES[b % 2], "h": hh, "eps_ew_bp": 1e4 * num_e[pc, b, j] / den_e[pc, b, j],
                         "eps_w_bp": 1e4 * num_w[pc, b, j] / den_w[pc, b, j]})
pd.DataFrame(rows).to_csv(OUT / f"profile_paths_{a.w}.csv", index=False)

# contested stock-days: is a consensus position (strong or unanimous) within five days before or after?
cons = (S == 2) | (S == 3); c = np.zeros((T + 1, n), dtype=np.int32); c[1:] = np.cumsum(cons, axis=0)
past = np.zeros((T, n), bool); nxt = np.zeros((T, n), bool)
past[1:] = (c[1:T] - c[np.clip(np.arange(1, T) - 5, 0, None)]) > 0
nxt[:-1] = (c[np.minimum(np.arange(1, T) + 5, T)] - c[1:T]) > 0
ctd = S == 0
rows = []
for per, mask in PERIODS.items():
    tb, tm, bb = TB[mask].astype(np.float64), TM[mask].astype(np.float64), np.abs(BAR[mask]).astype(np.float64)
    pl = (BAR[mask].astype(np.float64) * R[mask])
    for nm, sel in (("after consensus only (exit leg)", ctd & past & ~nxt), ("before consensus only (entry leg)", ctd & ~past & nxt), ("between consensus", ctd & past & nxt),
                    ("no consensus within 5 days (churn)", ctd & ~past & ~nxt), ("all contested", ctd), ("everything else", ~ctd)):
        s = sel[mask]
        rows.append({"period": per, "class": nm, "share_stockdays_contested": s.sum() / ctd[mask].sum(), "share_turn_bar": tb[s].sum() / tb.sum(), "share_turn_mem": tm[s].sum() / tm.sum(),
                     "share_expo_bar": bb[s].sum() / bb.sum(), "share_pnl": pl[s].sum() / pl.sum()})
pd.DataFrame(rows).to_csv(OUT / f"profile_churn_{a.w}.csv", index=False)

rows = []
for pi, per in enumerate(PERIODS):
    for gi, g_ in enumerate(GROUPS):
        tp, te = nm_pnl[pi, gi].sum(), nm_exp[pi, gi].sum()
        for by, arr in (("pnl", nm_pnl[pi, gi]), ("loss", -nm_pnl[pi, gi]), ("expo", nm_exp[pi, gi])):
            for r_, j in enumerate(np.argsort(-arr)[:15]):
                sid = str(ids[j]); tk, nm = (si.loc[sid, "ticker"], si.loc[sid, "securitynm"]) if sid in si.index else ("", "")
                rows.append({"period": per, "group": g_, "by": by, "rank": r_ + 1, "sec_id": sid, "ticker": tk, "name": str(nm).split(";")[0],
                             "sector": SEC_N[int(np.bincount(SEC[:, j], minlength=NS).argmax())], "pnl_pct": 100 * ANN * nm_pnl[pi, gi, j] / PERIODS[per].sum(),
                             "share_group_pnl": nm_pnl[pi, gi, j] / tp, "share_group_expo": nm_exp[pi, gi, j] / te})
pd.DataFrame(rows).to_csv(OUT / f"profile_names_{a.w}.csv", index=False)
conc = []
for pi, per in enumerate(PERIODS):
    for gi, g_ in enumerate(GROUPS):
        x = np.sort(nm_pnl[pi, gi])[::-1]; e = nm_exp[pi, gi]; nz = (e > 0).sum()
        conc.append({"period": per, "group": g_, "names_ever": int(nz), "top10_share_pnl": x[:10].sum() / x.sum(), "top50_share_pnl": x[:50].sum() / x.sum(),
                     "top10_share_expo": np.sort(e)[::-1][:10].sum() / e.sum(), "share_names_positive": (nm_pnl[pi, gi][e > 0] > 0).mean()})
pd.DataFrame(conc).to_csv(OUT / f"profile_conc_{a.w}.csv", index=False)
print(f"written, {time.time() - t0:.0f}s", flush=True)
