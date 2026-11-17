"""Are the two receipt-free universes what they claim? 2026-11-04. Run from the repo root, after both builds.

Month by month against the headline universe, from the raw CRSP flags, not from the build's own output:
  no line that was a receipt at its ranking date; drop = headline minus its receipts, exactly; refilled = 500 names
  that hold every non-receipt headline member; dense ranks; returns of the names they share unchanged.
Also how the build's rule (share type AD in the ranking month) compares with the re-booking's (ever a receipt).
"""
import numpy as np, pandas as pd
from _common import ROOT, OUT, listing_types

raw = ROOT / "data/us/private/raw"
ex = pd.read_parquet(raw / "crsp_monthly_extra.parquet", columns=["permno", "date", "sharetype"])
ex["ym"] = pd.to_datetime(ex["date"]).dt.to_period("M"); ad = ex[ex["sharetype"].eq("AD")]
ad = set(zip(ad["ym"], ad["permno"].astype("int64").astype(str)))
print(f"receipt line-months in the raw pull: {len(ad):,}", flush=True)


def load(name):
    u = pd.read_parquet(ROOT / "data/us" / name / "universe.parquet"); u["ym"] = pd.to_datetime(u["month"]).dt.to_period("M")
    return u[u["ym"] <= pd.Period("2021-12", "M")]


head, drop, fill = load("shared"), load("drop_receipts"), load("no_receipts")
assert (head.groupby("ym").size() == 500).all() and (fill.groupby("ym").size() == 500).all()
rows, bad = [], 0
for ym, h in head.groupby("ym"):
    hs, ds, fs = set(h["sec_id"]), set(drop.loc[drop["ym"] == ym, "sec_id"]), set(fill.loc[fill["ym"] == ym, "sec_id"])
    rec = {s for s in hs if (ym - 1, s) in ad}
    assert ds == hs - rec, ym
    assert (hs - rec) <= fs, ym
    assert not any((ym - 1, s) in ad for s in fs), ym
    rows.append({"ym": ym, "headline_receipts": len(rec), "drop_names": len(ds), "fill_new": len(fs - hs)})
    for u in (drop, fill):
        r = sorted(u.loc[u["ym"] == ym, "cap_rank"].tolist()); assert r == list(range(1, len(r) + 1)), ym
c = pd.DataFrame(rows); c["year"] = [p.year for p in c["ym"]]
print("every month: no receipt in either universe, drop = headline minus receipts, refilled keeps every non-receipt member, ranks dense", flush=True)
s = c.groupby(c["year"] // 4 * 4)[["headline_receipts", "drop_names", "fill_new"]].mean().round(1); print(s.to_string())
print(f"drop names per month: min {c.drop_names.min()}, mean {c.drop_names.mean():.1f}, max {c.drop_names.max()}", flush=True)

a = pd.read_parquet(ROOT / "data/us/shared/returns.parquet", columns=["date", "sec_id", "ret"])
for name in ("drop_receipts", "no_receipts"):
    b = pd.read_parquet(ROOT / "data/us" / name / "returns.parquet", columns=["date", "sec_id", "ret"])
    m = a.merge(b, on=["date", "sec_id"], suffixes=("", "_b"))
    d = (m["ret"] - m["ret_b"]).abs()
    print(f"{name}: {b['sec_id'].nunique()} names, {len(m):,} shared return rows, max |difference| {d.max():.2e}, rows differing {(d > 1e-12).sum()}, NaN mismatch {(m['ret'].isna() != m['ret_b'].isna()).sum()}", flush=True)

_, TYPE = listing_types(np.sort(head["sec_id"].unique())); ids = np.sort(head["sec_id"].unique())
ever = set(ids[TYPE == 1]); hm = head[head["sec_id"].isin(ever)]
inner = pd.Series([(y - 1, s) in ad for y, s in zip(hm["ym"], hm["sec_id"])])
print(f"name-months of ever-receipts in the headline: {len(hm):,}; a receipt at that date: {int(inner.sum()):,}; ordinary line by then: {int((~inner).sum()):,} ({(~inner).mean():.1%})", flush=True)
left = hm[~inner.to_numpy()].groupby("sec_id").size().sort_values(ascending=False).head(10)
si = pd.read_parquet(raw / "crsp_secinfo.parquet", columns=["permno", "secinfostartdt", "securitynm"]).sort_values("secinfostartdt").groupby("permno").tail(1)
si["sec_id"] = si["permno"].astype("int64").astype(str); nm = si.set_index("sec_id")["securitynm"].str.split(";").str[0]
print("ever-receipts that stay in the receipt-free universes as ordinary lines, name-months:", "; ".join(f"{nm.get(k, k)} {v}" for k, v in left.items()), flush=True)
c.to_csv(OUT / "check_no_receipts_universe.csv", index=False)
