"""Coverage tables and the build report of the European stage 2 (build_europe.py): what
the availability report (scripts/report_europe_availability.py) reads.

  member_month_coverage   per universe member-month, whether each characteristic has a value
                          as the features table sees it (after the last-observed carry)
  input_coverage          per member-month, the share of days with a return, volume,
                          high/low and a spread estimate, and the usable fiscal record
  jkp_coverage            per ranking year and country, the share of member-months with a
                          JKP row and with each JKP column
  build_report            build_report.txt of a version
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from afe.data import europe_panel as xp


def member_month_coverage(detail: pd.DataFrame, mc: pd.DataFrame, dc: pd.DataFrame, names: list[str],
                          calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per universe member-month: 1/0 per characteristic whether a value (after the
    last-observed carry) exists, as the features table sees it: monthly and annual ones at
    the end of M-1, daily ones as the share of the month's trading days with a value."""
    mem = detail[["month", "gvkey", "country", "cap_rank"]].copy()
    prev = pd.MultiIndex.from_arrays([mem["month"] - 1, mem["gvkey"]], names=["month", xp.ID])
    mon = mc.reindex(prev).notna().astype("float32")
    mon.index = mem.index
    out = pd.concat([mem, mon], axis=1)
    if len(dc.columns):
        d = dc.notna().astype("float32").reset_index()
        d["month"] = d["date"].dt.to_period("M")
        dm = d.drop(columns="date").groupby(["month", xp.ID]).mean()
        per_month_days = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).size()
        cnt = d.groupby(["month", xp.ID]).size()
        dm = dm.mul(cnt / cnt.index.get_level_values("month").map(per_month_days).to_numpy(), axis=0)
        idx = pd.MultiIndex.from_arrays([mem["month"], mem["gvkey"]], names=["month", xp.ID])
        dd = dm.reindex(idx).fillna(0.0)
        dd.index = mem.index
        for c in dd.columns:
            out[c] = dd[c].to_numpy()
    return out[["month", "gvkey", "country", "cap_rank"] + [c for c in names if c in out.columns]]


def input_coverage(detail: pd.DataFrame, cd: pd.DataFrame, f: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per universe member-month: share of the month's trading days with a return, a
    traded close, volume, a high/low, a spread estimate; and whether a fiscal record
    usable at the end of M-1 (June rule, <= 30 months old) exists, with its age."""
    c = cd.assign(month=cd["date"].dt.to_period("M"))
    c["has_ret"], c["has_vol"] = c["ret"].notna(), c["cshtrd"].notna()
    c["has_hl"] = c["prchd"].notna() & c["prcld"].notna()
    c["has_cs"] = c["cs"].notna()
    days = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).size()
    g = c.groupby(["month", "gvkey"])[["has_ret", "traded", "has_vol", "has_hl", "has_cs"]].sum()
    g = g.div(g.index.get_level_values("month").map(days).to_numpy(), axis=0)
    mem = detail[["month", "gvkey", "country", "cap_rank"]]
    out = mem.merge(g.reset_index(), on=["month", "gvkey"], how="left").fillna(
        {k: 0.0 for k in ["has_ret", "traded", "has_vol", "has_hl", "has_cs"]})
    fr = f[["gvkey", "datadate", "at"]].copy()
    fr["avail"] = pd.PeriodIndex.from_fields(year=fr["datadate"].dt.year + 1, month=6, freq="M")
    fr = fr[fr["at"].notna()].sort_values("avail")
    left = out[["month", "gvkey"]].assign(key=(out["month"] - 1)).sort_values("key")
    left["key_i"] = left["key"].dt.year * 12 + left["key"].dt.month
    fr["key_i"] = fr["avail"].dt.year * 12 + fr["avail"].dt.month
    m = pd.merge_asof(left.sort_values("key_i"), fr[["gvkey", "key_i", "datadate"]].sort_values("key_i"),
                      on="key_i", by="gvkey", direction="backward")
    m_end = m["key"].dt.to_timestamp(how="end")
    m["fund_age_m"] = (m_end.dt.year - m["datadate"].dt.year) * 12 + (m_end.dt.month - m["datadate"].dt.month)
    m.loc[m["fund_age_m"] > 30, ["datadate", "fund_age_m"]] = np.nan
    out = out.merge(m[["month", "gvkey", "fund_age_m"]], on=["month", "gvkey"], how="left")
    out["has_funda"] = out["fund_age_m"].notna()
    return out


JKP_META = {"id", "permno", "permco", "gvkey", "iid", "excntry", "exch_main", "common", "primary_sec", "bidask",
            "crsp_shrcd", "crsp_exchcd", "comp_tpci", "comp_exchg", "curcd", "fx", "date", "eom", "adjfct",
            "source_crsp", "obs_main", "gics", "sic", "naics", "ff49", "size_grp", "month", "jkp_line", "ret_usd"}


def jkp_coverage(jt: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    """Share of universe member-months with a JKP row, and with each JKP column non-missing,
    per ranking year (the year of M-1, i.e. JKP's eom) and country."""
    mem = detail[["month", "gvkey", "country"]]
    cols = [c for c in jt.columns if c not in JKP_META and pd.api.types.is_numeric_dtype(jt[c])]
    x = mem.merge(jt[["month", "gvkey"] + cols].drop_duplicates(["month", "gvkey"]), on=["month", "gvkey"], how="left",
                  indicator=True)
    nn = x[cols].notna().astype("float32")
    nn["has_row"] = (x["_merge"] == "both").astype("float32")
    nn["year"] = (x["month"] - 1).dt.year
    nn["country"] = x["country"].to_numpy()
    return nn.groupby(["year", "country"]).agg(["mean", "size"]).pipe(_flatten).reset_index()


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    keep = {c: c[0] for c in df.columns if c[1] == "mean"}
    sizes = [c for c in df.columns if c[1] == "size"]
    out = df[list(keep)].copy()
    out.columns = list(keep.values())
    out["n_member_months"] = df[sizes[0]].to_numpy()
    return out


# ------------------------------------------------------------------ report


def build_report(cfg, uni, detail, ret, calendar, mkt, ff, rf, coverage, cmm, fxtab, raw, f, jkp_cov) -> str:
    n = int(cfg["universe"]["size"])
    L = [f"{cfg['market']} stage 2 built {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"countries {cfg['countries']}; universe {cfg['universe']}; numeraire ECU to 1998-12-31, EUR from 1999-01-01", "",
         f"universe: {uni['month'].nunique()} months x {n}, {uni['sec_id'].nunique():,} companies; returns.parquet "
         f"{len(ret):,} rows, {ret['sec_id'].nunique():,} companies; calendar {len(calendar):,} days", ""]
    cal = pd.Series(calendar, index=calendar)
    L += ["Pooled trading days per year: " + str(cal.groupby(calendar.year).size().to_dict()), ""]

    u = uni.assign(month=uni["month"].dt.to_period("M"))
    r = ret.assign(month=ret["date"].dt.to_period("M"))
    days = r.groupby("month")["date"].nunique()
    have = r.merge(u[["month", "sec_id"]], on=["month", "sec_id"]).groupby(["month", "sec_id"]).size().rename("d").reset_index()
    cv = u[["month", "sec_id"]].merge(have, on=["month", "sec_id"], how="left").fillna({"d": 0})
    cv["full"] = cv["d"] >= 0.8 * cv["month"].map(days)
    L += ["Universe members with returns in their month, by year (any day / >= 80% of days):",
          cv.groupby(cv["month"].dt.year).agg(any=("d", lambda s: (s > 0).mean()), full=("full", "mean")).round(4).to_string(), ""]

    comp = detail.groupby([detail["month"].dt.year, "country"]).size().unstack("country").fillna(0)
    comp = comp.div(detail.groupby(detail["month"].dt.year)["month"].nunique(), axis=0)
    L += ["Universe composition, average members per month by home country:", comp.round(1).to_string(), ""]
    cut = detail.groupby(detail["month"].dt.year)["mktcap"].agg(
        cutoff_bn=lambda s: s.min() / 1e3, median_bn=lambda s: s.median() / 1e3, largest_bn=lambda s: s.max() / 1e3)
    L += ["Universe caps per year (numeraire bn; cutoff = smallest member cap of the year):", cut.round(2).to_string(), ""]

    y = ff.groupby(ff.index.year)
    fs = pd.DataFrame({"mkt_mean_ann": y["mktrf"].mean() * 252, "mkt_vol_ann": y["mktrf"].std() * np.sqrt(252),
                       "smb_mean_ann": y["smb"].mean() * 252, "hml_mean_ann": y["hml"].mean() * 252,
                       "days_no_smb_hml": y["hml"].apply(lambda s: int(s.isna().sum())),
                       "rf_year": y["rf"].apply(lambda s: (1.0 + s).prod() - 1.0),
                       "members_min": mkt["n"].groupby(mkt.index.year).min()})
    src = cfg["factors"].get("source", "top_n")
    L += [f"Factors (factors.source {src}; the top {n}'s own = its value-weighted market and 2x3 sort): market "
          "excess return, SMB, HML and rf by year, numeraire (rf_year = the year's daily rf compounded):",
          fs.round(3).to_string(), ""]
    if "mkt_ff" in mkt:
        both = mkt[["mkt_vw", "mkt_ff"]].dropna()
        mo = (1.0 + both).groupby(both.index.to_period("M")).prod() - 1.0
        yr = (1.0 + both).groupby(both.index.year).prod() - 1.0
        L += [f"FF Europe from {both.index.min().date()}, the top {n}'s own before. FF's market against the top {n}'s: "
              f"daily corr {both.corr().iloc[0, 1]:.3f}, monthly corr {mo.corr().iloc[0, 1]:.3f}, monthly tracking "
              f"error {(mo['mkt_vw'] - mo['mkt_ff']).std() * np.sqrt(12):.2%} a year; calendar-year returns:",
              (yr * 100).round(1).T.to_string(), ""]
        if "smb_top_n" in ff:
            s = ff.loc[ff.index >= both.index.min(), ["smb", "smb_top_n", "hml", "hml_top_n"]].dropna()
            L += [f"SMB / HML, FF against the top {n}'s sort: daily corr {s['smb'].corr(s['smb_top_n']):.3f} / "
                  f"{s['hml'].corr(s['hml_top_n']):.3f}; mean a year FF {s['smb'].mean() * 252:.2%} / "
                  f"{s['hml'].mean() * 252:.2%}, top {n} {s['smb_top_n'].mean() * 252:.2%} / "
                  f"{s['hml_top_n'].mean() * 252:.2%}", ""]
    spans = rf.dropna(subset=["series"]).groupby("series")["date"].agg(["min", "max", "size"])
    L += ["rf splice (Bundesbank 1-month series on the calendar):", spans.to_string(), ""]

    L += ["Coverage of each characteristic in features.parquet, share of member-days non-missing BEFORE the "
          "median fill (after the last-observed carry), by year:", coverage.round(3).to_string(), ""]
    early = cmm[cmm["month"].dt.year <= 1997]
    if len(early):
        cc = early.groupby("country")[[c for c in cmm.columns if c not in ("month", "gvkey", "country", "cap_rank")]].mean()
        L += ["Member-month coverage by home country, universe months up to 1997-12:", cc.T.round(2).to_string(), ""]

    fr = f.groupby([f["datadate"].dt.year, "curcd"]).size().unstack("curcd").fillna(0)
    L += ["g_funda records (union pull) by fiscal year end and reporting currency (converted to the numeraire at "
          "the fiscal year end):", fr.loc[:, fr.sum() > 50].astype(int).to_string(), ""]
    if jkp_cov is not None:
        j = jkp_cov.assign(w=jkp_cov["n_member_months"])
        yr = j.groupby("year").apply(lambda t: np.average(t["has_row"], weights=t["w"]))
        L += ["Universe member-months with a JKP row (as of the end of M-1), by year of M-1: " + str(yr.round(3).to_dict()), ""]
    return "\n".join(L)
