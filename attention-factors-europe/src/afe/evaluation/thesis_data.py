"""Inputs of the thesis tables and figures: the matrix results, the evaluation factors, the
universe statistics, and the paper's own numbers (labelled as such wherever they appear).

Matrix results: runs/_matrix/<name>/summary.csv lists one finished run per (spec, K, seed).
Its run_dir is the path on the machine that ran it; the run is found here by its directory
name under runs/. A run's daily series is its oos_daily.csv; net = gross - turnover_cost *
turnover - short_cost * short with the costs of its own manifest. The "combination" of a
spec is the equal-weighted average of its seeds' daily series (one portfolio of the seed
strategies), used where a single return series is needed (cumulative returns, alphas);
the tables report the mean and spread of the per-seed statistics.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from afe.data import europe_panel
from afe.evaluation import metrics

ROOT = Path(__file__).resolve().parents[3]

# Epstein, Wang, Choi & Pelger (2025), Table 2 (US, 1998-2021): K -> SR, mu, sigma, SR_net,
# mu_net, sigma_net, beta. Transcribed from arXiv:2510.11616v1; the paper's numbers, not ours.
PAPER_T2 = {
    "attention": {1: (3.05, 14.45, 4.74, 1.68, 7.94, 4.72, 0.05), 3: (3.05, 14.91, 4.89, 1.69, 8.25, 4.87, 0.06),
                  5: (2.92, 14.21, 4.87, 1.58, 7.66, 4.85, 0.07), 8: (3.35, 15.70, 4.68, 1.94, 9.05, 4.66, 0.07),
                  15: (3.81, 16.66, 4.37, 2.25, 9.78, 4.35, 0.06), 30: (3.97, 16.66, 4.20, 2.28, 9.52, 4.18, 0.05),
                  100: (4.52, 16.45, 3.64, 2.19, 7.93, 3.62, 0.05)},
    "pca_ou": {1: (0.40, 1.85, 4.57, -2.54, -11.62, 4.57, 0.02), 3: (1.26, 4.18, 3.33, -2.72, -9.04, 3.33, 0.01),
               5: (0.99, 2.91, 2.93, -3.44, -10.11, 2.93, 0.00), 8: (0.78, 2.04, 2.61, -4.15, -10.83, 2.61, 0.00),
               10: (0.80, 1.99, 2.50, -4.32, -10.80, 2.50, 0.00), 15: (0.51, 1.12, 2.20, -5.24, -11.53, 2.20, 0.00),
               30: (0.18, 0.40, 2.24, -6.45, -14.74, 2.29, -0.00), 100: (-0.35, -0.66, 1.87, -7.05, -13.23, 1.88, -0.00)},
    "pca_longconv": {1: (2.26, 13.10, 5.79, 1.19, 6.98, 5.78, 0.10), 3: (2.76, 14.61, 5.30, 1.57, 8.29, 5.28, 0.07),
                     5: (2.41, 14.10, 5.86, 1.30, 7.62, 5.84, 0.10), 8: (2.64, 14.88, 5.63, 1.50, 8.42, 5.61, 0.09),
                     10: (2.66, 14.94, 5.61, 1.52, 8.48, 5.59, 0.09), 15: (2.56, 14.74, 5.75, 1.41, 8.08, 5.73, 0.09),
                     30: (2.79, 15.15, 5.42, 1.57, 8.47, 5.40, 0.09), 100: (2.66, 14.36, 5.40, 1.44, 7.75, 5.38, 0.09)},
    "market": (0.42, 8.61, 20.37, 0.42, 8.61, 20.37, 1.00),
}
# Table 3 (K = 30): dropped group -> SR, SR seed sd, mu, sigma, SR_net, mu_net, beta.
PAPER_T3 = {"none": (3.97, 0.13, 16.66, 4.20, 2.28, 9.52, 0.05),
            "past_returns": (1.50, 0.07, 7.82, 5.23, 0.59, 3.09, 0.08),
            "investment": (3.88, 0.17, 17.93, 4.63, 2.19, 10.06, 0.06),
            "profitability": (3.94, 0.15, 18.39, 4.67, 2.26, 10.48, 0.05),
            "intangibles": (3.91, 0.15, 18.18, 4.65, 2.24, 10.34, 0.06),
            "value": (4.08, 0.12, 18.45, 4.53, 2.32, 10.44, 0.04),
            "trading_frictions": (2.90, 0.14, 13.36, 4.61, 1.34, 6.14, 0.06)}
# Table 4 (selected tuning parameters, as published)
PAPER_T4 = {"policy.hidden": 32, "policy.dropout": 0.1, "model.embedding_dim": 32, "training.epochs": 30,
            "policy.layers": 1, "objective.lambda_var": 100, "training.lr": 0.003, "training.weight_decay": 0.05,
            "policy.lambda_squash": 0.001}


class Matrix:
    def __init__(self, matrix_dir: Path, runs_dir: Path | None = None):
        self.dir = Path(matrix_dir)
        self.runs_dir = Path(runs_dir) if runs_dir else self.dir.parents[1]
        s = pd.read_csv(self.dir / "summary.csv")
        s["run"] = [self.runs_dir / Path(p.replace("\\", "/")).name for p in s["run_dir"]]
        missing = [str(p) for p in s["run"] if not (p / "oos_daily.csv").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} run directories missing, e.g. {missing[:3]}")
        self.summary = s

    def runs(self, spec: str, K: int, seeds: list[int] | None = None) -> pd.DataFrame:
        r = self.summary[(self.summary["spec"] == spec) & (self.summary["K"] == K)]
        if seeds is not None:
            r = r[r["seed"].isin(seeds)]
        if r.empty:
            raise KeyError(f"no runs for {spec} K={K}")
        return r.sort_values("seed")

    def Ks(self, spec: str) -> list[int]:
        return sorted(self.summary.loc[self.summary["spec"] == spec, "K"].unique().tolist())

    def per_seed(self, spec: str, K: int, seeds: list[int] | None = None) -> dict[int, pd.DataFrame]:
        return {int(r.seed): daily(r.run) for r in self.runs(spec, K, seeds).itertuples()}

    def combination(self, spec: str, K: int, seeds: list[int] | None = None) -> pd.DataFrame:
        ds = list(self.per_seed(spec, K, seeds).values())
        cols = ["gross", "net", "turnover", "short"]
        out = sum(d[cols] for d in ds) / len(ds)
        return out.assign(mkt_ew=ds[0]["mkt_ew"], rf=ds[0]["rf"]).dropna()


@lru_cache(maxsize=None)
def _costs(run: Path) -> tuple[float, float]:
    ob = json.loads((run / "manifest.json").read_text())["config"]["objective"]
    return float(ob["turnover_cost"]), float(ob["short_cost"])


@lru_cache(maxsize=512)
def daily(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "oos_daily.csv", parse_dates=["date"]).set_index("date")
    tc, sc = _costs(run)
    return d.assign(net=d["gross"] - tc * d["turnover"] - sc * d["short"])


def perf(d: pd.DataFrame, tc: float = 0.0005, sc: float = 0.0001) -> dict:
    """Table 2 statistics of one daily series (columns gross, net, turnover, short, mkt_ew)."""
    g, n = metrics.annualised(d["gross"]), metrics.annualised(d["net"])
    return {"SR": g["SR"], "mu": g["mu_pct"], "sigma": g["sigma_pct"], "SR_net": n["SR"], "mu_net": n["mu_pct"],
            "sigma_net": n["sigma_pct"], "beta": metrics.beta(d["gross"].to_numpy(), d["mkt_ew"].to_numpy()),
            "turnover": float(d["turnover"].mean()), "short": float(d["short"].mean()),
            "break_even_bps": metrics.break_even_turnover_cost(d["gross"].to_numpy(), d["turnover"].to_numpy(),
                                                                 d["short"].to_numpy(), sc)}


# ------------------------------------------------------------------ factors


def us_factors(root: Path = ROOT) -> pd.DataFrame:
    """Fama-French five factors + momentum (WRDS ff.fivefactors_daily, decimals)."""
    f = pd.read_parquet(root / "data/us/shared/raw/ff5_daily.parquet").set_index("date").astype("float64")
    return f.rename(columns={"umd": "mom"})[["mktrf", "smb", "hml", "rmw", "cma", "mom", "rf"]]


def _french_zip(path: Path) -> pd.DataFrame:
    """A Ken French daily file whose first table starts with a ',' header (any columns), in decimals."""
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head = next(i for i, s in enumerate(lines) if s.startswith(","))
    cols = [c.strip().lower().replace("-", "") for c in lines[head].split(",")[1:]]
    rows = []
    for s in lines[head + 1:]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows and s.strip():
            break
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype("float64").replace(-99.99, np.nan) / 100.0


def europe_factors(root: Path = ROOT, market: str = "europe17") -> pd.DataFrame:
    """Ken French's Europe five factors and momentum (USD) in the build's numeraire (ECU to
    1998, EUR after): market total return x (numeraire per USD) change, minus the Bundesbank
    rate of the build; long-short spreads x the same change (both legs convert alike)."""
    ext = root / "data/europe/private/raw/external"
    f5 = europe_panel.read_ff_daily(ext / "Europe_5_Factors_Daily_CSV.zip")
    mom = _french_zip(ext / "Europe_Mom_Factor_Daily_CSV.zip").rename(columns={"wml": "mom"})
    f = f5.join(mom, how="inner")
    fx = pd.read_parquet(root / f"data/{market}/shared/fx_to_numeraire_daily.parquet")
    usd = fx[fx["currency"] == "USD"].set_index("date")["eur_per_unit"].astype("float64").sort_index()
    x = usd.reindex(usd.index.union(f.index)).ffill().reindex(f.index)
    g = x / x.shift(1)
    rf = pd.read_parquet(root / f"data/{market}/shared/rf_daily.parquet").set_index("date")["rf"].astype("float64")
    rf_n = rf.reindex(rf.index.union(f.index)).ffill().reindex(f.index)
    out = pd.DataFrame({"mktrf": (1 + f["mktrf"] + f["rf"]) * g - 1 - rf_n})
    for c in ["smb", "hml", "rmw", "cma", "mom"]:
        out[c] = f[c] * g
    return out.assign(rf=rf_n).dropna()


def str_factor(data_dir: Path, lo: float = 0.3, hi: float = 0.7) -> pd.Series:
    """Own short-term reversal factor of a universe: each day, the equal-weighted return of the
    members in the bottom 30% of last month's return (features char_ST_Rev, known before the
    day) minus the top 30%; Ken French's has no European counterpart, so both markets use this."""
    f = pd.read_parquet(data_dir / "features.parquet", columns=["date", "sec_id", "char_ST_Rev"])
    r = pd.read_parquet(data_dir / "returns.parquet", columns=["date", "sec_id", "ret"])
    d = f.merge(r, on=["date", "sec_id"], how="inner").dropna(subset=["ret"])
    q = d.groupby("date")["char_ST_Rev"].rank(pct=True)
    long = d[q <= lo].groupby("date")["ret"].mean()
    short = d[q > hi].groupby("date")["ret"].mean()
    return (long - short).rename("str").astype("float64")


# ------------------------------------------------------------------ universe


def universe_by_month(data_dir: Path, market: str, root: Path = ROOT) -> pd.DataFrame:
    """Members on each month's first trading day: count, smallest / median / largest cap in USD
    bn (US build: USD millions; Europe: numeraire millions over numeraire per USD), country."""
    u = pd.read_parquet(data_dir / "universe.parquet")
    first = u.groupby("month").size().index
    cols = ["date", "sec_id", "mktcap_lag"] + (["country"] if market != "us" else [])
    r = pd.read_parquet(data_dir / "returns.parquet", columns=cols)
    r = r[r["date"].isin(first)]
    m = u.rename(columns={"month": "date"}).merge(r, on=["date", "sec_id"], how="left")
    if market != "us":
        fx = pd.read_parquet(root / f"data/{market}/shared/fx_to_numeraire_daily.parquet")
        usd = fx[fx["currency"] == "USD"].set_index("date")["eur_per_unit"].astype("float64").sort_index()
        m["cap_usd_bn"] = m["mktcap_lag"] / usd.reindex(usd.index.union(first)).ffill().reindex(m["date"]).to_numpy() / 1e3
    else:
        m["cap_usd_bn"] = m["mktcap_lag"] / 1e3
    return m
