"""The thesis figures from the run matrix (static, for print): PNG (300 dpi) and PDF.

    python scripts/thesis_figures.py --out <dir> [--matrix runs/_matrix/thesis] [--only f02_cumulative ...]

Writes <out>/figures/<name>.png/.pdf and <out>/figures/index.json (title, caption, source),
and the series behind each line chart as <out>/figures/data/<name>.csv. Colours follow the
model in every figure (attention blue, PCA + LongConv orange, PCA + OU aqua, market grey;
validated with the dataviz palette checker: all-pairs CVD dE 9.2, normal 24.0, the aqua below
3:1 on white so every line is direct-labelled); markets are panels, never colours. Scatter
plots highlight at most three groups over grey (three slots pass the all-pairs check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.evaluation import thesis_data as td  # noqa: E402
from afe.model.attention_pipeline import AttentionArb  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"attention": "#2a78d6", "pca_longconv": "#eb6834", "pca_ou": "#1baf7a", "market": "#52514e"}
NAME = {"attention": "Attention factors", "pca_longconv": "PCA + LongConv", "pca_ou": "PCA + OU threshold",
        "market": "Market (equal-weighted)"}
HI3 = ["#2a78d6", "#eb6834", "#1baf7a"]
STACK = ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # Other first: validated order
MARKETS = {"us": "US", "eu": "Europe"}
DATA = {"us": "data/us/shared", "eu": "data/europe17/holfill"}
W = 6.3                                                                       # text width, inches
COUNTRY = {"GB": "United Kingdom", "GBR": "United Kingdom", "FR": "France", "FRA": "France", "DE": "Germany",
           "DEU": "Germany", "CH": "Switzerland", "CHE": "Switzerland", "NL": "Netherlands", "NLD": "Netherlands",
           "IT": "Italy", "ITA": "Italy", "ES": "Spain", "ESP": "Spain", "SE": "Sweden", "SWE": "Sweden"}
INDEX: list[dict] = []

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none",
    "pdf.fonttype": 42})


def spec(m: str, model: str) -> str:
    return f"{m}_{model}"


def save(fig, out: Path, name: str, title: str, caption: str, source: str, data: pd.DataFrame | None = None) -> None:
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    files = [f"figures/{name}.png", f"figures/{name}.pdf"]
    if data is not None:
        (d / "data").mkdir(exist_ok=True)
        data.to_csv(d / "data" / f"{name}.csv")
        files.append(f"figures/data/{name}.csv")
    INDEX.append({"name": name, "title": title, "caption": caption, "source": source, "files": files})
    print(f"  {name}: {title}")


def year_axis(ax, step: int = 5) -> None:
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def log_ticks(ax, ticks: list[float]) -> None:
    lo, hi = ax.get_ylim()
    t = [v for v in ticks if lo <= v <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(t))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in t]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def end_labels(ax, items: list[tuple[str, float]], x, log: bool = False, min_gap: float = 0.06) -> None:
    """Labels at the right end of lines, in ink, nudged apart (axes-fraction units)."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    tf = (lambda v: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))) if log else (lambda v: (v - lo) / (hi - lo))
    pos = sorted([(tf(y), lab) for lab, y in items])
    ys = [p for p, _ in pos]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = max(0.0, ys[-1] - 0.98)
    ys = [y - shift for y in ys]
    lift = max(0.0, 0.02 - ys[0])
    ys = [y + lift for y in ys]
    for y, (_, lab) in zip(ys, pos):
        ax.annotate(lab, xy=(1.01, y), xycoords=("axes fraction", "axes fraction"), color=INK2, fontsize=7.5,
                    va="center", ha="left", annotation_clip=False)


# ------------------------------------------------------------------ figures


def f01_universe(out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    eu = td.universe_by_month("europe17")
    top = ["GBR", "FRA", "DEU", "CHE", "NLD"]
    lab = {"GBR": "United Kingdom", "FRA": "France", "DEU": "Germany", "CHE": "Switzerland", "NLD": "Netherlands"}
    cc = eu.groupby(["date", "country"]).size().unstack(fill_value=0)
    cc = pd.concat([cc.drop(columns=top, errors="ignore").sum(axis=1).rename("Other"), cc[top]], axis=1)
    cc = cc[(cc.index >= "1998-01-01") & (cc.index <= "2021-12-31")]
    share = 100 * cc.div(cc.sum(axis=1), axis=0)
    a.stackplot(share.index, share.T.values, colors=STACK, edgecolor="white", linewidth=0.6)
    a.set_ylim(0, 100)
    a.set_ylabel("Share of the 200 members (%)")
    a.set_title("European universe by country")
    year_axis(a)
    cum = share.cumsum(axis=1).iloc[-1]
    mids = (cum - share.iloc[-1] / 2)
    end_labels(a, [(("Other countries" if c == "Other" else lab[c]), mids[c]) for c in share.columns], share.index[-1],
               min_gap=0.075)
    caps = {}
    for m, mk in [("us", "us"), ("eu", "europe17")]:
        u = td.universe_by_month(mk)
        caps[MARKETS[m]] = u.groupby("date")["cap_usd_bn"].min()
    c = pd.DataFrame(caps)
    c = c[(c.index >= "1998-01-01") & (c.index <= "2021-12-31")]
    for (k, s), col in zip(c.items(), HI3[:2]):
        b.plot(s.index, s.values, color=col, lw=1.4)
    b.set_yscale("log")
    log_ticks(b, [2, 5, 10, 20, 50])
    year_axis(b)
    b.set_ylabel("Smallest member's cap ($bn, log)")
    b.set_title("Size of the universe's smallest member")
    end_labels(b, [("US (500th)", c["US"].iloc[-1]), ("Europe (200th)", c["Europe"].iloc[-1])], c.index[-1], log=True)
    fig.tight_layout(w_pad=6)
    save(fig, out, "F01_universe", "Figure 1. The two universes",
         "Left: country composition of the European universe (200 largest companies of europe17, first trading "
         "day of each month). Right: market capitalisation of the smallest member, billions of USD, log scale: "
         "the US 500th and the European 200th listing are of similar size throughout.",
         "data/us/private/universe_asof.parquet, data/europe17/private/universe_asof.parquet", pd.concat([share.add_prefix("share_"), c.add_prefix("min_cap_")], axis=1))


def cumulative(out: Path, M: td.Matrix, col: str, name: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    keep = {}
    for ax, m in zip(axes, MARKETS):
        items = []
        for model in ["attention", "pca_longconv", "pca_ou"]:
            d = M.combination(spec(m, model), 30)
            cum = (1 + d[col]).cumprod()
            ax.plot(cum.index, cum.values, color=COL[model], lw=1.4, label=NAME[model])
            items.append((NAME[model], cum.iloc[-1]))
            keep[f"{MARKETS[m]} {NAME[model]}"] = cum
        mk = (1 + d["mkt_ew"]).cumprod()
        ax.plot(mk.index, mk.values, color=COL["market"], lw=0.9, label=NAME["market"])
        items.append(("Market", mk.iloc[-1]))
        keep[f"{MARKETS[m]} market"] = mk
        ax.set_yscale("log")
        ax.set_title(MARKETS[m])
        ax.axhline(1, color=AXIS, lw=0.6)
        year_axis(ax)
        ax._items = items
    axes[0].set_ylabel("Value of 1 invested (log)")
    for ax in axes:
        log_ticks(ax, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
        end_labels(ax, ax._items, None, log=True)
    fig.tight_layout(w_pad=6)
    save(fig, out, name, title,
         f"Cumulative {'net' if col == 'net' else 'gross'} return of the K = 30 strategies, January 1998 - December "
         "2021, log scale, " + ("after 5bp per unit turnover and 1bp per unit short position; " if col == "net" else "before costs; ")
         + "each strategy is the equal-weighted combination of its seeds (PCA + OU: one deterministic run); the "
         "equal-weighted universe for reference. Portfolios are scaled to ||w||_1 = 1, so levels are not "
         "comparable with the market's.", "runs/_matrix/thesis (headline)", pd.DataFrame(keep))


def f02_cumulative(out, M):
    cumulative(out, M, "net", "F02_cumulative_net", "Figure 2. Cumulative net returns")


def f03_cumulative_gross(out, M):
    cumulative(out, M, "gross", "F03_cumulative_gross", "Figure 3. Cumulative gross returns")


def f04_sharpe_vs_k(out: Path, M: td.Matrix) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    rows = []
    for ax, m in zip(axes, MARKETS):
        items = []
        for model in ["attention", "pca_longconv", "pca_ou"]:
            sp = spec(m, model)
            Ks = M.Ks(sp)
            per = M.summary[M.summary["spec"] == sp].groupby("K")["SR_net"]
            mu, lo, hi = per.mean().loc[Ks], per.min().loc[Ks], per.max().loc[Ks]
            ax.fill_between(Ks, lo, hi, color=COL[model], alpha=0.18, lw=0)
            ax.plot(Ks, mu, color=COL[model], marker="o", ms=3.2, lw=1.4, label=NAME[model])
            items.append((NAME[model], mu.iloc[-1]))
            rows += [{"market": MARKETS[m], "model": NAME[model], "K": k, "mean": mu[k], "min": lo[k], "max": hi[k]} for k in Ks]
        if m == "us":
            pk = sorted(td.PAPER_T2["attention"])
            ax.plot(pk, [td.PAPER_T2["attention"][k][3] for k in pk], ls="none", marker="o", ms=4.2, mfc="white",
                    mec=COL["attention"], mew=1.0, label="Attention, paper's Table 2")
            ax.annotate("paper", xy=(pk[-1], td.PAPER_T2["attention"][pk[-1]][3]), xytext=(4, 4),
                        textcoords="offset points", color=INK2, fontsize=7)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 8, 15, 30, 100])
        ax.set_xticklabels(["1", "3", "5", "8", "15", "30", "100"])
        ax.minorticks_off()
        ax.set_xlabel("Number of factors K")
        ax.set_title(MARKETS[m])
        ax._items = items
    axes[0].set_ylabel("Net Sharpe ratio")
    for ax in axes:
        end_labels(ax, ax._items, None, min_gap=0.08)
    fig.tight_layout(w_pad=6)
    save(fig, out, "F04_net_sharpe_vs_k", "Figure 4. Net Sharpe ratio and the number of factors",
         "Out-of-sample net Sharpe ratio, 1998-2021, against K: mean over seeds (line) and the range over seeds "
         "(band); PCA + OU is one deterministic run. Hollow markers: the attention model in the paper's Table 2 (US).",
         "runs/_matrix/thesis (headline); Epstein et al. (2025) Table 2", pd.DataFrame(rows).set_index(["market", "model", "K"]))


def f05_break_even(out: Path, M: td.Matrix) -> None:
    costs = np.arange(0, 20.25, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), sharey=True)
    rows = {}
    for ax, m in zip(axes, MARKETS):
        for model in ["attention", "pca_longconv", "pca_ou"]:
            curves, bes = [], []
            for d in M.per_seed(spec(m, model), 30).values():
                g, t, s = d["gross"], d["turnover"], d["short"]
                curves.append([metrics.annualised(g - c * 1e-4 * t - 1e-4 * s)["SR"] for c in costs])
                bes.append(metrics.break_even_turnover_cost(g.to_numpy(), t.to_numpy(), s.to_numpy(), 1e-4))
            y, be = np.mean(curves, axis=0), float(np.mean(bes))
            ax.plot(costs, y, color=COL[model], lw=1.4, label=NAME[model])
            ax.plot([be], [0], ls="none", marker="o", ms=4.5, color=COL[model], mec="white", mew=0.8, zorder=5)
            ax.annotate(f"{be:.1f}bp", xy=(be, 0), xytext=(0, -11 if model != "pca_longconv" else 8),
                        textcoords="offset points", ha="center", color=INK2, fontsize=7,
                        bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"}, zorder=6)
            rows[f"{MARKETS[m]} {NAME[model]}"] = y
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(5, color=MUTED, lw=0.7)
        ax.annotate("paper's 5bp", xy=(5, 0.98), xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", color=INK2, fontsize=7, va="top")
        ax.set_xlim(0, 20)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("Cost per unit turnover (bp)")
        ax.set_title(MARKETS[m])
    axes[0].set_ylabel("Net Sharpe ratio")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig, out, "F05_break_even", "Figure 5. Net Sharpe ratio as a function of the trading cost",
         "K = 30, 1998-2021, mean over seeds: the net Sharpe ratio for a cost of 0-20bp per unit turnover (shorting "
         "cost 1bp kept); dots: the break-even cost (mean over seeds, Table 6). The statutory transaction taxes of "
         "the European markets (UK 50bp, France 20-30bp, Italy 10-12bp on purchases) are in Table 6.",
         "runs/_matrix/thesis (headline)", pd.DataFrame(rows, index=pd.Index(costs, name="cost_bp")))


def f06_annual(out: Path, M: td.Matrix) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True)
    rows = {}
    for ax, m in zip(axes, MARKETS):
        for i, model in enumerate(["attention", "pca_longconv"]):
            ys = [d.groupby(d.index.year)["net"].apply(lambda x: metrics.annualised(x)["SR"])
                  for d in M.per_seed(spec(m, model), 30).values()]
            y = pd.concat(ys, axis=1).mean(axis=1)
            rows[f"{MARKETS[m]} {NAME[model]}"] = y
            ax.bar(y.index + (i - 0.5) * 0.4, y.values, width=0.38, color=COL[model], label=NAME[model], lw=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(MARKETS[m])
        ax.set_ylabel("Net Sharpe ratio")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_xticks(range(1998, 2022, 2))
    fig.tight_layout(h_pad=1.5)
    save(fig, out, "F06_net_sharpe_by_year", "Figure 6. Net Sharpe ratio by year",
         "Net Sharpe ratio within each calendar year, K = 30, mean over seeds: attention factors and the two-step "
         "PCA + LongConv benchmark. Table A1 has the values, including PCA + OU and the 2022-2025 holdout.",
         "runs/_matrix/thesis (headline)", pd.DataFrame(rows))


def f07_turnover(out: Path, M: td.Matrix) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    rows = {}
    for ax, m in zip(axes, MARKETS):
        items = []
        for model in ["attention", "pca_longconv", "pca_ou"]:
            d = M.combination(spec(m, model), 30)
            r = d["turnover"].rolling(126, min_periods=60).mean()
            ax.plot(r.index, r.values, color=COL[model], lw=1.2, label=NAME[model])
            rows[f"{MARKETS[m]} {NAME[model]}"] = r
            items.append((NAME[model], r.iloc[-1]))
        ax.set_ylim(0, 1.0)
        ax.set_title(MARKETS[m])
        year_axis(ax)
        ax._items = items
    axes[0].set_ylabel("Daily turnover, 6-month mean")
    for ax in axes:
        end_labels(ax, ax._items, None, min_gap=0.08)
    fig.tight_layout(w_pad=6)
    save(fig, out, "F07_turnover", "Figure 7. Turnover over time",
         "Daily turnover ||w_t - w_t-1||_1 of the K = 30 strategies (||w||_1 = 1), 126-day moving average, "
         "equal-weighted combination of the seeds. A turnover of 0.5 at 5bp costs 6.3% a year.",
         "runs/_matrix/thesis (headline)", pd.DataFrame(rows))


def f08_ablation(out: Path, M: td.Matrix) -> None:
    groups = {"past_returns": "Past returns", "trading_frictions": "Trading frictions", "value": "Value",
              "investment": "Investment", "profitability": "Profitability", "intangibles": "Intangibles"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True, sharex=True)
    rows = []
    for ax, m in zip(axes, MARKETS):
        base = M.runs(spec(m, "attention"), 30, [0, 1, 2]).set_index("seed")["SR_net"]
        ys, err = [], []
        for g in groups:
            r = M.runs(f"{m}_drop_{g}", 30, [0, 1, 2]).set_index("seed")["SR_net"]
            dlt = r - base.reindex(r.index)
            ys.append(dlt.mean())
            err.append(dlt.std(ddof=1))
            rows.append({"market": MARKETS[m], "dropped": groups[g], "delta_SR_net": dlt.mean(), "sd": dlt.std(ddof=1)})
        pos = np.arange(len(groups))[::-1]
        ax.barh(pos, ys, height=0.55, color=COL["attention"], lw=0, xerr=err,
                error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        if m == "us":
            pv = [td.PAPER_T3[g][4] - td.PAPER_T3["none"][4] for g in groups]
            ax.plot(pv, pos, ls="none", marker="o", ms=4.2, mfc="white", mec=INK2, mew=1.0)
            ax.annotate("paper, Table 3", xy=(pv[0], pos[0]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=7)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(pos)
        ax.set_yticklabels(list(groups.values()))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Change in net Sharpe ratio")
        ax.set_title(MARKETS[m])
    fig.tight_layout(w_pad=2)
    save(fig, out, "F08_ablation", "Figure 8. Removing one characteristic group",
         "Change in the net Sharpe ratio of the attention model (K = 30, 1998-2021) when a characteristic group is "
         "removed, against the headline model on the same seeds 0-2 (mean and sd of the paired differences). "
         "Hollow markers: the same change in the paper's Table 3.", "runs/_matrix/thesis (ablation, headline)",
         pd.DataFrame(rows).set_index(["market", "dropped"]))


def f09_holdout(out: Path, M: td.Matrix) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    keep = {}
    for ax, m in zip(axes, MARKETS):
        items = []
        for col, lab in [("net", "Net"), ("gross", "Gross")]:
            d = M.combination(f"{m}_attention_2022_2025", 30)
            c = (1 + d[col]).cumprod()
            ax.plot(c.index, c.values, color=COL["attention"], lw=1.4 if col == "net" else 0.9,
                    alpha=1.0 if col == "net" else 0.55)
            items.append((lab, c.iloc[-1]))
            keep[f"{MARKETS[m]} {lab}"] = c
        ax.axhline(1, color=AXIS, lw=0.6)
        ax.set_title(MARKETS[m])
        year_axis(ax, 1)
        ax._items = items
    axes[0].set_ylabel("Value of 1 invested")
    for ax in axes:
        end_labels(ax, ax._items, None, min_gap=0.08)
    fig.tight_layout(w_pad=5)
    save(fig, out, "F09_holdout_2022_2025", "Figure 9. The 2022-2025 holdout",
         "Cumulative gross and net return of the attention model (K = 30, equal-weighted combination of seeds 0-4) "
         "after the paper's sample, same hyperparameters, annual refits on the preceding eight years.",
         "runs/_matrix/thesis (extension)", pd.DataFrame(keep))


# ------------------------------------------------------------------ factor structure


def sic_group(s) -> str:
    try:
        s = int(float(s))
    except (TypeError, ValueError):
        return "Other"
    if 6000 <= s <= 6799:
        return "Finance & real estate"
    if 1000 <= s <= 1499 or 2900 <= s <= 2999 or 4920 <= s <= 4925:
        return "Energy & mining"
    if 3570 <= s <= 3579 or 3600 <= s <= 3699 or 7370 <= s <= 7379:
        return "Technology"
    return "Other"


def factor_state(M: td.Matrix, m: str, K: int, day: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Members on `day` with name, industry, country; betas (S, K) and factor weights (K, S) of the
    seed-0 model refitted that January (trained on the eight preceding years)."""
    r = M.runs(spec(m, "attention"), K, [0]).iloc[0]
    cfg = json.loads((r.run / "manifest.json").read_text())["config"]
    mc, pc = cfg["model"], cfg["policy"]
    dd = td.ROOT / DATA[m]
    day = pd.Timestamp(day)
    p = slots.load_slots(dd, f"{day - pd.DateOffset(days=10):%Y-%m-%d}", f"{day:%Y-%m-%d}",
                         max_rank=cfg["sample"].get("universe_size"))
    t = p.dates.searchsorted(day)
    model = AttentionArb(n_features=len(p.features), n_factors=K, embedding_dim=mc["embedding_dim"],
                         lambda_ridge=mc["lambda_ridge"], level_hidden=mc.get("level_hidden", 0), hidden=pc["hidden"],
                         lookback=pc["residual_lookback"], dropout=pc["dropout"], lambda_squash=pc["lambda_squash"])
    model.load_state_dict(torch.load(r.run / "models" / f"{p.dates[t].year}.pt"))
    model.eval()
    with torch.no_grad():
        wF = model.factors.factor_weights(p.X[t:t + 1], p.in_universe[t:t + 1])
        beta = model.factors.loadings(wF)[0].numpy()
    wF = wF[0].numpy()
    mask = p.in_universe[t].numpy()
    sec = p.sec_ids[p.idx[t].numpy().clip(max=p.n_pool - 1)]
    if m == "us":
        info = pd.read_parquet(td.ROOT / "data/us/private/raw/crsp_secinfo.parquet")
        info = info[(info.secinfostartdt <= p.dates[t]) & (info.secinfoenddt >= p.dates[t])].drop_duplicates("permno", keep="last")
        info = info.assign(sec_id=info.permno.astype(str), name=info.securitynm, sic=info.siccd, country="US")
    else:
        info = pd.read_parquet(td.ROOT / "data/europe17/private/universe_asof.parquet")
        info = info[info.month == p.dates[t].to_period("M")].drop_duplicates("gvkey")
        info = info.assign(sec_id=info.gvkey, name=info.conm)
    info = info.set_index("sec_id")
    df = pd.DataFrame({"sec_id": sec[mask]})
    df["name"] = (info["name"].reindex(df["sec_id"]).fillna("?").str.split(";").str[0].str.strip()
                  .str.title().to_numpy())                       # CRSP: "NAME; COM A; CONS"
    df["industry"] = [sic_group(s) for s in info["sic"].reindex(df["sec_id"]).to_numpy()]
    df["country"] = info["country"].reindex(df["sec_id"]).fillna("?").to_numpy()
    return df, beta[mask], wF[:, mask]


def f10_factor_map(out: Path, M: td.Matrix, day: str = "2021-01-04") -> None:
    from sklearn.manifold import TSNE
    panels = []
    for m in MARKETS:
        df, beta, _ = factor_state(M, m, 8, day)
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(beta)
        df["x"], df["y"] = xy[:, 0], xy[:, 1]
        panels.append((m, "industry", df))
        if m == "eu":
            panels.append((m, "country", df))
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    for ax, (m, by, df) in zip(axes, panels):
        groups = (["Finance & real estate", "Energy & mining", "Technology"] if by == "industry"
                  else [c for c in df["country"].value_counts().index[:3]])
        other = ~df[by].isin(groups)
        ax.scatter(df.loc[other, "x"], df.loc[other, "y"], s=7, color=GRID, lw=0)
        for g, col in zip(groups, HI3):
            s = df[df[by] == g]
            ax.scatter(s["x"], s["y"], s=9, color=col, lw=0.4, edgecolor="white", label=COUNTRY.get(g, g))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{MARKETS[m]}, by {by}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.5, ncol=1, handletextpad=0.2,
                  markerscale=1.3)
    fig.tight_layout(w_pad=1.0)
    data = pd.concat([d.assign(market=MARKETS[m]) for m, by, d in panels if by == "industry"]).set_index(["market", "sec_id"])
    save(fig, out, "F10_factor_map", "Figure 10. What the attention factors group together",
         f"t-SNE map of the loadings of the members on the first {8} attention factors (K = 8, seed 0), "
         f"first trading day of 2021, model trained on 2013-2020 (the paper's Figure 3). Members close together load "
         "alike. Highlighted: three industries (SIC) in each market, and the three largest countries in Europe; "
         "grey: the rest.", "runs/_matrix/thesis (us/eu_attention K=8 seed 0, models/2021.pt); CRSP secinfo, "
         "Compustat Global universe table (names, SIC, country)", data)


def t14_top_names(out: Path, M: td.Matrix, day: str = "2021-01-04") -> None:
    rows = []
    for m in MARKETS:
        df, _, wF = factor_state(M, m, 8, day)
        for k in range(6):
            w = wF[k] / wF[k].sum()
            order = np.argsort(-w)[:10]
            for rank, j in enumerate(order, 1):
                rows.append({"market": MARKETS[m], "factor": k + 1, "rank": rank, "name": df["name"].iloc[j],
                             "industry": df["industry"].iloc[j], "country": df["country"].iloc[j], "weight": w[j]})
    t = pd.DataFrame(rows)
    d = out / "tables"
    d.mkdir(parents=True, exist_ok=True)
    t.to_csv(d / "T14_factor_top_names.csv", index=False)
    lines = ["## Table A3. The ten largest weights of the first six attention factors", "",
             f"K = 8, seed 0, first trading day of 2021 (model trained on 2013-2020), the paper's Figure 4. Weight = "
             "share of the factor's (positive, softmax) portfolio weights; share of the top ten in brackets.", ""]
    for m in MARKETS:
        lines += [f"### {MARKETS[m]}", "", "| Rank | " + " | ".join(
            f"Factor {k} ({100 * t[(t.market == MARKETS[m]) & (t.factor == k)].weight.sum():.0f}%)" for k in range(1, 7)) + " |",
            "|---|" + "---|" * 6]
        for rank in range(1, 11):
            cells = []
            for k in range(1, 7):
                r = t[(t.market == MARKETS[m]) & (t.factor == k) & (t["rank"] == rank)].iloc[0]
                cells.append(f"{r['name']} ({100 * r.weight:.1f}%)")
            lines.append(f"| {rank} | " + " | ".join(cells) + " |")
        lines.append("")
    (d / "T14_factor_top_names.md").write_text("\n".join(lines), encoding="utf-8")
    INDEX.append({"name": "T14_factor_top_names", "title": "Table A3. The ten largest weights of the first six attention factors",
                  "caption": lines[2], "source": "runs/_matrix/thesis (K=8 seed 0, models/2021.pt)",
                  "files": ["tables/T14_factor_top_names.csv", "tables/T14_factor_top_names.md"]})
    print("  T14_factor_top_names (table, written with the figures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "thesis")
    ap.add_argument("--matrix", type=Path, default=ROOT / "runs" / "_matrix" / "thesis")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    M = td.Matrix(args.matrix, ROOT / "runs")
    jobs = {"f01_universe": lambda: f01_universe(args.out), "f02_cumulative": lambda: f02_cumulative(args.out, M),
            "f03_cumulative_gross": lambda: f03_cumulative_gross(args.out, M),
            "f04_sharpe_vs_k": lambda: f04_sharpe_vs_k(args.out, M), "f05_break_even": lambda: f05_break_even(args.out, M),
            "f06_annual": lambda: f06_annual(args.out, M), "f07_turnover": lambda: f07_turnover(args.out, M),
            "f08_ablation": lambda: f08_ablation(args.out, M), "f09_holdout": lambda: f09_holdout(args.out, M),
            "f10_factor_map": lambda: f10_factor_map(args.out, M), "t14_top_names": lambda: t14_top_names(args.out, M)}
    for name, f in jobs.items():
        if not args.only or name in args.only:
            f()
    idx = args.out / "figures" / "index.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() and args.only else []
    keep = {e["name"]: e for e in old}
    keep.update({e["name"]: e for e in INDEX})
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(sorted(keep.values(), key=lambda e: e["name"]), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(INDEX)} items -> {args.out / 'figures'}")


if __name__ == "__main__":
    main()
