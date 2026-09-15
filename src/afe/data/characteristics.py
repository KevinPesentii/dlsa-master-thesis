"""The 39 firm characteristics of Epstein, Wang, Choi & Pelger (2025), as implemented.

Definitions follow Data Collection/DataCollectionDetails.docx, which transcribes Table
IA.XVIII of Chen, Pelger & Zhu (2024) plus four daily-frequency inputs from the attention
paper. Each characteristic is one function below, registered with its theme, update
frequency and reference. To change a definition, edit the function; to add one, write a
function and decorate it; to drop one, remove it from configs/us_data.yaml.
docs/us_characteristics.md is the human-readable version of this file and must be kept
in step with it.

Timing. A characteristic dated at month end m is built from information available at
m and is used for every trading day of month m+1 (the paper's "prior month"
convention). Three update frequencies:

  annual   from Compustat annual fundamentals. Input: one row per (gvkey, datadate) with
           the raw items, `*_lag` = the same item one fiscal year earlier, `be` = book
           equity and `me_dec` = market equity at the December end of the fiscal year's
           calendar year. When a fiscal year becomes usable is decided in build_us.py
           (config fundamentals.availability), not here.
  monthly  from the CRSP monthly file and from per-month statistics of daily data.
           Input: MonthlyInputs, wide frames indexed by month (Period) x permno.
  daily    from the CRSP daily file. Input: DailyInputs, wide frames indexed by trading
           date x permno. Values dated d use data through d.

All wide frames are float32 or float64 with NaN for "no data"; every rolling window
states its minimum number of observations, below which the value is NaN. Nothing here
fills NaN: the cross-sectional median fill happens after ranking, in build_us.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Characteristic:
    name: str
    theme: str
    frequency: str  # annual | monthly | daily
    reference: str
    fn: Callable
    doc: str


REGISTRY: dict[str, Characteristic] = {}


def register(name: str, theme: str, frequency: str, reference: str):
    def deco(fn):
        REGISTRY[name] = Characteristic(name, theme, frequency, reference, fn, (fn.__doc__ or "").strip())
        return fn
    return deco


def _ratio(num, den):
    """num / den with a zero or missing denominator giving NaN, never inf."""
    den = den.where(den != 0)
    return num / den


# --------------------------------------------------------------------------- inputs


@dataclass
class DailyInputs:
    """Wide daily frames (trading date x permno) plus the factor series on the same dates."""
    ret: pd.DataFrame        # simple return, delisting-adjusted (dlyret)
    prc: pd.DataFrame        # close, absolute value (CRSP stores bid-ask midpoints negative)
    high: pd.DataFrame       # daily high, falling back to the close where missing
    vol: pd.DataFrame        # shares traded
    cumfacpr: pd.DataFrame   # cumulative price adjustment factor, latest basis (prc/cumfacpr is split-adjusted)
    ff: pd.DataFrame         # columns mktrf, smb, hml, rf, indexed like the frames
    cfg: dict = field(default_factory=dict)   # the `windows` block of the config


@dataclass
class MonthlyInputs:
    """Wide monthly frames (Period[M] x permno) and the daily inputs they can draw on."""
    ret: pd.DataFrame        # monthly return (mthret)
    prc: pd.DataFrame        # month-end price, absolute value
    vol: pd.DataFrame        # monthly volume, shares
    shrout: pd.DataFrame     # shares outstanding, thousands
    cap_co: pd.DataFrame     # company market cap, USD millions, summed over share classes
    stats: pd.DataFrame      # per-(permno, month) sums of daily data; see build_us.monthly_daily_stats
    daily: DailyInputs
    cfg: dict = field(default_factory=dict)


def sample_month_end(daily_wide: pd.DataFrame) -> pd.DataFrame:
    """Last available daily value in each calendar month, indexed by Period[M]."""
    return daily_wide.groupby(daily_wide.index.to_period("M")).last()


def _stat(M: MonthlyInputs, name: str) -> pd.DataFrame:
    """One wide frame (month x permno) of a per-month daily statistic, on the same grid
    as M.ret so frames can be combined elementwise."""
    return M.stats[name].unstack("permno").reindex(index=M.ret.index, columns=M.ret.columns)


FF3_X = ["one", "mktrf", "smb", "hml"]  # regressors of the Resid_Var regression, in order


def cross_key(a: str, b: str) -> str:
    """Name of the per-month sum of x_a * x_b; stored once per unordered pair."""
    i, j = sorted([FF3_X.index(a), FF3_X.index(b)])
    return f"x_{FF3_X[i]}_{FF3_X[j]}"


# =========================================================================== PAST RETURNS


def _cum_return(ret_m: pd.DataFrame, lag_start: int, lag_end: int, min_obs: int) -> pd.DataFrame:
    """Compound monthly returns over months m-lag_start .. m-lag_end (both inclusive).

    Missing months inside the window are skipped (treated as a zero return); the value is
    NaN when fewer than min_obs months are present. Log-sum for numerical hygiene.
    """
    window = lag_start - lag_end + 1
    logs = np.log1p(ret_m).shift(lag_end)
    return np.expm1(logs.rolling(window, min_periods=min_obs).sum())


@register("r2_1", "Past Returns", "monthly", "Jegadeesh and Titman (1993)")
def r2_1(M: MonthlyInputs) -> pd.DataFrame:
    """Short-term momentum: the return of month m (the month before prediction month m+1)."""
    return M.ret


@register("ST_Rev", "Past Returns", "monthly", "Jegadeesh and Titman (1993)")
def st_rev(M: MonthlyInputs) -> pd.DataFrame:
    """Short-term reversal: prior-month return. Numerically identical to r2_1; the paper
    lists both, so both are kept."""
    return M.ret


@register("r12_2", "Past Returns", "monthly", "Fama and French (1996)")
def r12_2(M: MonthlyInputs) -> pd.DataFrame:
    """Momentum: cumulative return over months m-11 .. m-1 (t-12 .. t-2 for prediction
    month t). Following Fama-French the stock must have a price at m-12 (t-13) and a
    return at m-1 (t-2); other months may be missing, down to momentum_min_obs.r12_2."""
    mo = M.cfg["momentum_min_obs"]["r12_2"]
    out = _cum_return(M.ret, 11, 1, mo)
    ok = M.prc.shift(12).notna() & M.ret.shift(1).notna()
    return out.where(ok)


@register("r12_7", "Past Returns", "monthly", "Novy-Marx (2012)")
def r12_7(M: MonthlyInputs) -> pd.DataFrame:
    """Intermediate momentum: cumulative return over months m-11 .. m-6 (t-12 .. t-7)."""
    return _cum_return(M.ret, 11, 6, M.cfg["momentum_min_obs"]["r12_7"])


@register("r36_13", "Past Returns", "monthly", "De Bondt and Thaler (1985)")
def r36_13(M: MonthlyInputs) -> pd.DataFrame:
    """Long-term momentum / reversal: cumulative return over months m-35 .. m-12
    (t-36 .. t-13)."""
    return _cum_return(M.ret, 35, 12, M.cfg["momentum_min_obs"]["r36_13"])


@register("Ret_D1", "Past Returns", "daily", "Guijarro-Ordonez, Pelger and Zanotti (2022)")
def ret_d1(D: DailyInputs) -> pd.DataFrame:
    """Daily return on day d. Not in Table IA.XVIII; the raw input series of the
    statistical arbitrage papers."""
    return D.ret


@register("Ret_W1", "Past Returns", "daily", "attention paper")
def ret_w1(D: DailyInputs) -> pd.DataFrame:
    """Return over the past trading week: compounded over the last weekly_days (5) days
    ending on d, all of which must be present."""
    w = D.cfg["weekly_days"]
    return np.expm1(np.log1p(D.ret).rolling(w, min_periods=w).sum())


@register("STD_W1", "Past Returns", "daily", "attention paper")
def std_w1(D: DailyInputs) -> pd.DataFrame:
    """Realised volatility over the past trading week: standard deviation of the last
    weekly_days (5) daily returns ending on d."""
    w = D.cfg["weekly_days"]
    return D.ret.rolling(w, min_periods=w).std()


# =========================================================================== VALUE


def book_equity(f: pd.DataFrame) -> pd.Series:
    """Fama-French book equity: SH + TXDITC - PS.
    SH = SEQ; if missing CEQ + PS; if missing AT - LT.
    PS = PSTKRV; if missing PSTKL; if missing PSTK. Missing TXDITC counts as zero."""
    ps = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0.0)
    sh = f["seq"].fillna(f["ceq"] + ps).fillna(f["at"] - f["lt"])
    return sh + f["txditc"].fillna(0.0) - ps


@register("A2ME", "Value", "annual", "Bhandari (1988)")
def a2me(f):
    """Total assets over market equity at the December end of the fiscal year's calendar year."""
    return _ratio(f["at"], f["me_dec"])


@register("BEME", "Value", "annual", "Fama and French (1992)")
def beme(f):
    """Book equity (see book_equity) over December market equity."""
    return _ratio(f["be"], f["me_dec"])


@register("C", "Value", "annual", "Palazzo (2012)")
def cash(f):
    """Cash and short-term investments over total assets."""
    return _ratio(f["che"], f["at"])


@register("CF", "Value", "annual", "Hou, Karolyi and Kho (2011)")
def cf(f):
    """Free cash flow over book equity: (NI + DP - change in working capital - CAPX) / BE.
    The change in working capital is Compustat WCAPCH; it is missing for most firms after
    the 1980s funds-statement format, so the fallback is the change in balance-sheet
    working capital WCAP from the prior fiscal year."""
    wc_change = f["wcapch"].fillna(f["wcap"] - f["wcap_lag"])
    return _ratio(f["ni"] + f["dp"] - wc_change - f["capx"], f["be"])


@register("CF2P", "Value", "annual", "Desai, Rajgopal and Venkatachalam (2004)")
def cf2p(f):
    """Cash flow over December market equity, cash flow = IB + DP + TXDB (deferred taxes,
    zero when missing)."""
    return _ratio(f["ib"] + f["dp"] + f["txdb"], f["me_dec"])


@register("Q", "Value", "annual", "Kaldor (1966)")
def tobins_q(f):
    """Tobin's Q: (AT + ME_dec - CEQ - TXDB) / AT. The IA text calls CEQ "cash and
    short-term investments"; CEQ is common equity and the formula uses it as such (the
    standard Q of Kaplan-Zingales / Freyberger et al.)."""
    return _ratio(f["at"] + f["me_dec"] - f["ceq"] - f["txdb"], f["at"])


@register("Lev", "Value", "annual", "Lewellen (2015)")
def leverage(f):
    """Book leverage: (DLTT + DLC) / (DLTT + DLC + SEQ)."""
    debt = f["dltt"] + f["dlc"]
    return _ratio(debt, debt + f["seq"])


@register("E2P", "Value", "annual", "Basu (1983)")
def e2p(f):
    """Earnings before extraordinary items (IB) over December market equity."""
    return _ratio(f["ib"], f["me_dec"])


# =========================================================================== INVESTMENT


def net_operating_assets(f: pd.DataFrame) -> pd.Series:
    """Operating assets minus operating liabilities, unscaled (used lagged by RNA).
    OA = AT - CHE - IVAO.  OL = AT - DLC - DLTT - MIB - PSTK - CEQ."""
    oa = f["at"] - f["che"] - f["ivao"]
    ol = f["at"] - f["dlc"] - f["dltt"] - f["mib"] - f["pstk"] - f["ceq"]
    return oa - ol


@register("Investment", "Investment", "annual", "Cooper, Gulen and Schill (2008)")
def investment(f):
    """Asset growth: (AT - AT_lag) / AT_lag."""
    return _ratio(f["at"] - f["at_lag"], f["at_lag"])


@register("NOA", "Investment", "annual", "Hirshleifer, Hou, Teoh and Zhang (2004)")
def noa(f):
    """Net operating assets (see net_operating_assets) over lagged total assets."""
    return _ratio(f["noa_level"], f["at_lag"])


@register("DPI2A", "Investment", "annual", "Lyandres, Sun and Zhang (2008)")
def dpi2a(f):
    """Change in gross PP&E plus change in inventory, over lagged total assets."""
    return _ratio((f["ppegt"] - f["ppegt_lag"]) + (f["invt"] - f["invt_lag"]), f["at_lag"])


# =========================================================================== TRADING FRICTIONS


@register("AT", "Trading Frictions", "annual", "Gandhi and Lustig (2015)")
def total_assets(f):
    """Level of total assets, USD millions."""
    return f["at"]


@register("LME", "Trading Frictions", "monthly", "Fama and French (1992)")
def lme(M: MonthlyInputs) -> pd.DataFrame:
    """Size: company market cap at the end of month m, USD millions, summed over the
    company's eligible share classes (the same number that ranks the universe)."""
    return M.cap_co


@register("LTurnover", "Trading Frictions", "monthly", "Datar, Naik and Radcliffe (1998)")
def lturnover(M: MonthlyInputs) -> pd.DataFrame:
    """Turnover: shares traded in month m over shares outstanding (shrout is in thousands)."""
    return _ratio(M.vol, M.shrout * 1000.0)


@register("Rel2High", "Trading Frictions", "monthly", "George and Hwang (2004)")
def rel2high(M: MonthlyInputs) -> pd.DataFrame:
    """Closeness to the 52-week high: month-end price over the highest daily high of the
    past rel2high_days (252) trading days, both split-adjusted with CRSP's cumulative
    factor. Only the RATIO of adjusted prices is used, so a split after m cannot change
    the value at m (prefix invariance)."""
    D, c = M.daily, M.daily.cfg
    adj_prc = D.prc / D.cumfacpr
    adj_high = D.high / D.cumfacpr
    ratio = adj_prc / adj_high.rolling(c["rel2high_days"], min_periods=c["rel2high_min_obs"]).max()
    return sample_month_end(ratio)


@register("Variance", "Trading Frictions", "monthly", "Ang, Hodrick, Xing and Zhang (2006)")
def variance(M: MonthlyInputs) -> pd.DataFrame:
    """Sample variance of daily returns over the variance_months (2) calendar months
    ending at m, pooled from per-month sums (n, sum r, sum r^2)."""
    k, mo = M.cfg["variance_months"], M.cfg["variance_min_obs"]
    n = _stat(M, "n_ret").rolling(k, min_periods=1).sum()
    s1 = _stat(M, "sum_ret").rolling(k, min_periods=1).sum()
    s2 = _stat(M, "sum_ret2").rolling(k, min_periods=1).sum()
    var = (s2 - s1 ** 2 / n) / (n - 1)
    return var.where(n >= mo)


@register("Resid_Var", "Trading Frictions", "monthly", "Ang, Hodrick, Xing and Zhang (2006)")
def resid_var(M: MonthlyInputs) -> pd.DataFrame:
    """Residual variance of daily excess returns regressed on the Fama-French three
    factors (with intercept) over the variance_months (2) calendar months ending at m.
    Solved from per-month cross-product sums, one 4x4 system per stock-month."""
    k, mo = M.cfg["variance_months"], M.cfg["variance_min_obs"]
    roll = lambda s: _stat(M, s).rolling(k, min_periods=1).sum()  # noqa: E731
    n = roll("n_ret")
    xtx = np.empty(n.shape + (4, 4))
    xty = np.empty(n.shape + (4,))
    for i, a in enumerate(FF3_X):
        xty[..., i] = roll(f"xy_{a}").to_numpy()
        for j, b in enumerate(FF3_X):
            xtx[..., i, j] = roll(cross_key(a, b)).to_numpy()
    yty = roll("sum_exret2").to_numpy()
    xtx[n.to_numpy() < mo] = np.nan  # too few days: do not even try to solve
    beta = _batched_solve(xtx, xty)
    rss = yty - np.einsum("...i,...i->...", beta, xty)
    dof = n.to_numpy() - len(FF3_X)
    out = pd.DataFrame(rss / np.where(dof > 0, dof, np.nan), index=n.index, columns=n.columns)
    return out.where(n >= mo)


@register("Spread", "Trading Frictions", "monthly", "Chung and Zhang (2014)")
def spread(M: MonthlyInputs) -> pd.DataFrame:
    """Average relative bid-ask spread over month m: mean of (ask - bid) / midpoint from
    CRSP closing quotes, days with ask <= bid or a missing quote excluded, at least
    spread_min_obs days."""
    n = _stat(M, "n_spread")
    return _ratio(_stat(M, "sum_spread"), n).where(n >= M.cfg["spread_min_obs"])


@register("SUV", "Trading Frictions", "monthly", "Garfinkel (2009)")
def suv(M: MonthlyInputs) -> pd.DataFrame:
    """Standardised unexplained volume. Daily volume is regressed on a constant and the
    absolute values of positive and negative returns over the suv_estimation_months (6)
    months before m (at least suv_min_obs days). Unexplained volume in month m is the
    average residual under those coefficients, standardised by the estimation-window
    residual standard deviation. Positive and negative parts never overlap, so the 3x3
    normal equations have a zero off-diagonal block."""
    k, mo = M.cfg["suv_estimation_months"], M.cfg["suv_min_obs"]
    est = lambda s: _stat(M, s).rolling(k, min_periods=1).sum().shift(1)  # noqa: E731  months m-k..m-1
    n, srp, srn = est("n_vol"), est("sum_rp"), est("sum_rn")
    srp2, srn2, sy, srpy, srny, sy2 = (est(s) for s in ["sum_rp2", "sum_rn2", "sum_vol", "sum_rp_vol", "sum_rn_vol", "sum_vol2"])
    zero = n * 0.0
    xtx = np.stack([np.stack([n, srp, srn], -1), np.stack([srp, srp2, zero], -1), np.stack([srn, zero, srn2], -1)], -2)
    xty = np.stack([sy, srpy, srny], -1)
    xtx[n.to_numpy() < mo] = np.nan
    b = _batched_solve(xtx, xty)
    rss = sy2.to_numpy() - np.einsum("...i,...i->...", b, xty)
    dof = n.to_numpy() - 3
    sigma = np.sqrt(rss / np.where(dof > 0, dof, np.nan))
    # evaluation month m: mean residual = (sum y - b0 n - b1 sum rp - b2 sum rn) / n
    n_m, y_m, rp_m, rn_m = (_stat(M, s) for s in ["n_vol", "sum_vol", "sum_rp", "sum_rn"])
    unexplained = (y_m.to_numpy() - b[..., 0] * n_m.to_numpy() - b[..., 1] * rp_m.to_numpy()
                   - b[..., 2] * rn_m.to_numpy()) / n_m.to_numpy()
    out = pd.DataFrame(unexplained / sigma, index=n.index, columns=n.columns)
    return out.where((n >= mo) & (n_m >= M.cfg["spread_min_obs"]))


@register("Vol", "Trading Frictions", "daily", "attention paper")
def weekly_volume(D: DailyInputs) -> pd.DataFrame:
    """Shares traded over the last weekly_days (5) trading days ending on d."""
    w = D.cfg["weekly_days"]
    return D.vol.rolling(w, min_periods=w).sum()


@register("Beta", "Trading Frictions", "monthly", "Frazzini and Pedersen (2014)")
def beta(M: MonthlyInputs) -> pd.DataFrame:
    """CAPM beta as corr(stock, market) x sigma_stock / sigma_market. Volatilities are
    standard deviations of daily log excess returns over beta_vol_days (252) with at
    least beta_vol_min_obs (120); the correlation uses overlapping three-day log excess
    returns over beta_corr_days (1260) with at least beta_corr_min_obs (750). No
    shrinkage toward one (the IA does not mention it)."""
    D, c = M.daily, M.daily.cfg
    rf = D.ff["rf"]
    le = np.log1p(D.ret).sub(np.log1p(rf), axis=0)                  # stock log excess return
    lm = np.log1p(D.ff["mktrf"] + rf) - np.log1p(rf)                 # market log excess return
    sig_i = le.rolling(c["beta_vol_days"], min_periods=c["beta_vol_min_obs"]).std()
    sig_m = lm.rolling(c["beta_vol_days"], min_periods=c["beta_vol_min_obs"]).std()
    le3 = le.rolling(3, min_periods=3).sum()
    lm3 = lm.rolling(3, min_periods=3).sum()
    rho = le3.rolling(c["beta_corr_days"], min_periods=c["beta_corr_min_obs"]).corr(lm3)
    return sample_month_end((rho * sig_i).div(sig_m, axis=0))


# =========================================================================== PROFITABILITY


@register("PROF", "Profitability", "annual", "Ball, Gerakos, Linnainmaa and Nikolaev (2015)")
def prof(f):
    """Gross profitability: GP over book equity."""
    return _ratio(f["gp"], f["be"])


@register("CTO", "Profitability", "annual", "Haugen and Baker (1996)")
def cto(f):
    """Capital turnover: sales over lagged total assets."""
    return _ratio(f["sale"], f["at_lag"])


@register("FC2Y", "Profitability", "annual", "D'Acunto, Liu, Pflueger and Weber (2018)")
def fc2y(f):
    """Fixed costs over sales: (XSGA + XRD + XAD) / SALE, missing XRD and XAD as zero."""
    return _ratio(f["xsga"] + f["xrd"] + f["xad"], f["sale"])


@register("OP", "Profitability", "annual", "Fama and French (2015)")
def op(f):
    """Operating profitability: (REVT - COGS - XINT - XSGA) / BE. The spec writes the
    interest item as TIE; Compustat's item is XINT (interest and related expense)."""
    return _ratio(f["revt"] - f["cogs"] - f["xint"] - f["xsga"], f["be"])


@register("PM", "Profitability", "annual", "Soliman (2008)")
def pm(f):
    """Profit margin: operating income after depreciation over sales."""
    return _ratio(f["oiadp"], f["sale"])


@register("RNA", "Profitability", "annual", "Soliman (2008)")
def rna(f):
    """Return on net operating assets: OIADP over lagged net operating assets (unscaled
    level of the prior fiscal year, see net_operating_assets)."""
    return _ratio(f["oiadp"], f["noa_level_lag"])


@register("D2A", "Profitability", "annual", "Gorodnichenko and Weber (2016)")
def d2a(f):
    """Capital intensity: depreciation and amortisation over total assets."""
    return _ratio(f["dp"], f["at"])


# =========================================================================== INTANGIBLES


def noncash_working_capital(f: pd.DataFrame) -> pd.Series:
    """ACT - CHE - LCT - DLC - TXP, literally as in Table IA.XVIII. Sloan (1996) subtracts
    current liabilities NET of debt and taxes payable, i.e. ACT - CHE - (LCT - DLC - TXP);
    switch the signs of dlc and txp here to use that version."""
    return f["act"] - f["che"] - f["lct"] - f["dlc"] - f["txp"]


@register("OA", "Intangibles", "annual", "Sloan (1996)")
def oa(f):
    """Operating accruals: (change in non-cash working capital - DP) / lagged AT."""
    return _ratio((f["nwc"] - f["nwc_lag"]) - f["dp"], f["at_lag"])


@register("OL", "Intangibles", "annual", "Novy-Marx (2011)")
def ol(f):
    """Operating leverage: (COGS + XSGA) / AT."""
    return _ratio(f["cogs"] + f["xsga"], f["at"])


@register("PCM", "Intangibles", "annual", "Bustamante and Donangelo (2017)")
def pcm(f):
    """Price-to-cost margin: (SALE - COGS) / SALE."""
    return _ratio(f["sale"] - f["cogs"], f["sale"])


# =========================================================================== helpers


def _batched_solve(xtx: np.ndarray, xty: np.ndarray, max_cond: float = 1e12) -> np.ndarray:
    """Solve one small linear system per leading index. NaN where an entry is missing or
    the system is (near) singular, e.g. a stock with no negative return in the window."""
    out = np.full(xty.shape, np.nan)
    ok = np.isfinite(xtx).all(axis=(-2, -1)) & np.isfinite(xty).all(axis=-1)
    if ok.any():
        cond = np.linalg.cond(xtx[ok])
        good = ok.copy()
        good[ok] = np.isfinite(cond) & (cond < max_cond)
        if good.any():
            out[good] = np.linalg.solve(xtx[good], xty[good][..., None])[..., 0]
    return out


def rank_quantile(s: pd.Series) -> pd.Series:
    """Cross-sectional rank mapped to (-0.5, 0.5]; NaN stays NaN. Same formula as
    synthetic.py. CONFIRM 1 in docs/schemas.md: verify the range against the authors'
    code before results are compared to the paper."""
    return s.rank(method="average", pct=True) - 0.5


def by_frequency(names: list[str]) -> dict[str, list[Characteristic]]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"characteristics not in the registry: {unknown}")
    out: dict[str, list[Characteristic]] = {"annual": [], "monthly": [], "daily": []}
    for n in names:
        out[REGISTRY[n].frequency].append(REGISTRY[n])
    return out
