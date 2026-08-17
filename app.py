"""Crisis Tail-Risk & Volatility Shield.

An interactive portfolio decision app for a high-net-worth investor who holds a
$1M+ liquid growth portfolio and wants to know how much of it to move out of
equities - and into what - to survive the next crash without giving up too much
of the equity ride.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (ASSET_LABELS, BENCHMARK, DBMF_INCEPTION,
                        MANAGED_FUTURES_PROXY, PROVENANCE, STRESS_WINDOWS,
                        TICKERS)
from src.data import load_prices, to_returns
from src.metrics import (drawdown_series, max_drawdown, portfolio_returns,
                         summary)
from src.optimizer import equal_weight, optimise, slice_windows

st.set_page_config(page_title="Crisis Tail-Risk & Volatility Shield",
                   page_icon="🛡️", layout="wide")

PALETTE = {"Shield": "#0F766E", "Equal weight": "#B45309",
           "100% SPY": "#334155", "Your mix": "#7C3AED"}


# ---------------------------------------------------------------- data layer
@st.cache_data(show_spinner="Downloading prices…", ttl=60 * 60 * 12)
def get_data(use_snapshot: bool):
    prices, report, source, retrieved = load_prices(use_snapshot=use_snapshot)
    return prices, report, source, retrieved


@st.cache_data(show_spinner="Searching 12,341 allocations…")
def run_optimiser(returns: pd.DataFrame, te_cap: float, step: float,
                  bounds: tuple, windows_key: tuple, objective: str,
                  rebalance: str):
    windows = {k: STRESS_WINDOWS[k] for k in windows_key}
    return optimise(returns, te_cap=te_cap, step=step, bounds=list(bounds),
                    windows=windows, objective=objective, rebalance=rebalance)


def money(value: float) -> str:
    """Dollar string for tables and metrics."""
    return f"${value:,.0f}"


def md_money(value: float) -> str:
    """Dollar string for markdown text (escaped so Streamlit not read it as LaTeX)."""
    return money(value).replace("$", "\\$")


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ Shield controls")
st.sidebar.caption("Every control below changes the recommended allocation.")

portfolio_value = st.sidebar.number_input(
    "Portfolio value ($)", min_value=250_000, max_value=50_000_000,
    value=1_000_000, step=250_000, format="%d")

te_cap = st.sidebar.slider(
    "Tracking-error budget vs SPY (annualised)", 0.02, 0.20, 0.10, 0.005,
    format="%.1f%%",
    help="How far the shielded portfolio may drift from the equity market. "
         "Tighter budget = closer to SPY = less crash protection.")

equity_floor, equity_cap = st.sidebar.slider(
    "Equity (SPY) allocation limits", 0.0, 1.0, (0.30, 0.80), 0.05,
    help="Investment-policy limits on the growth sleeve.")

max_single_diversifier = st.sidebar.slider(
    "Max in any single diversifier", 0.10, 1.0, 0.40, 0.05,
    help="Concentration limit on TLT, GLD and DBMF individually.")

objective = st.sidebar.radio(
    "Objective", ["worst", "average"],
    format_func=lambda o: {"worst": "Minimise the deepest crisis drawdown",
                           "average": "Minimise the average crisis drawdown"}[o])

rebalance = st.sidebar.selectbox(
    "Rebalancing", ["M", "Q", "Y", "BH"], index=1,
    format_func=lambda f: {"M": "Monthly", "Q": "Quarterly", "Y": "Annual",
                           "BH": "Buy & hold (never)"}[f])

selected_crises = st.sidebar.multiselect(
    "Stress tests the shield must survive", list(STRESS_WINDOWS),
    default=list(STRESS_WINDOWS))

use_snapshot = st.sidebar.toggle(
    "Use offline data snapshot", value=False,
    help="Skip the live Yahoo Finance download and use the CSV committed with "
         "the code. Useful if the network is blocked during the demo.")

if not selected_crises:
    st.sidebar.error("Select at least one stress test.")
    st.stop()

# ------------------------------------------------------------------ analysis
prices, report, source, retrieved = get_data(use_snapshot)
returns = to_returns(prices)
bounds = ((equity_floor, equity_cap),) + ((0.0, max_single_diversifier),) * 3

try:
    best, grid = run_optimiser(returns, te_cap, 0.025, bounds,
                               tuple(selected_crises), objective, rebalance)
except ValueError as exc:
    st.error(f"{exc} Widen the equity limits or the concentration limit.")
    st.stop()

shield_w = best["weights"]
eq_w = equal_weight(TICKERS)
spy_w = {t: (1.0 if t == BENCHMARK else 0.0) for t in TICKERS}

series = {
    "Shield": portfolio_returns(returns, shield_w, rebalance),
    "Equal weight": portfolio_returns(returns, eq_w, rebalance),
    "100% SPY": returns[BENCHMARK],
}
stats = {name: summary(s, returns[BENCHMARK]) for name, s in series.items()}

# --------------------------------------------------------------------- header
st.title("Crisis Tail-Risk & Volatility Shield")
st.markdown(
    f"**Investor:** a high-net-worth individual with a {md_money(portfolio_value)} liquid "
    "growth portfolio.  \n**Decision:** *what mix of SPY, TLT, GLD and managed "
    "futures (DBMF) minimises the loss in a severe market crash, while staying "
    "within an agreed tracking-error budget versus the equity market?*")

if not best["te_budget_feasible"]:
    st.warning("No allocation inside your weight limits meets that tracking-error "
               "budget. Showing the closest-tracking portfolio instead - loosen the "
               "budget or the equity limits.")

c1, c2, c3, c4 = st.columns(4)
worst_shield = best["worst_crisis_dd"]
worst_spy = min(max_drawdown(returns[BENCHMARK].loc[s:e])
                for s, e in (STRESS_WINDOWS[k] for k in selected_crises))
worst_eq = min(max_drawdown(portfolio_returns(returns.loc[s:e], eq_w, rebalance))
               for s, e in (STRESS_WINDOWS[k] for k in selected_crises))
c1.metric("Worst crisis loss - Shield", f"{worst_shield:.1%}",
          f"{(worst_shield - worst_spy):.1%} vs SPY")
c2.metric("Worst crisis loss - 100% SPY", f"{worst_spy:.1%}")
c3.metric("Worst crisis loss - equal weight", f"{worst_eq:.1%}",
          f"{(worst_shield - worst_eq):.1%} vs Shield")
c4.metric("Tracking error vs SPY", f"{best['tracking_error']:.1%}",
          f"budget {te_cap:.1%}", delta_color="off")

st.success(
    f"**Recommendation:** hold "
    + ", ".join(f"**{w:.0%} {t}**" for t, w in shield_w.items() if w > 0)
    + f". On the worst of the selected crises this portfolio lost "
      f"{abs(worst_shield):.1%} ({md_money(portfolio_value * abs(worst_shield))} on "
      f"{md_money(portfolio_value)}) against {abs(worst_spy):.1%} "
      f"({md_money(portfolio_value * abs(worst_spy))}) for an all-equity portfolio.")

tab_decision, tab_stress, tab_frontier, tab_data, tab_verify = st.tabs(
    ["Decision", "Stress lab", "Trade-off & correlations", "Data & pipeline",
     "Verification & limits"])

# ------------------------------------------------------------------- decision
with tab_decision:
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Recommended allocation")
        alloc = pd.DataFrame({
            "Asset": [ASSET_LABELS[t] for t in TICKERS],
            "Shield": [shield_w[t] for t in TICKERS],
            "Equal weight": [eq_w[t] for t in TICKERS],
            "Dollars": [shield_w[t] * portfolio_value for t in TICKERS],
        })
        fig = px.bar(alloc, x="Shield", y="Asset", orientation="h", text_auto=".0%",
                     color_discrete_sequence=[PALETTE["Shield"]])
        fig.update_layout(height=280, xaxis_tickformat=".0%", showlegend=False,
                          margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None,
                          yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            alloc.assign(Shield=alloc["Shield"].map("{:.1%}".format),
                         **{"Equal weight": alloc["Equal weight"].map("{:.1%}".format),
                            "Dollars": alloc["Dollars"].map(money)}),
            hide_index=True, use_container_width=True)
        st.caption(f"Chosen from {best['candidates_evaluated']:,} allocations on a "
                   "2.5% weight grid - a global search, not a local optimum.")

    with right:
        st.subheader("Growth of the portfolio")
        levels = pd.DataFrame({k: (1 + v).cumprod() * portfolio_value
                               for k, v in series.items()})
        fig = px.line(levels, color_discrete_map=PALETTE, log_y=True)
        fig.update_layout(height=320, yaxis_title="Value (log scale)",
                          xaxis_title=None, legend_title=None,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Underwater plot - how deep, how long")
        dd = pd.DataFrame({k: drawdown_series(v) for k, v in series.items()})
        fig = px.area(dd, color_discrete_map=PALETTE)
        fig.update_layout(height=280, yaxis_tickformat=".0%", legend_title=None,
                          xaxis_title=None, yaxis_title="Drawdown",
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(opacity=0.45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full-sample statistics")
    table = pd.DataFrame(stats).T
    pct_rows = ["CAGR", "Volatility", "Max drawdown", "VaR 95% (daily)",
                "CVaR 95% (daily)", "Tracking error vs SPY"]
    shown = table.copy()
    for col in shown.columns:
        fmt = "{:.2%}" if col in pct_rows else "{:.2f}"
        shown[col] = shown[col].map(fmt.format)
    st.dataframe(shown, use_container_width=True)
    st.caption("Equal weight (25/25/25/25) is the mandated comparison portfolio. "
               "The Shield is allowed to differ from it only because the optimiser "
               "found a better drawdown-per-unit-of-tracking-error trade-off.")

# ---------------------------------------------------------------- stress lab
with tab_stress:
    st.subheader("Stress tests: peak-to-trough loss in each crisis")
    blocks = slice_windows(returns, {k: STRESS_WINDOWS[k] for k in selected_crises})
    rows = []
    for name, block in blocks.items():
        for label, w in [("Shield", shield_w), ("Equal weight", eq_w), ("100% SPY", spy_w)]:
            rows.append({"Crisis": name, "Portfolio": label,
                         "Max drawdown": max_drawdown(portfolio_returns(block, w, rebalance))})
    crisis_df = pd.DataFrame(rows)
    fig = px.bar(crisis_df, x="Crisis", y="Max drawdown", color="Portfolio",
                 barmode="group", color_discrete_map=PALETTE, text_auto=".1%")
    fig.update_layout(height=420, yaxis_tickformat=".0%", xaxis_title=None,
                      legend_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Replay one crisis dollar-by-dollar")
    crisis = st.selectbox("Crisis", list(blocks), index=len(blocks) - 1)
    block = blocks[crisis]
    replay = pd.DataFrame({
        label: (1 + portfolio_returns(block, w, rebalance)).cumprod() * portfolio_value
        for label, w in [("Shield", shield_w), ("Equal weight", eq_w), ("100% SPY", spy_w)]})
    fig = px.line(replay, color_discrete_map=PALETTE)
    fig.update_layout(height=340, yaxis_title="Value", xaxis_title=None,
                      legend_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for col, (label, w) in zip(cols, [("Shield", shield_w), ("Equal weight", eq_w),
                                      ("100% SPY", spy_w)]):
        r = portfolio_returns(block, w, rebalance)
        loss = max_drawdown(r)
        col.metric(f"{label} - worst loss in {crisis}", f"{loss:.1%}",
                   f"{money(portfolio_value * loss)}", delta_color="off")

    st.subheader("Build your own mix and compare")
    st.caption("Sliders are normalised to 100%, so you can compare any allocation "
               "with the optimiser's answer.")
    manual = {}
    slider_cols = st.columns(4)
    for col, t in zip(slider_cols, TICKERS):
        manual[t] = col.slider(ASSET_LABELS[t], 0.0, 1.0, float(shield_w[t]), 0.05)
    if sum(manual.values()) == 0:
        st.info("Give at least one asset a positive weight.")
    else:
        mine = portfolio_returns(returns, manual, rebalance)
        mine_stats = summary(mine, returns[BENCHMARK])
        crisis_dd = {n: max_drawdown(portfolio_returns(b, manual, rebalance))
                     for n, b in blocks.items()}
        m1, m2, m3 = st.columns(3)
        m1.metric("Your worst crisis loss", f"{min(crisis_dd.values()):.1%}",
                  f"{min(crisis_dd.values()) - worst_shield:.1%} vs Shield")
        m2.metric("Your tracking error", f"{mine_stats['Tracking error vs SPY']:.1%}",
                  "over budget" if mine_stats["Tracking error vs SPY"] > te_cap
                  else "within budget", delta_color="off")
        m3.metric("Your CAGR", f"{mine_stats['CAGR']:.2%}",
                  f"{mine_stats['CAGR'] - stats['Shield']['CAGR']:.2%} vs Shield")

# ------------------------------------------------------- trade-off & correls
with tab_frontier:
    st.subheader("The trade-off the investor is actually making")
    st.caption("Every dot is one of the allocations searched. Moving left buys "
               "crash protection; moving right buys closeness to the equity market.")
    plot_df = grid.copy()
    plot_df["Feasible"] = np.where(plot_df["tracking_error"] <= te_cap,
                                   "Within TE budget", "Outside budget")
    fig = px.scatter(
        plot_df, x="tracking_error", y="worst_crisis_dd", color="Feasible",
        opacity=0.45, hover_data={t: ":.0%" for t in TICKERS},
        color_discrete_map={"Within TE budget": "#0F766E", "Outside budget": "#CBD5E1"})
    fig.add_trace(go.Scatter(
        x=[best["tracking_error"]], y=[best["worst_crisis_dd"]], mode="markers+text",
        marker=dict(size=16, color="#DC2626", symbol="star"),
        text=["Shield"], textposition="top center", name="Recommended"))
    fig.update_layout(height=460, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      xaxis_title="Tracking error vs SPY (annualised)",
                      yaxis_title="Worst crisis drawdown",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Correlations - calm markets vs crises")
        crisis_idx = pd.Index([])
        for name in selected_crises:
            s, e = STRESS_WINDOWS[name]
            crisis_idx = crisis_idx.union(returns.loc[s:e].index)
        calm = returns.drop(index=crisis_idx).corr()
        stressed = returns.loc[crisis_idx].corr()
        which = st.radio("Regime", ["Crisis days", "Calm days", "Difference"],
                         horizontal=True)
        matrix = {"Crisis days": stressed, "Calm days": calm,
                  "Difference": stressed - calm}[which]
        fig = px.imshow(matrix.round(2), text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r",
                        x=[ASSET_LABELS[t] for t in matrix.columns],
                        y=[ASSET_LABELS[t] for t in matrix.index])
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Diversifiers only earn their place if their correlation with "
                   "equities stays low *when equities are falling*.")
    with right:
        st.subheader("Rolling 12-month volatility")
        roll = pd.DataFrame({k: v.rolling(252).std() * np.sqrt(252)
                             for k, v in series.items()}).dropna()
        fig = px.line(roll, color_discrete_map=PALETTE)
        fig.update_layout(height=380, yaxis_tickformat=".0%", xaxis_title=None,
                          yaxis_title="Annualised volatility", legend_title=None,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ data & pipeline
with tab_data:
    st.subheader("Provenance")
    st.table(pd.DataFrame({
        "Field": ["Provider", "Instruments", "Price field", "Frequency",
                  "Date range", "Retrieval date", "Source used now"],
        "Value": [PROVENANCE.provider, PROVENANCE.instruments, PROVENANCE.field,
                  PROVENANCE.frequency,
                  f"{report['first_date']} to {report['last_date']}",
                  retrieved, source]}))

    st.subheader("Cleaning log")
    st.json(report)
    st.markdown(
        f"""
**Decisions taken, not hidden**

* Prices are dividend- and split-adjusted closes, so returns are total returns.
* Missing prices are forward-filled for at most **3** consecutive days (holiday
  mismatches between an ETF and a mutual fund); anything longer is dropped, so
  every return in the analysis uses four real, simultaneous observations.
* **{MANAGED_FUTURES_PROXY} splice:** DBMF only launched {DBMF_INCEPTION}, which would
  exclude 2008 - the most important stress test for this investor. Before that
  date we chain-link the *returns* of {MANAGED_FUTURES_PROXY}, a managed-futures
  mutual fund live since 2007, onto DBMF's first traded price. Read the pre-2019
  sleeve as "a managed-futures allocation", not as DBMF itself.
""")

    st.subheader("Price history (rebased to 100)")
    rebased = prices / prices.iloc[0] * 100
    fig = px.line(rebased, log_y=True)
    fig.update_layout(height=340, xaxis_title=None, yaxis_title="Index (log)",
                      legend_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download the cleaned price data (CSV)",
                       prices.to_csv().encode(), "shield_prices.csv", "text/csv")

# ------------------------------------------------------ verification & limits
with tab_verify:
    st.subheader("Independent check, recomputed live")
    st.caption("The app's engine compounds monthly-rebalanced returns. The check "
               "below rebuilds the same number from raw prices with plain pandas, "
               "outside the optimiser's code path.")
    check_w = shield_w
    engine = float((1 + portfolio_returns(returns, check_w, "BH")).prod() - 1)
    manual = float(((prices / prices.iloc[0]) * pd.Series(check_w)).sum(axis=1).iloc[-1] - 1)
    v1, v2, v3 = st.columns(3)
    v1.metric("Engine (buy & hold total return)", f"{engine:.4%}")
    v2.metric("Recomputed from raw prices", f"{manual:.4%}")
    v3.metric("Difference", f"{abs(engine - manual):.2e}",
              "match" if abs(engine - manual) < 1e-6 else "MISMATCH", delta_color="off")

    dd_engine = max_drawdown(series["Shield"])
    level = (1 + series["Shield"]).cumprod().to_numpy()[::10]
    brute = min(float(level[j] / level[i] - 1)
                for i in range(len(level)) for j in range(i, len(level)))
    st.write(f"Max drawdown, engine: **{dd_engine:.2%}** · brute-force double loop "
             f"over a 10-day sampled path: **{brute:.2%}** (small gap expected "
             "from sampling).")
    st.caption("`verify.py` in the repository runs six such checks, including two "
               "edge cases, and writes VERIFICATION_OUTPUT.md.")

    st.subheader("Edge case handled in the app")
    st.write("If your tracking-error budget cannot be met inside the weight limits, "
             "the app says so and shows the closest-tracking portfolio rather than "
             "silently returning an infeasible answer. Try setting the budget to 2% "
             "with an equity ceiling of 40%.")

    st.subheader("What would make this conclusion unreliable")
    st.markdown("""
* **Six crises is a small sample.** A grid search over 12,341 allocations can fit
  the crises it is shown; the weights are a robust *region*, not a precise point.
* **The pre-2019 managed-futures sleeve is a proxy** (RYMFX), not DBMF itself.
* **Regimes change.** In 2008 Treasuries rallied hard; in 2022 they fell with
  equities. A shield built mainly on bonds would have failed in 2022 - which is
  why the optimiser keeps managed futures and gold in the mix.
* **Frictions are ignored:** no fees, spreads, taxes or slippage; rebalancing is
  assumed to happen at close with no cost.
* **Past drawdowns are not a cap on future ones.** Treat the numbers as a stress
  ranking of allocations, not as a guarantee.
""")

st.divider()
st.caption("IE New York College · Python for Finance · Portfolio Optimizer "
           "Challenge. Educational project, not investment advice.")
