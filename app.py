"""Streamlit prototype: perishable inventory heuristic of Gulecyuz, O'Sullivan and Tarim (2025).

Pages
  1 Input data   choose a paper instance (Section 6.3, Tables 2-3, Tables 4-5) and inspect/edit it
  2 Run model    Step 1 (Algorithm 1), Step 2 (simulation, Section 6.2), SDP benchmark (Eq. 10)
  3 Results      charts of Q_t, I_{t,b}, costs and the total cost distribution
  4 Paper check  Section 6.3 worked example, Table 7 and Table 6 against this implementation
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

import example63
import paper_data as P
import sdp
from heuristic import Heuristic, Instance, validate

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Perishable inventory heuristic", layout="wide")

# Okabe-Ito colours (colour-blind safe)
BLUE, ORANGE, GREEN, SKY, PINK, GREY = "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#8C8C8C"
COST_ORDER = ["setup", "ordering", "holding", "penalty", "outdating"]
COST_COLORS = [BLUE, SKY, GREEN, ORANGE, PINK]
PAGES = ["1 · Input data", "2 · Run model", "3 · Results", "4 · Paper check"]
SOURCES = ["Section 6.3 example", "Tables 2–3 instance (Poisson)", "Tables 4–5 instance (normal)"]

s = st.session_state
s.setdefault("page", PAGES[0])
s.setdefault("source", SOURCES[0])
s.setdefault("base", dict(P.EXAMPLE, name="Section 6.3 example", number=0, pattern="Example"))
s.setdefault("inst", dict(s["base"]))
s.setdefault("res", None)


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def vec(v):
    """Composition as text, e.g. (13, 0)."""
    return "(" + ", ".join(f"{x:g}" if float(x).is_integer() else f"{x:.2f}" for x in v) + ")"


def is_example(inst) -> bool:
    return inst.get("pattern") == "Example" and not modified()


def set_base(d: dict):
    s["base"] = dict(d)
    s["inst"] = dict(d)
    s["res"] = None
    for k in list(s.keys()):
        if k.startswith("in_"):
            del s[k]


def modified() -> bool:
    keys = ("M", "a", "c", "h", "p", "w", "dist", "cv", "lam")
    return any(s["inst"].get(k) != s["base"].get(k) for k in keys)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Perishable inventory heuristic")
    st.caption(f"{P.PAPER_AUTHORS} · *{P.PAPER_REF}*")
    st.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Current instance**")
    st.write(s["inst"].get("name", ""))
    if modified():
        st.warning("Edited: values differ from the paper.")
    st.caption(f"[Paper]({P.PAPER_DOI}) · notation as in Table 1")


# ================================================================ PAGE 1
def page_input():
    st.title("1 · Input data")
    st.write("Pick an instance from the paper. Every default value below is taken from the paper; "
             "you can edit them, and the app marks the instance as edited.")

    src = st.radio("Instance source", SOURCES, key="source", horizontal=True)
    if src == SOURCES[0]:
        st.info("Section 6.3: T = 4, M = 2, a = 20, c = 1, h = 1, p = 10, w = 0, Poisson λ_t = (6, 9, 3, 8). "
                "The text prints λ_t = (6, 9, 4, 8) once; the captions of Figs. 2–4 and E{d₃} = 3 use (6, 9, 3, 8).")
        if st.button("Load Section 6.3 example", type="primary"):
            set_base(dict(P.EXAMPLE, name="Section 6.3 example", number=0, pattern="Example"))
            st.rerun()
    elif src == SOURCES[1]:
        c1, c2, c3, c4, c5 = st.columns(5)
        pat = c1.selectbox("Demand pattern (Table 3)", list(P.TABLE3), key="f_pat")
        M = c2.selectbox("Shelf life M", P.TABLE2["M"], key="f_M")
        a = c3.selectbox("Setup cost a", P.TABLE2["a"], key="f_a")
        c = c4.selectbox("Unit ordering cost c", P.TABLE2["c"], key="f_c")
        p = c5.selectbox("Unit penalty cost p", P.TABLE2["p"], key="f_p")
        code = {v: k for k, v in P.PATTERN_CODE.items()}[pat]
        n = next(r[0] for r in P.APPENDIX if r[1] == code and r[2] == M and r[3] == a and r[4] == c and r[5] == p)
        st.caption(f"This is **instance {n}** in Appendix Table {7 + code} (h = 1, w = 0, T = 12).")
        if st.button(f"Load instance {n}", type="primary"):
            set_base(P.instance(n))
            st.rerun()
    else:
        c1, c2, c3 = st.columns(3)
        M = c1.selectbox("Shelf life M", P.TABLE4["M"], key="g_M")
        a = c2.selectbox("Setup cost a", P.TABLE4["a"], key="g_a")
        cv = c3.selectbox("Coefficient of variation CV", P.TABLE4["CV"], key="g_cv")
        d = next(x for x in P.instances_table4() if x["M"] == M and x["a"] == a and x["cv"] == cv)
        st.caption("Normal demand, mean = Table 5 rate, sd = CV × mean; c = 2, h = 0.5, p = 5, w = 0, T = 12.")
        if st.button("Load this Table 4 instance", type="primary"):
            set_base(d)
            st.rerun()

    st.divider()
    inst = s["inst"]
    st.subheader(inst.get("name", ""))
    st.markdown("**Parameters** (Table 1 notation)")
    cols = st.columns(8)
    T = len(inst["lam"])
    cols[0].number_input("T", value=T, disabled=True, help="Length of the planning horizon = number of λ_t values.")
    M = cols[1].number_input("M", 2, 6, int(inst["M"]), key="in_M", help="Fixed item lifetime.")
    a = cols[2].number_input("a", 0.0, 1000.0, float(inst["a"]), 0.5, key="in_a", help="Fixed setup cost per order.")
    c = cols[3].number_input("c", 0.0, 100.0, float(inst["c"]), 0.5, key="in_c", help="Unit ordering cost.")
    h = cols[4].number_input("h", 0.01, 100.0, float(inst["h"]), 0.5, key="in_h", help="Unit holding cost.")
    p = cols[5].number_input("p", 0.01, 1000.0, float(inst["p"]), 0.5, key="in_p", help="Unit penalty cost.")
    w = cols[6].number_input("w", 0.0, 100.0, float(inst["w"]), 0.5, key="in_w", help="Unit outdating cost (paper: w = 0).")
    dist = cols[7].selectbox("d_t", ["poisson", "normal"], index=0 if inst["dist"] == "poisson" else 1, key="in_dist")
    cv = inst.get("cv")
    if dist == "normal":
        cv = st.number_input("CV (sd = CV × mean)", 0.01, 1.0, float(cv or 0.25), 0.01, key="in_cv")
    else:
        cv = None

    st.markdown("**Expected demand E{d_t}** (λ_t for Poisson, mean for normal). Edit cells to change.")
    lam_df = pd.DataFrame([inst["lam"]], columns=[f"t={t + 1}" for t in range(T)], index=["E{d_t}"])
    edited = st.data_editor(lam_df, key=f"in_lam_{inst.get('name', '')}", width="stretch")
    lam = [float(v) for v in edited.iloc[0].tolist()]
    lam = [int(v) if float(v).is_integer() else v for v in lam]

    new = dict(inst, M=int(M), a=_num(a), c=_num(c), h=_num(h), p=_num(p), w=_num(w), dist=dist, cv=cv, lam=lam, T=T)
    if new != inst:
        s["inst"], s["res"] = new, None
    errs = validate(s["inst"])
    for e in errs:
        st.error(e)

    c1, c2 = st.columns([3, 2])
    with c1:
        df = pd.DataFrame({"t": range(1, T + 1), "E{d_t}": lam})
        chart = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("t:O", title="period t"), y=alt.Y("E{d_t}:Q", title="E{d_t}"),
            tooltip=["t", "E{d_t}"]).properties(height=240, title="Non-stationary expected demand")
        st.altair_chart(chart, width="stretch")
    with c2:
        if modified() and st.button("Reset to the paper's values"):
            set_base(s["base"])
            st.rerun()
        st.markdown(f"Critical ratio p/(h+p) = **{fmt(s['inst']['p'] / (s['inst']['h'] + s['inst']['p']), 3)}** "
                    "(right-hand side of Eq. 17 divided by N).")
        st.markdown(f"Cycles are at most M = {s['inst']['M']} periods long (Definition 4).")
        st.markdown("Opening composition I₀ = 0 (Eq. 6).")
    with st.expander("Table 1 · Summary of notation (from the paper)"):
        st.dataframe(pd.DataFrame(P.NOTATION, columns=["Group", "Notation", "Description"]), hide_index=True, width="stretch")
    if not errs:
        st.success("Inputs are valid. Go to **2 · Run model**.")


def _num(v):
    return int(v) if float(v).is_integer() else float(v)


# ================================================================ PAGE 2
def page_run():
    st.title("2 · Run model")
    inst = s["inst"]
    errs = validate(inst)
    if errs:
        st.error("Fix the inputs on page 1 first: " + " ".join(errs))
        return
    st.write(f"**{inst['name']}**{' (edited)' if modified() else ''} — T = {len(inst['lam'])}, M = {inst['M']}, "
             f"a = {inst['a']}, c = {inst['c']}, h = {inst['h']}, p = {inst['p']}, w = {inst['w']}, d_t ~ {inst['dist']}"
             + (f", CV = {inst['cv']}" if inst["dist"] == "normal" else ""))
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1])
    reps = c1.number_input("Replications", 100, 20000, 5000, 500, help="The paper uses 5000 (Section 6.3, Table 7).")
    seed = c2.number_input("Random seed", 0, 10_000, 1)
    x = Instance.from_dict(inst)
    ok, why = sdp.can_solve(dict(T=x.T, M=x.M, mu=x.lam, opening=None))
    run_sdp = c3.checkbox("Solve SDP benchmark (Eq. 10)", value=ok, disabled=not ok, help=why)
    go = c4.button("Run heuristic", type="primary", width="stretch")
    if not ok:
        st.caption(f"SDP not offered: {why}")
    S_override = None
    if is_example(inst):
        src = st.radio("Order-up-to levels S_ij in Step 1", ["As printed in the paper (Section 6.3 text and Fig. 3)",
                                                              "Computed from Eq. (17)"], horizontal=True, key="s_src",
                       help="The paper's printed S values do not all follow Eq. (17) with one rounding rule. "
                            "With the printed values Algorithm 1 returns the paper's Y_4 = (1, 1, 1, 0).")
        if src.startswith("As printed"):
            S_override = P.EXAMPLE_FIG3_S
            st.caption("S₁₁ = 10, S₁₂ = 19, S₂₂ = 13 (text); S₂₃ = 16, S₃₃ = 6, S₃₄ = 13, S₄₄ = 13 (Fig. 3). "
                       "Step 2 always uses Eq. (17) with the observed composition, as in Section 6.2.")
    if go:
        with st.spinner("Step 1: Algorithm 1 …"):
            h = Heuristic(x, S_override)
            s1 = h.step1()
        with st.spinner(f"Step 2: simulation with {reps} replications …"):
            s2 = h.step2(s1["Y"], int(reps), int(seed))
        opt = None
        if run_sdp:
            with st.spinner("SDP benchmark …"):
                opt = sdp.solve_instance(x)
        s["res"] = {"inst": dict(inst), "s1": s1, "s2": s2, "opt": opt, "S_src": "paper" if S_override else "eq17"}
    res = s["res"]
    if not res:
        st.info("Press **Run heuristic**.")
        return
    s1, s2, opt = res["s1"], res["s2"], res["opt"]

    st.subheader("Step 1 · Replenishment periods (Section 6.1, Algorithm 1)")
    k = st.columns(4)
    k[0].metric("Y_T", "(" + ", ".join(map(str, s1["Y"])) + ")")
    k[1].metric("Orders planned", sum(s1["Y"]))
    k[2].metric("Deterministic-equivalent TC(T)", fmt(s1["TC_det"]))
    k[3].metric("Step 1 time", f"{s1['runtime_s'] * 1000:.1f} ms")
    st.markdown("**B_T** = {" + ", ".join(f"R({c.i},{c.j})" for c in s1["B"]) + "}")
    s_lab = "S_ij (printed in paper)" if res.get("S_src") == "paper" else "S_ij (Eq. 17)"
    cyc = pd.DataFrame([{"R(i,j)": f"R({c.i},{c.j})", "I_{i-1}": vec(c.I_start), s_lab: c.S,
                         "Q_i": round(c.Q, 2), "EWA Σ I_{k,M-1}": vec(c.ewa), "c(i,j)": round(c.cost, 2),
                         "ending I_{t,b}, b=0..M-1, t=i..j": "  ".join(vec(e) for e in c.I_end)} for c in s1["B"]])
    st.dataframe(cyc, hide_index=True, width="stretch")
    with st.expander("Algorithm 1 trace: every R(i,j) compared at the end of each period j (Eq. 18–20)", expanded=len(s1["Y"]) <= 4):
        rows = [{k2: v for k2, v in r.items() if not k2.startswith("_")} for rr in s1["trace"] for r in rr]
        tr = pd.DataFrame(rows)
        for col in ("TC(i-1)", "Q_i", "c(i,j)", "c'", "AC"):
            tr[col] = tr[col].round(2)
        tr["I_j"] = tr["I_j"].map(vec)
        st.dataframe(tr.style.apply(lambda r: ["background-color: rgba(0,114,178,.15)" if r["chosen"] else "" for _ in r], axis=1)
                     .format(precision=2), hide_index=True, width="stretch")
        st.caption("AC = (TC(i−1) + c(i,j) + c′(j+1, T_ij)) / T_ij.  The row with the lowest AC is kept (Eq. 19), "
                   "TC(j) = TC(i*−1) + c(i*, j) (Eq. 20).")

    st.subheader("Step 2 · Order quantities by simulation (Section 6.2)")
    k = st.columns(5)
    k[0].metric("Expected total cost TC_H", fmt(s2["TC"]))
    k[1].metric("95% CI half-width", "± " + fmt(s2["ci95"]))
    k[2].metric("Avg. cycle length", fmt(s2["avg_cycle_length"]))
    k[3].metric("Replications", f"{s2['reps']:,}")
    k[4].metric("Step 2 time", f"{s2['runtime_s']:.2f} s")
    if opt:
        st.subheader("Benchmark · SDP (Section 3.1, Eq. 10)")
        gap = 100 * (s2["TC"] - opt[0]) / opt[0]
        k = st.columns(4)
        k[0].metric("Optimal TC_SDP = V₁(I₀)", fmt(opt[0]))
        k[1].metric("Optimality gap 100·(TC_H − TC_SDP)/TC_SDP", f"{gap:.2f}%")
        k[2].metric("States evaluated", f"{opt[1]:,}")
        k[3].metric("SDP time", f"{opt[2]:.2f} s")
    pr = res["inst"].get("paper_runtimes")
    if pr and not modified():
        st.caption(f"Paper runtimes for instance {res['inst']['number']} (Appendix): heuristic {pr['heur']} s, "
                   f"heuristic sim. {pr['heur_sim']} s, SDP {pr['sdp']} s, SDP sim. {pr['sdp_sim']} s "
                   "(MATLAB / C on an i7-1165G7).")
    st.success("Charts are on **3 · Results**.")


# ================================================================ PAGE 3
def page_results():
    st.title("3 · Results")
    res = s["res"]
    if not res:
        st.info("Run the model on page 2 first.")
        return
    s2 = res["s2"]
    required_step2 = {"reps", "TC", "ci95", "avg_cycle_length", "per_period", "parts", "costs"}
    if not required_step2.issubset(s2):
        st.warning("These saved results are from an older app version. Go to **2 · Run model** and run the heuristic again.")
        return
    s1, opt, inst = res["s1"], res["opt"], res["inst"]
    M = inst["M"]
    seed_label = f", seed {s2['seed']}" if "seed" in s2 else ""
    st.write(f"**{inst['name']}** · Y_T = ({', '.join(map(str, s1['Y']))}) · {s2['reps']:,} replications{seed_label}")
    k = st.columns(5)
    k[0].metric("TC_H (mean)", fmt(s2["TC"]), help="Mean total cost over the replications, Eq. (1).")
    k[1].metric("95% CI half-width", "± " + fmt(s2["ci95"]), help=f"[{fmt(s2['TC'] - s2['ci95'])}, {fmt(s2['TC'] + s2['ci95'])}]")
    if opt:
        k[2].metric("TC_SDP", fmt(opt[0]))
        k[3].metric("Gap", f"{100 * (s2['TC'] - opt[0]) / opt[0]:.2f}%")
    else:
        k[2].metric("TC_SDP", "—", help="SDP not run (M > 3 or turned off).")
        k[3].metric("Gap", "—")
    k[4].metric("Avg. cycle length", fmt(s2["avg_cycle_length"]))

    per = pd.DataFrame(s2["per_period"])
    c1, c2 = st.columns(2)
    with c1:
        long_q = per.melt(id_vars=["t", "P(Q_t>0)"], value_vars=["E{d_t}", "mean Q_t"], var_name="series", value_name="units")
        chart = alt.Chart(long_q).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("t:O", title="period t", axis=alt.Axis(labelAngle=0)), xOffset="series:N",
            y=alt.Y("units:Q", title="units"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["E{d_t}", "mean Q_t"], range=[SKY, BLUE]), title=None,
                            legend=alt.Legend(orient="bottom")),
            tooltip=["t", "series", alt.Tooltip("units:Q", format=".2f"), alt.Tooltip("P(Q_t>0):Q", format=".2f")])
        st.altair_chart(chart.properties(height=330, title="Expected demand and mean order quantity", padding={"top": 14}),
                        width="stretch")
        st.caption("Orders are placed only in periods with Y_t = 1; Q_t changes with the observed I_{t−1}, so it is an average.")
    with c2:
        cols = [f"I_t,{b}" for b in range(M - 1)]
        long = per.melt(id_vars="t", value_vars=cols + [f"I_t,{M - 1}"], var_name="age", value_name="units")
        long["series"] = long["age"].map(lambda a: f"waste I_t,{M - 1}" if a == f"I_t,{M - 1}" else a.replace("I_t,", "age b = "))
        order = [f"age b = {b}" for b in range(M - 1)] + [f"waste I_t,{M - 1}"]
        palette = [BLUE, GREEN, SKY, PINK, GREY][:M - 1] + [ORANGE]
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("t:O", title="period t", axis=alt.Axis(labelAngle=0)), y=alt.Y("units:Q", title="mean units at end of t"),
            color=alt.Color("series:N", sort=order, scale=alt.Scale(domain=order, range=palette), title=None,
                            legend=alt.Legend(orient="bottom")),
            order=alt.Order("series:N"), tooltip=["t", "series", alt.Tooltip("units:Q", format=".2f")])
        st.altair_chart(chart.properties(height=330, title="Mean ending inventory I_{t,b} by age", padding={"top": 14}),
                        width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        parts = pd.DataFrame({"component": COST_ORDER, "cost": [s2["parts"][k2] for k2 in COST_ORDER]})
        chart = alt.Chart(parts).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("component:N", sort=COST_ORDER, title=None), x=alt.X("cost:Q", title="mean cost per run"),
            color=alt.Color("component:N", scale=alt.Scale(domain=COST_ORDER, range=COST_COLORS), legend=None),
            tooltip=["component", alt.Tooltip("cost:Q", format=",.2f")])
        text = chart.mark_text(align="left", dx=4).encode(text=alt.Text("cost:Q", format=",.1f"), color=alt.value("#555"))
        st.altair_chart((chart + text).properties(height=240, title="Cost components of TC_H (Eq. 1)"), width="stretch")
    with c2:
        hist = pd.DataFrame({"TC": s2["costs"]})
        chart = alt.Chart(hist).mark_bar(color=BLUE).encode(
            x=alt.X("TC:Q", bin=alt.Bin(maxbins=40), title="total cost of one replication"),
            y=alt.Y("count():Q", title="replications"), tooltip=[alt.Tooltip("count():Q", title="replications")])
        rules = [alt.Chart(pd.DataFrame({"v": [s2["TC"]], "l": ["TC_H"]})).mark_rule(color=ORANGE, size=2).encode(x="v:Q")]
        if opt:
            rules.append(alt.Chart(pd.DataFrame({"v": [opt[0]]})).mark_rule(color=GREEN, size=2, strokeDash=[4, 3]).encode(x="v:Q"))
        st.altair_chart(alt.layer(chart, *rules).properties(height=240, title="Distribution of total cost"), width="stretch")
        st.caption("Orange: mean TC_H." + (" Green dashed: optimal TC_SDP." if opt else ""))

    st.markdown("**Per-period averages**")
    show = per.copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].round(2)
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button("Download per-period results (CSV)", show.to_csv(index=False).encode(), "per_period_results.csv", "text/csv")


# ================================================================ PAGE 4
def page_paper():
    st.title("4 · Paper check")
    t1, t2, t3 = st.tabs(["Section 6.3 example", "Table 7 · 216 instances", "Table 6 · normal demand"])
    with t1:
        example_tab()
    with t2:
        table_tab("table7_instances.csv", P.TABLE7, 7)
    with t3:
        table_tab("table6_instances.csv", P.TABLE6, 6)


@st.cache_data(show_spinner=False)
def example_runs():
    x = Instance.from_dict(P.EXAMPLE)
    ours = Heuristic(x).step1()
    paper_s = Heuristic(x, P.EXAMPLE_FIG3_S).step1()
    sim_paper_Y = Heuristic(x).step2(P.EXAMPLE_PUBLISHED["Y_4"], 5000, 1)
    sim_ours = Heuristic(x).step2(ours["Y"], 5000, 1)
    opt = sdp.solve_instance(x)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}
    return ({"Y": ours["Y"], "B": [(c.i, c.j) for c in ours["B"]], "rows": [strip(r) for rr in ours["trace"] for r in rr]},
            {"Y": paper_s["Y"], "TC_det": paper_s["TC_det"], "rows": [strip(r) for rr in paper_s["trace"] for r in rr]},
            sim_paper_Y["TC"], sim_paper_Y["ci95"], sim_ours["TC"], sim_ours["ci95"], opt[0])


def example_tab():
    st.markdown("T = 4, M = 2, a = 20, c = 1, h = 1, p = 10, w = 0, Poisson λ_t = (6, 9, 3, 8).")
    ours, paper_s, tc_pY, ci_pY, tc_o, ci_o, opt = example_runs()
    rep = example63.replay()

    st.markdown("#### A · Algorithm 1 with the paper's own arc costs (Figs. 2–4)")
    st.write("The cycle costs on the arcs of Fig. 2 and the compositions of Fig. 3 are fed into Eqs. (18)–(20). "
             "The decision rule returns the paper's answer.")
    df = pd.DataFrame(rep["rows"])
    st.dataframe(df.style.apply(lambda r: ["background-color: rgba(0,114,178,.15)" if r["chosen"] else "" for _ in r], axis=1)
                 .format(precision=2), hide_index=True, width="stretch")
    c = st.columns(3)
    c[0].metric("Y_4", "(" + ", ".join(map(str, rep["Y"])) + ")", help="Paper: (1, 1, 1, 0)")
    c[1].markdown(f"**B_4**  \n{rep['B']}")
    c[2].metric("TC(4), Step 2 with this Y_4, 5000 reps", f"{tc_pY:.1f} ± {ci_pY:.1f}", help="Paper: TC(4) = 116.2")
    st.caption("Fig. 2 prints c(3,4) = 43 after R(1,2); the text uses c′(3, T₁₂) = 45, giving AC = 24.25 instead of 23.75. "
               "Either value leads to the same choice at j = 2.")

    st.markdown("#### B · Published numbers vs. this implementation")
    st.write("This implementation computes every cycle cost itself (Φ of Eqs. 3–5, costs of Eq. 1, EWA, Eqs. 18–20). "
             "Column 3 feeds it the S values printed in the paper (text and Fig. 3); column 4 computes S_ij from Eq. (17) as the "
             "smallest whole number with Σ F_{i,t}(S − Σ I_{k,M−1}) ≥ N·p/(h+p).")
    r_ours = {r["cycle"] + f"@{r['j']}": r for r in ours["rows"]}
    r_ps = {r["cycle"] + f"@{r['j']}": r for r in paper_s["rows"]}
    E = P.EXAMPLE_PUBLISHED
    comp = [
        ("S₁₁", E["S_11"], r_ours["R(1,1)@1"]["S_ij"], r_ps["R(1,1)@1"]["S_ij"]),
        ("TC(1) = c(1,1)", E["TC(1)"], r_ours["R(1,1)@1"]["c(i,j)"], r_ps["R(1,1)@1"]["c(i,j)"]),
        ("S₁₂", E["S_12"], r_ours["R(1,2)@2"]["S_ij"], r_ps["R(1,2)@2"]["S_ij"]),
        ("TC(2), option R(1,2)", E["TC(2) option 1"], r_ours["R(1,2)@2"]["c(i,j)"], r_ps["R(1,2)@2"]["c(i,j)"]),
        ("c′(3, T₁₂)", E["c'(3,T_12)"], r_ours["R(1,2)@2"]["c'"], r_ps["R(1,2)@2"]["c'"]),
        ("AC(T₁₂)", E["AC(T_12)"], r_ours["R(1,2)@2"]["AC"], r_ps["R(1,2)@2"]["AC"]),
        ("S₂₂", E["S_22"], r_ours["R(2,2)@2"]["S_ij"], r_ps["R(2,2)@2"]["S_ij"]),
        ("TC(2), option R(1,1),R(2,2)", E["TC(2) option 2"],
         r_ours["R(2,2)@2"]["TC(i-1)"] + r_ours["R(2,2)@2"]["c(i,j)"], r_ps["R(2,2)@2"]["TC(i-1)"] + r_ps["R(2,2)@2"]["c(i,j)"]),
        ("c′(3, T₂₂)", E["c'(3,T_22)"], r_ours["R(2,2)@2"]["c'"], r_ps["R(2,2)@2"]["c'"]),
        ("AC(T₂₂)", E["AC(T_22)"], r_ours["R(2,2)@2"]["AC"], r_ps["R(2,2)@2"]["AC"]),
        ("Y₄", str(tuple(E["Y_4"])), str(tuple(ours["Y"])), str(tuple(paper_s["Y"]))),
        ("Deterministic TC(4) on B₄", "105 (Fig. 2: 34 + 33 + 38)", "—", paper_s["TC_det"]),
        ("TC(4), Step 2, 5000 reps", E["TC(4)"], f"{tc_o:.1f} ± {ci_o:.1f}", f"{tc_pY:.1f} ± {ci_pY:.1f}"),
    ]
    comp = [(q, pv, ps, oe) for (q, pv, oe, ps) in comp]
    out = pd.DataFrame(comp, columns=["Quantity", "Paper (Section 6.3)", "Implementation with printed S", "Implementation with Eq. (17)"])
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))
    st.dataframe(out, hide_index=True, width="stretch")
    st.markdown(
        f"- With the printed S values the implementation reproduces TC(1) = 34, TC(2) = 52 and 67, c′ = 1, AC = 22.67, "
        f"the chosen path B₄ = {{R(1,1), R(2,2), R(3,4)}} and its Fig. 2 costs 34, 33, 38. "
        f"c′(3, T₁₂) is 43 as on Fig. 2 (the text writes 45, so AC = 23.75 instead of 24.25; the choice is the same).\n"
        f"- Eq. (17) gives S₁₁ = 9 and S₁₂ = 18 (paper: 10 and 19). The paper's S values do not follow one rounding rule "
        f"(S₁₁, S₁₂, S₂₃ = 16, S₄₄ = 13 match P(d < S) ≥ ratio; S₂₂ = 13 and S₃₄ = 13 do not).\n"
        f"- At j = 3 our c(2,3) = 38 (paper 39) and at j = 4 our c(3,4) = 42 (paper 38), so this implementation ends with "
        f"Y₄ = {tuple(ours['Y'])}; its simulated TC(4) = {tc_o:.1f}, below the paper's 116.2.\n"
        f"- Optimal SDP cost for this example: {opt:.2f} (not reported in the paper).")


def table_tab(fname: str, paper_rows: list, num: int):
    path = os.path.join(HERE, "results", fname)
    if num == 7:
        cols = ["Parameter", "Avg. gap %", "CI low", "CI high", "Heuristic TC", "SDP TC", "Cycle length",
                "SDP runtime (s)", "SDP sim. (s)", "Heuristic runtime (s)", "Heuristic sim. (s)"]
        st.markdown("216 instances: 6 demand patterns (Table 3) × M ∈ {2,3,4} × a ∈ {25,50,75} × c ∈ {1,3} × p ∈ {10,20}, "
                    "h = 1, w = 0, T = 12, Poisson demand, 5000 replications.")
    else:
        cols = ["Parameter", "Avg. gap %", "CI low", "CI high", "Heuristic TC", "SDP TC", "Cycle length"]
        st.markdown("12 instances ([47] modified): Table 5 rates, M ∈ {2,3} × a ∈ {12.5, 37.5, 50} × CV ∈ {0.25, 0.333}, "
                    "c = 2, h = 0.5, p = 5, w = 0, normal demand.")
    paper = pd.DataFrame(paper_rows, columns=cols)
    st.markdown(f"**Table {num} as published**")
    fm = {c: "{:.2f}" for c in cols[1:7]}
    st.dataframe(paper.style.format(fm), hide_index=True, width="stretch")
    if not os.path.exists(path):
        st.warning("Run `python run_experiments.py` to create the reproduction file.")
        return
    df = pd.read_csv(path)
    groups = [("Total avg", df)]
    if num == 7:
        groups += [(pt, df[df.pattern == pt]) for pt in ["Stationary", "Increasing", "Decreasing", "Cyclic", "Seasonal", "Random"]]
        groups += [(f"M = {m}", df[df.M == m]) for m in (2, 3, 4)]
        groups += [(f"a = {a}", df[df.a == a]) for a in (25, 50, 75)]
        groups += [(f"c = {c}", df[df.c == c]) for c in (1, 3)]
        groups += [(f"p = {p}", df[df.p == p]) for p in (10, 20)]
    else:
        groups += [(f"M = {m}", df[df.M == m]) for m in (2, 3)]
        groups += [(f"a = {a}", df[df.a == a]) for a in (12.5, 37.5, 50)]
        groups += [(f"CV = {v}", df[df.cv == v]) for v in (0.25, 0.333)]
    ours = []
    for name, g in groups:
        gg = g.dropna(subset=["gap_pct"])
        pr = paper[paper.Parameter == name].iloc[0]
        ours.append({"Parameter": name, "n (gap)": f"{len(gg)}/{len(g)}",
                     "Our gap %": round(gg.gap_pct.mean(), 2) if len(gg) else None, "Paper gap %": pr["Avg. gap %"],
                     "Our heuristic TC": round(g.TC_H.mean(), 2), "Paper heuristic TC": pr["Heuristic TC"],
                     "Our SDP TC": round(gg.TC_SDP.mean(), 2) if len(gg) else None, "Paper SDP TC": pr["SDP TC"],
                     "Our cycle length": round(g.avg_cycle_length.mean(), 2), "Paper cycle length": pr["Cycle length"],
                     "Our Step 1 (ms)": round(1000 * g.step1_s.mean(), 1), "Our Step 2 (s)": round(g.step2_s.mean(), 2)})
    ours = pd.DataFrame(ours)
    st.markdown(f"**This implementation on the same instances** (`results/{fname}`)")
    st.dataframe(ours, hide_index=True, width="stretch")
    if num == 7:
        st.caption("Our SDP covers M = 2 and M = 3 (144 instances); for M = 4 the paper needed 4,954 s per instance in C. "
                   "Rows that mix M = 4 therefore average the gap over M ≤ 3 only, and the TC columns are not like-for-like with the paper for those rows.")
    plot = ours.dropna(subset=["Our gap %"]).melt(id_vars="Parameter", value_vars=["Our gap %", "Paper gap %"],
                                                  var_name="source", value_name="gap")
    chart = alt.Chart(plot).mark_bar().encode(
        y=alt.Y("Parameter:N", sort=list(ours.Parameter), title=None), x=alt.X("gap:Q", title="average optimality gap (%)"),
        yOffset="source:N", color=alt.Color("source:N", scale=alt.Scale(domain=["Paper gap %", "Our gap %"], range=[GREY, BLUE]),
                                            title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["Parameter", "source", alt.Tooltip("gap:Q", format=".2f")])
    st.altair_chart(chart.properties(height=26 * len(ours) + 40, title=f"Optimality gap: paper Table {num} vs. this implementation"),
                    width="stretch")
    with st.expander("Instance-level results"):
        st.dataframe(df, hide_index=True, width="stretch")


{"1 · Input data": page_input, "2 · Run model": page_run, "3 · Results": page_results,
 "4 · Paper check": page_paper}[s["page"]]()
