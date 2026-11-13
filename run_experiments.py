"""Re-run the paper's numerical study (Section 7) with this implementation.

  python run_experiments.py            # all 216 instances of Tables 2-3 and the 12 of Tables 4-5
  python run_experiments.py --reps 1000 --quick

Writes results/table7_instances.csv and results/table6_instances.csv, then prints the
summary rows next to the paper's Table 7 and Table 6.
Gap = 100 x (TC_H - TC_SDP) / TC_SDP   (Section 7.2).  SDP is solved for M = 2 and M = 3 only.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import paper_data as P
import sdp
from heuristic import Heuristic, Instance

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["set", "number", "pattern", "M", "a", "c", "h", "p", "w", "dist", "cv", "Y_T", "avg_cycle_length",
          "TC_det", "TC_H", "ci95_H", "TC_SDP", "gap_pct", "step1_s", "step2_s", "sdp_s",
          "paper_heur_s", "paper_heur_sim_s", "paper_sdp_sim_s", "paper_sdp_s"]


def evaluate(d: dict, reps: int, seed: int, tag: str) -> dict:
    inst = Instance.from_dict(d)
    h = Heuristic(inst)
    s1 = h.step1()
    s2 = h.step2(s1["Y"], reps, seed)
    opt = sdp.solve_instance(inst)
    pr = d.get("paper_runtimes", {})
    return {
        "set": tag, "number": d["number"], "pattern": d["pattern"], "M": inst.M, "a": inst.a, "c": inst.c,
        "h": inst.h, "p": inst.p, "w": inst.w, "dist": inst.dist, "cv": inst.cv or "",
        "Y_T": "".join(map(str, s1["Y"])), "avg_cycle_length": round(s2["avg_cycle_length"], 4),
        "TC_det": round(s1["TC_det"], 4), "TC_H": round(s2["TC"], 4), "ci95_H": round(s2["ci95"], 4),
        "TC_SDP": round(opt[0], 4) if opt else "",
        "gap_pct": round(100 * (s2["TC"] - opt[0]) / opt[0], 4) if opt else "",
        "step1_s": round(s1["runtime_s"], 5), "step2_s": round(s2["runtime_s"], 3), "sdp_s": round(opt[2], 3) if opt else "",
        "paper_heur_s": pr.get("heur", ""), "paper_heur_sim_s": pr.get("heur_sim", ""),
        "paper_sdp_sim_s": pr.get("sdp_sim", ""), "paper_sdp_s": pr.get("sdp", ""),
    }


def summary(rows: list, key, label: str) -> str:
    sel = [r for r in rows if key(r)]
    g = [r["gap_pct"] for r in sel if r["gap_pct"] != ""]
    s = f"{label:12s} n={len(sel):3d}  TC_H {st.mean(r['TC_H'] for r in sel):8.2f}  cycle {st.mean(r['avg_cycle_length'] for r in sel):.2f}"
    if g:
        s += f"  TC_SDP {st.mean(r['TC_SDP'] for r in sel if r['TC_SDP'] != ''):8.2f}  gap {st.mean(g):.2f}% (max {max(g):.2f}, n={len(g)})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="only instances with M <= 3")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    rows7 = []
    for n in range(1, 217):
        d = P.instance(n)
        if args.quick and d["M"] > 3:
            continue
        rows7.append(evaluate(d, args.reps, args.seed, "Tables 2-3"))
        print(f"instance {n:3d} done", flush=True)
    rows6 = [evaluate(d, args.reps, args.seed, "Tables 4-5") for d in P.instances_table4()]

    for name, rows in (("table7_instances.csv", rows7), ("table6_instances.csv", rows6)):
        with open(os.path.join(HERE, "results", name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)

    print("\nTable 7 set (paper Total avg: gap 1.87%, TC_H 852.92, TC_SDP 837.58, cycle 2.26)")
    print(summary(rows7, lambda r: True, "Total avg"))
    for pat in P.TABLE3:
        print(summary(rows7, lambda r, pat=pat: r["pattern"] == pat, pat))
    for M in (2, 3, 4):
        print(summary(rows7, lambda r, M=M: r["M"] == M, f"M = {M}"))
    for a in (25, 50, 75):
        print(summary(rows7, lambda r, a=a: r["a"] == a, f"a = {a}"))
    print("\nTable 6 set (paper Total avg: gap 3.21%, TC_H 691.34, TC_SDP 669.77, cycle 1.95)")
    print(summary(rows6, lambda r: True, "Total avg"))


if __name__ == "__main__":
    main()
