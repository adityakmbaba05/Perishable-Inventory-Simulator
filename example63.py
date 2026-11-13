"""Section 6.3 worked example replayed with the paper's own numbers.

Algorithm 1 is run with the cycle costs printed on the arcs of Fig. 2 and the
inventory compositions of Fig. 3 (which depend on the previous cycles, Fig. 4).
Nothing is recomputed: this shows that the decision rule of Eqs. (18)-(20)
turns the paper's arc costs into the paper's answer Y_4 = (1, 1, 1, 0).

T = 4, M = 2, a = 20, c = 1, h = 1, p = 10, w = 0, Poisson rates (6, 9, 3, 8).
"""
from __future__ import annotations

LAM = [6, 9, 3, 8]
H = 1
T, M = 4, 2

# (i, j, previous cycles) -> (c(i, j) from Fig. 2, items left for the next period from Fig. 3)
# Fig. 3 rows: start, after the order, end of each period (columns = ages 0 and 1; age 1 is waste when M = 2).
FIG2 = {
    (1, 1, ()): (34, 4),
    (1, 2, ()): (52, 0),
    (2, 2, ((1, 1),)): (33, 4),
    (2, 3, ((1, 1),)): (39, 0),
    (3, 3, ((1, 1), (2, 2))): (25, 3),
    (3, 3, ((1, 2),)): (29, 3),
    (3, 4, ((1, 1), (2, 2))): (38, 0),
    (3, 4, ((1, 2),)): (43, 0),
    (4, 4, ((1, 1), (2, 2), (3, 3))): (35, 5),
    (4, 4, ((1, 2), (3, 3))): (35, 5),
    (4, 4, ((1, 1), (2, 3))): (38, 5),
}


def _look_ahead(j: int, left: float, B: tuple):
    """c'(j+1, T_ij) and T_ij from Fig. 2 costs (Section 6.1)."""
    if j >= T:
        return 0, T, "end of horizon"
    if left >= LAM[j]:                                   # remaining inventory covers E{d_{j+1}}
        return H * (left - LAM[j]), j + 1, "remaining inventory"
    best = None
    for e in range(j + 1, min(T, j + M) + 1):
        key = (j + 1, e, B)
        if key not in FIG2:
            continue
        cost = FIG2[key][0]
        if best is None or cost / (e - j) < best[0]:
            best = (cost / (e - j), cost, e)
    return best[1], best[2], f"next cycle R({j + 1},{best[2]})"


def replay():
    best = {0: {"TC": 0, "B": ()}}
    rows = []
    for j in range(1, T + 1):
        avg, pick = float("inf"), None
        for i in range(max(1, j - M + 1), j + 1):
            prev = best[i - 1]
            cost, left = FIG2[(i, j, prev["B"])]
            B = prev["B"] + ((i, j),)
            cp, Tij, how = _look_ahead(j, left, B)
            ac = (prev["TC"] + cost + cp) / Tij
            row = {"j": j, "cycle": f"R({i},{j})", "previous cycles": _fmt(prev["B"]), "TC(i-1)": prev["TC"],
                   "c(i,j) [Fig. 2]": cost, "T_ij": Tij, "c'": cp, "c' from": how, "AC(T_ij)": round(ac, 2),
                   "chosen": False}
            rows.append(row)
            if ac < avg:
                avg, pick = ac, (i, cost, B, row)
        i, cost, B, row = pick
        row["chosen"] = True
        best[j] = {"TC": best[i - 1]["TC"] + cost, "B": B}
    Y = [0] * T
    for (i, _) in best[T]["B"]:
        Y[i - 1] = 1
    return {"rows": rows, "Y": Y, "B": _fmt(best[T]["B"]), "TC_det": best[T]["TC"]}


def _fmt(B):
    return "{" + ", ".join(f"R({i},{j})" for i, j in B) + "}" if B else "∅"


if __name__ == "__main__":
    r = replay()
    for x in r["rows"]:
        print(x)
    print(r["Y"], r["B"], r["TC_det"])
