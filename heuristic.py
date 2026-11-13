"""The two-step heuristic of Gulecyuz, O'Sullivan and Tarim (2025), Omega 133, 103267.

Notation follows Table 1 of the paper.

  T, M          horizon length, fixed item lifetime
  a, c, h, p, w setup, unit ordering, unit holding, unit penalty, unit outdating cost
  d_t           demand in period t (independent, non-stationary)
  I_{t,b}       items of age b at the end of period t, b = 0..M-1 (I_{t,M-1} is waste)
  I_{t-1}       inventory composition at the beginning of t: (I_{t-1,0}, ..., I_{t-1,M-2})
                A negative I_{t-1,0} is backlogged demand (Eq. 7 only forces b >= 1 to be >= 0).
  Q_t           order quantity in period t
  R(i, j)       replenishment cycle: order at the start of i, covers periods i..j (j <= i+M-1)
  c(i, j)       cost of R(i, j) (Eq. 11), deterministic-equivalent in Step 1
  S_ij          order-up-to level of R(i, j) (Eq. 17)
  T_{i,t}       last period covered by leftover stock or by the next cycle (Section 6.1)
  c'(t+1, T_{i,t})  cost of that coverage
  AC(T_{i,t})   (TC(i-1) + c(i,t) + c'(t+1,T_{i,t})) / T_{i,t}      (Eq. 18)
  TC(t)         TC(i*-1) + c(i*, t),  TC(0) = 0                      (Eq. 20)
  Y_t           1 if an order is planned in period t

Step 1 (Section 6.1, Algorithm 1) fixes the replenishment periods Y_T.
Step 2 (Section 6.2) computes Q_t from the observed composition I_{i-1} in a simulation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from demand import Demand

EPS = 1e-9


# ---------------------------------------------------------------- Eqs. (2)-(5)
def q_cost(Q: float, a: float, c: float) -> float:
    """Eq. (2): q(Q_t) = a + c Q_t if Q_t > 0, otherwise 0."""
    return a + c * Q if Q > EPS else 0.0


def phi(I_prev, Q: float, d: float) -> list:
    """Definition 3, Eqs. (3)-(5): Φ(I_{t-1}, Q_t, d_t) = I_t.

    I_prev = (I_{t-1,0}, ..., I_{t-1,M-2}).  Returns (I_{t,0}, ..., I_{t,M-1}); the last entry is waste.
    FIFO: the oldest items meet demand first, then the new order Q_t.
    """
    m1 = len(I_prev)                      # M - 1
    M = m1 + 1
    out = [0.0] * M
    # Eq. (4): I_{t,0} = Q_t - (d_t - sum_{b=0}^{M-2} I_{t-1,b})^+   (I_{t-1,0} < 0 adds the backlog to demand)
    out[0] = Q - max(d - sum(I_prev), 0.0)
    # Eq. (5): I_{t,b} = (I_{t-1,b-1} - (d_t - sum_{j=b}^{M-2} I_{t-1,j})^+)^+,  b = 1..M-1
    for b in range(1, M):
        older = sum(I_prev[b:m1])
        out[b] = max(I_prev[b - 1] - max(d - older, 0.0), 0.0)
    return out


def period_cost(I_t, h: float, p: float, w: float) -> dict:
    """The end-of-period terms of Eq. (1): h(I_{t,0})^+ + h sum_{b=1}^{M-2} I_{t,b} - p(I_{t,0})^- + w I_{t,M-1}."""
    M = len(I_t)
    return {
        "holding": h * max(I_t[0], 0.0) + h * sum(I_t[1:M - 1]),
        "penalty": -p * min(I_t[0], 0.0),
        "outdating": w * I_t[M - 1],
    }


def next_state(I_t) -> list:
    """Composition at the beginning of the next period: drop the waste I_{t,M-1}."""
    return list(I_t[:-1])


# ---------------------------------------------------------------- data classes
@dataclass
class Instance:
    T: int
    M: int
    lam: list            # E{d_t}: Poisson rate, or normal mean
    a: float
    c: float
    h: float
    p: float
    w: float = 0.0
    dist: str = "poisson"
    cv: float | None = None
    I0: list | None = None   # opening composition I_0; Eq. (6) sets it to zero
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        keys = ("T", "M", "lam", "a", "c", "h", "p", "w", "dist", "cv", "I0", "name")
        return cls(**{k: d[k] for k in keys if k in d})

    def opening(self) -> list:
        return list(self.I0) if self.I0 else [0.0] * (self.M - 1)


@dataclass
class CycleEval:
    i: int
    j: int
    S: int
    Q: float
    cost: float
    parts: dict
    I_start: list                       # I_{i-1}
    I_after_order: list                 # composition right after Q_i arrives (Fig. 3, second row)
    I_end: list = field(default_factory=list)       # I_{t} for t = i..j, each of length M
    ewa: list = field(default_factory=list)         # cumulative EWA waste used in Eq. (17)


# ---------------------------------------------------------------- the heuristic
class Heuristic:
    def __init__(self, inst: Instance, S_override: dict | None = None):
        self.x = inst
        self.T, self.M = inst.T, inst.M
        self.lam = [float(v) for v in inst.lam]
        self.dem = Demand(inst.dist, self.lam, inst.cv, max_len=inst.M)
        self.ratio = inst.p / (inst.h + inst.p)        # p / (h + p)
        self.S_override = S_override or {}

    # ---- EWA (Broekmeulen and van Donselaar): expected outdating of stock already on hand
    def ewa(self, i: int, j: int, I_start) -> list:
        """Cumulative sum_{k=i}^{t} I_{k,M-1} for t = i..j, from previous orders only,
        assuming expected demand E{d_t} occurs in each period (Section 5)."""
        I, cum, out = list(I_start), 0.0, []
        for t in range(i, j + 1):
            I_t = phi(I, 0.0, self.lam[t - 1])
            cum += I_t[-1]
            out.append(cum)
            I = next_state(I_t)
        return out

    # ---- Eq. (17): sum_{t=i}^{j} F_{i,t}(S - sum_{k=i}^{t} I_{k,M-1}) = N p / (h + p)
    def order_up_to(self, i: int, j: int, I_start, cum_w: list | None = None, use_override: bool = True) -> int:
        """Smallest integer S_ij that satisfies Eq. (17) as '>='. N = j - i + 1.

        S_override (Step 1 only) replaces Eq. (17) by S values printed in the paper, to replay Section 6.3.
        """
        if use_override and (i, j) in self.S_override:
            return int(self.S_override[(i, j)])
        if cum_w is None:
            cum_w = self.ewa(i, j, I_start)
        N = j - i + 1
        target = N * self.ratio - 1e-12

        def lhs(S):
            return sum(self.dem.cdf(i, t, S - cum_w[t - i]) for t in range(i, j + 1))

        lo = 0
        hi = int(math.ceil(self.dem.upper(i, j) + cum_w[-1])) + 2
        while lhs(hi) < target:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if lhs(mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ---- Eq. (11) under the deterministic-equivalent setting (Step 1)
    def cycle_cost(self, i: int, j: int, I_start) -> CycleEval:
        x = self.x
        cum_w = self.ewa(i, j, I_start)
        S = self.order_up_to(i, j, I_start, cum_w)
        position = sum(I_start)                         # on hand minus backlog
        Q = max(0.0, S - position)
        parts = {"setup": x.a if Q > EPS else 0.0, "ordering": x.c * Q, "holding": 0.0, "penalty": 0.0, "outdating": 0.0}
        I, ends = list(I_start), []
        for t in range(i, j + 1):
            I_t = phi(I, Q if t == i else 0.0, self.lam[t - 1])
            for k, v in period_cost(I_t, x.h, x.p, x.w).items():
                parts[k] += v
            ends.append(I_t)
            I = next_state(I_t)
        after_order = [Q] + list(I_start)               # ages 0..M-1 right after the order arrives
        return CycleEval(i, j, S, Q, sum(parts.values()), parts, list(I_start), after_order, ends, cum_w)

    # ---- c'(j+1, T_{i,j}) and T_{i,j}  (Section 6.1 and Algorithm 1)
    def look_ahead(self, j: int, I_j) -> tuple:
        """Returns (c', T_{i,j}, how).

        If the stock left after R(i, j) meets the expected demand of the next periods, c' is the cost of that
        remaining inventory while it suffices (Section 6.3 charges h on every unit left after the covered
        demand: c'(3, T_22) = 1).  Otherwise the next cycle is chosen with the modified Silver-Meal rule
        T* = argmin_{T' > j} c(j+1, T') / (T' - j) and c' = c(j+1, T*).
        """
        x = self.x
        if j >= self.T:
            return 0.0, self.T, "end of horizon"
        I, cost, last = list(I_j), 0.0, j
        for t in range(j + 1, self.T + 1):
            if sum(I) + EPS < self.lam[t - 1]:
                break
            I_t = phi(I, 0.0, self.lam[t - 1])
            cost += x.h * max(sum(I_t), 0.0) + x.w * I_t[-1]
            I = next_state(I_t)
            last = t
        if last > j:
            return cost, last, "remaining inventory"
        best = None
        for e in range(j + 1, min(self.T, j + self.M) + 1):
            cyc = self.cycle_cost(j + 1, e, I_j)
            avg = cyc.cost / (e - j)
            if best is None or avg < best[0] - 1e-12:
                best = (avg, cyc.cost, e)
        return best[1], best[2], f"next cycle R({j + 1},{best[2]})"

    # ---- Step 1: Algorithm 1
    def step1(self) -> dict:
        t0 = time.perf_counter()
        best = [{"TC": 0.0, "I": self.x.opening(), "B": []}]       # best[t]: TC(t), I_t, set B_t
        trace = []
        for j in range(1, self.T + 1):
            avgcost, pick, rows = math.inf, None, []
            for i in range(max(1, j - self.M + 1), j + 1):
                prev = best[i - 1]
                cyc = self.cycle_cost(i, j, prev["I"])
                I_j = next_state(cyc.I_end[-1])
                c_prime, T_ij, how = self.look_ahead(j, I_j)
                temp = (prev["TC"] + cyc.cost + c_prime) / T_ij               # Eq. (18)
                row = {"j": j, "i": i, "cycle": f"R({i},{j})", "TC(i-1)": prev["TC"], "S_ij": cyc.S, "Q_i": cyc.Q,
                       "c(i,j)": cyc.cost, "T_ij": T_ij, "c'": c_prime, "c' from": how, "AC": temp,
                       "I_j": [round(v, 2) for v in I_j], "chosen": False, "_cyc": cyc}
                rows.append(row)
                if temp < avgcost - 1e-12:                                    # Eq. (19)
                    avgcost, pick = temp, (cyc, I_j, prev, row)
            cyc, I_j, prev, row = pick
            row["chosen"] = True
            best.append({"TC": prev["TC"] + cyc.cost, "I": I_j, "B": prev["B"] + [cyc]})   # Eq. (20)
            trace.append(rows)
        B = best[self.T]["B"]
        Y = [0] * self.T
        for cyc in B:
            Y[cyc.i - 1] = 1
        return {"Y": Y, "B": B, "TC_det": best[self.T]["TC"], "trace": trace,
                "TC_path": [b["TC"] for b in best], "runtime_s": time.perf_counter() - t0}

    # ---- Step 2: Section 6.2 simulation
    def step2(self, Y: list, reps: int = 5000, seed: int = 1) -> dict:
        x, T, M = self.x, self.T, self.M
        t0 = time.perf_counter()
        cycles = cycles_from_Y(Y, T)
        D = self.dem.sample(reps, seed)                    # reps x T integer demand
        cache = {}
        tot = {"setup": 0.0, "ordering": 0.0, "holding": 0.0, "penalty": 0.0, "outdating": 0.0}
        costs = np.zeros(reps)
        Qsum = np.zeros(T); Qcnt = np.zeros(T)
        Isum = np.zeros((T, M)); back = np.zeros(T)
        for r in range(reps):
            d = D[r]
            I, cost = x.opening(), 0.0
            for (i, j) in cycles:
                key = (i, j, tuple(round(v, 6) for v in I))
                S = cache.get(key)
                if S is None:
                    S = cache[key] = self.order_up_to(i, j, I, use_override=False)
                Q = max(0.0, S - sum(I))
                if Q > EPS:
                    tot["setup"] += x.a
                    tot["ordering"] += x.c * Q
                    cost += x.a + x.c * Q
                    Qsum[i - 1] += Q
                    Qcnt[i - 1] += 1
                for t in range(i, j + 1):
                    I_t = phi(I, Q if t == i else 0.0, float(d[t - 1]))
                    pc = period_cost(I_t, x.h, x.p, x.w)
                    for k, v in pc.items():
                        tot[k] += v
                        cost += v
                    Isum[t - 1] += [max(v, 0.0) for v in I_t]
                    back[t - 1] += max(-I_t[0], 0.0)
                    I = next_state(I_t)
            costs[r] = cost
        mean = float(costs.mean())
        sd = float(costs.std(ddof=1)) if reps > 1 else 0.0
        per = []
        for t in range(T):
            row = {"t": t + 1, "E{d_t}": self.lam[t], "Y_t": Y[t], "mean Q_t": Qsum[t] / reps,
                   "P(Q_t>0)": Qcnt[t] / reps, "mean backlog": back[t] / reps}
            for b in range(M):
                row[f"I_t,{b}"] = Isum[t, b] / reps
            per.append(row)
        return {"reps": reps, "seed": seed, "TC": mean, "sd": sd, "ci95": 1.96 * sd / math.sqrt(reps),
                "parts": {k: v / reps for k, v in tot.items()}, "costs": costs, "per_period": per,
                "cycles": cycles, "avg_cycle_length": T / len(cycles), "runtime_s": time.perf_counter() - t0}


def cycles_from_Y(Y: list, T: int) -> list:
    """Replenishment cycles (i, j) implied by Y_T: each cycle runs until the next planned order."""
    starts = [t + 1 for t in range(T) if Y[t] == 1]
    if not starts or starts[0] != 1:
        starts = [1] + starts
    return [(s, (starts[k + 1] - 1) if k + 1 < len(starts) else T) for k, s in enumerate(starts)]


def run(inst: Instance, reps: int = 5000, seed: int = 1, S_override: dict | None = None) -> dict:
    h = Heuristic(inst, S_override)
    s1 = h.step1()
    s2 = h.step2(s1["Y"], reps, seed)
    return {"step1": s1, "step2": s2}


def validate(d: dict) -> list:
    """Input checks for the prototype. Returns a list of messages (empty = valid)."""
    errs = []
    T, M = d.get("T"), d.get("M")
    if not (isinstance(T, int) and 1 <= T <= 24):
        errs.append("T must be a whole number from 1 to 24.")
    if not (isinstance(M, int) and 2 <= M <= 6):
        errs.append("M must be a whole number from 2 to 6.")
    lam = d.get("lam") or []
    if isinstance(T, int) and len(lam) != T:
        errs.append(f"Enter one E{{d_t}} per period ({T} values).")
    if any((not isinstance(v, (int, float))) or v < 0 or v > 200 for v in lam):
        errs.append("Every E{d_t} must be between 0 and 200.")
    for k in ("a", "c", "w"):
        if d.get(k) is None or d[k] < 0:
            errs.append(f"{k} must be zero or more.")
    for k in ("h", "p"):
        if d.get(k) is None or d[k] <= 0:
            errs.append(f"{k} must be greater than zero (Lemma 2 needs h > 0 and p > 0).")
    if d.get("dist") == "normal" and not (d.get("cv") and 0 < d["cv"] <= 1):
        errs.append("CV must be between 0 and 1 for normal demand.")
    return errs
