"""Stochastic dynamic programming benchmark, Section 3.1, Eq. (10):

    V_t(I) = min_Q { E C(I, Q) + E V_{t+1}(Φ(I, Q, d_t)) },   V_{T+1} = 0,

with the single-period cost E C of Eq. (9) and the state I_{t-1} of Definition 2.
The paper solves it in C on a grid of 151^(M-1) x T states (each age from -50 to 100).
This prototype solves M = 2 and M = 3 in numpy on a grid sized from the demand; M = 4
took the paper 4,954 s per instance and is not offered here.

State used here (same information as I_{t-1}):
  M = 2: s = I_{t-1,0}; s > 0 items on hand (age 1 in period t), s < 0 backlog.
  M = 3: (s1, s2) = (I_{t-1,0}, I_{t-1,1}) when stock is on hand, or b > 0 backlog.
Costs per period follow Eq. (1): h(I_{t,0})^+ + h sum_{b=1}^{M-2} I_{t,b} - p(I_{t,0})^- + w I_{t,M-1}.
"""
from __future__ import annotations

import math

import numpy as np

from demand import Demand


def _pmfs(inp: dict) -> list:
    dem = Demand(inp.get("dist", "poisson"), inp["mu"], inp.get("cv"), max_len=1)
    out = []
    for pmf in dem.pmf:
        pmf = pmf.copy()
        pmf[pmf <= 1e-12] = 0.0
        out.append(pmf)
    return out


def can_solve(inp: dict):
    T, M, mu = inp["T"], inp["M"], inp["mu"]
    if any(v != 0 for v in (inp.get("opening") or [])):
        return False, "The SDP benchmark here starts from I_0 = 0 (Eq. 6)."
    if M == 2:
        if max(mu) > 80 or T > 36:
            return False, "Demand or horizon is too large for the exact benchmark here."
        return True, "under a second"
    if M == 3:
        m3 = max(sum(mu[t:t + 3]) for t in range(T))
        if m3 > 110 or T > 24:
            return False, "Demand or horizon is too large for the exact benchmark here."
        return True, "a few seconds"
    return False, (f"With M = {M} the SDP state has {M - 1} dimensions. "
                   "The paper reports 4,954 s per instance for M = 4 (Table 7, C code).")


def _solve_m2(inp: dict):
    """State s: s > 0 units of age 1 on hand, s < 0 backorders."""
    T, mu, a, c, h, p, w = inp["T"], inp["mu"], inp["a"], inp["c"], inp["h"], inp["p"], inp["w"]
    pmfs = _pmfs(inp)
    two = max(mu[t] + (mu[t + 1] if t + 1 < T else 0) for t in range(T))
    max_mu = max(mu)
    smax = math.ceil(two + 8 * math.sqrt(two) + 20)
    bmax = math.ceil(max_mu + 8 * math.sqrt(max_mu) + 20)
    ns = smax + bmax + 1
    s = (np.arange(ns) - bmax)[:, None, None]
    q = np.arange(smax + 1)[None, :, None]
    order_cost = np.where(q[0, :, 0] > 0, a + c * q[0, :, 0], 0.0)
    v_next, states = np.zeros(ns), 0
    for t in range(T, 0, -1):
        pmf = pmfs[t - 1]
        keep = np.nonzero(pmf)[0]
        d, pr = keep[None, None, :], pmf[keep]
        old = np.maximum(s, 0)
        fresh = q - np.maximum(d - old, 0) + np.minimum(s, 0)
        waste = np.maximum(old - d, 0)
        nxt = np.clip(fresh + bmax, 0, ns - 1)
        cost = np.where(fresh > 0, h * fresh, -p * fresh) + w * waste + v_next[nxt]
        v_next = (order_cost[None, :] + cost @ pr).min(axis=1)
        states += ns
    return float(v_next[bmax]), states


def _solve_m3(inp: dict):
    """State (s1, s2): units of age 1 and age 2 on hand, or b > 0 backorders with an empty shelf.

    For speed the expectation over demand is split in two. While demand is met from old
    stock, the outcome depends only on how much age-1 stock is left (matrix A). Once old
    stock runs out, it depends only on total old stock (matrix R).
    """
    T, mu, a, c, h, p, w = inp["T"], inp["mu"], inp["a"], inp["c"], inp["h"], inp["p"], inp["w"]
    pmfs = _pmfs(inp)
    m3 = max(sum(mu[t:t + 3]) for t in range(T))
    m2 = max(sum(mu[t:t + 2]) for t in range(T))
    max_mu = max(mu)
    pos = math.ceil(m3 + 6 * math.sqrt(m3) + 10)             # cap on stock after ordering
    s1_max, s2_max = pos, math.ceil(m2 + 6 * math.sqrt(m2) + 10)
    b_max = math.ceil(max_mu + 7 * math.sqrt(max_mu) + 15)
    n1, n2, nq = s1_max + 1, s2_max + 1, pos + 1
    qs = np.arange(nq)
    order_cost = np.where(qs > 0, a + c * qs, 0.0)
    s1g, s2g = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    total_old = (s1g + s2g).ravel()
    q_allowed = qs[None, :] <= np.maximum(0, pos - total_old)[:, None]
    v_next, vb_next = np.zeros((n1, n2)), np.zeros(b_max + 1)
    cache, states = {}, 0
    for t in range(T, 0, -1):
        m = (inp.get("dist", "poisson"), mu[t - 1])
        if m not in cache:
            pmf = pmfs[t - 1]
            k = len(pmf)
            ext = np.concatenate([pmf, np.zeros(n1 + n2 + 2)])
            cdf = np.cumsum(ext)
            a_mat = np.zeros((n1 * n2, n1))
            for s1 in range(n1):
                for s2 in range(n2):
                    row = s1 * n2 + s2
                    a_mat[row, s1] = cdf[s2]
                    if s1:
                        a_mat[row, :s1] = ext[s2 + 1:s2 + s1 + 1][::-1]
            e = np.arange(1, k + 1)
            p_tail = ext[np.minimum(np.arange(n1 + n2 - 1)[:, None] + e[None, :], len(ext) - 1)]
            p_tail[np.arange(n1 + n2 - 1)[:, None] + e[None, :] >= k] = 0.0
            waste_cost = w * np.array([np.dot(pmf[:s2 + 1], s2 - np.arange(min(s2 + 1, k))) if s2 + 1 <= k
                                       else np.dot(pmf, s2 - np.arange(k)) for s2 in range(n2)])
            cache[m] = (pmf, k, a_mat, p_tail, waste_cost)
        pmf, k, a_mat, p_tail, waste_cost = cache[m]
        # value of ending the period with `fresh` new units (z < 0: backorders) and l units of age 1 left
        g = h * (qs[:, None] + np.arange(n1)[None, :]) + v_next[np.minimum(qs, s1_max)][:, np.minimum(np.arange(n1), s2_max)]
        z = np.arange(-(b_max + k), nq)
        g0 = np.where(z >= 0, h * z + v_next[np.clip(z, 0, s1_max), 0], -p * z + vb_next[np.clip(-z, 0, b_max)])
        off = b_max + k                                         # g0[z + off]
        e = np.arange(1, k + 1)
        g_shift = g0[(qs[None, :] - e[:, None]) + off]          # [e, q] = g0(q - e)
        r_mat = p_tail @ g_shift                                # [total old stock, q]
        w_all = a_mat @ g.T + r_mat[total_old] + order_cost[None, :] + np.repeat(waste_cost[None, :], n1, axis=0).ravel()[:, None]
        w_all[~q_allowed] = np.inf
        v = w_all.min(axis=1).reshape(n1, n2)
        d_idx = np.nonzero(pmf)[0]
        vb = np.zeros(b_max + 1)
        for b in range(1, b_max + 1):
            qb = np.arange(pos + b + 1)
            vals = g0[(qb[:, None] - b - d_idx[None, :]) + off] @ pmf[d_idx]
            vb[b] = (np.where(qb > 0, a + c * qb, 0.0) + vals).min()
        vb[0] = v[0, 0]
        v_next, vb_next = v, vb
        states += n1 * n2 + b_max
    return float(v_next[0, 0]), states


def solve(inp: dict):
    """Returns (V_1(I_0), number of states evaluated, runtime in seconds), or None if not offered."""
    import time
    ok, _ = can_solve(inp)
    if not ok:
        return None
    t0 = time.perf_counter()
    v, states = _solve_m2(inp) if inp["M"] == 2 else _solve_m3(inp)
    return v, states, time.perf_counter() - t0


def solve_instance(inst) -> tuple | None:
    """Convenience wrapper for heuristic.Instance."""
    return solve(dict(T=inst.T, M=inst.M, mu=list(inst.lam), a=inst.a, c=inst.c, h=inst.h, p=inst.p, w=inst.w,
                      dist=inst.dist, cv=inst.cv, opening=inst.I0))
