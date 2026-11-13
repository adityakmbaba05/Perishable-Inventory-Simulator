"""Demand distributions used in the paper.

Section 7.1: Poisson demand with the rates of Table 3 (and Section 6.3).
Tables 4-5 ([47] modified): normal demand, mean = rate in Table 5, standard deviation = CV x mean.
Normal demand is discretised to whole units (and truncated at 0) so the simulation and the SDP
work on the same integer grid as the Poisson case.

F_{i,t} is the cdf of d_{i,t} = d_i + ... + d_t (Section 4, Eq. 13), obtained by convolution.
"""
from __future__ import annotations

import math

import numpy as np


def poisson_pmf(lam: float) -> np.ndarray:
    if lam <= 0:
        return np.array([1.0])
    hi = int(math.ceil(lam + 12 * math.sqrt(lam) + 30))
    k = np.arange(hi + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(v + 1) for v in k])
    pmf = np.exp(logp)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf / pmf.sum()


def normal_pmf(mu: float, sd: float) -> np.ndarray:
    if mu <= 0:
        return np.array([1.0])
    sd = max(sd, 1e-9)
    hi = int(math.ceil(mu + 8 * sd + 5))
    edges = (np.arange(hi + 1) + 0.5 - mu) / (sd * math.sqrt(2))
    cdf = 0.5 * (1 + np.array([math.erf(v) for v in edges]))
    pmf = np.diff(np.concatenate([[0.0], cdf]))
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf / pmf.sum()


class Demand:
    def __init__(self, dist: str, lam: list, cv: float | None = None, max_len: int = 4):
        self.dist, self.lam, self.cv = dist, list(lam), cv
        self.T = len(lam)
        if dist == "poisson":
            self.pmf = [poisson_pmf(v) for v in lam]
        elif dist == "normal":
            self.pmf = [normal_pmf(v, (cv or 0) * v) for v in lam]
        else:
            raise ValueError("dist must be 'poisson' or 'normal'")
        # cdf of d_{i,t} for t - i + 1 <= max_len
        self.max_len = max_len
        self._cdf = {}
        for i in range(1, self.T + 1):
            acc = np.array([1.0])
            for t in range(i, min(self.T, i + max_len - 1) + 1):
                acc = np.convolve(acc, self.pmf[t - 1])
                self._cdf[(i, t)] = np.minimum(np.cumsum(acc), 1.0)

    def cdf(self, i: int, t: int, x: float) -> float:
        """F_{i,t}(x) = P(d_i + ... + d_t <= x)."""
        arr = self._cdf.get((i, t))
        if arr is None:                      # longer than max_len: build on demand
            acc = np.array([1.0])
            for k in range(i, t + 1):
                acc = np.convolve(acc, self.pmf[k - 1])
            arr = self._cdf[(i, t)] = np.minimum(np.cumsum(acc), 1.0)
        k = math.floor(x + 1e-9)
        if k < 0:
            return 0.0
        if k >= len(arr):
            return 1.0
        return float(arr[k])

    def upper(self, i: int, j: int) -> int:
        arr = self._cdf.get((i, j))
        return len(arr) if arr is not None else int(sum(len(p) for p in self.pmf[i - 1:j]))

    def sample(self, reps: int, seed: int) -> np.ndarray:
        """reps x T matrix of independent demand draws (inverse-cdf sampling)."""
        rng = np.random.default_rng(seed)
        u = rng.random((reps, self.T))
        out = np.empty((reps, self.T), dtype=np.int64)
        for t in range(self.T):
            out[:, t] = np.searchsorted(np.cumsum(self.pmf[t]), u[:, t], side="left")
        return out
