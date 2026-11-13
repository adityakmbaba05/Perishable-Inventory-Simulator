"""Test cases built only from data printed in the paper.  Run: python -m pytest -q"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import example63  # noqa: E402
import paper_data as P  # noqa: E402
import sdp  # noqa: E402
from heuristic import Heuristic, Instance, cycles_from_Y, period_cost, phi, validate  # noqa: E402

EX = Instance.from_dict(P.EXAMPLE)


# ---- TC1-TC3: model equations
def test_tc01_phi_fifo_matches_section63_option1():
    """Section 6.3: order up to S_12 = 19, E{d_1} = 6 -> I_{1,0} = 13; E{d_2} = 9 -> I_{2,1} = 4 outdated."""
    I1 = phi([0], 19, 6)
    assert I1 == [13, 0]
    I2 = phi(I1[:-1], 0, 9)
    assert I2 == [0, 4]


def test_tc02_phi_backlog_and_old_stock_first():
    """Eqs. (4)-(5), M = 3: old items (age 1) are used before new ones; unmet demand makes I_{t,0} negative."""
    assert phi([2, 3], 5, 4) == [5, 1, 0]          # 3 of age 1 used first, then 1 of age 0
    assert phi([0, 0], 0, 7) == [-7, 0, 0]          # backlog
    assert phi([-7, 0], 10, 2) == [1, 0, 0]          # backlog of 7 met first, then demand 2


def test_tc03_period_cost_eq1():
    """Eq. (1): waste I_{t,M-1} is not held; h on I_{t,0}^+ and ages 1..M-2; p on backlog."""
    c = period_cost([0, 4], h=1, p=10, w=0)                 # Section 6.3 option 1, end of period 2
    assert c == {"holding": 0, "penalty": 0, "outdating": 0}
    c = period_cost([-3, 0, 0], h=1, p=10, w=2)
    assert c["penalty"] == 30 and c["holding"] == 0


# ---- TC4-TC6: Section 6.3 worked example
def test_tc04_example_arithmetic_with_paper_S_values():
    """Using the S values printed in Section 6.3, every period-1 and period-2 number matches the text."""
    tr = Heuristic(EX, P.EXAMPLE_PUBLISHED_S).step1()["trace"]
    r = {f"{x['cycle']}@{x['j']}": x for rows in tr for x in rows}
    assert r["R(1,1)@1"]["c(i,j)"] == 34                    # TC(1) = 20 + 10 + 4 + 0
    assert r["R(1,2)@2"]["c(i,j)"] == 52                    # TC(2) = 20 + 19 + 13 + 0
    assert r["R(1,2)@2"]["c'"] == 45 and r["R(1,2)@2"]["T_ij"] == 4
    assert round(r["R(1,2)@2"]["AC"], 2) == 24.25
    assert r["R(2,2)@2"]["S_ij"] == 13 and r["R(2,2)@2"]["Q_i"] == 9
    assert r["R(2,2)@2"]["TC(i-1)"] + r["R(2,2)@2"]["c(i,j)"] == 67
    assert r["R(2,2)@2"]["c'"] == 1 and r["R(2,2)@2"]["T_ij"] == 3
    assert round(r["R(2,2)@2"]["AC"], 2) == 22.67
    assert r["R(2,2)@2"]["chosen"] and not r["R(1,2)@2"]["chosen"]


def test_tc05_example_replay_of_fig2_gives_paper_path():
    """Algorithm 1 fed with the Fig. 2 arc costs returns Y_4 = (1,1,1,0) and B_4 = {R(1,1), R(2,2), R(3,4)}."""
    rep = example63.replay()
    assert rep["Y"] == P.EXAMPLE_PUBLISHED["Y_4"]
    assert rep["B"] == "{R(1,1), R(2,2), R(3,4)}"


def test_tc05b_full_implementation_with_fig3_S_gives_paper_path():
    """With the S values of Fig. 3, Algorithm 1 computes the Fig. 2 costs on the chosen path and Y_4 = (1,1,1,0)."""
    s1 = Heuristic(EX, P.EXAMPLE_FIG3_S).step1()
    assert s1["Y"] == [1, 1, 1, 0]
    assert [round(c.cost) for c in s1["B"]] == [34, 33, 38]          # Fig. 2 red arcs
    assert s1["TC_det"] == 105


def test_tc06_example_step2_close_to_published_TC4():
    """Section 6.3: with Y_4 = (1,1,1,0) and 5000 replications the paper reports TC(4) = 116.2."""
    r = Heuristic(EX).step2(P.EXAMPLE_PUBLISHED["Y_4"], 5000, 1)
    assert abs(r["TC"] - 116.2) / 116.2 < 0.02


# ---- TC7-TC9: Eq. (17) and EWA
def test_tc07_order_up_to_satisfies_eq17():
    h = Heuristic(EX)
    S = h.order_up_to(1, 2, [0])
    lhs = lambda x: h.dem.cdf(1, 1, x) + h.dem.cdf(1, 2, x)
    assert lhs(S) >= 2 * 10 / 11 > lhs(S - 1)


def test_tc08_order_up_to_increases_with_penalty():
    lo = Heuristic(Instance.from_dict(dict(P.EXAMPLE, p=10))).order_up_to(1, 2, [0])
    hi = Heuristic(Instance.from_dict(dict(P.EXAMPLE, p=20))).order_up_to(1, 2, [0])
    assert hi >= lo


def test_tc09_ewa_counts_expected_outdating_of_old_stock():
    """Section 6.3, R(3,3) after R(1,1),R(2,2): 4 items of age 1, E{d_3} = 3 -> 1 item expires."""
    assert Heuristic(EX).ewa(3, 3, [4]) == [1]


# ---- TC10-TC12: Algorithm 1 structure on the 216-instance set
@pytest.mark.parametrize("n", [1, 13, 25, 100, 216])
def test_tc10_cycles_never_longer_than_M(n):
    d = P.instance(n)
    s1 = Heuristic(Instance.from_dict(d)).step1()
    assert s1["Y"][0] == 1
    assert all(c.j - c.i + 1 <= d["M"] for c in s1["B"])
    assert sum(c.j - c.i + 1 for c in s1["B"]) == 12


def test_tc11_instance_numbering_matches_appendix():
    assert P.instance(1)["pattern"] == "Stationary" and P.instance(1)["M"] == 2
    assert P.instance(216) == P.instance(216) and P.instance(216)["pattern"] == "Random"
    assert P.instance(216)["M"] == 4 and P.instance(216)["a"] == 75 and P.instance(216)["p"] == 20
    assert len(P.APPENDIX) == 216 and len(P.instances_table4()) == 12


def test_tc12_cycles_from_Y():
    assert cycles_from_Y([1, 1, 1, 0], 4) == [(1, 1), (2, 2), (3, 4)]


# ---- TC13-TC15: SDP benchmark and gap
def test_tc13_sdp_is_lower_bound():
    d = P.instance(1)
    inst = Instance.from_dict(d)
    h = Heuristic(inst)
    tc = h.step2(h.step1()["Y"], 2000, 3)["TC"]
    opt = sdp.solve_instance(inst)[0]
    assert opt <= tc + 3 * h.step2(h.step1()["Y"], 2000, 3)["ci95"]
    assert 100 * (tc - opt) / opt < 5.4                     # paper: gap never above 5.4%


def test_tc14_sdp_not_offered_for_M4():
    ok, msg = sdp.can_solve(dict(T=12, M=4, mu=P.TABLE3["Stationary"]))
    assert not ok and "4,954" in msg


def test_tc15_input_validation():
    assert validate(dict(T=4, M=2, lam=[6, 9, 3, 8], a=20, c=1, h=1, p=10, w=0, dist="poisson")) == []
    bad = validate(dict(T=4, M=1, lam=[6, 9, 3], a=-1, c=1, h=0, p=10, w=0, dist="poisson"))
    assert len(bad) == 4
