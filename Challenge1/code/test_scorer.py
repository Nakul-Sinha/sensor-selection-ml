"""Tests for scorer.py.

Run with:  python test_scorer.py
Plain asserts; the file is also pytest-compatible.
"""

import itertools
import math
import sys
import time
import traceback

import numpy as np

from scorer import (
    PAIRS,
    PLANS,
    best_plan_index,
    chance_constant,
    matrix_for_plan,
    matrix_from_distances,
    plan_distances,
    q_values,
    redundancy,
    score_cases,
    utility_all_plans,
)


# --------------------------------------------------------------- helpers

def _random_catalog(rng, scale=None):
    if scale is None:
        scale = rng.uniform(0.6, 2.0)
    return rng.normal(size=(12, 6)) * scale


def _reference_utilities(R):
    """Straightforward slow implementation, written directly from the spec."""
    R = np.asarray(R, dtype=np.float64)
    out = []
    for plan in itertools.combinations(range(12), 3):
        ds = []
        for i in range(6):
            for j in range(i + 1, 6):
                s = sum((R[w, i] - R[w, j]) ** 2 for w in plan)
                ds.append(math.sqrt(s / 3.0))
        ds = np.asarray(ds)
        cs = []
        for a, b in itertools.combinations(plan, 2):
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(R[a], R[b])[0, 1]
            c = abs(c)
            if not np.isfinite(c):
                c = 1.0
            cs.append(c)
        red = sum(cs) / 3.0
        out.append(ds.min() + 0.18 * ds.mean() - 0.035 * red)
    return np.asarray(out)


# --------------------------------------------------------------- tests

def test_1_plans_and_pairs():
    assert PLANS.shape == (220, 3)
    assert np.issubdtype(PLANS.dtype, np.integer)
    assert np.all(PLANS[:, 0] < PLANS[:, 1])
    assert np.all(PLANS[:, 1] < PLANS[:, 2])
    assert PLANS.min() == 0 and PLANS.max() == 11
    rows = [tuple(r) for r in PLANS.tolist()]
    assert len(set(rows)) == 220                    # all unique
    assert rows == sorted(rows)                     # lexicographic ascending
    assert rows == list(itertools.combinations(range(12), 3))
    expected_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                      (1, 2), (1, 3), (1, 4), (1, 5),
                      (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    assert PAIRS.shape == (15, 2)
    assert [tuple(r) for r in PAIRS.tolist()] == expected_pairs


def test_2_chance_constant_analytic():
    rng = np.random.default_rng(7)
    R = _random_catalog(rng, scale=1.0)
    u = utility_all_plans(R)
    assert np.unique(u).size == 220, "seed must yield 220 distinct utilities"
    q = q_values(R)
    # with distinct utilities the sorted q values are exactly 1/220 .. 220/220
    assert np.array_equal(np.sort(q), np.arange(1, 221) / 220.0)
    analytic = sum(k * k for k in range(1, 221)) / 220.0 ** 3
    c = chance_constant(R)
    print(f"        chance_constant = {c:.12f}   analytic = {analytic:.12f}")
    assert abs(c - analytic) < 1e-9


def test_3_vectorized_matches_reference():
    rng = np.random.default_rng(123)
    for _ in range(4):
        R = _random_catalog(rng)
        u = utility_all_plans(R)
        assert u.shape == (220,) and u.dtype == np.float64
        ref = _reference_utilities(R)
        np.testing.assert_allclose(u, ref, rtol=0.0, atol=1e-10)
        assert best_plan_index(R) == int(np.argmax(ref))
    # degenerate rows: a constant row (nonfinite corr -> 1.0) and a duplicate row
    R = _random_catalog(rng, scale=1.0)
    R[4, :] = 2.5
    R[7, :] = R[2, :]
    np.testing.assert_allclose(utility_all_plans(R), _reference_utilities(R),
                               rtol=0.0, atol=1e-10)


def test_3b_single_plan_functions():
    rng = np.random.default_rng(11)
    R = _random_catalog(rng, scale=1.0)
    plan = (2, 5, 9)
    d = plan_distances(R, plan)
    assert d.shape == (15,)
    k = [tuple(r) for r in PAIRS.tolist()].index((0, 3))
    manual = math.sqrt(sum((R[w, 0] - R[w, 3]) ** 2 for w in plan) / 3.0)
    assert abs(d[k] - manual) < 1e-14
    C = np.corrcoef(R[list(plan)])
    manual_red = (abs(C[0, 1]) + abs(C[0, 2]) + abs(C[1, 2])) / 3.0
    assert abs(redundancy(R, plan) - manual_red) < 1e-14
    idx = [tuple(r) for r in PLANS.tolist()].index(plan)
    expect = d.min() + 0.18 * d.mean() - 0.035 * redundancy(R, plan)
    assert abs(utility_all_plans(R)[idx] - expect) < 1e-12
    # a plan is a set: the order of its indices must not matter
    assert np.array_equal(plan_distances(R, (9, 2, 5)), d)
    assert redundancy(R, (5, 9, 2)) == redundancy(R, plan)


def test_4_matrix_properties():
    rng = np.random.default_rng(5)
    for _ in range(10):
        R = _random_catalog(rng)
        plan = rng.choice(12, size=3, replace=False)
        M = matrix_for_plan(R, plan)
        assert M.shape == (6, 6)
        assert np.issubdtype(M.dtype, np.integer)
        assert np.array_equal(M, M.T)                # symmetric
        assert np.all(np.diag(M) == 0)               # zero diagonal
        off = M[PAIRS[:, 0], PAIRS[:, 1]]
        assert np.all((off >= 1) & (off <= 4))       # values in 1..4
        assert np.array_equal(M, matrix_from_distances(plan_distances(R, plan)))


def test_5_bucket_boundaries():
    d = np.zeros(15)
    d[0], d[1], d[2] = 0.75, 1.25, 1.85              # exact edges (inclusive below)
    d[3] = np.nextafter(0.75, 0.0)                   # just below each edge
    d[4] = np.nextafter(1.25, 0.0)
    d[5] = np.nextafter(1.85, 0.0)
    d[6], d[7] = 0.0, 100.0
    M = matrix_from_distances(d)
    v = M[PAIRS[:, 0], PAIRS[:, 1]]
    assert v[0] == 2 and v[1] == 3 and v[2] == 4     # lower bound inclusive
    assert v[3] == 1 and v[4] == 2 and v[5] == 3
    assert v[6] == 1 and v[7] == 4
    assert np.all(v[8:] == 1)


def test_5b_q_ties_and_definition():
    rng = np.random.default_rng(3)
    # integer-valued catalog + a duplicated window -> exact utility ties
    R = rng.integers(0, 5, size=(12, 6)).astype(np.float64)
    R[6, :] = R[1, :]
    u = utility_all_plans(R)
    assert np.unique(u).size < 220                   # ties really exist
    q = q_values(R)
    brute = (u[None, :] <= u[:, None]).sum(axis=1) / 220.0
    assert np.array_equal(q, brute)                  # q == fraction with u <= own u
    assert q.max() == 1.0
    # fully degenerate catalog: identical (nonconstant) rows -> all plans tie
    R2 = np.tile(np.array([0.0, 0.4, 1.1, 2.0, 2.6, 3.5]), (12, 1))
    assert np.all(q_values(R2) == 1.0)
    assert chance_constant(R2) == 1.0
    assert best_plan_index(R2) == 0                  # ties -> lowest index


def test_6_perfect_submission():
    rng = np.random.default_rng(42)
    n = 40
    R_list = [_random_catalog(rng) for _ in range(n)]
    plans, mats = [], []
    for R in R_list:
        p = tuple(int(w) for w in PLANS[best_plan_index(R)])
        plans.append(p)
        mats.append(matrix_for_plan(R, p))
    res = score_cases(R_list, plans, mats)
    for key in ("MeasurementPlanScore", "MatrixCertificateScore",
                "JointOptimalScore", "Score", "P_raw", "A", "A_corrected",
                "M_exact"):
        assert abs(res[key] - 1.0) < 1e-12, (key, res[key])
    assert 0.30 < res["P_chance"] < 0.40


def test_7_random_submission_near_chance():
    rng = np.random.default_rng(2026)
    n = 400
    R_list = [_random_catalog(rng) for _ in range(n)]
    plans = [tuple(int(w) for w in rng.choice(12, size=3, replace=False))
             for _ in range(n)]
    mats = []
    for _ in range(n):
        vals = rng.integers(1, 5, size=15)
        M = np.zeros((6, 6), dtype=np.int64)
        M[PAIRS[:, 0], PAIRS[:, 1]] = vals
        M[PAIRS[:, 1], PAIRS[:, 0]] = vals
        mats.append(M)
    res = score_cases(R_list, plans, mats)
    print(f"        random submission: Score={res['Score']:.4f}  "
          f"MPS={res['MeasurementPlanScore']:.4f}  P_raw={res['P_raw']:.4f}  "
          f"P_chance={res['P_chance']:.4f}  A={res['A']:.4f}  "
          f"M_exact={res['M_exact']:.4f}  JOS={res['JointOptimalScore']:.4f}")
    assert res["MeasurementPlanScore"] < 0.1
    assert res["Score"] < 0.2
    assert abs(res["P_raw"] - res["P_chance"]) < 0.08   # chance correction works
    assert abs(res["A"] - 0.25) < 0.08                  # random matrix ~ 25% agreement
    assert res["JointOptimalScore"] < 0.01


def test_8_speed_1000_cases():
    rng = np.random.default_rng(99)
    n = 1000
    R_all = rng.normal(size=(n, 12, 6)) * 1.2
    plans = [tuple(int(w) for w in rng.choice(12, size=3, replace=False))
             for _ in range(n)]
    mats = [matrix_for_plan(R_all[i], plans[i]) for i in range(n)]
    t0 = time.perf_counter()
    res = score_cases(R_all, plans, mats)
    dt = time.perf_counter() - t0
    print(f"        score_cases on {n} cases: {dt * 1e3:.0f} ms")
    assert dt < 60.0
    # the matrices above are exactly correct for the submitted plans
    assert abs(res["A"] - 1.0) < 1e-12
    assert abs(res["M_exact"] - 1.0) < 1e-12
    assert abs(res["MatrixCertificateScore"] - 1.0) < 1e-12


# --------------------------------------------------------------- runner

def main():
    tests = [
        test_1_plans_and_pairs,
        test_2_chance_constant_analytic,
        test_3_vectorized_matches_reference,
        test_3b_single_plan_functions,
        test_4_matrix_properties,
        test_5_bucket_boundaries,
        test_5b_q_ties_and_definition,
        test_6_perfect_submission,
        test_7_random_submission_near_chance,
        test_8_speed_1000_cases,
    ]
    print(f"python {sys.version.split()[0]} | numpy {np.__version__}")
    failed = 0
    for t in tests:
        t0 = time.perf_counter()
        try:
            t()
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS  {t.__name__}  ({(time.perf_counter() - t0) * 1e3:.1f} ms)")
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
