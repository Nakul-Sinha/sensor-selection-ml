"""Exact grading math for the measurement-plan / observability-matrix competition.

Pure numpy + stdlib itertools. Importable as a module; no I/O, no data files.

Spec summary
------------
* 12 candidate windows (indices 0..11), 6 episodes (indices 0..5).
* A response catalog ``R`` is a float array of shape (12, 6): R[w, e].
* A plan is a set of exactly 3 distinct window indices; the C(12, 3) = 220
  legal plans are the rows of ``PLANS`` in lexicographic ascending order.
* For a plan P and episode pair (i, j), i < j:

      d(i, j, P) = sqrt( (1/3) * sum_{w in P} (R[w, i] - R[w, j])**2 )

  The 15 pairs are enumerated in ``PAIRS`` (row-major upper-triangle) order.
* redundancy(P) = mean of |Pearson correlation| over the 3 unordered pairs of
  the selected 6-element rows R[w, :] (np.corrcoef semantics); a nonfinite
  correlation (nan/inf, e.g. a constant row) counts as 1.0 before averaging.
* utility(P) = min(D(P)) + 0.18 * mean(D(P)) - 0.035 * redundancy(P)
* q(P) = (# of the 220 plans with utility <= utility(P)) / 220  (ties <=,
  so the single best plan has q = 1.0).
* Per-case chance constant  c = (1/220) * sum over all 220 plans of q**2.
* Observability matrix for a plan: symmetric 6x6 int, zero diagonal; the
  (i, j) entry is the bucket of d(i, j, P) with lower-bound-inclusive edges:

      d < 0.75            -> 1
      0.75 <= d < 1.25    -> 2
      1.25 <= d < 1.85    -> 3
      d >= 1.85           -> 4

``score_cases`` aggregates submitted plans/matrices over many cases; see its
docstring for the exact formulas.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "PLANS",
    "PAIRS",
    "plan_distances",
    "redundancy",
    "utility_all_plans",
    "best_plan_index",
    "matrix_for_plan",
    "matrix_from_distances",
    "q_values",
    "chance_constant",
    "score_cases",
]

N_WINDOWS = 12
N_EPISODES = 6
N_PLANS = 220  # C(12, 3)
N_PAIRS = 15   # C(6, 2)

# All 220 legal plans (rows sorted ascending), lexicographic ascending order.
PLANS = np.array(list(itertools.combinations(range(N_WINDOWS), 3)), dtype=np.int64)

# The 15 episode pairs (i, j), i < j, in row-major upper-triangle order:
# (0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)
PAIRS = np.array(list(itertools.combinations(range(N_EPISODES), 2)), dtype=np.int64)

# O(1) lookup: sorted 3-tuple of window indices -> row index in PLANS.
_PLAN_INDEX = {tuple(int(w) for w in row): k for k, row in enumerate(PLANS)}

# utility(P) = min(D) + _MEAN_W * mean(D) - _RED_W * redundancy(P)
_MEAN_W = 0.18
_RED_W = 0.035

# Lower-bound-inclusive bucket edges for the observability matrix.
_BUCKET_EDGES = (0.75, 1.25, 1.85)


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _as_catalog(R):
    """Validate and return R as a float64 (12, 6) array."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (N_WINDOWS, N_EPISODES):
        raise ValueError(
            f"catalog must have shape ({N_WINDOWS}, {N_EPISODES}), got {R.shape}"
        )
    return R


def _as_plan(plan):
    """Validate a plan (any order) and return it as a sorted int64 (3,) array."""
    p = np.sort(np.asarray(plan, dtype=np.int64).reshape(-1))
    if p.shape[0] != 3 or tuple(int(w) for w in p) not in _PLAN_INDEX:
        raise ValueError(
            f"plan must be 3 distinct window indices in 0..{N_WINDOWS - 1}, got {plan!r}"
        )
    return p


def _plan_row(plan):
    """Return (sorted plan array, its row index in PLANS)."""
    p = _as_plan(plan)
    return p, _PLAN_INDEX[(int(p[0]), int(p[1]), int(p[2]))]


def _pair_sq_diffs(R):
    """Per-window squared episode-pair differences, shape (12, 15), PAIRS order."""
    d = R[:, PAIRS[:, 0]] - R[:, PAIRS[:, 1]]
    return d * d


def _abs_corr(R):
    """(12, 12) matrix of |Pearson correlation| between the window rows.

    Uses np.corrcoef semantics (same ddof for both rows; result clipped to
    [-1, 1]). Any nonfinite entry (e.g. a constant row gives 0/0 -> nan) is
    replaced by 1.0, as the spec requires.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R)
    A = np.abs(C)
    A[~np.isfinite(A)] = 1.0
    return A


def _q_from_utilities(u):
    """q for each plan: (# utilities <= own utility) / 220, ties counted <=."""
    return np.searchsorted(np.sort(u), u, side="right") / float(N_PLANS)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def plan_distances(R, plan):
    """The 15 distances d(i, j, plan), shape (15,), in PAIRS order."""
    R = _as_catalog(R)
    p = _as_plan(plan)
    return np.sqrt(_pair_sq_diffs(R)[p].mean(axis=0))


def redundancy(R, plan):
    """Mean |Pearson correlation| over the 3 unordered row pairs of the plan."""
    R = _as_catalog(R)
    p = _as_plan(plan)
    A = _abs_corr(R)
    return float((A[p[0], p[1]] + A[p[0], p[2]] + A[p[1], p[2]]) / 3.0)


def utility_all_plans(R):
    """Utility of every legal plan, shape (220,), in PLANS order.

    Fully vectorized: no Python loop over the 220 plans.
    """
    R = _as_catalog(R)
    sq = _pair_sq_diffs(R)                      # (12, 15)
    D = np.sqrt(sq[PLANS].mean(axis=1))         # (220, 15): distances per plan
    A = _abs_corr(R)                            # (12, 12)
    red = (
        A[PLANS[:, 0], PLANS[:, 1]]
        + A[PLANS[:, 0], PLANS[:, 2]]
        + A[PLANS[:, 1], PLANS[:, 2]]
    ) / 3.0                                     # (220,)
    return D.min(axis=1) + _MEAN_W * D.mean(axis=1) - _RED_W * red


def best_plan_index(R):
    """Row index into PLANS of the max-utility plan (ties -> lowest index)."""
    return int(np.argmax(utility_all_plans(R)))


def matrix_from_distances(d15):
    """Symmetric zero-diagonal 6x6 int bucket matrix from 15 distances (PAIRS order)."""
    d = np.asarray(d15, dtype=np.float64).reshape(N_PAIRS)
    b = np.ones(N_PAIRS, dtype=np.int64)
    for edge in _BUCKET_EDGES:  # d<0.75 -> 1, [0.75,1.25) -> 2, [1.25,1.85) -> 3, >=1.85 -> 4
        b += d >= edge
    M = np.zeros((N_EPISODES, N_EPISODES), dtype=np.int64)
    M[PAIRS[:, 0], PAIRS[:, 1]] = b
    M[PAIRS[:, 1], PAIRS[:, 0]] = b
    return M


def matrix_for_plan(R, plan):
    """Correct observability matrix for a plan on catalog R, shape (6, 6) int."""
    return matrix_from_distances(plan_distances(R, plan))


def q_values(R):
    """q(P) for every plan, shape (220,), in PLANS order."""
    return _q_from_utilities(utility_all_plans(R))


def chance_constant(R):
    """Per-case chance constant c = mean over the 220 plans of q(P)**2."""
    q = q_values(R)
    return float(np.mean(q * q))


def score_cases(R_list, sub_plans, sub_matrices):
    """Score submitted plans and matrices against the true catalogs.

    Parameters
    ----------
    R_list : sequence of (12, 6) float arrays (or an (n, 12, 6) array)
        The true response catalogs, one per case.
    sub_plans : sequence of length-3 index tuples/arrays
        The submitted plans (treated as sets; order of indices is irrelevant).
    sub_matrices : sequence of (6, 6) int arrays
        The submitted observability matrices.

    Returns
    -------
    dict with keys 'MeasurementPlanScore', 'MatrixCertificateScore',
    'JointOptimalScore', 'Score', 'P_raw', 'P_chance', 'A', 'A_corrected',
    'M_exact' (all Python floats).

    Math
    ----
    P_raw     = mean_i q_i(submitted plan)**2
    P_chance  = mean_i c_i                    (c_i = chance constant of case i)
    MeasurementPlanScore = clip((P_raw - P_chance) / (1 - P_chance), 0, 1)
    a_i       = (# of the 15 upper-triangle entries where the submitted matrix
                 equals the correct matrix built from the SUBMITTED plan) / 15
    A         = mean_i a_i;   A_corrected = clip((A - 0.25) / 0.75, 0, 1)
    M_exact   = mean_i [full 6x6 submitted matrix == correct matrix]
    MatrixCertificateScore = 0.90 * A_corrected + 0.10 * M_exact
    joint_i   = 1 iff the submitted plan attains the MAX utility among all 220
                plans AND the submitted matrix is exactly correct for it
    JointOptimalScore = mean_i joint_i
    Score = 0.55 * MeasurementPlanScore + 0.40 * MatrixCertificateScore
            + 0.05 * JointOptimalScore
    """
    n = len(R_list)
    if len(sub_plans) != n or len(sub_matrices) != n:
        raise ValueError("R_list, sub_plans and sub_matrices must have the same length")
    if n == 0:
        raise ValueError("score_cases needs at least one case")

    iu, ju = PAIRS[:, 0], PAIRS[:, 1]
    q_sub_sq = np.empty(n, dtype=np.float64)
    c_all = np.empty(n, dtype=np.float64)
    a_frac = np.empty(n, dtype=np.float64)
    exact = np.empty(n, dtype=np.float64)
    joint = np.empty(n, dtype=np.float64)

    for k in range(n):
        R = _as_catalog(R_list[k])
        p, idx = _plan_row(sub_plans[k])

        u = utility_all_plans(R)
        q = _q_from_utilities(u)
        q_sub_sq[k] = q[idx] * q[idx]
        c_all[k] = np.mean(q * q)

        M_true = matrix_for_plan(R, p)
        M_sub = np.asarray(sub_matrices[k])
        if M_sub.shape != (N_EPISODES, N_EPISODES):
            raise ValueError(
                f"submitted matrix {k} must have shape (6, 6), got {M_sub.shape}"
            )
        a_frac[k] = np.count_nonzero(M_sub[iu, ju] == M_true[iu, ju]) / float(N_PAIRS)
        is_exact = bool(np.array_equal(M_sub, M_true))
        exact[k] = 1.0 if is_exact else 0.0
        joint[k] = 1.0 if (is_exact and u[idx] == u.max()) else 0.0

    P_raw = float(np.mean(q_sub_sq))
    P_chance = float(np.mean(c_all))
    denom = 1.0 - P_chance
    if denom > 0.0:
        mps = float(np.clip((P_raw - P_chance) / denom, 0.0, 1.0))
    else:
        # Degenerate: every plan ties on every case (P_chance == 1); the spec's
        # normalization is 0/0 and no skill above chance is expressible.
        mps = 0.0

    A_mean = float(np.mean(a_frac))
    A_corrected = float(np.clip((A_mean - 0.25) / 0.75, 0.0, 1.0))
    M_exact = float(np.mean(exact))
    mcs = 0.90 * A_corrected + 0.10 * M_exact
    jos = float(np.mean(joint))
    score = 0.55 * mps + 0.40 * mcs + 0.05 * jos

    return {
        "MeasurementPlanScore": mps,
        "MatrixCertificateScore": mcs,
        "JointOptimalScore": jos,
        "Score": score,
        "P_raw": P_raw,
        "P_chance": P_chance,
        "A": A_mean,
        "A_corrected": A_corrected,
        "M_exact": M_exact,
    }
