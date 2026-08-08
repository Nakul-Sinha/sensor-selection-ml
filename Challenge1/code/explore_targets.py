"""
Reverse-engineer the exact target construction from TRAIN data.

Answers three questions, using the TRUE packet_response_matrix (so model error is excluded):
  Q1. Which standardization convention did the organizer use?
  Q2. What is the ORACLE CEILING -- i.e. how much does the hidden 0.20*clean component cost us,
      even with a perfect tile->response regressor?
  Q3. What thresholds are actually implied by the data (read them off empirically)?

Self-contained on purpose: implemented independently of scorer.py so that agreement between the
two implementations is meaningful evidence of correctness.

Usage:  python explore_targets.py <public_dir> [n_cases]
"""
import sys
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)   # (220,3)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)  # (15,2)
THRESH = (0.75, 1.25, 1.85)


def bucketize(d):
    """d -> category 1..4 with lower-bound-inclusive thresholds."""
    return np.digitize(d, THRESH, right=False) + 1


# ------------------------------------------------------- standardizers
def _std_mean_pop(x, axis=0):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, ddof=0, keepdims=True) + 1e-12)


def _std_mean_samp(x, axis=0):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, ddof=1, keepdims=True) + 1e-12)


def _std_median_mad(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1.4826 * mad + 1e-12)


def _std_median_iqr(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    q75 = np.percentile(x, 75, axis=axis, keepdims=True)
    q25 = np.percentile(x, 25, axis=axis, keepdims=True)
    return (x - med) / ((q75 - q25) / 1.349 + 1e-12)


def _std_median_stdpop(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return (x - med) / (x.std(axis=axis, ddof=0, keepdims=True) + 1e-12)


STDZ = {
    "mean_pop": _std_mean_pop,
    "mean_samp": _std_mean_samp,
    "median_mad": _std_median_mad,
    "median_iqr": _std_median_iqr,
    "median_stdpop": _std_median_stdpop,
}


# ------------------------------------------------------- core math
def catalog_from_P(P, inner, outer):
    """P: (6,12) episode x window  ->  R: (12,6) window x episode.

    inner  = the 'robust' standardization applied to the visible packet response
    outer  = the second standardization applied after mixing in the hidden clean part
    We cannot see the clean part, so R_hat = outer(inner(P)).
    Standardization is always over the 6 EPISODES within each window (axis 0 of P).
    """
    S = STDZ[inner](P, axis=0)
    G = STDZ[outer](S, axis=0)
    return G.T.copy()          # (12,6)


def pair_sqdiff_per_window(R):
    """R (12,6) -> (12,15) squared differences for each window and episode pair."""
    return (R[:, PAIRS[:, 0]] - R[:, PAIRS[:, 1]]) ** 2


def distances_for_plan(R, plan):
    sq = pair_sqdiff_per_window(R)          # (12,15)
    return np.sqrt(sq[list(plan), :].mean(axis=0))


def abscorr_matrix(R):
    """(12,12) matrix of |pearson| between window response rows."""
    Rc = R - R.mean(axis=1, keepdims=True)
    denom = np.sqrt((Rc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (Rc @ Rc.T) / np.outer(denom, denom)
    C = np.abs(C)
    C[~np.isfinite(C)] = 1.0
    return C


def utility_all_plans(R):
    """(220,) utilities, vectorized."""
    sq = pair_sqdiff_per_window(R)                    # (12,15)
    d = np.sqrt(sq[PLANS].mean(axis=1))               # (220,15)
    C = abscorr_matrix(R)
    red = (C[PLANS[:, 0], PLANS[:, 1]] +
           C[PLANS[:, 0], PLANS[:, 2]] +
           C[PLANS[:, 1], PLANS[:, 2]]) / 3.0
    return d.min(axis=1) + 0.18 * d.mean(axis=1) - 0.035 * red


def q_of_plan(util, plan_idx):
    return float((util <= util[plan_idx]).sum()) / len(util)


def plan_to_idx(plan_tuple):
    key = tuple(sorted(plan_tuple))
    hit = np.where((PLANS == np.array(key)).all(axis=1))[0]
    return int(hit[0])


# ------------------------------------------------------- main
def main():
    public_dir = Path(sys.argv[1])
    n_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    tr = pd.read_csv(public_dir / "train.csv")
    print("train.csv rows:", len(tr))
    print("columns:", list(tr.columns))
    tr = tr.head(n_cases)

    Ps, scheds, mats = [], [], []
    for _, row in tr.iterrows():
        P = np.asarray(json.loads(row["packet_response_matrix"]), dtype=np.float64)  # (6,12)
        M = np.asarray(json.loads(row["observability_matrix"]), dtype=np.int64)      # (6,6)
        plan = tuple(sorted(int(t[1:]) for t in str(row["window_schedule"]).split("|")))
        Ps.append(P); scheds.append(plan); mats.append(M)
    Ps = np.stack(Ps); mats = np.stack(mats)
    print("P shape:", Ps.shape, " P finite:", np.isfinite(Ps).all())
    print("P stats: min %.4f max %.4f mean %.4f std %.4f"
          % (Ps.min(), Ps.max(), Ps.mean(), Ps.std()))

    true_cat = mats[:, PAIRS[:, 0], PAIRS[:, 1]]        # (N,15) true buckets for the TRUE plan
    print("\ntrue margin distribution (train):",
          {v: int((true_cat == v).sum()) for v in (1, 2, 3, 4)})

    print("\n" + "=" * 78)
    print("Q1/Q2: convention sweep  (using TRUE packet_response_matrix -> ORACLE CEILING)")
    print("=" * 78)
    print("%-14s %-14s | %8s %8s %8s | %8s" %
          ("inner", "outer", "entryAcc", "exactMat", "planTop1", "meanQ^2"))
    print("-" * 78)

    results = []
    for inner in STDZ:
        for outer in STDZ:
            ent_ok = 0; ent_tot = 0; exact = 0; top1 = 0; q2sum = 0.0
            for k in range(len(Ps)):
                R = catalog_from_P(Ps[k], inner, outer)
                d = distances_for_plan(R, scheds[k])
                pred = bucketize(d)
                ent_ok += int((pred == true_cat[k]).sum()); ent_tot += 15
                exact += int((pred == true_cat[k]).all())
                util = utility_all_plans(R)
                bi = int(np.argmax(util))
                top1 += int(tuple(PLANS[bi]) == scheds[k])
                q2sum += q_of_plan(util, plan_to_idx(scheds[k])) ** 2
            n = len(Ps)
            row = (inner, outer, ent_ok / ent_tot, exact / n, top1 / n, q2sum / n)
            results.append(row)
            print("%-14s %-14s | %8.4f %8.4f %8.4f | %8.4f" % row)

    best = max(results, key=lambda r: r[2])
    print("\nBEST by entry accuracy: inner=%s outer=%s  entryAcc=%.4f exact=%.4f top1=%.4f"
          % (best[0], best[1], best[2], best[3], best[4]))

    # ---------------------------------------------------------------- Q3
    inner, outer = best[0], best[1]
    print("\n" + "=" * 78)
    print("Q3: empirical thresholds under inner=%s outer=%s" % (inner, outer))
    print("=" * 78)
    all_d, all_c = [], []
    for k in range(len(Ps)):
        R = catalog_from_P(Ps[k], inner, outer)
        all_d.append(distances_for_plan(R, scheds[k]))
        all_c.append(true_cat[k])
    all_d = np.concatenate(all_d); all_c = np.concatenate(all_c)
    for v in (1, 2, 3, 4):
        m = all_c == v
        if m.sum():
            dd = all_d[m]
            print("cat %d  n=%6d  min=%.4f  p01=%.4f  p25=%.4f  med=%.4f  p75=%.4f  p99=%.4f  max=%.4f"
                  % (v, m.sum(), dd.min(), np.percentile(dd, 1), np.percentile(dd, 25),
                     np.median(dd), np.percentile(dd, 75), np.percentile(dd, 99), dd.max()))
    for lo, hi, nm in ((1, 2, "1|2"), (2, 3, "2|3"), (3, 4, "3|4")):
        a = all_d[all_c == lo]; b = all_d[all_c == hi]
        if len(a) and len(b):
            print("boundary %s : max(cat%d)=%.4f  min(cat%d)=%.4f  midpoint=%.4f  (published %.2f)"
                  % (nm, lo, a.max(), hi, b.min(), 0.5 * (a.max() + b.min()),
                     THRESH[lo - 1]))

    # global scale check: ratio of our d to the published thresholds
    print("\nIf a uniform scale factor s maps our d onto the true d, the best s is estimated by")
    print("matching quantiles at the published cut points:")
    for ti, t in enumerate(THRESH):
        frac_below = (all_c <= ti + 1).mean()
        our_cut = np.quantile(all_d, frac_below)
        print("  cut %.2f : true frac below = %.4f -> our d quantile = %.4f -> implied scale %.4f"
              % (t, frac_below, our_cut, t / max(our_cut, 1e-9)))

    # ---------------------------------------------------------------- Q4
    # THEORY: every candidate standardizer is affine in x with data-dependent
    # coefficients, and standardization is affine-invariant => outer(inner(P)) == outer(P).
    # Centering cancels in the differences R[w,i]-R[w,j], and correlation is scale free.
    # Therefore the ONLY thing that can differ between conventions is a single global
    # SCALE constant applied to d. Sweep it directly.
    print("\n" + "=" * 78)
    print("Q4: direct global-scale sweep (the only real degree of freedom)")
    print("=" * 78)
    ref = "mean_pop"
    R_all = np.stack([catalog_from_P(Ps[k], ref, ref) for k in range(len(Ps))])   # (N,12,6)
    d_true_plan = np.stack([distances_for_plan(R_all[k], scheds[k]) for k in range(len(Ps))])

    print("known exact ratios vs mean_pop:  sample-std = sqrt(5/6) = %.6f" % np.sqrt(5.0 / 6.0))
    print("%8s | %9s | %9s" % ("scale", "entryAcc", "exactMat"))
    print("-" * 34)
    best_s, best_acc = None, -1.0
    for s in np.arange(0.80, 1.36, 0.005):
        pred = bucketize(d_true_plan * s)
        acc = float((pred == true_cat).mean())
        ex = float((pred == true_cat).all(axis=1).mean())
        if acc > best_acc:
            best_acc, best_s, best_ex = acc, s, ex
    for s in [np.sqrt(5.0 / 6.0), 1.0, np.sqrt(6.0 / 5.0), best_s]:
        pred = bucketize(d_true_plan * s)
        acc = float((pred == true_cat).mean())
        ex = float((pred == true_cat).all(axis=1).mean())
        tag = " <-- BEST" if abs(s - best_s) < 1e-9 else ""
        print("%8.5f | %9.4f | %9.4f%s" % (s, acc, ex, tag))
    print("\nBEST global scale = %.5f  entryAcc=%.4f  exactMatrix=%.4f" % (best_s, best_acc, best_ex))
    print("=> ORACLE CEILING on MatrixCertificate entry accuracy is ~%.4f" % best_acc)
    print("   (this is the cost of the hidden 0.20*clean component alone,")
    print("    with a PERFECT tile->response regressor)")

    # plan-side oracle at the best scale
    top1 = 0; q2 = 0.0
    for k in range(len(Ps)):
        util = utility_all_plans(R_all[k] * best_s)
        top1 += int(tuple(PLANS[int(np.argmax(util))]) == scheds[k])
        q2 += q_of_plan(util, plan_to_idx(scheds[k])) ** 2
    n = len(Ps)
    mps = (q2 / n - 0.335609504132) / (1 - 0.335609504132)
    print("\nplan oracle @ best scale: top1=%.4f  meanQ^2=%.4f  => MeasurementPlanScore~%.4f"
          % (top1 / n, q2 / n, mps))


if __name__ == "__main__":
    main()
