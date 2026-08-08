"""
Fair comparison of per-window standardization transforms.

Each candidate transform gets its OWN optimal thresholds, so conventions are compared on the
information their d carries, not on accidental scale alignment. Uses the TRUE
packet_response_matrix => everything here is an ORACLE CEILING.

Usage: python explore3.py <public_dir> [n_cases]
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
CHANCE = 0.335609504132
EPS = 1e-12


# ---------------- per-window transforms.  P: (N,6,12) -> (N,6,12) standardized over axis=1
def t_raw(P):
    return P - np.median(P, axis=1, keepdims=True)

def t_std_pop(P):
    return (P - P.mean(1, keepdims=True)) / (P.std(1, ddof=0, keepdims=True) + EPS)

def t_std_samp(P):
    return (P - P.mean(1, keepdims=True)) / (P.std(1, ddof=1, keepdims=True) + EPS)

def t_mad(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return (P - m) / (1.4826 * mad + EPS)

def t_iqr(P):
    m = np.median(P, axis=1, keepdims=True)
    iqr = np.percentile(P, 75, axis=1, keepdims=True) - np.percentile(P, 25, axis=1, keepdims=True)
    return (P - m) / (iqr / 1.349 + EPS)

def t_range(P):
    m = np.median(P, axis=1, keepdims=True)
    rng = P.max(1, keepdims=True) - P.min(1, keepdims=True)
    return (P - m) / (rng + EPS)

def t_meanad(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.abs(P - m).mean(1, keepdims=True)
    return (P - m) / (mad + EPS)

def t_rank(P):
    """Rank -> normal scores (van der Waerden). Fully bounded, maximally robust."""
    order = P.argsort(axis=1).argsort(axis=1).astype(np.float64)      # 0..5
    u = (order + 0.5) / 6.0
    from math import sqrt
    # inverse normal CDF via erfinv
    from scipy.special import erfinv
    return sqrt(2.0) * erfinv(2 * u - 1)

def _mad_raw(P):
    m = np.median(P, axis=1, keepdims=True)
    return np.median(np.abs(P - m), axis=1, keepdims=True)

def t_mad_floor(P, frac=0.25):
    """MAD with a per-case floor -> prevents blow-up when 4+ of 6 values cluster."""
    m = np.median(P, axis=1, keepdims=True)
    mad = _mad_raw(P)
    floor = frac * np.median(mad, axis=2, keepdims=True)              # median MAD across windows
    return (P - m) / (1.4826 * np.maximum(mad, floor) + EPS)

def t_winsor_std(P):
    lo = np.percentile(P, 10, axis=1, keepdims=True)
    hi = np.percentile(P, 90, axis=1, keepdims=True)
    Pw = np.clip(P, lo, hi)
    return (P - Pw.mean(1, keepdims=True)) / (Pw.std(1, ddof=0, keepdims=True) + EPS)


TRANSFORMS = {
    "raw_centered": t_raw,
    "std_pop": t_std_pop,
    "std_samp": t_std_samp,
    "mad": t_mad,
    "iqr": t_iqr,
    "range": t_range,
    "meanad": t_meanad,
    "rank_normal": t_rank,
    "mad_floor.25": lambda P: t_mad_floor(P, 0.25),
    "mad_floor.50": lambda P: t_mad_floor(P, 0.50),
    "winsor_std": t_winsor_std,
}


def best_thresholds(d, true_cat, n_grid=300, rounds=40):
    """Coordinate-ascent search for 3 thresholds maximizing entry accuracy."""
    qs = [np.quantile(d, 0.02), np.quantile(d, 0.25), np.quantile(d, 0.55)]
    t = list(qs)
    def acc(th):
        return (np.digitize(d, th) + 1 == true_cat).mean()
    cur = acc(t)
    for _ in range(rounds):
        moved = False
        for k in range(3):
            lo = (t[k - 1] + 1e-6) if k > 0 else max(d.min(), 1e-6)
            hi = (t[k + 1] - 1e-6) if k < 2 else np.quantile(d, 0.999)
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, n_grid)
            bv, ba = t[k], cur
            for v in grid:
                tt = list(t); tt[k] = v
                a = acc(tt)
                if a > ba:
                    ba, bv = a, v
            if ba > cur + 1e-12:
                t[k], cur, moved = bv, ba, True
        if not moved:
            break
    return t, cur


def main():
    public = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 4800
    tr = pd.read_csv(public / "train.csv").head(n)
    P = np.stack([np.asarray(json.loads(s), np.float64).reshape(6, 12)
                  for s in tr["packet_response_matrix"]])
    M = np.stack([np.asarray(json.loads(s), np.int64) for s in tr["observability_matrix"]])
    sched = np.stack([np.array(sorted(int(t[1:]) for t in s.split("|")))
                      for s in tr["window_schedule"]])
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N = len(P); rows = np.arange(N)
    true_idx = np.array([np.where((PLANS == s).all(axis=1))[0][0] for s in sched])
    print("cases:", N)
    print("true cat distribution:", {v: int((true_cat == v).sum()) for v in (1, 2, 3, 4)})

    print("\n%-14s | %8s %8s %8s %8s | %8s %8s" %
          ("transform", "pubAcc", "bestScl", "freeAcc", "exactM", "top1", "PlanScore"))
    print("-" * 84)

    results = {}
    for name, fn in TRANSFORMS.items():
        try:
            S = fn(P)                                   # (N,6,12)
        except Exception as e:
            print("%-14s  ERROR %s" % (name, e)); continue
        R = np.transpose(S, (0, 2, 1))                  # (N,12,6)
        sq = (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2      # (N,12,15)
        d_sched = np.sqrt(sq[rows[:, None], sched, :].mean(axis=1))  # (N,15)

        # published thresholds with best global scale
        pub_best, pub_s = -1, 1.0
        for s in np.arange(0.05, 4.0, 0.01):
            a = (np.digitize(d_sched * s, (0.75, 1.25, 1.85)) + 1 == true_cat).mean()
            if a > pub_best:
                pub_best, pub_s = a, s
        th, free_acc = best_thresholds(d_sched.ravel(), true_cat.ravel())
        pred = np.digitize(d_sched, th) + 1
        exact = (pred == true_cat).all(axis=1).mean()

        # plan side
        Rc = R - R.mean(2, keepdims=True)
        nn = np.sqrt((Rc ** 2).sum(2))
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.abs(np.einsum('nwe,nve->nwv', Rc, Rc) / (nn[:, :, None] * nn[:, None, :]))
        C[~np.isfinite(C)] = 1.0
        red = (C[:, PLANS[:, 0], PLANS[:, 1]] + C[:, PLANS[:, 0], PLANS[:, 2]]
               + C[:, PLANS[:, 1], PLANS[:, 2]]) / 3.0
        Dp = np.sqrt(sq[:, PLANS, :].mean(axis=2))       # (N,220,15)
        bt1, bmq = -1, 0
        for s in (0.25, 0.5, pub_s, 1.0, 2.0):
            u = (Dp.min(2) + 0.18 * Dp.mean(2)) * s - 0.035 * red
            t1 = (u.argmax(1) == true_idx).mean()
            if t1 > bt1:
                own = u[rows, true_idx][:, None]
                q = (u <= own).sum(1) / 220.0
                bt1, bmq = t1, (q ** 2).mean()
        results[name] = (pub_best, free_acc, exact, bt1, bmq)
        print("%-14s | %8.4f %8.2f %8.4f %8.4f | %8.4f %8.4f" %
              (name, pub_best, pub_s, free_acc, exact, bt1, (bmq - CHANCE) / (1 - CHANCE)))
        print("               thresholds: %.4f / %.4f / %.4f" % tuple(th))

    bestname = max(results, key=lambda k: results[k][1])
    print("\nBEST transform by free-threshold accuracy: %s  (%.4f)" % (bestname, results[bestname][1]))


if __name__ == "__main__":
    main()
