"""
Focused follow-up: fix outer=median/MAD (identified as the organizer's convention) and find the
best achievable matrix rule + plan rule using the TRUE packet_response_matrix (oracle ceiling).

Usage: python explore2.py <public_dir> [n_cases]
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
CHANCE = 0.335609504132


def med_mad(x, axis=0, c=1.4826):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (c * mad + 1e-12), mad


def catalogs(P):                       # P (N,6,12) -> R (N,12,6)
    S, mad = med_mad(P, axis=1)
    return np.transpose(S, (0, 2, 1)), mad


def dists_all_windows(R):              # R (N,12,6) -> (N,12,15) squared diffs
    return (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2


def abscorr(R):                        # (N,12,6) -> (N,12,12)
    Rc = R - R.mean(axis=2, keepdims=True)
    n = np.sqrt((Rc ** 2).sum(axis=2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.einsum('nwe,nve->nwv', Rc, Rc) / (n[:, :, None] * n[:, None, :])
    C = np.abs(C)
    C[~np.isfinite(C)] = 1.0
    return C


def main():
    public = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 4800
    tr = pd.read_csv(public / "train.csv").head(n)
    P = np.stack([np.asarray(json.loads(s), np.float64).reshape(6, 12)
                  for s in tr["packet_response_matrix"]])
    M = np.stack([np.asarray(json.loads(s), np.int64) for s in tr["observability_matrix"]])
    sched = np.stack([np.array(sorted(int(t[1:]) for t in s.split("|")))
                      for s in tr["window_schedule"]])
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]                    # (N,15)
    N = len(P)
    print("cases:", N)

    R, mad = catalogs(P)
    print("MAD stats: min %.3e  p01 %.3e  med %.3e  max %.3e"
          % (mad.min(), np.percentile(mad, 1), np.median(mad), mad.max()))
    print("R abs stats: p50 %.3f p99 %.3f max %.3f" %
          (np.percentile(np.abs(R), 50), np.percentile(np.abs(R), 99), np.abs(R).max()))

    sq = dists_all_windows(R)                                     # (N,12,15)
    rows = np.arange(N)[:, None]
    d_sched = np.sqrt(sq[rows, sched, :].mean(axis=1))            # (N,15) distances for TRUE plan

    print("\n--- d for true plan, by true category ---")
    for v in (1, 2, 3, 4):
        m = true_cat == v
        dd = d_sched[m]
        print("cat %d n=%6d  p05=%.3f p25=%.3f med=%.3f p75=%.3f p95=%.3f"
              % (v, m.sum(), *[np.percentile(dd, q) for q in (5, 25, 50, 75, 95)]))

    # ---- best global scale with PUBLISHED thresholds
    print("\n--- scale sweep, published thresholds 0.75/1.25/1.85 ---")
    best = (-1, None)
    for s in np.arange(0.30, 2.01, 0.005):
        pred = np.digitize(d_sched * s, (0.75, 1.25, 1.85)) + 1
        acc = (pred == true_cat).mean()
        if acc > best[0]:
            best = (acc, s)
    print("best scale %.3f -> entryAcc %.4f" % (best[1], best[0]))

    # ---- free 3-threshold search (coordinate ascent from quantiles)
    print("\n--- FREE threshold search on raw d ---")
    t = [np.quantile(d_sched, 0.02), np.quantile(d_sched, 0.25), np.quantile(d_sched, 0.50)]
    def acc_of(th):
        pred = np.digitize(d_sched, th) + 1
        return (pred == true_cat).mean()
    cur = acc_of(t)
    for _ in range(60):
        improved = False
        for k in range(3):
            lo = t[k - 1] + 1e-4 if k > 0 else 0.01
            hi = t[k + 1] - 1e-4 if k < 2 else d_sched.max()
            grid = np.linspace(lo, hi, 200)
            best_v, best_a = t[k], cur
            for v in grid:
                tt = list(t); tt[k] = v
                a = acc_of(tt)
                if a > best_a:
                    best_a, best_v = a, v
            if best_a > cur + 1e-12:
                t[k] = best_v; cur = best_a; improved = True
        if not improved:
            break
    print("thresholds %.4f / %.4f / %.4f -> entryAcc %.4f" % (t[0], t[1], t[2], cur))
    pred = np.digitize(d_sched, t) + 1
    print("exact-matrix rate %.4f" % (pred == true_cat).all(axis=1).mean())
    print("confusion (rows=true 1..4, cols=pred 1..4):")
    for a in (1, 2, 3, 4):
        print("  ", [int(((true_cat == a) & (pred == b)).sum()) for b in (1, 2, 3, 4)])

    # ---- plan side: sweep scale (changes redundancy weighting) for top1 / meanQ^2
    print("\n--- plan selection oracle ---")
    C = abscorr(R)
    red = (C[:, PLANS[:, 0], PLANS[:, 1]] + C[:, PLANS[:, 0], PLANS[:, 2]]
           + C[:, PLANS[:, 1], PLANS[:, 2]]) / 3.0                # (N,220)
    Dp = np.sqrt(sq[:, PLANS, :].mean(axis=2))                    # (N,220,15)
    true_idx = np.array([np.where((PLANS == s).all(axis=1))[0][0] for s in sched])
    print("%7s | %7s | %8s | %8s" % ("scale", "top1", "meanQ^2", "PlanScore"))
    for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        u = (Dp.min(axis=2) + 0.18 * Dp.mean(axis=2)) * s - 0.035 * red
        top1 = (u.argmax(axis=1) == true_idx).mean()
        q = (u <= u[rows, true_idx][:, None]).sum(axis=1) / 220.0
        mq = (q ** 2).mean()
        print("%7.2f | %7.4f | %8.4f | %8.4f" % (s, top1, mq, (mq - CHANCE) / (1 - CHANCE)))


if __name__ == "__main__":
    main()
