"""
Targeted fix: correct the per-window scale (MAD) estimate.

Diagnosis: R = (P - med) / (1.4826 * MAD). We estimate MAD from noisy predictions P_hat, and
adding independent noise INFLATES a dispersion estimate. So MAD_hat is biased high => R_hat is
shrunk => d_hat too small. Evidence: the fitted global scale is 0.94 for the oracle but >1.0 for
every model run. A single global scale cannot fix it, because the inflation is relatively worst
where the true MAD is small -- and those windows dominate d.

Here we learn a correction MAD_hat -> MAD_true on out-of-fold predictions (we know the true MAD on
train), and check the downstream effect. Everything is fitted on train only.

Usage: python test_madfix.py <cache_dir>
"""
import sys, itertools
from pathlib import Path
import numpy as np

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
THRESH = (0.75, 1.25, 1.85)
CHANCE = 0.335609504132


def med_mad(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return m, mad


def catalog(P, mad_override=None):
    m, mad = med_mad(P)
    if mad_override is not None:
        mad = mad_override
    return np.transpose((P - m) / (1.4826 * mad + 1e-12), (0, 2, 1))


def sqdiff(R):
    return (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2


def abscorr(R):
    Rc = R - R.mean(2, keepdims=True)
    n = np.sqrt((Rc ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.einsum('nwe,nve->nwv', Rc, Rc) / (n[:, :, None] * n[:, None, :]))
    C[~np.isfinite(C)] = 1.0
    return C


def evaluate(R, sched, true_cat, tag):
    N = len(R); rows = np.arange(N)
    sq = sqdiff(R)
    d = np.sqrt(sq[rows[:, None], sched, :].mean(axis=1))
    best = max(((np.digitize(d * s, THRESH) + 1 == true_cat).mean(), s)
               for s in np.arange(0.5, 2.5, 0.01))
    acc, sb = best
    pred = np.digitize(d * sb, THRESH) + 1
    exact = (pred == true_cat).all(axis=1).mean()
    Dp = np.sqrt(sq[:, PLANS, :].mean(axis=2))
    C = abscorr(R)
    red = (C[:, PLANS[:, 0], PLANS[:, 1]] + C[:, PLANS[:, 0], PLANS[:, 2]]
           + C[:, PLANS[:, 1], PLANS[:, 2]]) / 3.0
    u = Dp.min(2) + 0.18 * Dp.mean(2) - 0.035 * red
    ti = np.array([np.where((PLANS == x).all(axis=1))[0][0] for x in sched])
    top1 = (u.argmax(1) == ti).mean()
    q = (u <= u[rows, ti][:, None]).sum(1) / 220.0
    ps = ((q ** 2).mean() - CHANCE) / (1 - CHANCE)
    mcs = 0.9 * np.clip((acc - 0.25) / 0.75, 0, 1) + 0.1 * exact
    ap = 0.55 * ps + 0.40 * mcs
    print("  %-16s entryAcc=%.4f(s=%.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f | approx=%.4f"
          % (tag, acc, sb, exact, top1, ps, mcs and ap))
    return ap


def main():
    cache = Path(sys.argv[1])
    P = np.load(cache / "P_train.npy").astype(np.float64)
    Ph = np.load(cache / "oof_P.npy").astype(np.float64)      # out-of-fold predictions (v1)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N = len(P)

    _, mad_true = med_mad(P)       # (N,1,12)
    _, mad_hat = med_mad(Ph)
    mt = mad_true[:, 0, :].ravel(); mh = mad_hat[:, 0, :].ravel()
    print("MAD true  : med %.5f  p10 %.5f  p90 %.5f" % (np.median(mt), *np.percentile(mt, [10, 90])))
    print("MAD hat   : med %.5f  p10 %.5f  p90 %.5f" % (np.median(mh), *np.percentile(mh, [10, 90])))
    print("inflation ratio med(mh/mt) = %.4f" % np.median(mh / (mt + 1e-12)))
    print("corr(log mh, log mt) = %.4f"
          % np.corrcoef(np.log(mh + 1e-9), np.log(mt + 1e-9))[0, 1])

    # split cases: fit the correction on half, evaluate on the other half
    rng = np.random.RandomState(0)
    perm = rng.permutation(N); a, b = perm[:N // 2], perm[N // 2:]

    print("\n=== baselines (evaluated on held-out half) ===")
    evaluate(catalog(P[b]), sched[b], true_cat[b], "ORACLE")
    evaluate(catalog(Ph[b]), sched[b], true_cat[b], "model, raw MAD")

    # --- correction 1: single multiplicative factor on MAD (fit on half a)
    ratio = np.median(mad_true[a] / (mad_hat[a] + 1e-12))
    print("\nfitted global MAD factor = %.4f" % ratio)
    evaluate(catalog(Ph[b], mad_override=mad_hat[b] * ratio), sched[b], true_cat[b],
             "MAD * const")

    # --- correction 2: variance deconvolution  mad_true^2 ~ mad_hat^2 - c
    resid = (P[a] - Ph[a])
    sigma2 = float((resid ** 2).mean())
    print("prediction noise var = %.3e  (sd %.5f)" % (sigma2, np.sqrt(sigma2)))
    best = None
    for c in np.linspace(0.0, 3.0, 61):
        corrected = np.sqrt(np.maximum(mad_hat[a] ** 2 - c * sigma2, (0.25 * mad_hat[a]) ** 2))
        err = np.abs(np.log(corrected + 1e-12) - np.log(mad_true[a] + 1e-12)).mean()
        if best is None or err < best[0]:
            best = (err, c)
    c_best = best[1]
    print("fitted deconvolution c = %.2f (log-MAD abs err %.4f)" % (c_best, best[0]))
    corr_b = np.sqrt(np.maximum(mad_hat[b] ** 2 - c_best * sigma2, (0.25 * mad_hat[b]) ** 2))
    evaluate(catalog(Ph[b], mad_override=corr_b), sched[b], true_cat[b], "MAD deconv")

    # --- correction 3: isotonic-style binned map log(mad_hat) -> log(mad_true)
    lh_a, lt_a = np.log(mad_hat[a].ravel() + 1e-12), np.log(mad_true[a].ravel() + 1e-12)
    edges = np.quantile(lh_a, np.linspace(0, 1, 41))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    idx = np.clip(np.digitize(lh_a, edges) - 1, 0, len(edges) - 2)
    means = np.array([lt_a[idx == k].mean() if (idx == k).sum() > 20 else np.nan
                      for k in range(len(edges) - 1)])
    ok = ~np.isnan(means)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lh_b = np.log(mad_hat[b].ravel() + 1e-12)
    mapped = np.interp(lh_b, centres[ok], means[ok])
    mad_map_b = np.exp(mapped).reshape(mad_hat[b].shape)
    evaluate(catalog(Ph[b], mad_override=mad_map_b), sched[b], true_cat[b], "MAD binned map")

    # --- correction 4: shrink MAD toward the case median MAD
    print()
    for lam in (0.15, 0.3, 0.5):
        tgt = np.median(mad_hat, axis=2, keepdims=True)
        shrunk = (1 - lam) * mad_hat[b] + lam * tgt[b]
        evaluate(catalog(Ph[b], mad_override=shrunk), sched[b], true_cat[b],
                 "MAD shrink %.2f" % lam)


if __name__ == "__main__":
    main()
