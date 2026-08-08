"""
Hypothesis: packet_response_matrix is a SMOOTH (near-linear) combination of a handful of named tile
statistics -- "frequency centroid, low-to-middle and high-band balance, temporal centroid, spectral
roughness, and pulse contrast". If so, a linear/polynomial model will beat a GBM by orders of
magnitude on precision, because trees are piecewise-constant and the quantity that matters
(the within-window contrast, scale ~0.02) needs very high precision.

Usage: python model_linear.py <cache_dir> [--poly]
"""
import sys, time, itertools
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES  # noqa

PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
EPS = 1e-12


def med_mad_std(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return (P - m) / (1.4826 * mad + EPS)


def report(name, y, pred, contrast_true, contrast_pred):
    ssr = ((y - pred) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ssr / sst
    cr = 1 - ((contrast_true - contrast_pred) ** 2).sum() / \
             ((contrast_true - contrast_true.mean()) ** 2).sum()
    print("  %-26s  P_R2=%.6f  RMSE=%.3e | CONTRAST_R2=%.6f" % (name, r2, np.sqrt(ssr / len(y)), cr))
    return cr


def main():
    cache = Path(sys.argv[1])
    X = np.load(cache / "X_train.npy").astype(np.float64)      # (N,72,F)
    P = np.load(cache / "P_train.npy").astype(np.float64)      # (N,6,12)
    N, T, F = X.shape
    y = P.reshape(N, 72).reshape(-1)
    Xf = X.reshape(N * 72, F)
    groups = np.repeat(np.arange(N), 72)

    # sanitize -> WINSORIZE (critical: raw features like total_energy / pulse_contrast have
    # extreme tails that destroy the conditioning of the normal equations) -> standardize
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(Xf, 0.1, axis=0)
    hi = np.percentile(Xf, 99.9, axis=0)
    Xf = np.clip(Xf, lo, hi)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xs = (Xf - mu) / sd
    keep = sd > 1e-8
    Xs = Xs[:, keep]
    print("kept %d/%d non-degenerate features after winsorizing" % (keep.sum(), F))

    def contrast_of(v):
        vv = v.reshape(N, 6, 12)
        return (vv - vv.mean(axis=1, keepdims=True)).reshape(-1)

    ct = contrast_of(y)
    print("cases %d  features %d  target std %.5f  contrast std %.5f"
          % (N, F, y.std(), ct.std()))

    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(Xs, y, groups))

    def cv_ridge(Xmat, tag, alphas=(1e-2, 1.0, 10.0, 100.0, 1000.0)):
        best = None
        for a in alphas:
            oof = np.zeros(len(y))
            for tr, va in splits:
                m = Ridge(alpha=a, fit_intercept=True)
                m.fit(Xmat[tr], y[tr])
                oof[va] = m.predict(Xmat[va])
            cr = report("%s a=%g" % (tag, a), y, oof, ct, contrast_of(oof))
            if best is None or cr > best[0]:
                best = (cr, a, oof)
        return best

    print("\n--- linear on 59 raw features ---")
    t0 = time.time()
    best_lin = cv_ridge(Xs, "ridge_raw")
    print("  (%.1fs)" % (time.time() - t0))

    # squares + interactions of the most predictive features
    print("\n--- linear + squares of all features ---")
    Xq = np.hstack([Xs, Xs ** 2])
    best_q = cv_ridge(Xq, "ridge_sq")

    if "--poly" in sys.argv:
        print("\n--- linear + squares + pairwise interactions of top-16 by |corr| ---")
        c = np.array([abs(np.corrcoef(Xs[::17, i], y[::17])[0, 1]) for i in range(F)])
        c = np.nan_to_num(c)
        top = np.argsort(-c)[:16]
        print("  top features:", [FEATURE_NAMES[i] for i in top[:8]], "...")
        inter = []
        for i in range(len(top)):
            for j in range(i, len(top)):
                inter.append(Xs[:, top[i]] * Xs[:, top[j]])
        Xp = np.hstack([Xs, Xs ** 2, np.column_stack(inter)])
        print("  design matrix:", Xp.shape)
        best_p = cv_ridge(Xp, "ridge_poly")
        np.save(cache / "oof_linear.npy", best_p[2].reshape(N, 6, 12))
        print("saved oof_linear.npy (poly)")
    else:
        np.save(cache / "oof_linear.npy", best_q[2].reshape(N, 6, 12))
        print("saved oof_linear.npy (sq)")


if __name__ == "__main__":
    main()
