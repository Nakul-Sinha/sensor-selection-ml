"""
Stage A v3: predict the STANDARDIZED response R[w,e] directly (clipped), letting the model learn
the /MAD normalization itself.

Motivation: d is dominated by low-MAD windows, where dividing a small absolute error by a small MAD
amplifies it. Predicting the post-normalization quantity directly optimizes what we actually use.

Also fits a separate per-window log-MAD model (v4) as an alternative route.

Usage: python model_response3.py <cache_dir> [n_folds] [--clip C]
"""
import sys, time, itertools
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
THRESH = (0.75, 1.25, 1.85)
CHANCE = 0.335609504132
EPS = 1e-12


def med_mad_parts(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return m, mad


def med_mad_std(P):
    m, mad = med_mad_parts(P)
    return (P - m) / (1.4826 * mad + EPS)


def sqdiff(R):
    return (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2


def abscorr(R):
    Rc = R - R.mean(2, keepdims=True)
    n = np.sqrt((Rc ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.einsum('nwe,nve->nwv', Rc, Rc) / (n[:, :, None] * n[:, None, :]))
    C[~np.isfinite(C)] = 1.0
    return C


def eval_R(R, sched, true_cat, tag):
    N = R.shape[0]; rows = np.arange(N)
    sq = sqdiff(R)
    d = np.sqrt(sq[rows[:, None], sched, :].mean(axis=1))
    best = (-1, 1.0)
    for s in np.arange(0.50, 2.00, 0.01):
        a = (np.digitize(d * s, THRESH) + 1 == true_cat).mean()
        if a > best[0]:
            best = (a, s)
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
    print("  %-9s entryAcc=%.4f(s=%.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f | MCS=%.4f "
          "| approx=%.4f" % (tag, acc, sb, exact, top1, ps, mcs, 0.55 * ps + 0.40 * mcs))
    return 0.55 * ps + 0.40 * mcs


def build_features(X):
    N, T, F = X.shape
    Xg = X.reshape(N, 6, 12, F)
    mu_w = Xg.mean(1, keepdims=True); sd_w = Xg.std(1, keepdims=True) + 1e-6
    med_w = np.median(Xg, axis=1, keepdims=True)
    mad_w = np.median(np.abs(Xg - med_w), axis=1, keepdims=True)
    Xc = Xg - mu_w
    Xz = Xc / sd_w
    Xr = (Xg - med_w) / (1.4826 * mad_w + 1e-6)          # mirrors the target's own normalization
    Xsd = np.broadcast_to(sd_w, Xg.shape)                 # window-level dispersion as context
    out = np.concatenate([Xg, Xc, Xz, Xr, Xsd], axis=3).reshape(N, T, -1)
    return np.ascontiguousarray(np.nan_to_num(out, posinf=0, neginf=0), dtype=np.float32)


def main():
    cache = Path(sys.argv[1])
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 3
    clip = 8.0
    if "--clip" in sys.argv:
        clip = float(sys.argv[sys.argv.index("--clip") + 1])

    X0 = np.load(cache / "X_train.npy")
    P = np.load(cache / "P_train.npy").astype(np.float64)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N = len(P)

    X = build_features(X0)
    Xf = X.reshape(N * 72, -1)
    groups = np.repeat(np.arange(N), 72)
    print("features:", X.shape, " clip:", clip)

    Z = med_mad_std(P)                                    # (N,6,12) target
    print("target R: std=%.3f  |R|>8: %.3f%%  max=%.1f"
          % (Z.std(), 100 * (np.abs(Z) > 8).mean(), np.abs(Z).max()))
    y = np.clip(Z, -clip, clip).reshape(-1)

    params = dict(objective="regression", metric="l2", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1)
    oof = np.zeros(N * 72)
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(Xf, y, groups)):
        ds = lgb.Dataset(Xf[tr], y[tr]); dv = lgb.Dataset(Xf[va], y[va], reference=ds)
        bst = lgb.train(params, ds, num_boost_round=900, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = bst.predict(Xf[va], num_iteration=bst.best_iteration)
        print("  fold %d it=%d" % (k, bst.best_iteration))
    print("fit %.1f s" % (time.time() - t0))

    Zp = oof.reshape(N, 6, 12)
    yc = y.reshape(N, 6, 12)
    print("direct-R R^2 (vs clipped target) = %.5f  corr = %.5f"
          % (1 - ((yc - Zp) ** 2).sum() / ((yc - yc.mean()) ** 2).sum(),
             np.corrcoef(yc.ravel(), Zp.ravel())[0, 1]))

    print("\n=== downstream ===")
    eval_R(np.transpose(Z, (0, 2, 1)), sched, true_cat, "ORACLE")
    eval_R(np.transpose(Zp, (0, 2, 1)), sched, true_cat, "DIRECT")
    # re-standardizing the direct prediction sometimes helps
    eval_R(np.transpose(med_mad_std(Zp), (0, 2, 1)), sched, true_cat, "DIRECT+std")
    np.save(cache / "oof_directR.npy", Zp)
    print("saved oof_directR.npy")


if __name__ == "__main__":
    main()
