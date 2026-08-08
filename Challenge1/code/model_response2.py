"""
Stage A v2: predict the WITHIN-WINDOW CONTRAST directly.

Why: the target R is  (P - median_e P) / (1.4826 * MAD_e P), standardized over the 6 episodes
within each window. Raw P is dominated by across-window level (easy, R^2=0.997) while the
within-window contrast -- the only thing that survives standardization -- has scale ~0.017 and was
swamped by a 0.0074 RMSE.

Two changes:
  1. TARGET  = P - mean over the 6 episodes at the same window  (the contrast itself)
  2. FEATURES gain window-relative blocks: each tile stat minus / standardized by the same window's
     stats across its 6 episodes. This mirrors the target's construction.

Usage: python model_response2.py <cache_dir> [n_folds] [--eprel]
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


def med_mad_std(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return (P - m) / (1.4826 * mad + EPS)


def catalog(P):
    return np.transpose(med_mad_std(P), (0, 2, 1))


def sqdiff(R):
    return (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2


def abscorr(R):
    Rc = R - R.mean(2, keepdims=True)
    n = np.sqrt((Rc ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.einsum('nwe,nve->nwv', Rc, Rc) / (n[:, :, None] * n[:, None, :]))
    C[~np.isfinite(C)] = 1.0
    return C


def utilities(R):
    sq = sqdiff(R)
    Dp = np.sqrt(sq[:, PLANS, :].mean(axis=2))
    C = abscorr(R)
    red = (C[:, PLANS[:, 0], PLANS[:, 1]] + C[:, PLANS[:, 0], PLANS[:, 2]]
           + C[:, PLANS[:, 1], PLANS[:, 2]]) / 3.0
    return Dp.min(2) + 0.18 * Dp.mean(2) - 0.035 * red


def eval_block(P, sched, true_cat, tag):
    N = len(P); rows = np.arange(N)
    R = catalog(P)
    d = np.sqrt(sqdiff(R)[rows[:, None], sched, :].mean(axis=1))
    best = (-1, 1.0)
    for s in np.arange(0.60, 1.60, 0.01):
        a = (np.digitize(d * s, THRESH) + 1 == true_cat).mean()
        if a > best[0]:
            best = (a, s)
    acc, s_best = best
    pred = np.digitize(d * s_best, THRESH) + 1
    exact = (pred == true_cat).all(axis=1).mean()
    u = utilities(R)
    true_idx = np.array([np.where((PLANS == x).all(axis=1))[0][0] for x in sched])
    top1 = (u.argmax(1) == true_idx).mean()
    q = (u <= u[rows, true_idx][:, None]).sum(1) / 220.0
    ps = ((q ** 2).mean() - CHANCE) / (1 - CHANCE)
    mcs = 0.9 * np.clip((acc - 0.25) / 0.75, 0, 1) + 0.1 * exact
    print("  %-8s entryAcc=%.4f(s=%.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f | MCS=%.4f "
          "| approx=%.4f" % (tag, acc, s_best, exact, top1, ps, mcs, 0.55 * ps + 0.40 * mcs))
    return acc, top1, ps


def build_features(X, use_eprel=False):
    """X (N,72,F) -> (N,72,F*k) with window-relative blocks appended."""
    N, T, F = X.shape
    Xg = X.reshape(N, 6, 12, F)                       # episode, window
    mu_w = Xg.mean(axis=1, keepdims=True)             # over the 6 episodes at each window
    sd_w = Xg.std(axis=1, keepdims=True) + 1e-6
    Xc = Xg - mu_w                                    # centered contrast
    Xz = Xc / sd_w                                    # standardized contrast
    blocks = [Xg, Xc, Xz]
    if use_eprel:
        mu_e = Xg.mean(axis=2, keepdims=True)         # over the 12 windows at each episode
        sd_e = Xg.std(axis=2, keepdims=True) + 1e-6
        blocks.append((Xg - mu_e) / sd_e)
    out = np.concatenate(blocks, axis=3).reshape(N, T, -1)
    return np.ascontiguousarray(out, dtype=np.float32)


def main():
    cache = Path(sys.argv[1])
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 3
    use_eprel = "--eprel" in sys.argv

    X0 = np.load(cache / "X_train.npy")
    P = np.load(cache / "P_train.npy").astype(np.float64)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N = len(P)

    X = build_features(X0, use_eprel)
    print("features:", X.shape, " eprel:", use_eprel)

    # TARGET = within-window contrast
    Pmean_w = P.mean(axis=1, keepdims=True)           # (N,1,12)
    Ccon = P - Pmean_w                                 # (N,6,12)
    y = Ccon.reshape(N, 72).reshape(-1)
    Xf = X.reshape(N * 72, -1)
    groups = np.repeat(np.arange(N), 72)
    print("contrast target: std=%.6f  mad=%.6f" % (Ccon.std(), np.median(np.abs(Ccon))))

    oof = np.zeros(N * 72)
    params = dict(objective="regression", metric="l2", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1)
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(Xf, y, groups)):
        ds = lgb.Dataset(Xf[tr], y[tr]); dv = lgb.Dataset(Xf[va], y[va], reference=ds)
        bst = lgb.train(params, ds, num_boost_round=1200, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = bst.predict(Xf[va], num_iteration=bst.best_iteration)
        print("  fold %d best_iter=%d" % (k, bst.best_iteration))
    print("fit %.1f s" % (time.time() - t0))

    ssr = ((y - oof) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    print("\nCONTRAST R^2 = %.6f  RMSE = %.6f  (target std %.6f)"
          % (1 - ssr / sst, np.sqrt(ssr / len(y)), y.std()))

    P_hat = oof.reshape(N, 6, 12)     # already a contrast; standardization is shift-invariant
    Zt = med_mad_std(P); Zp = med_mad_std(P_hat)
    ok = np.isfinite(Zt) & np.isfinite(Zp)
    print("STD R R^2 = %.6f   corr = %.6f"
          % (1 - ((Zt[ok] - Zp[ok]) ** 2).sum() / ((Zt[ok] - Zt[ok].mean()) ** 2).sum(),
             np.corrcoef(Zt[ok], Zp[ok])[0, 1]))

    print("\n=== downstream ===")
    eval_block(P, sched, true_cat, "ORACLE")
    eval_block(P_hat, sched, true_cat, "MODEL2")
    np.save(cache / "oof_contrast.npy", P_hat)
    print("saved oof_contrast.npy")


if __name__ == "__main__":
    main()
