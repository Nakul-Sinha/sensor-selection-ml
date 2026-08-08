"""
Stage A: tile -> packet_response regression, then measure the DOWNSTREAM damage.

Reports both:
  * raw R^2 on P
  * R^2 on the within-window standardized response (the quantity that actually matters, since
    per-window offsets/scales cancel in the median/MAD standardization)
  * downstream matrix entry accuracy and plan top-1 using out-of-fold predictions,
    side by side with the oracle (true P).

Usage: python model_response.py <cache_dir> [n_folds] [--rel]
"""
import sys, time, itertools
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
THRESH = (0.75, 1.25, 1.85)
CHANCE = 0.335609504132
EPS = 1e-12


def med_mad_std(P):
    """P (N,6,12) -> standardized over the 6 episodes within each window."""
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
    return (P - m) / (1.4826 * mad + EPS)


def catalog(P):
    return np.transpose(med_mad_std(P), (0, 2, 1))          # (N,12,6)


def sqdiff(R):
    return (R[:, :, PAIRS[:, 0]] - R[:, :, PAIRS[:, 1]]) ** 2   # (N,12,15)


def abscorr(R):
    Rc = R - R.mean(2, keepdims=True)
    n = np.sqrt((Rc ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.einsum('nwe,nve->nwv', Rc, Rc) / (n[:, :, None] * n[:, None, :]))
    C[~np.isfinite(C)] = 1.0
    return C


def utilities(R, scale=1.0):
    sq = sqdiff(R)
    Dp = np.sqrt(sq[:, PLANS, :].mean(axis=2))              # (N,220,15)
    C = abscorr(R)
    red = (C[:, PLANS[:, 0], PLANS[:, 1]] + C[:, PLANS[:, 0], PLANS[:, 2]]
           + C[:, PLANS[:, 1], PLANS[:, 2]]) / 3.0
    return (Dp.min(2) + 0.18 * Dp.mean(2)) * scale - 0.035 * red


def eval_block(P, sched, true_cat, tag, scale_grid=np.arange(0.70, 1.40, 0.01)):
    N = len(P); rows = np.arange(N)
    R = catalog(P)
    sq = sqdiff(R)
    d = np.sqrt(sq[rows[:, None], sched, :].mean(axis=1))    # (N,15) for the TRUE plan
    best = (-1, 1.0)
    for s in scale_grid:
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
    mq = (q ** 2).mean()
    plan_score = (mq - CHANCE) / (1 - CHANCE)
    A_corr = np.clip((acc - 0.25) / 0.75, 0, 1)
    mcs = 0.9 * A_corr + 0.1 * exact
    approx = 0.55 * plan_score + 0.40 * mcs
    print("  %-10s entryAcc=%.4f (scale %.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f "
          "| MatrixCert=%.4f | approxScore=%.4f"
          % (tag, acc, s_best, exact, top1, plan_score, mcs, approx))
    return acc, s_best, top1, plan_score, approx


def main():
    cache = Path(sys.argv[1])
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 4
    use_rel = "--rel" in sys.argv

    X = np.load(cache / "X_train.npy")            # (N,72,F)
    P = np.load(cache / "P_train.npy").astype(np.float64)   # (N,6,12)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N, T, F = X.shape
    print("X", X.shape, "P", P.shape, "rel_features:", use_rel)

    if use_rel:
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True) + 1e-8
        X = np.concatenate([X, (X - mu) / sd], axis=2)
        F = X.shape[2]
        print("expanded to", X.shape)

    y = P.reshape(N, 72)                          # episode-major: e*12+w  matches feature rows
    Xf = X.reshape(N * 72, F)
    yf = y.reshape(N * 72)
    groups = np.repeat(np.arange(N), 72)

    oof = np.zeros(N * 72, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_folds)
    params = dict(objective="regression", metric="l2", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=40, feature_fraction=0.85,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1)
    t0 = time.time()
    for k, (tr, va) in enumerate(gkf.split(Xf, yf, groups)):
        ds = lgb.Dataset(Xf[tr], yf[tr])
        dv = lgb.Dataset(Xf[va], yf[va], reference=ds)
        bst = lgb.train(params, ds, num_boost_round=900, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = bst.predict(Xf[va], num_iteration=bst.best_iteration)
        print("  fold %d best_iter=%d" % (k, bst.best_iteration))
    print("fit %.1f s" % (time.time() - t0))

    ss_res = ((yf - oof) ** 2).sum(); ss_tot = ((yf - yf.mean()) ** 2).sum()
    print("\nRAW  P   R^2 = %.6f   RMSE = %.6f" % (1 - ss_res / ss_tot, np.sqrt(ss_res / len(yf))))

    P_hat = oof.reshape(N, 6, 12)
    Zt = med_mad_std(P);      Zp = med_mad_std(P_hat)
    ok = np.isfinite(Zt) & np.isfinite(Zp)
    ss_res2 = ((Zt[ok] - Zp[ok]) ** 2).sum(); ss_tot2 = ((Zt[ok] - Zt[ok].mean()) ** 2).sum()
    print("STD  R   R^2 = %.6f   (this is what actually matters)" % (1 - ss_res2 / ss_tot2))
    print("corr(std true, std pred) = %.6f" % np.corrcoef(Zt[ok], Zp[ok])[0, 1])

    print("\n=== downstream ===")
    eval_block(P, sched, true_cat, "ORACLE")
    eval_block(P_hat, sched, true_cat, "MODEL")

    np.save(cache / ("oof_P%s.npy" % ("_rel" if use_rel else "")), P_hat)
    print("\nsaved oof predictions")


if __name__ == "__main__":
    main()
