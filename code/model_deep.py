"""
Find the PLATEAU: every previous LightGBM run hit its round cap, so all reported numbers were
undertrained. Run a single grouped split with a large round budget and watch where validation
stops improving, for both the 59-feature and the window-relative feature sets.

Also reports the error-amplification diagnostic: how much of the standardized-R error is
concentrated in low-MAD windows.

Usage: python model_deep.py <cache_dir> [--rounds N] [--wide] [--lr X]
"""
import sys, time, itertools
from pathlib import Path
import numpy as np
import lightgbm as lgb

PLANS = np.array(list(itertools.combinations(range(12), 3)), dtype=np.int64)
PAIRS = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
THRESH = (0.75, 1.25, 1.85)
CHANCE = 0.335609504132
EPS = 1e-12


def med_mad_std(P):
    m = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - m), axis=1, keepdims=True)
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


def eval_P(P, sched, true_cat, tag):
    N = len(P); rows = np.arange(N)
    R = np.transpose(med_mad_std(P), (0, 2, 1))
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
    ap = 0.55 * ps + 0.40 * mcs
    print("  %-9s entryAcc=%.4f(s=%.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f | MCS=%.4f "
          "| approx=%.4f" % (tag, acc, sb, exact, top1, ps, mcs, ap))
    return ap


def main():
    cache = Path(sys.argv[1])
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 6000
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.05
    wide = "--wide" in sys.argv

    X = np.load(cache / "X_train.npy").astype(np.float32)
    P = np.load(cache / "P_train.npy").astype(np.float64)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N, T, F = X.shape

    if wide:
        Xg = X.reshape(N, 6, 12, F)
        med_w = np.median(Xg, axis=1, keepdims=True)
        mad_w = np.median(np.abs(Xg - med_w), axis=1, keepdims=True)
        Xr = (Xg - med_w) / (1.4826 * mad_w + 1e-6)
        Xc = Xg - Xg.mean(1, keepdims=True)
        X = np.concatenate([Xg, Xc, Xr], axis=3).reshape(N, T, -1)
        X = np.nan_to_num(X, posinf=0, neginf=0).astype(np.float32)
    print("design:", X.shape, "rounds", rounds, "lr", lr)

    Xf = X.reshape(N * 72, -1)
    y = P.reshape(-1)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    tr_cases, va_cases = perm[:int(0.75 * N)], perm[int(0.75 * N):]
    tile_case = np.repeat(np.arange(N), 72)
    tr = np.isin(tile_case, tr_cases); va = ~tr

    params = dict(objective="regression", metric="l2", learning_rate=lr,
                  num_leaves=511, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.5,
                  num_threads=10, verbose=-1, seed=0)
    ds = lgb.Dataset(Xf[tr], y[tr])
    dv = lgb.Dataset(Xf[va], y[va], reference=ds)
    hist = {}
    t0 = time.time()
    bst = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[ds, dv],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(200, verbose=False),
                               lgb.log_evaluation(500),
                               lgb.record_evaluation(hist)])
    print("fit %.0fs  best_iter=%d" % (time.time() - t0, bst.best_iteration))

    pv = bst.predict(Xf[va], num_iteration=bst.best_iteration)
    yv = y[va]
    rmse = np.sqrt(((yv - pv) ** 2).mean())
    print("valid P R2 = %.6f  RMSE = %.6f" % (1 - ((yv - pv) ** 2).sum() /
                                              ((yv - yv.mean()) ** 2).sum(), rmse))

    P_hat = P.copy(); P_hat.reshape(-1)[va] = pv
    Pv, Phv = P[va_cases], P_hat[va_cases]
    Zt, Zp = med_mad_std(Pv), med_mad_std(Phv)
    ok = np.isfinite(Zt) & np.isfinite(Zp)
    print("STD R R2 = %.5f  corr = %.5f"
          % (1 - ((Zt[ok] - Zp[ok]) ** 2).sum() / ((Zt[ok] - Zt[ok].mean()) ** 2).sum(),
             np.corrcoef(Zt[ok], Zp[ok])[0, 1]))

    # error amplification diagnostic
    mad = np.median(np.abs(Pv - np.median(Pv, axis=1, keepdims=True)), axis=1)   # (n,12)
    err = np.abs(Zt - Zp).mean(axis=1)                                           # (n,12)
    qs = np.quantile(mad, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(mad, qs)
    print("\nMAD quintile -> mean |R error| :",
          ["%.3f" % err[b == i].mean() for i in range(5)])
    print("MAD quintile medians          :", ["%.4f" % np.median(mad[b == i]) for i in range(5)])

    print("\n=== downstream on %d held-out cases ===" % len(va_cases))
    eval_P(Pv, sched[va_cases], true_cat[va_cases], "ORACLE")
    eval_P(Phv, sched[va_cases], true_cat[va_cases], "GBM")


if __name__ == "__main__":
    main()
