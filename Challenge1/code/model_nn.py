"""
Stage A v4: neural net for the tile -> response map.

Rationale: packet_response_matrix is a SMOOTH deterministic function of one tile. GBMs are
piecewise-constant, which caps their precision -- and precision is exactly what we need, because
the quantity that survives standardization is a within-window contrast of scale ~0.03 while the
GBM's RMSE is ~0.0074. An MLP can represent a smooth function far more precisely.

Single train/valid split (grouped by case) for fast iteration.

Usage: python model_nn.py <cache_dir> [--epochs N] [--raw] [--hidden H]
"""
import sys, time, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

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
    print("  %-9s entryAcc=%.4f(s=%.2f) exactM=%.4f | top1=%.4f PlanScore=%.4f | MCS=%.4f "
          "| approx=%.4f" % (tag, acc, sb, exact, top1, ps, mcs, 0.55 * ps + 0.40 * mcs))
    return 0.55 * ps + 0.40 * mcs


def main():
    cache = Path(sys.argv[1])
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 40
    hidden = int(sys.argv[sys.argv.index("--hidden") + 1]) if "--hidden" in sys.argv else 384
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(10)

    X = np.load(cache / "X_train.npy").astype(np.float64)     # (N,72,F)
    P = np.load(cache / "P_train.npy").astype(np.float64)
    sched = np.load(cache / "sched_train.npy").astype(np.int64)
    M = np.load(cache / "M_train.npy").astype(np.int64)
    true_cat = M[:, PAIRS[:, 0], PAIRS[:, 1]]
    N, T, F = X.shape

    # window-relative context blocks (same idea as v2 but the TARGET stays per-tile P)
    Xg = X.reshape(N, 6, 12, F)
    med_w = np.median(Xg, axis=1, keepdims=True)
    mad_w = np.median(np.abs(Xg - med_w), axis=1, keepdims=True)
    Xr = (Xg - med_w) / (1.4826 * mad_w + 1e-6)
    Xc = Xg - Xg.mean(1, keepdims=True)
    Xall = np.concatenate([Xg, Xc, Xr], axis=3).reshape(N, T, -1)
    Xall = np.nan_to_num(Xall, posinf=0, neginf=0)

    Xf = Xall.reshape(N * 72, -1)
    y = P.reshape(-1)

    # robust preprocessing: winsorize then z-score (NNs need bounded inputs)
    lo = np.percentile(Xf, 0.5, axis=0); hi = np.percentile(Xf, 99.5, axis=0)
    Xf = np.clip(Xf, lo, hi)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xf = (Xf - mu) / sd
    ymu, ysd = y.mean(), y.std()
    yn = (y - ymu) / ysd
    print("design:", Xf.shape, " epochs", epochs, " hidden", hidden)

    ncase = N
    rng = np.random.RandomState(0)
    perm = rng.permutation(ncase)
    n_tr = int(0.75 * ncase)
    tr_cases, va_cases = perm[:n_tr], perm[n_tr:]
    tile_of = np.repeat(np.arange(ncase), 72)
    tr = np.isin(tile_of, tr_cases); va = ~tr

    Xt = torch.tensor(Xf[tr], dtype=torch.float32)
    yt = torch.tensor(yn[tr], dtype=torch.float32)
    Xv = torch.tensor(Xf[va], dtype=torch.float32)

    D = Xf.shape[1]
    model = nn.Sequential(
        nn.Linear(D, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, hidden // 2), nn.GELU(),
        nn.Linear(hidden // 2, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched_lr = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=epochs * ((len(Xt) + 4095) // 4096))
    lossf = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 4096):
            b = idx[i:i + 4096]
            opt.zero_grad()
            out = model(Xt[b]).squeeze(1)
            loss = lossf(out, yt[b])
            loss.backward(); opt.step(); sched_lr.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xv).squeeze(1).numpy()
            pv = pv * ysd + ymu
            yv = y[va]
            r2 = 1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
            print("  ep %2d  train_mse %.5f  valid P_R2 %.6f  rmse %.5f  (%.0fs)"
                  % (ep + 1, tot / len(Xt), r2, np.sqrt(((yv - pv) ** 2).mean()), time.time() - t0))

    model.eval()
    with torch.no_grad():
        pv = model(Xv).squeeze(1).numpy() * ysd + ymu
    P_hat = P.copy()
    P_hat.reshape(-1)[va] = pv

    Pv = P[va_cases]; Phv = P_hat[va_cases]
    sv, cv = sched[va_cases], true_cat[va_cases]
    print("\n=== downstream on %d held-out cases ===" % len(va_cases))
    eval_P(Pv, sv, cv, "ORACLE")
    eval_P(Phv, sv, cv, "NN")
    np.save(cache / "nn_valid_P.npy", Phv)
    np.save(cache / "nn_valid_cases.npy", va_cases)


if __name__ == "__main__":
    main()
