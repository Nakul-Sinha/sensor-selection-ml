"""
Extra tile features, to test whether the residual response error is a FEATURE gap rather than a
model-capacity gap.

Evidence for a feature gap: going from 500 -> 4000 boosting rounds moved valid l2 only
5.64e-5 -> 4.77e-5 and calibration R^2 only 0.9966 -> 0.9970. If capacity were the binding
constraint, more rounds would have kept paying. So the model probably cannot see something it needs.

The documented target combines "frequency centroid, low-to-middle and high-band balance, temporal
centroid, spectral roughness, and pulse contrast". The base set covers those in ONE parameterisation
each. These extras cover alternative parameterisations (finer bands, marginal quantiles, log-domain
stats, 2D joint structure), which is where a mismatch would hide.

Usage: python extra_features.py <public_dir> <out_dir> [--limit N]
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

EPS = 1e-8
NT, NF = 24, 24
_K = np.arange(NF, dtype=np.float64)
_T = np.arange(NT, dtype=np.float64)

EXTRA_NAMES = (
    ["band6_%d" % i for i in range(6)] +
    ["logband6_%d" % i for i in range(6)] +
    ["f_q10", "f_q25", "f_q50", "f_q75", "f_q90",
     "t_q10", "t_q25", "t_q50", "t_q75", "t_q90",
     "tf_cov", "tf_corr",
     "log_freq_centroid", "log_freq_spread", "log_spectral_entropy",
     "logF_roughness", "logG_roughness",
     "f_acf1", "f_acf2", "t_acf1", "t_acf2",
     "f_nmax", "t_nmax",
     "quad_tl", "quad_tr", "quad_bl", "quad_br",
     "row_energy_gini", "col_energy_gini",
     "hi_lo_centroid_gap", "peak_prom_f", "peak_prom_t"]
)
N_EXTRA = len(EXTRA_NAMES)


def _quantile_positions(p, qs):
    """For a normalized profile p (n,24), the bin index where the cdf first crosses each q."""
    cum = np.cumsum(p, axis=1)
    out = []
    for q in qs:
        out.append(np.argmax(cum >= q, axis=1).astype(np.float64))
    return out


def _gini(x):
    """Concentration of a non-negative profile (n,24)."""
    xs = np.sort(np.maximum(x, 0.0), axis=1)
    n = xs.shape[1]
    idx = np.arange(1, n + 1, dtype=np.float64)
    denom = n * xs.sum(axis=1) + EPS
    return (2.0 * (xs * idx).sum(axis=1)) / denom - (n + 1.0) / n


def extra_packet_features(packet):
    X = np.asarray(packet).reshape(72, NT, NF).astype(np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        Fm = X.sum(axis=1); Gm = X.sum(axis=2)
        sF = Fm.sum(axis=1) + EPS; sG = Gm.sum(axis=1) + EPS
        pf = Fm / sF[:, None]; pt = Gm / sG[:, None]

        cols = []
        # finer 6-band split of the frequency marginal
        b6 = [Fm[:, i * 4:(i + 1) * 4].sum(axis=1) / sF for i in range(6)]
        cols += b6
        cols += [np.log(b + EPS) for b in b6]

        # marginal quantile positions
        cols += _quantile_positions(pf, [0.10, 0.25, 0.50, 0.75, 0.90])
        cols += _quantile_positions(pt, [0.10, 0.25, 0.50, 0.75, 0.90])

        # joint time-frequency structure
        P2 = X / (X.sum(axis=(1, 2))[:, None, None] + EPS)
        mt = (P2.sum(axis=2) @ _T); mf = (P2.sum(axis=1) @ _K)
        TT, KK = np.meshgrid(_T, _K, indexing="ij")
        cov = (P2 * (TT[None] - mt[:, None, None]) * (KK[None] - mf[:, None, None])).sum(axis=(1, 2))
        st = np.sqrt((P2.sum(axis=2) * (_T[None] - mt[:, None]) ** 2).sum(axis=1))
        sf_ = np.sqrt((P2.sum(axis=1) * (_K[None] - mf[:, None]) ** 2).sum(axis=1))
        cols += [cov, cov / (st * sf_ + EPS)]

        # log-domain spectral shape (the target may combine log quantities)
        Lf = np.log(Fm + EPS); Lg = np.log(Gm + EPS)
        pl = np.exp(Lf); pl = pl / (pl.sum(axis=1, keepdims=True) + EPS)
        lc = pl @ _K
        cols += [lc,
                 np.sqrt((pl * (_K[None] - lc[:, None]) ** 2).sum(axis=1)),
                 -(pl * np.log(pl + EPS)).sum(axis=1),
                 np.abs(np.diff(Lf, axis=1)).mean(axis=1),
                 np.abs(np.diff(Lg, axis=1)).mean(axis=1)]

        # autocorrelation of the marginals
        def acf(p, lag):
            a = p - p.mean(axis=1, keepdims=True)
            num = (a[:, :-lag] * a[:, lag:]).sum(axis=1)
            den = (a * a).sum(axis=1) + EPS
            return num / den
        cols += [acf(Fm, 1), acf(Fm, 2), acf(Gm, 1), acf(Gm, 2)]

        # ruggedness: count of interior local maxima
        cols += [((Fm[:, 1:-1] > Fm[:, :-2]) & (Fm[:, 1:-1] > Fm[:, 2:])).sum(axis=1).astype(float),
                 ((Gm[:, 1:-1] > Gm[:, :-2]) & (Gm[:, 1:-1] > Gm[:, 2:])).sum(axis=1).astype(float)]

        # quadrant energy shares
        tot = X.sum(axis=(1, 2)) + EPS
        cols += [X[:, :12, :12].sum(axis=(1, 2)) / tot, X[:, :12, 12:].sum(axis=(1, 2)) / tot,
                 X[:, 12:, :12].sum(axis=(1, 2)) / tot, X[:, 12:, 12:].sum(axis=(1, 2)) / tot]

        # concentration of the two marginals
        cols += [_gini(Gm), _gini(Fm)]

        # gap between high-band and low-band centroids
        Fh = Fm[:, 12:]; Fl = Fm[:, :12]
        ch = (Fh @ _K[12:]) / (Fh.sum(axis=1) + EPS)
        cl = (Fl @ _K[:12]) / (Fl.sum(axis=1) + EPS)
        cols += [ch - cl]

        # peak prominence
        cols += [(Fm.max(axis=1) - np.median(Fm, axis=1)) / (Fm.std(axis=1) + EPS),
                 (Gm.max(axis=1) - np.median(Gm, axis=1)) / (Gm.std(axis=1) + EPS)]

    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == N_EXTRA, (out.shape, N_EXTRA)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _chunk(public_dir, paths):
    o = np.empty((len(paths), 72, N_EXTRA), dtype=np.float32)
    b = Path(public_dir)
    for i, p in enumerate(paths):
        o[i] = extra_packet_features(np.load(b / p))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir"); ap.add_argument("out_dir")
    ap.add_argument("--n-jobs", type=int, default=10)
    a = ap.parse_args()
    pub = Path(a.public_dir); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("extra features:", N_EXTRA)
    for split in ("train", "test"):
        df = pd.read_csv(pub / f"{split}.csv")
        paths = df["response_packet_path"].tolist()
        ch = max(32, len(paths) // (a.n_jobs * 4))
        parts = [paths[i:i + ch] for i in range(0, len(paths), ch)]
        t0 = time.time()
        res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(_chunk)(str(pub), p) for p in parts)
        F = np.concatenate(res, axis=0)
        np.save(out / f"Xe_{split}.npy", F)
        print(split, F.shape, "%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
