"""Fast, fully-vectorized feature extraction for 24x24 time-frequency tiles.

Packet layout
-------------
A "packet" is an array of shape ``(6, 12, 24, 24)``:

* axis 0 -> 6 episodes
* axis 1 -> 12 windows
* axis 2 -> 24 TIME bins
* axis 3 -> 24 FREQUENCY bins

so a packet holds ``6 * 12 = 72`` tiles, each a ``(24 time, 24 freq)`` array.
``packet_features`` returns one feature row per tile in EPISODE-MAJOR order:
tile ``packet[e, w]`` lands in row ``e * 12 + w``.  That is exactly the
ordering produced by ``packet.reshape(72, 24, 24)`` (verified by the tests).

Target / degradation context
----------------------------
The per-tile scalar to be predicted summarizes the visible frequency centroid,
low-to-middle vs high band balance, temporal centroid, spectral roughness and
pulse contrast of the tile.  Tiles are degraded by multiplicative contrast
scaling, circular time rolls, additive noise and occasional local dropouts, so
this set emphasizes SCALE-INVARIANT statistics (normalized-marginal moments,
band ratios, entropies, flatness, crest/contrast factors, sorted-profile
shares, and circular time statistics that are additionally ROLL-invariant),
while keeping raw-scale statistics as well.

Conventions
-----------
* Input may be any float dtype; it is converted to float64 before any
  arithmetic (float16 accumulation over 576 values would overflow/underflow).
* ``EPS = 1e-8`` guards every division and logarithm.
* skew / kurtosis are the plain standardized moments ``m3 / sigma^3`` and
  ``m4 / sigma^4`` (Pearson convention, no ``-3`` excess correction).
* Every statistic is computed with numpy axis operations over the whole
  ``(72, 24, 24)`` block at once -- there is no Python loop over tiles.
* Degenerate tiles (all-zero, constant) yield finite values; any residual
  nan/inf (e.g. log of a negative marginal under heavy additive noise) is
  replaced by 0.0 in the returned matrix.

Only numpy and the standard library are used.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "FEATURE_NAMES",
    "N_FEATURES",
    "packet_features",
    "add_packet_relative",
    "RELATIVE_NAMES",
]

EPS = 1e-8

N_EPISODES = 6
N_WINDOWS = 12
N_TIME = 24
N_FREQ = 24
N_TILES = N_EPISODES * N_WINDOWS  # 72 tiles; row r = e * 12 + w

FEATURE_NAMES: list[str] = [
    # -- raw scale / distribution of all 576 tile values ----------------------
    "total_energy",          #  1 S = sum of all values
    "log_energy",            #  2 log1p(max(S, 0))
    "mean_val",              #  3
    "std_val",               #  4
    "min_val",               #  5
    "max_val",               #  6
    "median_val",            #  7
    # -- frequency-marginal moments (scale-invariant) -------------------------
    "freq_centroid",         #  8 sum(k * pf)
    "freq_spread",           #  9 sqrt(sum(pf * (k - centroid)^2))
    "freq_skew",             # 10 third standardized moment of pf
    "freq_kurtosis",         # 11 fourth standardized moment of pf
    # -- time-marginal moments (scale-invariant, NOT roll-invariant) ----------
    "time_centroid",         # 12
    "time_spread",           # 13
    "time_skew",             # 14
    "time_kurtosis",         # 15
    # -- band balance (scale-invariant) ---------------------------------------
    "band_low",              # 16 sum(F[0:8])   / (sum F + eps)
    "band_mid",              # 17 sum(F[8:16])  / (sum F + eps)
    "band_high",             # 18 sum(F[16:24]) / (sum F + eps)
    "lowmid_over_high",      # 19 (band_low + band_mid) / (band_high + eps)
    "log_lowmid_over_high",  # 20
    "low_over_mid",          # 21 band_low / (band_mid + eps)
    "log_low_over_mid",      # 22
    # -- spectral shape (scale-invariant) -------------------------------------
    "spectral_roughness_1",  # 23 sum|diff F| / (sum F + eps)
    "spectral_roughness_2",  # 24 sum diff(F)^2 / (sum F^2 + eps)
    "spectral_flatness",     # 25 geo-mean(F) / arith-mean(F)
    "spectral_entropy",      # 26 -sum pf log pf
    # -- temporal shape (scale-invariant) -------------------------------------
    "time_roughness_1",      # 27 sum|diff G| / (sum G + eps)
    "time_roughness_2",      # 28 sum diff(G)^2 / (sum G^2 + eps)
    "time_entropy",          # 29 -sum pt log pt  (roll-invariant too)
    # -- pulse contrast (scale-invariant) -------------------------------------
    "pulse_contrast_1",      # 30 max(G) / (median(G) + eps)
    "pulse_contrast_2",      # 31 (max(G) - median(G)) / (std(G) + eps)
    "pulse_crest",           # 32 max(G) / (mean(G) + eps)
    "time_kurtosis_raw",     # 33 kurtosis of the raw G values
    # -- whole-tile higher moments (scale-invariant) --------------------------
    "tile_kurtosis",         # 34 kurtosis of all 576 tile values
    "tile_skew",             # 35 skew of all 576 tile values
    # -- raw-scale percentiles -------------------------------------------------
    "p10",                   # 36
    "p25",                   # 37
    "p75",                   # 38
    "p90",                   # 39
    "p99",                   # 40
    "iqr",                   # 41 p75 - p25
    "rel_std",               # 42 std / (|mean| + eps)
    # -- local gradients / dropout indicator ----------------------------------
    "grad_t",                # 43 mean|diff along time| / (mean + eps)
    "grad_f",                # 44 mean|diff along freq| / (mean + eps)
    "frac_near_zero",        # 45 fraction of values <= min + 1% of range
    # -- restricted centroids & peak locations --------------------------------
    "freq_centroid_hi",      # 46 centroid of pf renormalized on k >= 12
    "freq_centroid_lo",      # 47 centroid of pf renormalized on k <  12
    "max_freq_bin",          # 48 argmax F (as float)
    "max_time_bin",          # 49 argmax G (as float)
    "spectral_rolloff85",    # 50 smallest k with cumsum(pf)[k] >= 0.85
    "spectral_rolloff50",    # 51 smallest k with cumsum(pf)[k] >= 0.50
    # -- roll-invariant circular time statistics -------------------------------
    "time_spread_circ",      # 52 circular std of pt
    "time_conc_circ",        # 53 circular concentration R of pt
    # -- sorted-profile shares (scale-invariant; time ones roll-invariant) -----
    "top3_time_share",       # 54 sum of top-3 of G / (sum G + eps)
    "top6_freq_share",       # 55 sum of top-6 of F / (sum F + eps)
    # -- extras (clearly useful for the documented target/degradations) --------
    "top1_time_share",       # 56 max(G) / (sum G + eps)  pulse share, roll-inv
    "top1_freq_share",       # 57 max(F) / (sum F + eps)  peak-band share
    "time_flatness",         # 58 geo-mean(G) / arith-mean(G)  pulsiness
    "spectral_rolloff15",    # 59 smallest k with cumsum(pf) >= 0.15 (low edge)
]

N_FEATURES = len(FEATURE_NAMES)

# Precomputed indices for linear-interpolated percentiles of the 576 sorted
# tile values (identical to np.percentile's default "linear" method, but
# served from ONE row-sort instead of repeated partitions -- measurably
# faster, and the same sort also yields min/max for free).
_P_Q = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 0.99], dtype=np.float64)
_P_POS = _P_Q * (N_TIME * N_FREQ - 1)                  # virtual index q*(n-1)
_P_LO = np.floor(_P_POS).astype(np.int64)
_P_HI = np.minimum(_P_LO + 1, N_TIME * N_FREQ - 1)
_P_FRAC = _P_POS - _P_LO


def packet_features(packet: np.ndarray) -> np.ndarray:
    """Extract per-tile features from one packet.

    Parameters
    ----------
    packet : array-like, shape (6, 12, 24, 24), any float dtype

    Returns
    -------
    np.ndarray, float32, shape (72, N_FEATURES)
        Row ``e * 12 + w`` holds the features of tile ``packet[e, w]``
        (episode-major).  Guaranteed finite (nan/inf replaced by 0.0).
    """
    X = np.asarray(packet)
    if X.shape != (N_EPISODES, N_WINDOWS, N_TIME, N_FREQ):
        raise ValueError(
            f"expected packet of shape (6, 12, 24, 24), got {X.shape}"
        )
    # Episode-major flattening: row r = e*12 + w  <->  X[r] == packet[e, w].
    # float16 would overflow/underflow when summing 576 values: go float64.
    X = X.reshape(N_TILES, N_TIME, N_FREQ).astype(np.float64, copy=False)

    k_idx = np.arange(N_FREQ, dtype=np.float64)
    t_idx = np.arange(N_TIME, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore",
                     under="ignore"):
        flat = X.reshape(N_TILES, N_TIME * N_FREQ)          # (72, 576)

        # ---- raw-scale distribution stats -------------------------------
        S = flat.sum(axis=1)                                 # total energy
        log_energy = np.log1p(np.maximum(S, 0.0))
        mean_val = S / float(N_TIME * N_FREQ)

        # one sort per tile row serves min, max and all six percentiles
        flat_sorted = np.sort(flat, axis=1)
        min_val = flat_sorted[:, 0]
        max_val = flat_sorted[:, -1]
        q6 = (flat_sorted[:, _P_LO] * (1.0 - _P_FRAC)
              + flat_sorted[:, _P_HI] * _P_FRAC)             # (72, 6)
        p10, p25, median_val, p75, p90, p99 = q6.T
        iqr = p75 - p25

        # central moments of all 576 values (reuse products, no ** powers)
        dz = flat - mean_val[:, None]
        dz2 = dz * dz
        z2 = dz2.mean(axis=1)                                # variance
        z3 = (dz2 * dz).mean(axis=1)
        z4 = (dz2 * dz2).mean(axis=1)
        std_val = np.sqrt(z2)
        tile_skew = z3 / (z2 * std_val + EPS)                # m3 / sigma^3
        tile_kurtosis = z4 / (z2 * z2 + EPS)                 # m4 / sigma^4
        rel_std = std_val / (np.abs(mean_val) + EPS)

        # ---- marginals ---------------------------------------------------
        Fm = X.sum(axis=1)                                   # (72, 24) freq marginal
        Gm = X.sum(axis=2)                                   # (72, 24) time marginal
        sumF = Fm.sum(axis=1)
        sumG = Gm.sum(axis=1)
        den_F = sumF + EPS
        den_G = sumG + EPS
        pf = Fm / den_F[:, None]
        pt = Gm / den_G[:, None]

        # ---- frequency-marginal moments ----------------------------------
        freq_centroid = pf @ k_idx
        dk = k_idx[None, :] - freq_centroid[:, None]
        wf2 = pf * dk * dk
        var_f = wf2.sum(axis=1)
        freq_spread = np.sqrt(np.maximum(var_f, 0.0))
        m3_f = (wf2 * dk).sum(axis=1)
        m4_f = (wf2 * dk * dk).sum(axis=1)
        freq_skew = m3_f / (var_f * freq_spread + EPS)       # / sigma^3
        freq_kurtosis = m4_f / (var_f * var_f + EPS)         # / sigma^4

        # ---- time-marginal moments ---------------------------------------
        time_centroid = pt @ t_idx
        dtc = t_idx[None, :] - time_centroid[:, None]
        wt2 = pt * dtc * dtc
        var_t = wt2.sum(axis=1)
        time_spread = np.sqrt(np.maximum(var_t, 0.0))
        m3_t = (wt2 * dtc).sum(axis=1)
        m4_t = (wt2 * dtc * dtc).sum(axis=1)
        time_skew = m3_t / (var_t * time_spread + EPS)
        time_kurtosis = m4_t / (var_t * var_t + EPS)

        # ---- band balance ------------------------------------------------
        band_low = Fm[:, 0:8].sum(axis=1) / den_F
        band_mid = Fm[:, 8:16].sum(axis=1) / den_F
        band_high = Fm[:, 16:24].sum(axis=1) / den_F
        lowmid_over_high = (band_low + band_mid) / (band_high + EPS)
        log_lowmid_over_high = np.log(
            (band_low + band_mid + EPS) / (band_high + EPS)
        )
        low_over_mid = band_low / (band_mid + EPS)
        log_low_over_mid = np.log((band_low + EPS) / (band_mid + EPS))

        # ---- spectral shape ----------------------------------------------
        dF = np.diff(Fm, axis=1)
        spectral_roughness_1 = np.abs(dF).sum(axis=1) / den_F
        spectral_roughness_2 = (dF * dF).sum(axis=1) / (
            (Fm * Fm).sum(axis=1) + EPS
        )
        F_mean = sumF / float(N_FREQ)
        spectral_flatness = np.exp(np.log(Fm + EPS).mean(axis=1)) / (
            F_mean + EPS
        )
        spectral_entropy = -(pf * np.log(pf + EPS)).sum(axis=1)

        # ---- temporal shape ----------------------------------------------
        dG = np.diff(Gm, axis=1)
        time_roughness_1 = np.abs(dG).sum(axis=1) / den_G
        time_roughness_2 = (dG * dG).sum(axis=1) / ((Gm * Gm).sum(axis=1) + EPS)
        time_entropy = -(pt * np.log(pt + EPS)).sum(axis=1)

        # ---- pulse contrast (one sort of G serves max/median/top-k) ------
        Gs = np.sort(Gm, axis=1)
        G_max = Gs[:, -1]
        G_med = 0.5 * (Gs[:, 11] + Gs[:, 12])                # median of 24
        G_mean = sumG / float(N_TIME)
        dGm = Gm - G_mean[:, None]
        dGm2 = dGm * dGm
        g2 = dGm2.mean(axis=1)
        g4 = (dGm2 * dGm2).mean(axis=1)
        G_std = np.sqrt(g2)
        pulse_contrast_1 = G_max / (G_med + EPS)
        pulse_contrast_2 = (G_max - G_med) / (G_std + EPS)
        pulse_crest = G_max / (G_mean + EPS)
        time_kurtosis_raw = g4 / (g2 * g2 + EPS)

        # ---- gradients / dropout indicator -------------------------------
        grad_t = np.abs(np.diff(X, axis=1)).mean(axis=(1, 2)) / (mean_val + EPS)
        grad_f = np.abs(np.diff(X, axis=2)).mean(axis=(1, 2)) / (mean_val + EPS)
        thresh = min_val + 0.01 * (max_val - min_val)
        frac_near_zero = (flat <= thresh[:, None]).mean(axis=1)

        # ---- restricted centroids & peak locations -----------------------
        F_hi = Fm[:, 12:]
        F_lo = Fm[:, :12]
        freq_centroid_hi = (F_hi @ k_idx[12:]) / (F_hi.sum(axis=1) + EPS)
        freq_centroid_lo = (F_lo @ k_idx[:12]) / (F_lo.sum(axis=1) + EPS)
        max_freq_bin = Fm.argmax(axis=1).astype(np.float64)
        max_time_bin = Gm.argmax(axis=1).astype(np.float64)

        cum = np.cumsum(pf, axis=1)
        # argmax over booleans = first True; all-False (degenerate) -> 0
        rolloff85 = np.argmax(cum >= 0.85, axis=1).astype(np.float64)
        rolloff50 = np.argmax(cum >= 0.50, axis=1).astype(np.float64)
        rolloff15 = np.argmax(cum >= 0.15, axis=1).astype(np.float64)

        # ---- roll-invariant circular time statistics ---------------------
        ang = (2.0 * np.pi / N_TIME) * t_idx
        Cc = pt @ np.cos(ang)
        Ss = pt @ np.sin(ang)
        Rlen = np.sqrt(Cc * Cc + Ss * Ss)
        time_conc_circ = Rlen
        # clamp: Rlen + EPS can slightly exceed 1 for a one-hot pt
        time_spread_circ = np.sqrt(np.maximum(-2.0 * np.log(Rlen + EPS), 0.0))

        # ---- sorted-profile shares ---------------------------------------
        Fs = np.sort(Fm, axis=1)
        top3_time_share = Gs[:, -3:].sum(axis=1) / den_G
        top6_freq_share = Fs[:, -6:].sum(axis=1) / den_F
        top1_time_share = G_max / den_G
        top1_freq_share = Fs[:, -1] / den_F
        time_flatness = np.exp(np.log(Gm + EPS).mean(axis=1)) / (G_mean + EPS)

    vals = {
        "total_energy": S,
        "log_energy": log_energy,
        "mean_val": mean_val,
        "std_val": std_val,
        "min_val": min_val,
        "max_val": max_val,
        "median_val": median_val,
        "freq_centroid": freq_centroid,
        "freq_spread": freq_spread,
        "freq_skew": freq_skew,
        "freq_kurtosis": freq_kurtosis,
        "time_centroid": time_centroid,
        "time_spread": time_spread,
        "time_skew": time_skew,
        "time_kurtosis": time_kurtosis,
        "band_low": band_low,
        "band_mid": band_mid,
        "band_high": band_high,
        "lowmid_over_high": lowmid_over_high,
        "log_lowmid_over_high": log_lowmid_over_high,
        "low_over_mid": low_over_mid,
        "log_low_over_mid": log_low_over_mid,
        "spectral_roughness_1": spectral_roughness_1,
        "spectral_roughness_2": spectral_roughness_2,
        "spectral_flatness": spectral_flatness,
        "spectral_entropy": spectral_entropy,
        "time_roughness_1": time_roughness_1,
        "time_roughness_2": time_roughness_2,
        "time_entropy": time_entropy,
        "pulse_contrast_1": pulse_contrast_1,
        "pulse_contrast_2": pulse_contrast_2,
        "pulse_crest": pulse_crest,
        "time_kurtosis_raw": time_kurtosis_raw,
        "tile_kurtosis": tile_kurtosis,
        "tile_skew": tile_skew,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p99": p99,
        "iqr": iqr,
        "rel_std": rel_std,
        "grad_t": grad_t,
        "grad_f": grad_f,
        "frac_near_zero": frac_near_zero,
        "freq_centroid_hi": freq_centroid_hi,
        "freq_centroid_lo": freq_centroid_lo,
        "max_freq_bin": max_freq_bin,
        "max_time_bin": max_time_bin,
        "spectral_rolloff85": rolloff85,
        "spectral_rolloff50": rolloff50,
        "time_spread_circ": time_spread_circ,
        "time_conc_circ": time_conc_circ,
        "top3_time_share": top3_time_share,
        "top6_freq_share": top6_freq_share,
        "top1_time_share": top1_time_share,
        "top1_freq_share": top1_freq_share,
        "time_flatness": time_flatness,
        "spectral_rolloff15": rolloff15,
    }
    if set(vals) != set(FEATURE_NAMES):  # pragma: no cover - dev guard
        raise AssertionError("feature dict out of sync with FEATURE_NAMES")

    out = np.stack([vals[name] for name in FEATURE_NAMES], axis=1)
    out = out.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def add_packet_relative(feats: np.ndarray) -> np.ndarray:
    """Concatenate packet-standardized copies of each feature column.

    Parameters
    ----------
    feats : array-like, shape (72, F)

    Returns
    -------
    np.ndarray, float32, shape (72, 2F)
        ``[:, :F]`` is the original matrix; ``[:, F:]`` is per-column
        ``(x - mean over the 72 tiles) / (std over the 72 tiles + 1e-8)``.
        These within-packet z-scores expose how a tile ranks against its
        packet-mates, which survives any packet-level contrast change.
    """
    f = np.asarray(feats, dtype=np.float32)
    if f.ndim != 2:
        raise ValueError(f"expected 2-D (n_tiles, F) feature matrix, got {f.shape}")
    f64 = f.astype(np.float64)
    mu = f64.mean(axis=0)
    sd = f64.std(axis=0)
    rel = (f64 - mu) / (sd + EPS)
    out = np.concatenate([f64, rel], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def RELATIVE_NAMES() -> list[str]:
    """Names for the doubled feature set returned by ``add_packet_relative``."""
    return list(FEATURE_NAMES) + [name + "_rel" for name in FEATURE_NAMES]
