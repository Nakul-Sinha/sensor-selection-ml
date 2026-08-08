"""Test-suite + benchmark for features.py.   Run:  python test_features.py

Covers:
  1. episode-major ordering (row e*12+w  <->  packet[e, w])
  2. shapes / dtypes / FEATURE_NAMES consistency
  3. finiteness (no nan/inf) and zero warnings on degenerate + random inputs
  4. scale invariance of the scale-invariant subset under a x3.7 contrast change
  5. speed benchmark: 200 random float16 packets -> projected time for 5800
  6. add_packet_relative shape + zero-mean / unit-std of the standardized block
  7. semantic sanity + circular-roll invariance of the circular time features
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (  # noqa: E402
    EPS,
    FEATURE_NAMES,
    N_FEATURES,
    RELATIVE_NAMES,
    add_packet_relative,
    packet_features,
)

SHAPE = (6, 12, 24, 24)

# Features that must be unchanged by a global multiplicative contrast change.
SCALE_INVARIANT = [
    "freq_centroid", "freq_spread", "freq_skew", "freq_kurtosis",
    "time_centroid", "time_spread", "time_skew", "time_kurtosis",
    "band_low", "band_mid", "band_high",
    "lowmid_over_high", "log_lowmid_over_high",
    "low_over_mid", "log_low_over_mid",
    "spectral_roughness_1", "spectral_roughness_2",
    "spectral_flatness", "spectral_entropy",
    "time_roughness_1", "time_roughness_2", "time_entropy",
    "pulse_contrast_1", "pulse_contrast_2", "pulse_crest",
    "time_kurtosis_raw", "tile_kurtosis", "tile_skew",
    "rel_std", "grad_t", "grad_f", "frac_near_zero",
    "freq_centroid_hi", "freq_centroid_lo",
    "max_freq_bin", "max_time_bin",
    "spectral_rolloff85", "spectral_rolloff50", "spectral_rolloff15",
    "time_spread_circ", "time_conc_circ",
    "top3_time_share", "top6_freq_share",
    "top1_time_share", "top1_freq_share",
    "time_flatness",
]


def col(feats: np.ndarray, name: str) -> np.ndarray:
    return feats[:, FEATURE_NAMES.index(name)]


# ---------------------------------------------------------------- test 1 --
def test_ordering():
    pkt = np.zeros(SHAPE, dtype=np.float32)
    for e in range(6):
        for w in range(12):
            pkt[e, w] = float(e * 12 + w)
    # the reshape convention itself is episode-major
    r = pkt.reshape(72, 24, 24)
    for e, w in [(0, 0), (0, 11), (3, 5), (5, 0), (5, 11)]:
        assert np.array_equal(r[e * 12 + w], pkt[e, w]), "reshape not episode-major"
    feats = packet_features(pkt)
    want = np.arange(72, dtype=np.float32)
    assert np.array_equal(col(feats, "mean_val"), want), "row order wrong (mean_val)"
    assert np.array_equal(col(feats, "median_val"), want)
    assert np.array_equal(col(feats, "max_val"), want)


# ---------------------------------------------------------------- test 2 --
def test_shapes_names():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES, "duplicate feature names"
    assert all(isinstance(n, str) and n for n in FEATURE_NAMES)
    rng = np.random.default_rng(123)
    pkt = rng.standard_normal(SHAPE).astype(np.float16)
    feats = packet_features(pkt)
    assert feats.shape == (72, N_FEATURES), feats.shape
    assert feats.dtype == np.float32, feats.dtype
    # any float dtype accepted
    assert packet_features(pkt.astype(np.float32)).shape == (72, N_FEATURES)
    assert packet_features(pkt.astype(np.float64)).shape == (72, N_FEATURES)
    # percentile columns match np.percentile (default linear interpolation)
    flat64 = pkt.astype(np.float64).reshape(72, 576)
    q = np.percentile(flat64, (10, 25, 50, 75, 90, 99), axis=1)
    for nm, row in zip(("p10", "p25", "median_val", "p75", "p90", "p99"), q):
        assert np.allclose(col(feats, nm), row, rtol=1e-5, atol=1e-5), (
            f"'{nm}' disagrees with np.percentile"
        )
    # wrong shape rejected
    try:
        packet_features(np.zeros((6, 12, 24, 23), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong input shape was not rejected")


# ---------------------------------------------------------------- test 3 --
def _degenerate_packets() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    zeros = np.zeros(SHAPE, dtype=np.float16)
    const = np.full(SHAPE, 2.5, dtype=np.float16)
    neg_const = np.full(SHAPE, -1.25, dtype=np.float16)
    hot = np.zeros(SHAPE, dtype=np.float16)
    hot[2, 5, 10, 3] = 1000.0
    rand = rng.standard_normal(SHAPE).astype(np.float16)
    dropout = (rng.random(SHAPE) * (rng.random(SHAPE) > 0.3)).astype(np.float16)
    return {
        "all_zeros": zeros,
        "constant": const,
        "constant_negative": neg_const,
        "single_hot_pixel": hot,
        "random_float16": rand,
        "random_with_dropouts": dropout,
    }


def test_no_nan_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning becomes a failure
        for name, pkt in _degenerate_packets().items():
            feats = packet_features(pkt)
            assert np.isfinite(feats).all(), f"non-finite features for '{name}'"
            rel = add_packet_relative(feats)
            assert np.isfinite(rel).all(), f"non-finite relative feats for '{name}'"


# ---------------------------------------------------------------- test 4 --
def test_scale_invariance():
    rng = np.random.default_rng(42)
    base = rng.random(SHAPE) + 0.05          # positive float64 tiles
    a = packet_features(base)
    b = packet_features(base * 3.7)          # exact multiplicative contrast
    worst_name, worst_dev = "", 0.0
    for name in SCALE_INVARIANT:
        xa = col(a, name).astype(np.float64)
        xb = col(b, name).astype(np.float64)
        dev = float(np.max(np.abs(xa - xb) / (np.abs(xa) + 1e-6)))
        if dev > worst_dev:
            worst_name, worst_dev = name, dev
        assert np.allclose(xa, xb, rtol=1e-4, atol=1e-5), (
            f"'{name}' is not scale-invariant (max rel dev {dev:.3e})"
        )
    # raw-scale features must scale by exactly ~3.7
    for name in ("total_energy", "mean_val", "std_val", "max_val",
                 "median_val", "p90", "iqr"):
        xa = col(a, name).astype(np.float64)
        xb = col(b, name).astype(np.float64)
        ratio = xb / (xa + EPS)
        assert np.allclose(ratio, 3.7, rtol=1e-3), f"'{name}' ratio {ratio[:4]}"
    assert not np.allclose(col(a, "total_energy"), col(b, "total_energy")), (
        "total_energy should change under scaling"
    )
    print(f"    scale invariance: {len(SCALE_INVARIANT)} features checked; "
          f"worst rel deviation {worst_dev:.2e} ({worst_name})")


# ---------------------------------------------------------------- test 5 --
def test_benchmark():
    rng = np.random.default_rng(2024)
    n = 200
    packets = rng.standard_normal((n,) + SHAPE).astype(np.float16)
    packet_features(packets[0])              # warm-up (allocator, caches)
    t0 = time.perf_counter()
    for i in range(n):
        packet_features(packets[i])
    total = time.perf_counter() - t0
    per = total / n
    proj = per * 5800.0
    print(f"    benchmark: {n} packets in {total:.3f} s  "
          f"({per * 1e3:.3f} ms/packet, {N_FEATURES} features/tile, 72 tiles)")
    print(f"    projected 5800 packets: {proj:.2f} s   (budget: 120 s, "
          f"headroom {120.0 / max(proj, 1e-9):.0f}x)")
    assert proj < 120.0, f"too slow: projected {proj:.1f} s for 5800 packets"


# ---------------------------------------------------------------- test 6 --
def test_add_packet_relative():
    rng = np.random.default_rng(5)
    pkt = rng.random(SHAPE).astype(np.float16)
    feats = packet_features(pkt)
    rel = add_packet_relative(feats)
    assert rel.shape == (72, 2 * N_FEATURES), rel.shape
    assert rel.dtype == np.float32
    assert np.array_equal(rel[:, :N_FEATURES], feats), (
        "first block must equal the original features"
    )
    block = rel[:, N_FEATURES:].astype(np.float64)
    colmean = np.abs(block.mean(axis=0))
    assert colmean.max() < 1e-4, f"relative block not ~zero-mean: {colmean.max():.2e}"
    sd_orig = feats.astype(np.float64).std(axis=0)
    varying = sd_orig > 1e-4
    sd_block = block.std(axis=0)
    assert np.allclose(sd_block[varying], 1.0, atol=1e-3), (
        "relative block not ~unit-std on varying columns"
    )
    names = RELATIVE_NAMES()
    assert len(names) == 2 * N_FEATURES
    assert names[:N_FEATURES] == list(FEATURE_NAMES)
    assert all(names[N_FEATURES + i] == FEATURE_NAMES[i] + "_rel"
               for i in range(N_FEATURES))
    print(f"    relative block: max |col mean| = {colmean.max():.2e}, "
          f"{int(varying.sum())}/{N_FEATURES} varying cols have std ~= 1")


# ---------------------------------------------------------------- test 7 --
def test_semantics_and_roll_invariance():
    pkt = np.zeros(SHAPE, dtype=np.float32)
    pkt[0, 0, :, 23] = 1.0   # tile 0: all energy in highest freq bin
    pkt[0, 1, :, 0] = 1.0    # tile 1: all energy in lowest freq bin
    pkt[0, 2, 7, :] = 1.0    # tile 2: single pulse at t=7, flat in freq
    f = packet_features(pkt)
    assert abs(col(f, "freq_centroid")[0] - 23.0) < 1e-5
    assert col(f, "band_high")[0] > 0.999 and col(f, "band_low")[0] < 1e-6
    assert abs(col(f, "freq_centroid")[1] - 0.0) < 1e-5
    assert col(f, "band_low")[1] > 0.999
    assert abs(col(f, "spectral_rolloff85")[1] - 0.0) < 1e-9
    assert abs(col(f, "time_centroid")[2] - 7.0) < 1e-5
    assert abs(col(f, "max_time_bin")[2] - 7.0) < 1e-9
    assert col(f, "pulse_crest")[2] > 20.0            # 24x the mean
    assert abs(col(f, "time_conc_circ")[2] - 1.0) < 1e-6   # one-hot in time

    # circular-roll invariance: roll tile 2 by 5 time bins
    pkt2 = pkt.copy()
    pkt2[0, 2] = np.roll(pkt[0, 2], 5, axis=0)
    f2 = packet_features(pkt2)
    for nm in ("time_conc_circ", "time_spread_circ", "top3_time_share",
               "top1_time_share", "time_entropy", "pulse_crest",
               "freq_centroid", "band_low", "band_mid", "band_high"):
        assert abs(col(f, nm)[2] - col(f2, nm)[2]) < 1e-5, (
            f"'{nm}' not invariant under circular time roll"
        )
    assert abs(col(f2, "max_time_bin")[2] - 12.0) < 1e-9   # 7 + 5


# --------------------------------------------------------------------------
def main() -> None:
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"features per tile: {N_FEATURES} | with packet-relative: {2 * N_FEATURES}")
    tests = [
        ("1. episode-major ordering (row = e*12 + w)", test_ordering),
        ("2. shapes / dtypes / FEATURE_NAMES", test_shapes_names),
        ("3. finiteness + zero warnings on degenerate inputs", test_no_nan_no_warnings),
        ("4. scale invariance under x3.7 contrast", test_scale_invariance),
        ("5. speed benchmark (200 packets -> 5800 projection)", test_benchmark),
        ("6. add_packet_relative / RELATIVE_NAMES", test_add_packet_relative),
        ("7. semantic sanity + circular-roll invariance", test_semantics_and_roll_invariance),
    ]
    for label, fn in tests:
        fn()
        print(f"PASS  {label}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
