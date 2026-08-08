"""
Load every response packet, extract tile features, cache to disk.

EXPERIMENT-ONLY caching: the final solution.py recomputes everything end-to-end from raw data
(guidebook 3.8 forbids relying on artifacts from a previous run). Caching here just makes local
iteration fast.

Usage:
    python build_features.py <public_dir> <out_dir> [--limit N] [--split train|test|both]
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from features import packet_features, FEATURE_NAMES  # noqa: E402


def _chunk_worker(public_dir, rel_paths):
    """Load + featurize a chunk of packets -> (n, 72, F) float32."""
    out = np.empty((len(rel_paths), 72, len(FEATURE_NAMES)), dtype=np.float32)
    for i, rp in enumerate(rel_paths):
        pk = np.load(Path(public_dir) / rp)
        out[i] = packet_features(pk)
    return out


def build(public_dir, rel_paths, n_jobs=-1, chunk=64, verbose=True):
    public_dir = Path(public_dir)
    chunks = [rel_paths[i:i + chunk] for i in range(0, len(rel_paths), chunk)]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(_chunk_worker)(str(public_dir), c) for c in chunks
    )
    feats = np.concatenate(res, axis=0)
    if verbose:
        dt = time.time() - t0
        print("featurized %d packets in %.1f s (%.2f ms/packet)"
              % (len(rel_paths), dt, 1000 * dt / max(len(rel_paths), 1)))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    public = Path(a.public_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in (["train", "test"] if a.split == "both" else [a.split]):
        df = pd.read_csv(public / f"{split}.csv")
        if a.limit:
            df = df.head(a.limit)
        print(f"\n=== {split}: {len(df)} cases ===")
        feats = build(public, df["response_packet_path"].tolist(), n_jobs=a.n_jobs)
        np.save(out / f"X_{split}.npy", feats)
        df[["case_id"]].to_csv(out / f"ids_{split}.csv", index=False)
        print("saved", out / f"X_{split}.npy", feats.shape, feats.dtype)

        if split == "train":
            import json
            P = np.stack([np.asarray(json.loads(s), dtype=np.float32).reshape(6, 12)
                          for s in df["packet_response_matrix"]])
            np.save(out / "P_train.npy", P)
            print("saved P_train.npy", P.shape)
            # targets
            sched = np.stack([np.array(sorted(int(t[1:]) for t in s.split("|")), dtype=np.int8)
                              for s in df["window_schedule"]])
            np.save(out / "sched_train.npy", sched)
            M = np.stack([np.asarray(json.loads(s), dtype=np.int8)
                          for s in df["observability_matrix"]])
            np.save(out / "M_train.npy", M)
            print("saved sched_train.npy", sched.shape, " M_train.npy", M.shape)


if __name__ == "__main__":
    main()
