"""CLI: extract E1-E4 evidence features for a dumped candidate cache.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.extract_features \
        --data-dir ../data/ICEWS14 --split test \
        --cache ../candidates/ICEWS14_test_topk50.pkl \
        --out ../candidates/ICEWS14_test_topk50_feats.npz

E4 features need the Top-K block context, so they are computed here rather
than in EvidenceRetriever.query_features:
  e4_orig_rank : (block position + 1) / K, the candidate's original rank
  e4_path_gap  : max e2_conf_sum inside the block minus own e2_conf_sum
"""

import argparse
import pickle
import time

import numpy as np

from src.evidence.retriever import EvidenceRetriever, FEATURE_NAMES

E4_NAMES = ["e4_orig_rank", "e4_path_gap"]
ALL_FEATURE_NAMES = FEATURE_NAMES + E4_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.cache, "rb") as fin:
        cache = pickle.load(fin)
    queries, cand_ids = cache["queries"], cache["cand_ids"]
    n, k = cand_ids.shape

    retriever = EvidenceRetriever(args.data_dir)
    feats = np.zeros((n, k, len(ALL_FEATURE_NAMES)), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        base = retriever.query_features(args.split, queries[i], cand_ids[i])
        e4_rank = (np.arange(k, dtype=np.float32) + 1.0) / k
        conf = base[:, 6]  # e2_conf_sum
        e4_gap = conf.max() - conf
        feats[i, :, :base.shape[1]] = base
        feats[i, :, base.shape[1]] = e4_rank
        feats[i, :, base.shape[1] + 1] = e4_gap
        if (i + 1) % 1000 == 0:
            print("[feats] {}/{} queries ({:.1f}s)".format(i + 1, n, time.time() - t0))

    np.savez_compressed(args.out, features=feats, feature_names=ALL_FEATURE_NAMES)
    print("[feats] saved {} with shape {} ({:.1f}s)".format(args.out, feats.shape, time.time() - t0))


if __name__ == "__main__":
    main()
