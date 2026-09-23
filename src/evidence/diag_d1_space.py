"""Task 4 / D1: headroom diagnostics on the candidate caches (no retraining).

Answers: is there room for Top-K internal reranking at all?
  - Recall@K curve (K=1/3/5/10/20/50) of config A
  - gold rank histogram (1 / 2-3 / 4-10 / 11-50 / not in Top-50)
  - Oracle upper bound: gold moved to rank 1 whenever it is inside Top-50
  - config-A absolute metrics bucketed by subject history event count
    ({0, 1-2, 3-9, 10-99, 100+}): n, MRR, H@1, H@3, Recall@50

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.diag_d1_space \
        --tags seed1 seed2 seed3 --cache-dir ../candidates \
        --data-dir ../data/ICEWS14 --out ../results/task4_diagnostics/d1_space.json
"""

import argparse
import json
import os
import pickle

import numpy as np

from src.evidence.run_experiments import subject_history_counts

KS = (1, 3, 5, 10, 20, 50)
BUCKETS = (("0", 0, 0), ("1-2", 1, 2), ("3-9", 3, 9), ("10-99", 10, 99), ("100+", 100, 10 ** 9))


def run_tag(cache_dir, data_dir, tag):
    with open(os.path.join(cache_dir, tag, "ICEWS14_test_topk50.pkl"), "rb") as fin:
        cache = pickle.load(fin)
    gold_rank = cache["gold_filter_rank"].astype(np.float64)
    n = len(gold_rank)

    recall = {str(k): float(np.mean(gold_rank <= k)) for k in KS}
    hist = {
        "1": float(np.mean(gold_rank == 1)),
        "2-3": float(np.mean((gold_rank >= 2) & (gold_rank <= 3))),
        "4-10": float(np.mean((gold_rank >= 4) & (gold_rank <= 10))),
        "11-50": float(np.mean((gold_rank >= 11) & (gold_rank <= 50))),
        "not_in_top50": float(np.mean(gold_rank > 50)),
    }
    mrr_a = float(np.mean(1.0 / gold_rank))
    oracle_rank = np.where(gold_rank <= 50, 1.0, gold_rank)
    mrr_oracle = float(np.mean(1.0 / oracle_rank))

    subj_counts = subject_history_counts(data_dir, "test", cache["queries"])
    buckets = {}
    for name, lo, hi in BUCKETS:
        mask = (subj_counts >= lo) & (subj_counts <= hi)
        nb = int(mask.sum())
        if nb == 0:
            buckets[name] = None
            continue
        r = gold_rank[mask]
        buckets[name] = {
            "n": nb,
            "mrr": float(np.mean(1.0 / r)),
            "hits1": float(np.mean(r <= 1)),
            "hits3": float(np.mean(r <= 3)),
            "recall50": float(np.mean(r <= 50)),
        }
    return {
        "n": n,
        "recall_at_k": recall,
        "gold_rank_hist": hist,
        "mrr_A": mrr_a,
        "mrr_oracle_top50": mrr_oracle,
        "headroom": mrr_oracle - mrr_a,
        "buckets": buckets,
    }


def _mean_std(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def aggregate(per_tag):
    tags = list(per_tag)
    agg = {
        "mrr_A": _mean_std([per_tag[t]["mrr_A"] for t in tags]),
        "mrr_oracle_top50": _mean_std([per_tag[t]["mrr_oracle_top50"] for t in tags]),
        "headroom": _mean_std([per_tag[t]["headroom"] for t in tags]),
        "recall_at_k": {k: _mean_std([per_tag[t]["recall_at_k"][k] for t in tags]) for k in map(str, KS)},
        "gold_rank_hist": {h: _mean_std([per_tag[t]["gold_rank_hist"][h] for t in tags])
                           for h in ("1", "2-3", "4-10", "11-50", "not_in_top50")},
        "buckets": {},
    }
    for name, _lo, _hi in BUCKETS:
        vals = [per_tag[t]["buckets"][name] for t in tags if per_tag[t]["buckets"][name]]
        agg["buckets"][name] = {
            "n_mean": float(np.mean([v["n"] for v in vals])),
            "mrr": _mean_std([v["mrr"] for v in vals]),
            "hits1": _mean_std([v["hits1"] for v in vals]),
            "hits3": _mean_std([v["hits3"] for v in vals]),
            "recall50": _mean_std([v["recall50"] for v in vals]),
        }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="../candidates")
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_tag = {t: run_tag(args.cache_dir, args.data_dir, t) for t in args.tags}
    out = {"per_tag": per_tag, "aggregate": aggregate(per_tag)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)

    a = out["aggregate"]
    print("[D1] saved {}".format(args.out))
    print("MRR_A = {:.4f}+-{:.4f} | oracle@50 = {:.4f} | headroom = {:.4f}".format(
        a["mrr_A"]["mean"], a["mrr_A"]["std"], a["mrr_oracle_top50"]["mean"], a["headroom"]["mean"]))
    print("Recall@K:", {k: round(v["mean"], 4) for k, v in a["recall_at_k"].items()})
    print("hist:", {k: round(v["mean"], 4) for k, v in a["gold_rank_hist"].items()})
    for name, b in a["buckets"].items():
        print("bucket {:>5s} n={:7.0f} MRR={:.4f} H@1={:.4f} H@3={:.4f} R@50={:.4f}".format(
            name, b["n_mean"], b["mrr"]["mean"], b["hits1"]["mean"], b["hits3"]["mean"], b["recall50"]["mean"]))


if __name__ == "__main__":
    main()
