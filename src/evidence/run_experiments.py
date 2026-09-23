"""Task 3: Phase-2 comparison experiments on candidate caches (no LLM).

Feature layout (14 dims, see retriever.py / extract_features.py):
  0 e1_count            1 e1_recent_count      2 e1_decayed_count   3 e1_log_recency
  4 e2_support          5 e2_log_support       6 e2_conf_sum        7 e2_num_rules
  8 e3_recent_activity  9 e3_log_recent_act   10 e3_interaction    11 e3_log_interaction
 12 e4_orig_rank       13 e4_path_gap

Configurations:
  A TiRGN               : no evidence (lambda = 0)
  B +Recent             : [1]  short-window repeat count
  C +Long               : [0]  full-history repeat count
  D +TemporalEvidence   : [2,3] decayed repeat count + recency
  E +CandidateEvidence  : E1 + E3 (candidate-conditioned, no relation-aligned paths)
  F +RCEV               : E2 + [e4_path_gap] (relation-aligned paths + contrastive gap)
  abl_E1/E2/E3/E4       : single-group ablations

e4_orig_rank is excluded from every config: it is a monotone function of the
original in-block order and therefore cannot carry new ranking information.

For each config and seed tag: logistic-regression weights are fit on the
validation split only, lambda is selected on validation MRR from a fixed grid,
then applied once to the test split. Reranking is restricted to the Top-K
block (see fuse.py), so filtered metrics stay valid.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.run_experiments \
        --tags seed1 seed2 seed3 --cache-dir ../candidates \
        --data-dir ../data/ICEWS14 --out ../results/task3_icews14.json
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from src.evidence.fuse import fit_logistic, standardize, rerank_metrics

CONFIGS = {
    "A_TiRGN": [],
    "B_Recent": [1],
    "C_Long": [0],
    "D_TemporalEvidence": [2, 3],
    "E_CandidateEvidence": [0, 1, 2, 3, 8, 9, 10, 11],
    "F_RCEV": [4, 5, 6, 7, 13],
    "abl_E1": [0, 1, 2, 3],
    "abl_E2": [4, 5, 6, 7],
    "abl_E3": [8, 9, 10, 11],
    "abl_E4": [13],
}
LAMBDA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def load_tag(cache_dir, tag):
    out = {}
    for split in ("valid", "test"):
        with open(os.path.join(cache_dir, tag, "ICEWS14_{}_topk50.pkl".format(split)), "rb") as fin:
            cache = pickle.load(fin)
        feats = np.load(os.path.join(cache_dir, tag, "ICEWS14_{}_topk50_feats.npz".format(split)))["features"]
        out[split] = (cache, feats)
    return out


def subject_history_counts(data_dir, split, queries):
    """Number of history events with the query subject as subject, t' < t."""
    train = np.loadtxt(os.path.join(data_dir, "train.txt"), dtype=np.int64)[:, :4]
    quads = train
    if split == "test":
        valid = np.loadtxt(os.path.join(data_dir, "valid.txt"), dtype=np.int64)[:, :4]
        quads = np.concatenate([train, valid], axis=0)
    subj_times = defaultdict(list)
    for s, _r, _o, t in quads:
        subj_times[s].append(t)
    for s in subj_times:
        subj_times[s].sort()
    counts = np.zeros(len(queries), dtype=np.int64)
    for i, (s, r, _o, t) in enumerate(queries):
        # inverse queries (r >= num_rels) predict the subject; use the
        # *queried* entity, which is column 0 after inversion either way.
        times = subj_times.get(int(s), [])
        counts[i] = np.searchsorted(times, t, side="left")
    return counts


def diagnostics(cache_a, res_a, res_f, subj_counts):
    """F vs A diagnostics. res_* are rerank_metrics outputs."""
    gold_pos = cache_a["gold_block_pos"]
    in_block = gold_pos >= 0
    rank_a, rank_f = res_a["ranks"], res_f["ranks"]
    diag = {
        "mrr_gold_in_topk_A": float(np.mean(1.0 / rank_a[in_block])),
        "mrr_gold_in_topk_F": float(np.mean(1.0 / rank_f[in_block])),
        "beneficial_flip_rate": float(np.mean(rank_f[in_block] < rank_a[in_block])),
        "harmful_flip_rate": float(np.mean(rank_f[in_block] > rank_a[in_block])),
    }
    for name, lo, hi in (("0-2", 0, 2), ("3-9", 3, 9), ("10+", 10, 10 ** 9)):
        mask = in_block & (subj_counts >= lo) & (subj_counts <= hi)
        key = "bucket_{}".format(name)
        if mask.sum() == 0:
            diag[key] = None
            continue
        diag[key] = {
            "n": int(mask.sum()),
            "mrr_A": float(np.mean(1.0 / rank_a[mask])),
            "mrr_F": float(np.mean(1.0 / rank_f[mask])),
            "gain": float(np.mean(1.0 / rank_f[mask]) - np.mean(1.0 / rank_a[mask])),
        }
    return diag


def run_tag(cache_dir, data_dir, tag, fit_seed):
    data = load_tag(cache_dir, tag)
    (cv, fv), (ct, ft) = data["valid"], data["test"]
    fvs, fts = standardize(fv, ft)
    results = {"configs": {}}
    res_a_test = res_f_test = None
    for cfg, cols in CONFIGS.items():
        if not cols:
            res_v = rerank_metrics(cv, np.zeros(fv.shape[:2]), lam=0.0, use_rho=False)
            res_t = rerank_metrics(ct, np.zeros(ft.shape[:2]), lam=0.0, use_rho=False)
            entry = {"lambda": 0.0, "valid_mrr": res_v["mrr"],
                     "test": {"mrr": res_t["mrr"], "hits": res_t["hits"]}}
            if cfg == "A_TiRGN":
                res_a_test = res_t
        else:
            w = fit_logistic(fvs[:, :, cols], cv["gold_block_pos"], seed=fit_seed)
            ev = (fvs[:, :, cols] @ w)
            et = (fts[:, :, cols] @ w)
            best_lam, best_vm = None, -1.0
            for lam in LAMBDA_GRID:
                res_v = rerank_metrics(cv, ev, lam=lam, feats=fv)
                if res_v["mrr"] > best_vm:
                    best_vm, best_lam = res_v["mrr"], lam
            res_t = rerank_metrics(ct, et, lam=best_lam, feats=ft)
            entry = {"lambda": best_lam, "valid_mrr": best_vm,
                     "weights": [float(x) for x in w],
                     "test": {"mrr": res_t["mrr"], "hits": res_t["hits"]}}
            if cfg == "F_RCEV":
                res_f_test = res_t
        results["configs"][cfg] = entry
    subj_counts = subject_history_counts(data_dir, "test", ct["queries"])
    results["diagnostics_F_vs_A"] = diagnostics(ct, res_a_test, res_f_test, subj_counts)
    return results


def aggregate(per_tag):
    agg = {}
    for cfg in CONFIGS:
        mrrs = [per_tag[t]["configs"][cfg]["test"]["mrr"] for t in per_tag]
        hits = {h: [per_tag[t]["configs"][cfg]["test"]["hits"][h] for t in per_tag]
                for h in (1, 3, 10)}
        agg[cfg] = {
            "test_mrr_mean": float(np.mean(mrrs)), "test_mrr_std": float(np.std(mrrs)),
            "lambdas": [per_tag[t]["configs"][cfg]["lambda"] for t in per_tag],
        }
        for h in (1, 3, 10):
            agg[cfg]["test_hits{}_mean".format(h)] = float(np.mean(hits[h]))
            agg[cfg]["test_hits{}_std".format(h)] = float(np.std(hits[h]))
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True,
                        help="cache tags; expects ICEWS14_{tag}_{split}_topk50.pkl")
    parser.add_argument("--cache-dir", default="../candidates")
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_tag = {}
    for tag in args.tags:
        print("[exp] running tag {}".format(tag))
        per_tag[tag] = run_tag(args.cache_dir, args.data_dir, tag, args.fit_seed)
    out = {"per_tag": per_tag, "aggregate": aggregate(per_tag)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)
    print("[exp] saved {}".format(args.out))
    print("{:24s} {:>14s} {:>10s} {:>10s} {:>10s}".format("config", "MRR", "H@1", "H@3", "H@10"))
    for cfg in CONFIGS:
        a = out["aggregate"][cfg]
        print("{:24s} {:>7.4f}+-{:<5.4f} {:>7.4f}+-{:<5.4f} {:>7.4f}+-{:<5.4f} {:>7.4f}+-{:<5.4f}".format(
            cfg, a["test_mrr_mean"], a["test_mrr_std"],
            a["test_hits1_mean"], a["test_hits1_std"],
            a["test_hits3_mean"], a["test_hits3_std"],
            a["test_hits10_mean"], a["test_hits10_std"]))


if __name__ == "__main__":
    main()
