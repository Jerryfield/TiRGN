"""Task 6 / S1: predict when TiRGN will be wrong (selective prediction).

Label : top-1 candidate != gold (error=1), seed1 test cache.
Feats : 14 E1-E4 evidence features + TiRGN score-gap features
        (top1-top2, top1-mean(top2-10), top1-mean(top2-50), top1, std(top10))
        + P2 LLM-confidence features (only for the 1500 sampled queries).
Model : LightGBM binary classifier, TEMPORAL split (first 70% of the
        timeline = train, last 30% = test; random splits are forbidden to
        avoid leakage across time).

Reports overall and per-bucket (0 / 1-9 / 10+) AUROC + AUPRC, with a
gaps-only ablation (does evidence add anything over TiRGN's own margins?).
Predicted error probabilities on the temporal-test part are stored for S2.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.sel_s1_error_pred \
        --cache ../candidates/seed1/ICEWS14_test_topk50.pkl \
        --feats ../candidates/seed1/ICEWS14_test_topk50_feats.npz \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --scores ../results/task5_pilot/p2_llm_scores.jsonl \
        --data-dir ../data/ICEWS14 \
        --out-dir ../results/task6_selective
"""

import argparse
import json
import os
import pickle

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.evidence.run_experiments import subject_history_counts


def gap_features(raw_scores):
    z = raw_scores
    z1 = z[:, 0]
    return np.stack([
        z1,
        z1 - z[:, 1],
        z1 - z[:, 1:10].mean(axis=1),
        z1 - z[:, 1:].mean(axis=1),
        z[:, :10].std(axis=1),
    ], axis=1)


def fit_eval(x_tr, y_tr, x_te, y_te, seed=0):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed)
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    out = {}
    if len(np.unique(y_te)) > 1:
        out["auroc"] = float(roc_auc_score(y_te, prob))
        out["auprc"] = float(average_precision_score(y_te, prob))
    return out, prob


def bucket_metrics(y, prob, buckets):
    out = {}
    for name, mask in buckets.items():
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            out[name] = None
            continue
        out[name] = {
            "n": int(mask.sum()),
            "base_rate": float(y[mask].mean()),
            "auroc": float(roc_auc_score(y[mask], prob[mask])),
            "auprc": float(average_precision_score(y[mask], prob[mask])),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--feats", required=True)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.cache, "rb") as fin:
        cache = pickle.load(fin)
    feats = np.load(args.feats)["features"]  # (N,50,14)
    queries = cache["queries"]
    n = len(queries)

    y = (cache["gold_block_pos"] != 0).astype(np.int32)  # 1 = error
    top1_feats = feats[:, 0, :]  # evidence features of the top-1 candidate
    block_mean = feats[:, :10, :].mean(axis=1)
    gaps = gap_features(cache["cand_raw_scores"].astype(np.float64))
    x_full = np.concatenate([gaps, top1_feats, block_mean], axis=1)
    x_gaps = gaps

    times = queries[:, 3].astype(np.float64)
    cutoff = np.quantile(times, 0.7)
    tr = times <= cutoff
    te = ~tr
    print("[S1] temporal split: train={} test={} (t<={})".format(tr.sum(), te.sum(), cutoff))

    subj = subject_history_counts(args.data_dir, "test", queries)
    buckets = {
        "all": np.ones(n, dtype=bool),
        "bucket_0": subj == 0,
        "bucket_1-9": (subj >= 1) & (subj <= 9),
        "bucket_10+": subj >= 10,
    }
    te_buckets = {k: v[te] for k, v in buckets.items()}

    results = {"temporal_cutoff": float(cutoff), "n_train": int(tr.sum()), "n_test": int(te.sum())}

    res_gaps, prob_gaps = fit_eval(x_gaps[tr], y[tr], x_gaps[te], y[te])
    results["gaps_only"] = {"overall": res_gaps, "buckets": bucket_metrics(y[te], prob_gaps, te_buckets)}

    res_full, prob_full = fit_eval(x_full[tr], y[tr], x_full[te], y[te])
    results["full_features"] = {"overall": res_full, "buckets": bucket_metrics(y[te], prob_full, te_buckets)}
    results["auroc_gain_full_vs_gaps"] = (
        res_full.get("auroc", float("nan")) - res_gaps.get("auroc", float("nan")))

    if args.samples and args.scores:
        llm = {}
        for line in open(args.samples):
            r = json.loads(line)
            if r["stratum"] != "S3_gold_outside":
                llm[r["sample_id"]] = r["query_index"]
        conf_map = {}
        for line in open(args.scores):
            s = json.loads(line)
            if s["sample_id"] in llm:
                conf_map[llm[s["sample_id"]]] = s["llm_conf"]
        idx_sub = np.array(sorted(conf_map.keys()))
        conf = np.array([conf_map[i] for i in idx_sub])
        conf_feats = np.stack([
            conf.max(axis=1), conf[:, 0], conf.std(axis=1),
            conf.max(axis=1) - np.sort(conf, axis=1)[:, -2],
        ], axis=1)
        x_llm = np.concatenate([x_full[idx_sub], conf_feats], axis=1)
        y_sub = y[idx_sub]
        tr_s, te_s = tr[idx_sub], te[idx_sub]
        res_llm, _prob = fit_eval(x_llm[tr_s], y_sub[tr_s], x_llm[te_s], y_sub[te_s])
        res_full_sub, _ = fit_eval(x_full[idx_sub][tr_s], y_sub[tr_s],
                                   x_full[idx_sub][te_s], y_sub[te_s])
        results["llm_subset"] = {
            "n_train": int(tr_s.sum()), "n_test": int(te_s.sum()),
            "full_plus_llm": res_llm,
            "full_no_llm": res_full_sub,
            "auroc_gain_llm": res_llm.get("auroc", float("nan")) - res_full_sub.get("auroc", float("nan")),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "s1_error_pred.json"), "w") as fout:
        json.dump(results, fout, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "s1_probs.npz"),
                        prob_gaps=prob_gaps, prob_full=prob_full,
                        y_test=y[te], queries_test=queries[te],
                        gold_rank_test=cache["gold_filter_rank"][te])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
