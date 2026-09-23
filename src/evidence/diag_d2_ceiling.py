"""Task 4 / D2: feature ceiling with a strong supervised reranker.

Question: is the Phase-2 failure caused by uninformative features or by the
linear fusion? A LightGBM reranker is trained on the validation split with:
  - all 14 E1-E4 features (standardized on valid)
  - all pairwise interaction terms (105)
  - per-query z-normalized raw TiRGN score (plus its interactions)
target = candidate is gold (queries with gold inside Top-50 only).

Evaluation on test follows the exact Phase-2 protocol: rerank strictly
inside the Top-50 block, filtered MRR / Hits recomputed from the cache.

Reports per seed and 3-seed mean+-std:
  A (no rerank), F (Phase-2 logistic on E2+gap), GBDT rerank.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.diag_d2_ceiling \
        --tags seed1 seed2 seed3 --cache-dir ../candidates \
        --out ../results/task4_diagnostics/d2_ceiling.json
"""

import argparse
import json
import os
import pickle

import numpy as np
import lightgbm as lgb

from src.evidence.fuse import fit_logistic, standardize, rerank_metrics

F_COLS = [4, 5, 6, 7, 13]  # Phase-2 F_RCEV feature subset
LAMBDA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def build_matrix(feats_std, raw_scores):
    """(N,K,14) standardized feats + (N,K) raw scores -> (N,K,120) matrix."""
    n, k, f = feats_std.shape
    mu = raw_scores.mean(axis=1, keepdims=True)
    sd = raw_scores.std(axis=1, keepdims=True) + 1e-8
    score_norm = ((raw_scores - mu) / sd)[:, :, None]
    base = np.concatenate([feats_std, score_norm], axis=2)  # (N,K,15)
    inter = []
    for i in range(base.shape[2]):
        for j in range(i + 1, base.shape[2]):
            inter.append(base[:, :, i] * base[:, :, j])
    return np.concatenate([base] + [x[:, :, None] for x in inter], axis=2)


def rerank_mrr(cache, score):
    gold_pos = cache["gold_block_pos"]
    gold_rank = cache["gold_filter_rank"].astype(np.float64)
    new_rank = gold_rank.copy()
    in_block = gold_pos >= 0
    rows = np.arange(len(gold_pos))[in_block]
    gold_s = score[rows, gold_pos[in_block]]
    better = (score[rows] > gold_s[:, None]).sum(axis=1)
    new_rank[in_block] = 1.0 + better
    return {"mrr": float(np.mean(1.0 / new_rank)),
            "hits": {h: float(np.mean(new_rank <= h)) for h in (1, 3, 10)}}


def run_tag(cache_dir, tag, seed):
    data = {}
    for split in ("valid", "test"):
        with open(os.path.join(cache_dir, tag, "ICEWS14_{}_topk50.pkl".format(split)), "rb") as fin:
            cache = pickle.load(fin)
        feats = np.load(os.path.join(cache_dir, tag, "ICEWS14_{}_topk50_feats.npz".format(split)))["features"]
        data[split] = (cache, feats)
    (cv, fv), (ct, ft) = data["valid"], data["test"]
    fvs, fts = standardize(fv, ft)

    # --- A baseline
    res_a = rerank_mrr(ct, ct["cand_raw_scores"].astype(np.float64))

    # --- F (Phase-2 logistic, identical protocol)
    w = fit_logistic(fvs[:, :, F_COLS], cv["gold_block_pos"], seed=0)
    ev, et = fvs[:, :, F_COLS] @ w, fts[:, :, F_COLS] @ w
    best_lam, best_vm = None, -1.0
    for lam in LAMBDA_GRID:
        res_v = rerank_metrics(cv, ev, lam=lam, feats=fv)
        if res_v["mrr"] > best_vm:
            best_vm, best_lam = res_v["mrr"], lam
    res_f_raw = rerank_metrics(ct, et, lam=best_lam, feats=ft)
    res_f = {"mrr": res_f_raw["mrr"], "hits": res_f_raw["hits"]}

    # --- GBDT reranker
    xv = build_matrix(fvs, cv["cand_raw_scores"])
    xt = build_matrix(fts, ct["cand_raw_scores"])
    n, k, d = xv.shape
    mask = cv["gold_block_pos"] >= 0
    x = xv[mask].reshape(-1, d)
    y = np.zeros(x.shape[0], dtype=np.int32)
    y[np.arange(mask.sum()) * k + cv["gold_block_pos"][mask]] = 1

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_hold = int(0.2 * len(y))
    hold, tr = perm[:n_hold], perm[n_hold:]
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed)
    clf.fit(x[tr], y[tr], eval_set=[(x[hold], y[hold])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    score_t = clf.predict_proba(xt.reshape(-1, xt.shape[2]))[:, 1].reshape(xt.shape[0], xt.shape[1])
    res_g = rerank_mrr(ct, score_t)
    res_g_valid = rerank_mrr(cv, clf.predict_proba(xv.reshape(-1, d))[:, 1].reshape(xv.shape[0], xv.shape[1]))

    importances = dict(zip(
        ["f{}".format(i) for i in range(x.shape[1])],
        [int(v) for v in clf.feature_importances_]))
    return {
        "A": res_a, "F_logistic": dict(res_f, **{"lambda": best_lam}),
        "GBDT": res_g, "GBDT_valid_mrr": res_g_valid["mrr"],
        "gbdt_best_iteration": int(clf.best_iteration_ or 500),
        "gbdt_top_features": sorted(importances.items(), key=lambda kv: -kv[1])[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="../candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_tag = {}
    for i, tag in enumerate(args.tags):
        print("[D2] training GBDT for {}".format(tag))
        per_tag[tag] = run_tag(args.cache_dir, tag, args.seed + i)

    agg = {}
    for cfg in ("A", "F_logistic", "GBDT"):
        agg[cfg] = {
            "mrr_mean": float(np.mean([per_tag[t][cfg]["mrr"] for t in args.tags])),
            "mrr_std": float(np.std([per_tag[t][cfg]["mrr"] for t in args.tags])),
            "hits": {h: {"mean": float(np.mean([per_tag[t][cfg]["hits"][h] for t in args.tags])),
                         "std": float(np.std([per_tag[t][cfg]["hits"][h] for t in args.tags]))}
                     for h in (1, 3, 10)},
        }
    out = {"per_tag": per_tag, "aggregate": agg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)
    print("[D2] saved {}".format(args.out))
    for cfg in ("A", "F_logistic", "GBDT"):
        a = agg[cfg]
        print("{:12s} MRR={:.4f}+-{:.4f} H@1={:.4f} H@3={:.4f} H@10={:.4f}".format(
            cfg, a["mrr_mean"], a["mrr_std"],
            a["hits"][1]["mean"], a["hits"][3]["mean"], a["hits"][10]["mean"]))


if __name__ == "__main__":
    main()
