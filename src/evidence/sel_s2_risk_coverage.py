"""Task 6 / S2: risk-coverage curves (the value form of selective prediction).

Using S1's predicted error probabilities as the abstention signal:
coverage 1.0 -> 0.5 (step 0.05), keep the lowest-risk queries, report the
MRR of the kept set. Three curves:
  (a) TiRGN gap features only (GBDT)
  (b) full features (GBDT)
  (c) LLM confidence of the top-1 candidate as certainty signal
      (frozen model, no training; evaluated on P2's 1500-query subset)
Reports AURC (trapezoid over coverage) and MRR after abstaining 20%.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.sel_s2_risk_coverage \
        --probs ../results/task6_selective/s1_probs.npz \
        --cache ../candidates/seed1/ICEWS14_test_topk50.pkl \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --scores ../results/task5_pilot/p2_llm_scores.jsonl \
        --out ../results/task6_selective/s2_risk_coverage.json
"""

import argparse
import json
import pickle

import numpy as np

COVERAGES = [round(1.0 - 0.05 * i, 2) for i in range(11)]


def rc_curve(risk, gold_rank, coverages=COVERAGES):
    order = np.argsort(risk)  # keep lowest risk first
    ranks = gold_rank[order].astype(np.float64)
    n = len(ranks)
    curve = {}
    for c in coverages:
        k = max(1, int(round(n * c)))
        curve["{:.2f}".format(c)] = float(np.mean(1.0 / ranks[:k]))
    xs = [c for c in coverages][::-1]
    ys = [curve["{:.2f}".format(c)] for c in xs]
    aurc = float(np.trapz(ys, xs))
    return curve, aurc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = np.load(args.probs)
    gold_rank = data["gold_rank_test"]

    out = {}
    for key, label in (("prob_gaps", "gaps_only"), ("prob_full", "full_features")):
        curve, aurc = rc_curve(data[key], gold_rank)
        out[label] = {"curve": curve, "aurc": aurc,
                      "mrr_at_coverage_0.8": curve["0.80"],
                      "mrr_at_coverage_1.0": curve["1.00"]}

    # LLM confidence curve on the P2 subset
    with open(args.cache, "rb") as fin:
        cache = pickle.load(fin)
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
    conf0 = np.array([conf_map[i][0] for i in idx_sub])  # conf of top-1 cand
    gold_sub = cache["gold_filter_rank"][idx_sub]
    curve, aurc = rc_curve(-conf0, gold_sub)  # high conf = low risk
    out["llm_conf_subset"] = {"curve": curve, "aurc": aurc,
                              "mrr_at_coverage_0.8": curve["0.80"],
                              "mrr_at_coverage_1.0": curve["1.00"],
                              "n": int(len(idx_sub))}

    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)
    for label, res in out.items():
        print("{:18s} AURC={:.4f} | MRR@cov1.0={:.4f} MRR@cov0.8={:.4f} MRR@cov0.5={:.4f}".format(
            label, res["aurc"], res["mrr_at_coverage_1.0"],
            res["mrr_at_coverage_0.8"], res["curve"]["0.50"]))


if __name__ == "__main__":
    main()
