"""Task 5 / P2 evaluation: fuse LLM confidences with TiRGN scores.

Protocol: rerank strictly inside the shown Top-10 prefix; positions 11+
keep their original ranks. lambda is selected on the calibration half
(even sample ids) and applied to the eval half (odd sample ids).

Reports on the eval half: A (lambda=0), F_fused (best lambda),
pure-LLM (rerank by confidence only); overall and per stratum.
Also applies the go/no-go decision rule from the pilot prompt.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.pilot_p2_eval \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --scores ../results/task5_pilot/p2_llm_scores.jsonl \
        --out ../results/task5_pilot/p2_eval.json
"""

import argparse
import json
import os

import numpy as np

LAMBDA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DECISION_MRR_CEILING = 0.4415
DECISION_SPARSE_GAIN = 0.01


def new_ranks(recs, score_fn):
    """score_fn(rec) -> (K,) fused scores over shown candidates."""
    ranks = np.zeros(len(recs))
    for i, rec in enumerate(recs):
        pos = rec["gold"]["block_pos"]
        if pos < 0 or pos >= len(rec["candidates"]):
            ranks[i] = rec["gold"]["filter_rank"]  # not shown: unchanged
            continue
        z = score_fn(rec)
        ranks[i] = 1.0 + (z > z[pos]).sum()
    return ranks


def mrr(ranks):
    return float(np.mean(1.0 / ranks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    recs = {}
    for line in open(args.samples):
        r = json.loads(line)
        if r["stratum"] != "S3_gold_outside":
            recs[r["sample_id"]] = r
    for line in open(args.scores):
        s = json.loads(line)
        if s["sample_id"] in recs:
            recs[s["sample_id"]]["llm_conf"] = s["llm_conf"]
    recs = [r for r in recs.values() if "llm_conf" in r]

    calib = [r for r in recs if r["sample_id"] % 2 == 0]
    ev = [r for r in recs if r["sample_id"] % 2 == 1]

    def z_raw(rec):
        return np.array([c["raw_score"] for c in rec["candidates"]])

    def e_llm(rec):
        return np.array(rec["llm_conf"])

    calib_mrr = {}
    for lam in LAMBDA_GRID:
        calib_mrr[lam] = mrr(new_ranks(calib, lambda r: z_raw(r) + lam * e_llm(r)))
    best_lam = max(calib_mrr, key=calib_mrr.get)

    def report(subset):
        out = {
            "n": len(subset),
            "A": mrr(new_ranks(subset, z_raw)),
            "pure_llm": mrr(new_ranks(subset, e_llm)),
            "fused": mrr(new_ranks(subset, lambda r: z_raw(r) + best_lam * e_llm(r))),
        }
        out["gain_vs_A"] = out["fused"] - out["A"]
        return out

    strata = sorted(set(r["stratum"] for r in recs))
    summary = {
        "best_lambda": best_lam,
        "calib_mrr_by_lambda": {str(k): v for k, v in calib_mrr.items()},
        "eval_all": report(ev),
        "eval_by_stratum": {s: report([r for r in ev if r["stratum"] == s]) for s in strata},
        "eval_sparse": report([r for r in ev if r["stratum"] in ("S0_bucket0", "S1_bucket1-9")]),
    }
    sparse_gain = summary["eval_sparse"]["gain_vs_A"]
    verdict = []
    verdict.append("sample fused MRR = {:.4f} vs ceiling 0.4415 -> {}".format(
        summary["eval_all"]["fused"],
        "PASS" if summary["eval_all"]["fused"] > DECISION_MRR_CEILING else "FAIL"))
    verdict.append("sparse bucket gain = {:+.4f} vs required +0.01 -> {}".format(
        sparse_gain, "PASS" if sparse_gain >= DECISION_SPARSE_GAIN else "FAIL"))
    summary["decision"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(summary, fout, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
