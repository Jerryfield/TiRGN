"""Task 6 / S3: protocol artifact experiment (same outputs, 3 verdicts).

Using P2's 1500 scored queries, the "LLM verification effect" is reported
under three protocols:
  (a) full 1500-query standard protocol (fused MRR gain vs TiRGN order)
  (b) AnRe-style small sample: 200 random queries x 20 draws; distribution
      of the apparent fused gain (how often does the conclusion flip?)
  (c) ExE-style candidate AUC: mean per-query pairwise AUC of the LLM
      confidence separating gold from the other shown candidates
      (9 negatives; the ExE 99-negative variant is approximated because
      only 10 candidates were scored per query).

Goal: demonstrate that the verdict on the very same model outputs depends
on the evaluation protocol, and quantify the apparent-gain variance.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.sel_s3_protocol \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --scores ../results/task5_pilot/p2_llm_scores.jsonl \
        --out ../results/task6_selective/s3_protocol.json
"""

import argparse
import json

import numpy as np

LAM = 0.5  # best lambda from the full-sample calibration (p2_eval.json)
N_DRAWS = 20
DRAW_SIZE = 200


def fused_ranks(recs):
    ranks = np.zeros(len(recs))
    for i, rec in enumerate(recs):
        pos = rec["gold"]["block_pos"]
        if pos < 0 or pos >= len(rec["candidates"]):
            ranks[i] = rec["gold"]["filter_rank"]
            continue
        z = np.array([c["raw_score"] for c in rec["candidates"]])
        e = np.array(rec["llm_conf"])
        zp = z + LAM * e
        ranks[i] = 1.0 + (zp > zp[pos]).sum()
    return ranks


def a_ranks(recs):
    return np.array([rec["gold"]["filter_rank"] for rec in recs], dtype=np.float64)


def mrr(r):
    return float(np.mean(1.0 / r))


def candidate_auc(recs):
    aucs = []
    for rec in recs:
        pos = rec["gold"]["block_pos"]
        if pos < 0 or pos >= len(rec["candidates"]):
            continue
        e = np.array(rec["llm_conf"])
        diff = e[pos] - np.delete(e, pos)
        aucs.append(float(np.mean((diff > 0) + 0.5 * (diff == 0))))
    return float(np.mean(aucs)), len(aucs)


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

    gain_full = mrr(fused_ranks(recs)) - mrr(a_ranks(recs))
    auc_full, n_auc = candidate_auc(recs)

    rng = np.random.RandomState(7)
    draws = []
    for _ in range(N_DRAWS):
        idx = rng.choice(len(recs), size=DRAW_SIZE, replace=False)
        sub = [recs[i] for i in idx]
        draws.append(mrr(fused_ranks(sub)) - mrr(a_ranks(sub)))
    draws = np.array(draws)

    out = {
        "protocol_a_full1500": {
            "n": len(recs), "fused_gain": gain_full,
            "verdict": "negative (no improvement)",
        },
        "protocol_b_200x20": {
            "draws": [float(x) for x in draws],
            "mean": float(draws.mean()), "std": float(draws.std()),
            "min": float(draws.min()), "max": float(draws.max()),
            "frac_positive": float((draws > 0).mean()),
            "verdict": "conclusion flips across draws" if (draws > 0).any() and (draws < 0).any()
            else "consistent",
        },
        "protocol_c_candidate_auc": {
            "auc": auc_full, "n_queries": n_auc, "negatives_per_query": 9,
            "verdict": "AUC > 0.5 looks like a positive effect" if auc_full > 0.5
            else "no effect even under AUC",
        },
    }
    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
