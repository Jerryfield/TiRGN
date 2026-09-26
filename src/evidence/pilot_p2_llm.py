"""Task 5 / P2: frozen LLM candidate verification (Qwen3-8B 4bit, no finetuning).

For every sampled query (strata S0/S1/S2, gold in Top-50) the Top-10
candidates are verified with a listwise prompt (configs/p2_listwise_prompt.md).
Each query runs twice with different candidate shuffles (position-bias
calibration); confidences are averaged per candidate.

Fusion and evaluation follow Phase-2: z' = z + lambda * e, rerank strictly
inside the shown Top-10 prefix (positions 11+ untouched), filtered MRR from
the cache. lambda is selected on a deterministic 50% calibration split of
the samples (even sample ids) and applied to the other half (eval split).
Rationale: running the LLM on the full validation set would double the GPU
cost; the split keeps tuning and evaluation disjoint within the pilot.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.pilot_p2_llm \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --out-dir ../results/task5_pilot --gpu 3 [--max-queries N]
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = """You are verifying candidate answers for temporal knowledge graph queries.
You will read a query about a political event prediction and evidence blocks
for 10 candidate entities. Judge which candidate is most likely the correct
missing entity, based ONLY on the evidence content (recurrence, temporal
paths, recency) and your world knowledge about these political entities.
Output strict JSON only, no other text:
{"ranking": [candidate ids from most to least likely, all 10],
 "confidence": [10 floats between 0 and 1, aligned with ranking order],
 "reasoning": "one sentence"}"""

USER_TEMPLATE = """Query: at day {t}, predict the missing entity: {qtext}
Context: recent events involving the queried entity:
{history}

Candidates (id. name):
{blocks}

Return the JSON now."""

LAMBDA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def build_user(rec, order):
    history = "\n".join("- " + h for h in rec["recent_history"][:5]) or "none"
    blocks = []
    for cid, j in enumerate(order):
        c = rec["candidates"][j]
        blocks.append("[{}] {}\n{}".format(cid, c["name"], c["evidence"]))
    return USER_TEMPLATE.format(t=rec["query"]["t"], qtext=rec["query"]["text"],
                                history=history, blocks="\n\n".join(blocks))


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def confidences_for_pass(rec, order, output):
    """Map model output to conf aligned with the original top-10 order."""
    conf = np.full(len(order), np.nan)
    parsed = parse_json(output)
    if not parsed or "confidence" not in parsed or "ranking" not in parsed:
        return conf, False
    try:
        ranking = [int(x) for x in parsed["ranking"]]
        confs = [float(x) for x in parsed["confidence"]]
        for cid, cf in zip(ranking, confs):
            if 0 <= cid < len(order):
                conf[order.index(cid)] = cf
    except (ValueError, TypeError):
        return conf, False
    ok = not np.isnan(conf).any()
    return conf, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="skip samples already present in the scores file")
    args = parser.parse_args()

    recs = [json.loads(l) for l in open(args.samples)]
    recs = [r for r in recs if r["stratum"] != "S3_gold_outside"]
    if args.max_queries:
        recs = recs[: args.max_queries]
    print("[P2] {} queries to verify".format(len(recs)))

    torch.cuda.set_device(args.gpu)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": args.gpu},
        torch_dtype=torch.float16)
    model.eval()

    scores_path = os.path.join(args.out_dir, "p2_llm_scores.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    done_ids = set()
    if args.resume and os.path.exists(scores_path):
        for line in open(scores_path):
            done_ids.add(json.loads(line)["sample_id"])
        print("[P2] resume: {} samples already done, skipping".format(len(done_ids)))
        recs = [r for r in recs if r["sample_id"] not in done_ids]
        print("[P2] {} queries remaining".format(len(recs)))
    n_parse_fail = 0
    with open(scores_path, "a" if args.resume else "w") as fout:
        for qi, rec in enumerate(recs):
            conf_sum = np.zeros(len(rec["candidates"]))
            n_ok = 0
            for pass_id in (0, 1):
                rng = np.random.RandomState(1000 * pass_id + rec["sample_id"])
                order = list(rng.permutation(len(rec["candidates"])))
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user(rec, order)}]
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
                inputs = tok(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                conf, ok = confidences_for_pass(rec, order, gen)
                if ok:
                    conf_sum += conf
                    n_ok += 1
                else:
                    n_parse_fail += 1
                    print("[P2] parse fail sample {} pass {}: {}".format(
                        rec["sample_id"], pass_id, gen[:120].replace("\n", " ")))
            llm_conf = (conf_sum / n_ok).tolist() if n_ok else [0.0] * len(conf_sum)
            fout.write(json.dumps({
                "sample_id": rec["sample_id"], "stratum": rec["stratum"],
                "llm_conf": llm_conf, "n_ok_passes": n_ok,
            }) + "\n")
            if (qi + 1) % 50 == 0:
                print("[P2] {}/{} queries done, parse_fail={}".format(qi + 1, len(recs), n_parse_fail))
    print("[P2] scores saved to {}, total parse fails: {}".format(scores_path, n_parse_fail))


if __name__ == "__main__":
    main()
