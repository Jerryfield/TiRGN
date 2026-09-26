"""Task 5 / P3: recovery test for queries whose gold is outside Top-50.

The frozen LLM (Qwen3-8B 4bit) sees only the query plus verbalized recent
history (no candidates) and freely nominates up to 5 entities
(configs/p3_recovery_prompt.md). Nominations are mapped back to entity ids
by exact / normalized name matching.

Reports: recovery rate (gold nominated), and the hypothetical MRR change on
the S3 sample if a nominated gold were inserted into the ranking at its
nomination position (others pushed down).

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.pilot_p3_recover \
        --samples ../results/task5_pilot/p1_samples.jsonl \
        --data-dir ../data/ICEWS14 \
        --out-dir ../results/task5_pilot --gpu 2
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = """You are answering temporal knowledge graph queries about political events.
Given a query and the recent event history of the queried entity, nominate
up to 5 real-world entities that could plausibly be the missing entity.
Use exact entity names as they appear in the history or well-known English
names of countries, organizations, and public figures.
Output strict JSON only: {"nominations": ["name1", "name2", ...],
"reasoning": "one sentence"}"""

USER_TEMPLATE = """Query: at day {t}, predict the missing entity: {qtext}
Recent events involving the queried entity:
{history}

Nominate up to 5 entities. Return the JSON now."""


def normalize(name):
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    recs = [json.loads(l) for l in open(args.samples)]
    recs = [r for r in recs if r["stratum"] == "S3_gold_outside"]
    if args.max_queries:
        recs = recs[: args.max_queries]
    print("[P3] {} recovery queries".format(len(recs)))

    name2id = {}
    with open(os.path.join(args.data_dir, "entity2id.txt")) as fin:
        for line in fin:
            name, eid = line.rstrip("\n").rsplit("\t", 1)
            name2id[name] = int(eid)
            name2id[normalize(name)] = int(eid)

    torch.cuda.set_device(args.gpu)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": args.gpu},
        torch_dtype=torch.float16)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "p3_nominations.jsonl")
    n_recovered, n_parsed = 0, 0
    ranks_old, ranks_new = [], []
    with open(out_path, "w") as fout:
        for qi, rec in enumerate(recs):
            history = "\n".join("- " + h for h in rec["recent_history"]) or "none"
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            t=rec["query"]["t"], qtext=rec["query"]["text"],
                            history=history)}]
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            parsed = parse_json(gen)
            noms = []
            if parsed and isinstance(parsed.get("nominations"), list):
                noms = [str(x) for x in parsed["nominations"][:5]]
                n_parsed += 1
            nom_ids = []
            for nm in noms:
                eid = name2id.get(nm, name2id.get(normalize(nm)))
                if eid is not None and eid not in nom_ids:
                    nom_ids.append(eid)
            gold = rec["gold"]["ent"]
            hit_pos = nom_ids.index(gold) + 1 if gold in nom_ids else 0
            old_rank = rec["gold"]["filter_rank"]
            new_rank = float(hit_pos) if hit_pos else float(old_rank)
            if hit_pos:
                n_recovered += 1
            ranks_old.append(float(old_rank))
            ranks_new.append(new_rank)
            fout.write(json.dumps({
                "sample_id": rec["sample_id"], "nominations": noms,
                "nom_ids": nom_ids, "gold": gold, "hit_pos": hit_pos,
                "old_rank": old_rank,
            }) + "\n")
            if (qi + 1) % 50 == 0:
                print("[P3] {}/{} done, recovered={}".format(qi + 1, len(recs), n_recovered))

    summary = {
        "n": len(recs), "n_parsed": n_parsed,
        "recovered": n_recovered,
        "recovery_rate": n_recovered / max(len(recs), 1),
        "mrr_old": float(np.mean(1.0 / np.array(ranks_old))),
        "mrr_new_with_insert": float(np.mean(1.0 / np.array(ranks_new))),
        "decision": "PASS" if n_recovered / max(len(recs), 1) >= 0.05 else "FAIL (<5% recovery)",
    }
    with open(os.path.join(args.out_dir, "p3_summary.json"), "w") as fout:
        json.dump(summary, fout, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
