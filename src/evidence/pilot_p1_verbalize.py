"""Task 5 / P1: stratified sampling + evidence verbalization (CPU only).

From the seed1 test cache draw 2000 queries in 4 strata (500 each):
  S0: subject-history bucket 0, gold in Top-50
  S1: bucket 1-9, gold in Top-50
  S2: bucket 10+, gold in Top-50
  S3: gold NOT in Top-50 (any bucket; reserved for P3 recovery test)

For every query, the Top-10 candidates get a fixed-format verbalized
evidence block (entity/relation *names*, relative time "Xd ago"):
  - direct history (s,r,o,t') occurrences (up to 3 most recent + total count)
  - 1-2 hop relation paths ending at the candidate (up to 3)
Inverse queries (r >= num_rels, i.e. subject prediction) are phrased as
"(?, r_base, entity)" and their evidence uses the mirrored direction.

Output: results/task5_pilot/p1_samples.jsonl (one JSON per sample).

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.pilot_p1_verbalize \
        --cache ../candidates/seed1/ICEWS14_test_topk50.pkl \
        --data-dir ../data/ICEWS14 --n-per-stratum 500 \
        --out ../results/task5_pilot/p1_samples.jsonl
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from src.evidence.run_experiments import subject_history_counts

NUM_RELS = 230
SAMPLE_SEED = 42
MAX_DIRECT = 3
MAX_PATHS = 3


def load_names(data_dir):
    def read(fname):
        table = {}
        with open(os.path.join(data_dir, fname)) as fin:
            for line in fin:
                name, idx = line.rstrip("\n").rsplit("\t", 1)
                table[int(idx)] = name
        return table
    return read("entity2id.txt"), read("relation2id.txt")


def build_index(quads):
    sro = defaultdict(list)          # (s,r,o) -> [t]
    out_edges = defaultdict(list)    # s -> [(t, r, o)] sorted by t
    xo = defaultdict(list)           # (x,o) -> [(t, r)]
    ent_events = defaultdict(list)   # entity -> [(t, r, other, as_subject)]
    for s, r, o, t in quads:
        sro[(s, r, o)].append(t)
        out_edges[s].append((t, r, o))
        xo[(s, o)].append((t, r))
        ent_events[s].append((t, r, o, True))
        ent_events[o].append((t, r, s, False))
    for d in (sro, out_edges, xo, ent_events):
        for k in d:
            d[k].sort()
    return sro, out_edges, xo, ent_events


def _ago(t, tp):
    return "{}d ago".format(int(t - tp))


def verbalize_candidate(s_q, r_base, cand, t, forward, idx, ent, rel):
    """Evidence text for one candidate. For inverse queries the candidate
    plays the *subject* role of the base relation."""
    sro, out_edges, xo, _ent_events = idx
    lines = []
    if forward:
        subj, obj = s_q, cand
    else:
        subj, obj = cand, s_q
    times = sro.get((subj, r_base, obj), [])
    times = [tp for tp in times if tp < t]
    if times:
        recent = ", ".join(_ago(t, tp) for tp in times[-MAX_DIRECT:][::-1])
        lines.append("direct: \"{} {} {}\" x{} before query ({})".format(
            ent[subj], rel[r_base], ent[obj], len(times), recent))
    else:
        lines.append("direct: none")
    paths = []
    # 1-hop: subj -r_any-> obj
    for tp, r1 in reversed(xo.get((subj, obj), [])):
        if tp >= t:
            continue
        paths.append("{} -[{}]-> {} ({})".format(ent[subj], rel[r1], ent[obj], _ago(t, tp)))
        if len(paths) >= MAX_PATHS:
            break
    # 2-hop: subj -r1-> x -r2-> obj, t1 <= t2 < t
    if len(paths) < MAX_PATHS:
        for t1, r1, x in reversed(out_edges.get(subj, [])):
            if t1 >= t:
                continue
            for t2, r2 in reversed(xo.get((x, obj), [])):
                if t2 >= t or t2 < t1:
                    continue
                paths.append("{} -[{}]-> {} -[{}]-> {} ({}/{})".format(
                    ent[subj], rel[r1], ent[x], rel[r2], ent[obj], _ago(t, t1), _ago(t, t2)))
                break
            if len(paths) >= MAX_PATHS:
                break
    lines.append("paths: " + (" | ".join(paths) if paths else "none"))
    return "\n".join(lines)


def query_history_brief(s_q, t, idx, ent, rel, limit=10):
    """Most recent events involving the queried entity, either role (for P3)."""
    _sro, _out, _xo, ent_events = idx
    lines = []
    for tp, r, other, as_subject in reversed(ent_events.get(s_q, [])):
        if tp >= t:
            continue
        if as_subject:
            lines.append("{} -[{}]-> {} ({})".format(ent[s_q], rel[r], ent[other], _ago(t, tp)))
        else:
            lines.append("{} -[{}]-> {} ({})".format(ent[other], rel[r], ent[s_q], _ago(t, tp)))
        if len(lines) >= limit:
            break
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--n-per-stratum", type=int, default=500)
    parser.add_argument("--topk-evidence", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.cache, "rb") as fin:
        cache = pickle.load(fin)
    queries, cand_ids = cache["queries"], cache["cand_ids"]
    gold_pos, gold_rank = cache["gold_block_pos"], cache["gold_filter_rank"]
    subj_counts = subject_history_counts(args.data_dir, "test", queries)

    rng = np.random.RandomState(SAMPLE_SEED)
    in_top = gold_pos >= 0
    strata = {
        "S0_bucket0": np.where((subj_counts == 0) & in_top)[0],
        "S1_bucket1-9": np.where((subj_counts >= 1) & (subj_counts <= 9) & in_top)[0],
        "S2_bucket10+": np.where((subj_counts >= 10) & in_top)[0],
        "S3_gold_outside": np.where(~in_top)[0],
    }
    ent, rel = load_names(args.data_dir)
    train = np.loadtxt(os.path.join(args.data_dir, "train.txt"), dtype=np.int64)[:, :4]
    valid = np.loadtxt(os.path.join(args.data_dir, "valid.txt"), dtype=np.int64)[:, :4]
    idx = build_index(np.concatenate([train, valid], axis=0))

    n_written = 0
    with open(args.out, "w") as fout:
        for stratum, pool in strata.items():
            take = min(args.n_per_stratum, len(pool))
            chosen = rng.choice(pool, size=take, replace=False)
            for qi in chosen:
                s, r, o, t = (int(v) for v in queries[qi])
                forward = r < NUM_RELS
                r_base = r % NUM_RELS
                if forward:
                    qtext = "({}, {}, ?)".format(ent[s], rel[r_base])
                else:
                    qtext = "(?, {}, {})".format(rel[r_base], ent[s])
                cands = []
                kk = args.topk_evidence if stratum != "S3_gold_outside" else 0
                for j in range(kk):
                    c = int(cand_ids[qi, j])
                    cands.append({
                        "ent": c, "name": ent[c], "orig_rank": j + 1,
                        "raw_score": float(cache["cand_raw_scores"][qi, j]),
                        "evidence": verbalize_candidate(s, r_base, c, t, forward, idx, ent, rel),
                    })
                rec = {
                    "sample_id": n_written,
                    "stratum": stratum,
                    "query_index": int(qi),
                    "query": {"s": s, "r": r, "o_gold": o, "t": t,
                              "forward": forward, "text": qtext,
                              "subject_history_count": int(subj_counts[qi])},
                    "gold": {"ent": o, "name": ent[o],
                             "block_pos": int(gold_pos[qi]),
                             "filter_rank": int(gold_rank[qi])},
                    "recent_history": query_history_brief(s, t, idx, ent, rel),
                    "candidates": cands,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            print("[P1] stratum {} pool={} taken={}".format(stratum, len(pool), take))
    print("[P1] wrote {} samples to {}".format(n_written, args.out))


if __name__ == "__main__":
    main()
