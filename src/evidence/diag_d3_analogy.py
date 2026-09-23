"""Task 4 / D3: analogy-based evidence prototype (AnRe-style, no LLM).

Motivation: F gained exactly 0 in the 0-2 history bucket because sparse
entities have no own history to mine. The only untested evidence source is
the history of *similar* entities.

Entity similarity (union of two channels):
  (a) semantic: sentence-transformers all-MiniLM-L6-v2 over entity names
  (b) structural: cosine over per-entity relation profile vectors
      (counts over 2*num_rels dims, as-subject / as-object, train split)
For each sparse query subject s: top-M similar entities per channel, union;
weight = max(channel similarities), clamped at 0.

Analogy score for query (s,r,?,t) and candidate o:
  score(o) = sum_{s'} sim(s,s') * #{(s', r, o, t') : t' < t}
Enabled only for queries whose subject history event count <= 9; all other
queries keep the config-A ranking. Fusion z' = z + lambda*rho*e with rho =
in-block evidence coverage, lambda selected on valid (same protocol as
Phase-2). M sensitivity: {5, 10, 20}.

Analogy evidence depends only on (query, history), not on the model, so it
is computed once per split and reused across seeds.

Usage (from src/):
    PYTHONPATH=.. python -m src.evidence.diag_d3_analogy \
        --tags seed1 seed2 seed3 --cache-dir ../candidates \
        --data-dir ../data/ICEWS14 \
        --out ../results/task4_diagnostics/d3_analogy.json
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from src.evidence.run_experiments import subject_history_counts

MS = (5, 10, 20)
LAMBDA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SPARSE_MAX = 9
NUM_RELS = 230


def load_names(data_dir):
    names = [None] * 80000
    max_id = 0
    with open(os.path.join(data_dir, "entity2id.txt")) as fin:
        for line in fin:
            name, eid = line.rstrip("\n").rsplit("\t", 1)
            names[int(eid)] = name
            max_id = max(max_id, int(eid))
    return names[:max_id + 1]


def relation_profiles(quads, num_ents):
    prof = np.zeros((num_ents, 2 * NUM_RELS), dtype=np.float64)
    for s, r, o, _t in quads:
        prof[s, r] += 1
        prof[o, r + NUM_RELS] += 1
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return prof / norm


def semantic_embeddings(names):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(names, batch_size=256, show_progress_bar=False,
                       normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float64)


def topm_union(s, sem, prof, m):
    """Union of top-M similar entities from both channels, weight = max sim."""
    sem_sim = sem @ sem[s]
    prof_sim = prof @ prof[s]
    sem_sim[s] = -1
    prof_sim[s] = -1
    out = {}
    for arr in (sem_sim, prof_sim):
        idx = np.argpartition(-arr, m)[:m]
        for j in idx:
            if arr[j] <= 0:
                continue
            out[int(j)] = max(out.get(int(j), 0.0), float(arr[j]))
    return out


def build_history(quads):
    fwd = defaultdict(lambda: defaultdict(list))  # (s,r) -> o -> [t]
    inv = defaultdict(lambda: defaultdict(list))  # (o,r) -> s -> [t]
    for s, r, o, t in quads:
        fwd[(s, r)][o].append(t)
        inv[(o, r)][s].append(t)
    for hist in (fwd, inv):
        for key in hist:
            for e in hist[key]:
                hist[key][e].sort()
    return fwd, inv


def analogy_scores(queries, cand_ids, subj_counts, hist, neighbors):
    """(N,K) analogy evidence; zero rows for non-sparse queries."""
    n, k = cand_ids.shape
    scores = np.zeros((n, k), dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    for i in range(n):
        if subj_counts[i] > SPARSE_MAX:
            continue
        s, r, _o, t = (int(v) for v in queries[i])
        r_rel = r % NUM_RELS
        forward = r < NUM_RELS
        neigh = neighbors.get(int(s))
        if not neigh:
            continue
        acc = defaultdict(float)
        for sp, w in neigh.items():
            # forward query (s,r,?): evidence from (s',r,o') counting objects;
            # inverse query (s,r+num_rels,?): evidence from (x,r,s') counting
            # subjects x (the queried entity s plays the object role here)
            rel_hist = hist[0].get((sp, r_rel)) if forward else hist[1].get((sp, r_rel))
            if not rel_hist:
                continue
            for o, times in rel_hist.items():
                cnt = np.searchsorted(times, t, side="left")
                if cnt:
                    acc[o] += w * cnt
        if not acc:
            continue
        covered[i] = True
        for j in range(k):
            scores[i, j] = acc.get(int(cand_ids[i, j]), 0.0)
    return scores, covered


def rerank_sparse(cache, evid, lam, mask):
    """Rerank inside Top-50 block only for masked (sparse) queries."""
    z = cache["cand_raw_scores"].astype(np.float64)
    gold_pos = cache["gold_block_pos"]
    new_rank = cache["gold_filter_rank"].astype(np.float64).copy()
    in_block = gold_pos >= 0
    rows = np.arange(len(gold_pos))[in_block & mask]
    for i in rows:
        cov = float((evid[i] > 0).mean())
        zprime = z[i] + lam * cov * evid[i]
        new_rank[i] = 1.0 + (zprime > zprime[gold_pos[i]]).sum()
    return new_rank


def mrr_of(ranks):
    return float(np.mean(1.0 / ranks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="../candidates")
    parser.add_argument("--data-dir", default="../data/ICEWS14")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train = np.loadtxt(os.path.join(args.data_dir, "train.txt"), dtype=np.int64)[:, :4]
    valid_q = np.loadtxt(os.path.join(args.data_dir, "valid.txt"), dtype=np.int64)[:, :4]
    histories = {"valid": train, "test": np.concatenate([train, valid_q], axis=0)}

    names = load_names(args.data_dir)
    num_ents = len(names)
    print("[D3] computing relation profiles and name embeddings for {} entities".format(num_ents))
    prof = relation_profiles(train, num_ents)
    sem = semantic_embeddings(names)

    caches = {}
    for tag in args.tags:
        caches[tag] = {}
        for split in ("valid", "test"):
            with open(os.path.join(args.cache_dir, tag, "ICEWS14_{}_topk50.pkl".format(split)), "rb") as fin:
                caches[tag][split] = pickle.load(fin)

    subj_counts = {}
    for split in ("valid", "test"):
        subj_counts[split] = subject_history_counts(args.data_dir, split, caches[args.tags[0]][split]["queries"])

    sparse_subjects = set()
    for split in ("valid", "test"):
        qs = caches[args.tags[0]][split]["queries"]
        for i in range(len(qs)):
            if subj_counts[split][i] <= SPARSE_MAX:
                sparse_subjects.add(int(qs[i][0]))
    print("[D3] {} unique sparse subjects".format(len(sparse_subjects)))

    neighbors = {m: {s: topm_union(s, sem, prof, m) for s in sparse_subjects} for m in MS}

    hist_idx = {split: build_history(histories[split]) for split in ("valid", "test")}

    out = {"coverage": {}, "per_m": {}}
    for m in MS:
        evid, covered = {}, {}
        for split in ("valid", "test"):
            evid[split], covered[split] = analogy_scores(
                caches[args.tags[0]][split]["queries"],
                caches[args.tags[0]][split]["cand_ids"],
                subj_counts[split], hist_idx[split], neighbors[m])
        sparse_mask = {s: subj_counts[s] <= SPARSE_MAX for s in ("valid", "test")}
        cov_stats = {
            split: float(covered[split][sparse_mask[split]].mean())
            for split in ("valid", "test")
        }
        out["coverage"][str(m)] = cov_stats

        per_tag = {}
        for tag in args.tags:
            cv, ct = caches[tag]["valid"], caches[tag]["test"]
            base_rank_v = cv["gold_filter_rank"].astype(np.float64)
            base_rank_t = ct["gold_filter_rank"].astype(np.float64)
            best_lam, best_vm = None, -1.0
            for lam in LAMBDA_GRID:
                rv = rerank_sparse(cv, evid["valid"], lam, sparse_mask["valid"])
                if mrr_of(rv) > best_vm:
                    best_vm, best_lam = mrr_of(rv), lam
            rt = rerank_sparse(ct, evid["test"], best_lam, sparse_mask["test"])
            buckets = {}
            for bname, lo, hi in (("0", 0, 0), ("1-2", 1, 2), ("3-9", 3, 9)):
                bmask = (subj_counts["test"] >= lo) & (subj_counts["test"] <= hi)
                buckets[bname] = {
                    "n": int(bmask.sum()),
                    "mrr_A": mrr_of(base_rank_t[bmask]),
                    "mrr_D3": mrr_of(rt[bmask]),
                    "gain": mrr_of(rt[bmask]) - mrr_of(base_rank_t[bmask]),
                }
            per_tag[tag] = {
                "lambda": best_lam, "valid_mrr": best_vm,
                "test_mrr": mrr_of(rt), "test_mrr_A": mrr_of(base_rank_t),
                "buckets": buckets,
            }
        agg = {
            "test_mrr_mean": float(np.mean([per_tag[t]["test_mrr"] for t in args.tags])),
            "test_mrr_std": float(np.std([per_tag[t]["test_mrr"] for t in args.tags])),
            "buckets": {
                b: {
                    "gain_mean": float(np.mean([per_tag[t]["buckets"][b]["gain"] for t in args.tags])),
                    "gain_std": float(np.std([per_tag[t]["buckets"][b]["gain"] for t in args.tags])),
                    "mrr_A_mean": float(np.mean([per_tag[t]["buckets"][b]["mrr_A"] for t in args.tags])),
                    "mrr_D3_mean": float(np.mean([per_tag[t]["buckets"][b]["mrr_D3"] for t in args.tags])),
                } for b in ("0", "1-2", "3-9")
            },
        }
        out["per_m"][str(m)] = {"per_tag": per_tag, "aggregate": agg}
        print("[D3] M={}: coverage valid={:.3f} test={:.3f} | test MRR={:.4f}+-{:.4f} | bucket gains {}".format(
            m, cov_stats["valid"], cov_stats["test"],
            agg["test_mrr_mean"], agg["test_mrr_std"],
            {b: round(agg["buckets"][b]["gain_mean"], 4) for b in agg["buckets"]}))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fout:
        json.dump(out, fout, indent=2)
    print("[D3] saved {}".format(args.out))


if __name__ == "__main__":
    main()
