"""Dump Top-K entity candidates per query after model.predict().

Mounted in main.py's test() loop. For every evaluated query (object
prediction, ground-truth history setting) we record:

  - the query quadruple (s, r, o_gold, t)
  - Top-K candidate entity ids under the *filtered* score (same filtering as
    the official evaluation: other true objects at the same timestamp are
    pushed to -1e7), ordered by descending filtered score
  - each candidate's raw (pre-filter) final_score value
  - each candidate's filtered score value
  - the gold entity's raw rank and filtered rank (1-indexed, computed with the
    same torch.sort based sort_and_rank as the official eval)
  - whether the gold entity appears inside the Top-K block

Reranking constraint used downstream: only the order *inside* the Top-K block
may change. Since the block holds filtered-ranks 1..K, any internal rerank
keeps every entity outside the block at its original rank, so standard
filtered MRR / Hits@K recomputed from this cache remain valid.
"""

import os
import pickle

import numpy as np
import torch

from rgcn import utils


class CandidateDumper:
    def __init__(self, topk):
        self.topk = topk
        self.queries = []
        self.cand_ids = []
        self.cand_filtered_scores = []
        self.cand_raw_scores = []
        self.gold_raw_rank = []
        self.gold_filter_rank = []
        self.gold_block_pos = []  # position inside top-K block, -1 if absent

    @torch.no_grad()
    def update(self, test_triples, final_score, all_ans_snap):
        """test_triples: (N, 4) array/tensor of (s, r, o, t).
        final_score: (N, num_nodes) tensor, the raw model output.
        all_ans_snap: all_ans_list[time_idx] used by the official filtering.
        """
        if not isinstance(test_triples, np.ndarray):
            test_triples = test_triples.cpu().numpy()
        raw_score = final_score.detach().clone()
        filter_score = utils.filter_score(
            torch.LongTensor(test_triples), final_score.detach().clone(), all_ans_snap)

        target = torch.LongTensor(test_triples[:, 2]).to(raw_score.device)
        raw_rank = utils.sort_and_rank(raw_score, target) + 1
        filt_rank = utils.sort_and_rank(filter_score.clone(), target) + 1

        k = min(self.topk, filter_score.size(1))
        top_score, top_idx = torch.topk(filter_score, k, dim=1)
        rows = torch.arange(filter_score.size(0))
        top_raw = raw_score[rows.unsqueeze(1), top_idx]

        gold = target.view(-1, 1)
        block_pos = (top_idx == gold).int().argmax(dim=1)
        in_block = (top_idx == gold).any(dim=1)
        block_pos = torch.where(in_block, block_pos, torch.full_like(block_pos, -1))

        self.queries.append(test_triples.astype(np.int64))
        self.cand_ids.append(top_idx.cpu().numpy().astype(np.int64))
        self.cand_filtered_scores.append(top_score.cpu().numpy().astype(np.float32))
        self.cand_raw_scores.append(top_raw.cpu().numpy().astype(np.float32))
        self.gold_raw_rank.append(raw_rank.cpu().numpy().astype(np.int64))
        self.gold_filter_rank.append(filt_rank.cpu().numpy().astype(np.int64))
        self.gold_block_pos.append(block_pos.cpu().numpy().astype(np.int64))

    def save(self, path, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "queries": np.concatenate(self.queries, axis=0),
            "cand_ids": np.concatenate(self.cand_ids, axis=0),
            "cand_filtered_scores": np.concatenate(self.cand_filtered_scores, axis=0),
            "cand_raw_scores": np.concatenate(self.cand_raw_scores, axis=0),
            "gold_raw_rank": np.concatenate(self.gold_raw_rank, axis=0),
            "gold_filter_rank": np.concatenate(self.gold_filter_rank, axis=0),
            "gold_block_pos": np.concatenate(self.gold_block_pos, axis=0),
            "topk": self.topk,
            "meta": meta,
        }
        with open(path, "wb") as fout:
            pickle.dump(payload, fout)
        n = len(payload["queries"])
        in_topk = int((payload["gold_block_pos"] >= 0).sum())
        print("[dump] saved {} queries to {} (gold in top-{}: {}, {:.4f})".format(
            n, path, self.topk, in_topk, in_topk / max(n, 1)))
