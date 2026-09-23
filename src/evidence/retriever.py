"""RCEV-NoLLM evidence retriever (prototype, no LLM).

Operates purely offline on the candidate cache produced by
src/evidence/dumper.py plus the raw dataset quadruples. For every
(query, candidate) pair it extracts four feature groups:

  E1 direct evidence   : count and recency of historical (s, r, o, t')
  E2 relation paths    : 1-hop / 2-hop temporal paths with rule head r
                         (TLogic-style, body times ordered, body < query t),
                         aggregated as grounding support and sum of rule
                         confidences (rule stats mined on train split only)
  E3 candidate history : recent activity frequency of candidate o and
                         historical interaction between s and o
  E4 contrastive       : candidate's original rank inside the Top-K block and
                         its path-evidence gap to the best candidate in the
                         block (computed per query over the cached Top-K)

History scope mirrors the official evaluation: valid queries see train
quadruples only, test queries see train+valid quadruples; in both cases only
events with t' < query time t are used.
"""

import numpy as np
from collections import defaultdict


RULE_BODY_CAP_PER_ENTITY = 200000  # cap on (in, out) edge-pair joins per entity
RECENCY_MISSING = 100000.0
ACTIVITY_WINDOW = 10
DECAY_TAU = 7.0  # exponential decay constant (in timestamps) for E1 decayed count


def _load_quads(path):
    data = np.loadtxt(path, dtype=np.int64)
    return data[:, :4]


class EvidenceRetriever:
    def __init__(self, data_dir):
        self.train = _load_quads(data_dir + "/train.txt")
        self.valid = _load_quads(data_dir + "/valid.txt")
        self.test = _load_quads(data_dir + "/test.txt")
        self.histories = {
            "valid": self.train,
            "test": np.concatenate([self.train, self.valid], axis=0),
        }
        self._index = {}
        self.rule_stat = self._mine_rules(self.train)

    # ------------------------------------------------------------- indexing
    def _build_index(self, quads):
        idx = {
            "sro_times": defaultdict(list),   # (s,r,o) -> sorted [t]
            "out_edges": defaultdict(list),   # s -> sorted [(t, r, o)]
            "pair_times": defaultdict(list),  # (min,max) unordered pair -> sorted [t]
            "ent_times": defaultdict(list),   # entity -> sorted [t] (as s or o)
            "xo_edges": defaultdict(list),    # (x,o) -> [(t, r)]
        }
        for s, r, o, t in quads:
            idx["sro_times"][(s, r, o)].append(t)
            idx["out_edges"][s].append((t, r, o))
            idx["xo_edges"][(s, o)].append((t, r))
            idx["ent_times"][s].append(t)
            idx["ent_times"][o].append(t)
            idx["pair_times"][(min(s, o), max(s, o))].append(t)
        for key in ("sro_times", "out_edges", "pair_times", "ent_times"):
            for k in idx[key]:
                idx[key][k].sort()
        return idx

    def _get_index(self, split):
        if split not in self._index:
            self._index[split] = self._build_index(self.histories[split])
        return self._index[split]

    # ----------------------------------------------------------- rule mining
    def _mine_rules(self, quads):
        """Approximate TLogic-style rule stats on the train split.

        Bodies: length-1 (r1) and length-2 (r1, r2) chains with t1 <= t2.
        Head grounding: (a, r_head, c) exists at any train time >= t2.
        Per-entity join cost is capped to keep hubs tractable; stats are
        therefore a deterministic approximation (prototype scope).
        """
        in_edges = defaultdict(list)   # b -> [(t, r, a)]
        out_edges = defaultdict(list)  # b -> [(t, r, c)]
        head_times = defaultdict(list)  # (a, r, c) -> [t]
        for s, r, o, t in quads:
            out_edges[s].append((t, r, o))
            in_edges[o].append((t, r, s))
            head_times[(s, r, o)].append(t)
        head_by_pair = defaultdict(list)  # (a, c) -> [(r_head, max_t)]
        for (a, r, c), times in head_times.items():
            head_by_pair[(a, c)].append((r, max(times)))

        pair_count = defaultdict(int)   # (r1, r2 or None) -> body groundings
        head_count = defaultdict(int)   # (r1, r2 or None, r_head) -> grounded heads
        for b in list(in_edges.keys()):
            ins, outs = in_edges[b], out_edges.get(b, [])
            if len(ins) * len(outs) > RULE_BODY_CAP_PER_ENTITY:
                continue  # skip extreme hubs (documented approximation)
            for t1, r1, a in ins:
                for t2, r2, c in outs:
                    if t1 > t2:
                        continue
                    pair_count[(r1, r2)] += 1
                    for r_head, mt in head_by_pair.get((a, c), []):
                        if mt >= t2:
                            head_count[(r1, r2, r_head)] += 1
        # length-1 bodies: (a, r1, c, t1) with head at t3 >= t1
        for s, r1, o, t1 in quads:
            pair_count[(r1, None)] += 1
            for r_head, mt in head_by_pair.get((s, o), []):
                if mt >= t1:
                    head_count[(r1, None, r_head)] += 1

        rule_stat = {}
        for key, hc in head_count.items():
            bc = pair_count[(key[0], key[1])]
            rule_stat[key] = (hc, hc / bc)
        return rule_stat

    # ------------------------------------------------------ feature extraction
    @staticmethod
    def _count_before(times, t):
        return np.searchsorted(times, t, side="left")

    def _e1(self, idx, s, r, o, t):
        times = idx["sro_times"].get((s, r, o))
        if not times:
            return 0.0, 0.0, 0.0, RECENCY_MISSING
        cnt = self._count_before(times, t)
        if cnt == 0:
            return 0.0, 0.0, 0.0, RECENCY_MISSING
        recent = cnt - self._count_before(times, t - ACTIVITY_WINDOW)
        decayed = sum(np.exp(-(t - tt) / DECAY_TAU) for tt in times[:cnt])
        return float(cnt), float(recent), float(decayed), float(t - times[cnt - 1])

    def _e2(self, idx, s, r, o, t):
        support, conf_sum, n_rules = 0, 0.0, 0
        fired = set()
        # length-1 paths: (s, r1, o, t1 < t)
        for tt, r1 in idx["xo_edges"].get((s, o), []):
            if tt >= t:
                continue
            key = (r1, None, r)
            if key in self.rule_stat and key not in fired:
                fired.add(key)
                support += self.rule_stat[key][0]
                conf_sum += self.rule_stat[key][1]
        # length-2 paths: s -r1-> x -r2-> o, t1 <= t2 < t
        for t1, r1, x in idx["out_edges"].get(s, []):
            if t1 >= t:
                break
            for t2, r2 in idx["xo_edges"].get((x, o), []):
                if t2 >= t:
                    continue
                if t2 < t1:
                    continue
                key = (r1, r2, r)
                if key in self.rule_stat and key not in fired:
                    fired.add(key)
                    support += self.rule_stat[key][0]
                    conf_sum += self.rule_stat[key][1]
        n_rules = len(fired)
        return float(support), conf_sum, float(n_rules)

    def _e3(self, idx, s, o, t):
        ent_times = idx["ent_times"].get(o, [])
        recent = self._count_before(ent_times, t) - self._count_before(ent_times, t - ACTIVITY_WINDOW)
        pair_times = idx["pair_times"].get((min(s, o), max(s, o)), [])
        inter = self._count_before(pair_times, t)
        return float(recent), float(inter)

    def query_features(self, split, query, cand_ids):
        """Feature matrix (K, n_features) for one query over its Top-K."""
        s, r, _gold, t = (int(v) for v in query)
        idx = self._get_index(split)
        feats = np.zeros((len(cand_ids), 12), dtype=np.float32)
        for j, o in enumerate(cand_ids):
            o = int(o)
            e1_cnt, e1_recent, e1_decay, e1_rec = self._e1(idx, s, r, o, t)
            e2_sup, e2_conf, e2_rules = self._e2(idx, s, r, o, t)
            e3_act, e3_inter = self._e3(idx, s, o, t)
            feats[j] = (e1_cnt, e1_recent, e1_decay, np.log1p(e1_rec),
                        e2_sup, np.log1p(e2_sup), e2_conf, e2_rules,
                        e3_act, np.log1p(e3_act), e3_inter, np.log1p(e3_inter))
        return feats


FEATURE_NAMES = [
    "e1_count", "e1_recent_count", "e1_decayed_count", "e1_log_recency",
    "e2_support", "e2_log_support", "e2_conf_sum", "e2_num_rules",
    "e3_recent_activity", "e3_log_recent_activity",
    "e3_interaction", "e3_log_interaction",
]
