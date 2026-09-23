"""Statistical fusion of evidence with TiRGN scores (no LLM).

Fusion rule (per query q, candidate j inside the Top-K block):

    z'_j = z_j + lambda * rho_q * e_j

  z_j    : raw final_score of the candidate (from the cache)
  e_j    : evidence score, a linear combination of standardized E1-E4
           features; weights are fit on the validation split only
           (logistic regression over block candidates, positive = gold)
  lambda : fixed scale hyperparameter
  rho_q  : reliability from evidence coverage, i.e. the fraction of block
           candidates carrying any nonzero direct/path/interaction evidence

Hard constraint: only the order *inside* the Top-K block is changed. Entities
outside the block keep their original filtered ranks, so recomputed filtered
MRR / Hits@K remain valid.
"""

import numpy as np


def evidence_coverage(block_feats):
    """Fraction of candidates with any nonzero E1/E2/E3 raw evidence."""
    raw = block_feats[:, [0, 4, 6, 10]]  # e1_count, e2_support, e2_conf_sum, e3_interaction
    return float((raw > 0).any(axis=1).mean())


def fit_logistic(feats, gold_pos, epochs=200, lr=0.1, l2=1e-4, seed=0):
    """Binary logistic regression over block candidates of queries whose gold
    is inside the Top-K. feats: (N,K,F) standardized; gold_pos: (N,) block
    position of gold or -1. Returns weight vector (F,)."""
    rng = np.random.RandomState(seed)
    mask = gold_pos >= 0
    x = feats[mask].reshape(-1, feats.shape[2])
    y = np.zeros(x.shape[0], dtype=np.float32)
    pos_rows = np.arange(mask.sum()) * feats.shape[1] + gold_pos[mask]
    y[pos_rows] = 1.0
    w = rng.normal(scale=0.01, size=feats.shape[2]).astype(np.float32)
    for _ in range(epochs):
        z = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = x.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def standardize(train_feats, *other_feats):
    mu = train_feats.reshape(-1, train_feats.shape[2]).mean(axis=0)
    sd = train_feats.reshape(-1, train_feats.shape[2]).std(axis=0) + 1e-8
    out = [(train_feats - mu) / sd] + [(f - mu) / sd for f in other_feats]
    return out if len(out) > 1 else out[0]


def rerank_metrics(cache, evid_score, lam, use_rho=True, feats=None):
    """Recompute filtered MRR / Hits under Top-K-internal reranking.

    cache       : candidate cache dict
    evid_score  : (N,K) evidence scores e_j
    lam         : fixed fusion scale
    use_rho     : multiply by per-query coverage reliability
    feats       : raw (unstandardized) features, needed when use_rho=True
    """
    z = cache["cand_raw_scores"]
    gold_pos = cache["gold_block_pos"]
    gold_rank = cache["gold_filter_rank"]
    n, k = z.shape

    zprime = z.astype(np.float64).copy()
    for i in range(n):
        rho = evidence_coverage(feats[i]) if (use_rho and feats is not None) else 1.0
        zprime[i] += lam * rho * evid_score[i]

    new_rank = gold_rank.astype(np.float64).copy()
    in_block = gold_pos >= 0
    rows = np.arange(n)[in_block]
    gp = gold_pos[in_block]
    gold_z = zprime[rows, gp]
    better = (zprime[rows] > gold_z[:, None]).sum(axis=1)
    new_rank[in_block] = 1.0 + better

    mrr = float(np.mean(1.0 / new_rank))
    hits = {h: float(np.mean(new_rank <= h)) for h in (1, 3, 10)}
    return {"mrr": mrr, "hits": hits, "ranks": new_rank}
