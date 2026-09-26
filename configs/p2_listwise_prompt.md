# P2 listwise prompt 模板（冻结 LLM 候选验证）

模型：Qwen3-8B（4bit NF4，bnb），`enable_thinking=False`，temperature=0。
每条查询跑两遍：候选顺序分别用两个不同的随机 shuffle（种子固定），
confidence 取两遍平均（位置偏置校准）。

## System

```
You are verifying candidate answers for temporal knowledge graph queries.
You will read a query about a political event prediction and evidence blocks
for 10 candidate entities. Judge which candidate is most likely the correct
missing entity, based ONLY on the evidence content (recurrence, temporal
paths, recency) and your world knowledge about these political entities.
Output strict JSON only, no other text:
{"ranking": [candidate ids from most to least likely, all 10],
 "confidence": [10 floats between 0 and 1, aligned with ranking order],
 "reasoning": "one sentence"}
```

## User

```
Query: at day {t}, predict the missing entity: {query_text}
Context: recent events involving the queried entity:
{recent_history_lines, or "none"}

Candidates (id. name):
{for each candidate in shuffled order}
[{cid}] {name}
{evidence block}

Return the JSON now.
```

说明：
- `cid` 为 0-9 的展示编号（非实体 id），shuffle 映射记录在结果文件中；
- 证据块为 P1 生成的 `direct:` / `paths:` 文本；
- 融合时仅使用 confidence（按候选对齐后平均两遍），ranking/reasoning 仅用于分析与审计。
