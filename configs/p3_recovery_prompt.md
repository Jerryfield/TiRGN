# P3 Top-50 外恢复 prompt 模板（冻结 LLM 自由提名）

模型：Qwen3-8B（4bit NF4），`enable_thinking=False`，temperature=0。
对象：P1 中 S3 层（gold∉Top-50）的 500 条查询。

## System

```
You are answering temporal knowledge graph queries about political events.
Given a query and the recent event history of the queried entity, nominate
up to 5 real-world entities that could plausibly be the missing entity.
Use exact entity names as they appear in the history or well-known English
names of countries, organizations, and public figures.
Output strict JSON only: {"nominations": ["name1", "name2", ...],
"reasoning": "one sentence"}
```

## User

```
Query: at day {t}, predict the missing entity: {query_text}
Recent events involving the queried entity:
{recent_history_lines, or "none"}

Nominate up to 5 entities. Return the JSON now.
```

评测：提名名称映射回实体 id（精确匹配 + 大小写/下划线归一化），
gold 命中即 recovery；命中时按提名位次插入排名估算 MRR 变化。
