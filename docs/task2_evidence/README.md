# 任务 2：候选缓存 + RCEV-NoLLM 证据检索器原型

日期：2026-09-21。全部无 LLM。未修改 `model.py` / `rrgcn.py` / `decoder.py` / `rgcn/` 的任何原始逻辑。

## 代码改动清单

| 文件 | 性质 | 说明 |
|---|---|---|
| `src/evidence/__init__.py` | 新增 | 包说明 |
| `src/evidence/dumper.py` | 新增 | `CandidateDumper`：predict 之后 dump Top-K 候选 |
| `src/evidence/retriever.py` | 新增 | E1-E3 证据检索 + TLogic 风格规则统计 |
| `src/evidence/extract_features.py` | 新增 | 特征抽取 CLI（含 E4） |
| `src/evidence/fuse.py` | 新增 | 统计融合 `z' = z + λ·ρ·e` + Top-K 内部重排评测 |
| `src/main.py` | 最小改动 | 新增 `--dump-candidates` / `--dump-topk` 两个参数；`test()` 内三处挂载；`--test` 分支下追加 valid dump。不加 flag 时行为与原始代码完全一致 |

## 候选缓存格式

位置：`candidates/ICEWS14_{valid,test}_topk50.pkl`（pickle dict）。

| 键 | shape | 含义 |
|---|---|---|
| `queries` | (N,4) int64 | 查询四元组 (s,r,o_gold,t)，含逆向查询（subject 预测，r+num_rels），N=2×split 大小 |
| `cand_ids` | (N,50) int64 | Top-50 候选实体 id，按 **filtered score** 降序 |
| `cand_filtered_scores` | (N,50) float32 | 候选的 filtered score |
| `cand_raw_scores` | (N,50) float32 | 候选的原始 final_score |
| `gold_raw_rank` / `gold_filter_rank` | (N,) int64 | gold 的 raw / filtered 排名（1-indexed，与官方 eval 同一套 `sort_and_rank`） |
| `gold_block_pos` | (N,) int64 | gold 在 Top-50 块内位置，不在块内为 -1 |
| `meta` | dict | dataset / split / model_name / argv |

挂载点：`main.py` `test()` 中 `model.predict()` 之后。dump 命令（在 `src/` 下）：

```
PYTHONPATH=.. LD_LIBRARY_PATH=$CONDA_PREFIX/lib python main.py -d ICEWS14 \
  <与基线相同超参> --test --dump-candidates ../candidates --dump-topk 50
```

注意：checkpoint 文件名含 gpu 编号，评测需用 `--gpu 0`（与训练一致）。

## 缓存规模与覆盖率

| split | N | gold∈Top-50 比例 |
|---|---|---|
| valid | 27646 | 0.8024 |
| test | 26444 | 0.7955 |

## 缓存正确性验证

λ=0（不重排）时从缓存重算的指标与官方评测逐位一致：

| split | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| valid | 0.450140 | 0.344860 | 0.506475 | 0.647363 |
| test | 0.435611 | 0.333573 | 0.485933 | 0.632733 |

（test 与 `logs/train_ICEWS14_gpu0.log` 的官方输出完全相同。）

## 证据特征（14 维，`candidates/{tag}/ICEWS14_{split}_topk50_feats.npz`）

| 组 | 特征 | 定义 |
|---|---|---|
| E1 | `e1_count`, `e1_recent_count`, `e1_decayed_count`, `e1_log_recency` | 历史 (s,r,o,t'<t) 出现次数（全历史 / 近 10 个时间戳 / 指数衰减 τ=7）；最近一次距查询时间的间隔（log1p） |
| E2 | `e2_support`, `e2_log_support`, `e2_conf_sum`, `e2_num_rules` | 以 r 为规则头的 1 跳/2 跳时间路径（体部 t1≤t2<t），按规则聚合的支持度与置信度和；规则统计只在 train 上挖掘 |
| E3 | `e3_recent_activity`, `e3_log_recent_activity`, `e3_interaction`, `e3_log_interaction` | 候选 o 近 10 个时间戳活动频率；s-o 历史交互次数 |
| E4 | `e4_orig_rank`, `e4_path_gap` | 候选在 Top-K 内原始排名（归一化）；与块内最优候选的 E2 置信度差 |

历史范围与官方评测一致：valid 查询只用 train，test 查询用 train+valid，且均只取 t'<t。

已知近似（原型范围，如实记录）：
- 规则挖掘对 15 个高度数 hub 实体跳过（in×out > 20 万的连接对），支持度/置信度为确定性近似；
- 头部 grounding 用「任意 train 时间 ≥ 体部末端时间」判定。

## 融合与重排（`fuse.py`）

- 证据分数 e：12 维特征标准化（统计量来自 valid）后做逻辑回归（正例=gold，仅 gold∈Top-K 的查询参与），**权重只在 valid 上拟合**；
- 可靠性 ρ：查询 Top-K 块内携带任何 E1/E2/E3 非零证据的候选比例；
- 融合 z' = z + λ·ρ·e，λ 为固定超参；**只重排 Top-K 内部顺序**，块外实体原位不动，filtered MRR/Hits@K 仍有效。

## 管线冒烟测试（非任务 3 正式结果）

逻辑回归权重 + 不同 λ 下重排（test 只用于观察，正式实验在任务 3）：

| λ | valid MRR | test MRR | test H@1 | test H@10 |
|---|---|---|---|---|
| 0（基线） | 0.4501 | 0.4356 | 0.3336 | 0.6327 |
| 0.05 | 0.4497 | 0.4365 | 0.3346 | 0.6329 |
| 0.10 | 0.4487 | 0.4361 | 0.3341 | 0.6325 |
| 0.20 | 0.4472 | 0.4345 | 0.3318 | 0.6314 |
| 0.50 | 0.4413 | 0.4279 | 0.3252 | 0.6258 |

观察：valid 上 λ=0 最优、test 上小 λ 略有提升，说明当前单一组权重在 valid 上过拟合证据信号有限——正式结论以任务 3 的多配置多 seed 实验为准，此处仅验证管线正确。

## 运行时间

- 候选 dump（valid+test，含模型加载）：约 4 分钟（GPU 0）；
- 特征抽取：valid 约 11.4 分钟，test 约 14.1 分钟（纯 CPU，含规则挖掘约 1 分钟）。
