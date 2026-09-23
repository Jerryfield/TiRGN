# 任务 1：TiRGN ICEWS14 基线复现

日期：2026-09-21。环境：`tirgn4090`（见 `docs/task0_environment/environment.md`）。

## 结论

通过验收：filtered entity MRR = 0.4356，与论文报告值（约 0.438x）相差 0.3 个点以内（< 1 个点）。

## 训练配置（与 README 默认超参一致）

```
python main.py -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 \
  --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
  --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 \
  --entity-prediction --relation-prediction --add-static-graph --angle 14 \
  --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint
```

完整 Namespace（取自日志首行）：

```
batch_size=1, n_epochs=500, lr=0.001, grad_norm=1.0, evaluate_every=1,
encoder=convgcn, decoder=timeconvtranse, n_hidden=200, n_layers=2, n_bases=100, n_basis=100,
opn=sub, dropout=0.2, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2,
self_loop=True, layer_norm=True, skip_connect=False, aggregation=none,
train_history_len=9, test_history_len=9, dilate_len=1, history_rate=0.3,
weight=0.5, task_weight=0.7, discount=1.0, angle=14,
entity_prediction=True, relation_prediction=True, add_static_graph=True,
multi_step=False, topk=50, split_by_relation=False
```

## 随机种子

官方代码无 `--seed`；仅 `np.random.seed(123)`（`rgcn/knowledge_graph.py:15`）。
torch / cudnn 随机性未固定，逐次运行会有小幅波动。

## 训练过程

- 完整训练 500 epoch（无早停），每 epoch 后在验证集评测。
- 最优验证集 raw MRR = 0.4379，出现在 epoch 10；之后 500 epoch 未再超过。
- 训练日志：`logs/train_ICEWS14_gpu0.log`（9525 行）。

## 测试结果（加载 best epoch 10，`--test`，ground-truth history 设置）

| 指标 | raw_ent | filter_ent | raw_rel | filter_rel |
|---|---|---|---|---|
| MRR | 0.4251 | **0.4356** | 0.4129 | 0.4644 |
| Hits@1 | 0.3189 | **0.3336** | 0.2598 | 0.3364 |
| Hits@3 | 0.4790 | **0.4859** | 0.4805 | 0.5153 |
| Hits@10 | 0.6305 | **0.6327** | 0.7298 | 0.7428 |

与论文报告值（ICEWS14，entity prediction，ground-truth history）对比：

| 指标 | 论文 | 复现 | 差值 |
|---|---|---|---|
| MRR (filter) | ~0.438 | 0.4356 | -0.003 |
| Hits@1 | ~0.336 | 0.3336 | -0.002 |
| Hits@3 | ~0.487 | 0.4859 | -0.001 |
| Hits@10 | ~0.638 | 0.6327 | -0.006 |

全部差值在 1 个点以内，达标。未做任何调参。

## Checkpoint

`models/gl_rate_0.3-ICEWS14-convgcn-timeconvtranse-ly2-dilate1-his9-weight_0.5-discount_1.0-angle_14-dp0.2_0.2_0.2_0.2-gpu0-checkpoint`
（目录内含 best epoch 10 的模型参数；任务 2 的候选缓存将基于该 checkpoint）。

## 备注

- 同目录下另有 `angle_8` 的 ICEWS14 checkpoint，为任务 0 期间试跑产物，非本基线。
- 测试集 51 个 snapshot，单次测试约 34 秒（GPU 0）。
