# 任务 3 训练配置（ICEWS14，3 seeds）

所有超参与 README 基线一致，唯一新增超参为 `--seed`（任务 3 要求 3 个 seed）。
seed 注入点在 `src/main.py` 参数解析之后（random / numpy / torch / cuda 全部固定），
不改任何模型逻辑。原始默认行为（不传 `--seed`）保持不变。

| 参数 | 值 |
|---|---|
| dataset | ICEWS14 |
| history-rate | 0.3 |
| train-history-len / test-history-len | 9 / 9 |
| dilate-len | 1 |
| lr | 0.001 |
| n-layers | 2 |
| n-hidden | 200 |
| evaluate-every | 1 |
| decoder / encoder | timeconvtranse / convgcn |
| weight / task-weight / discount / angle | 0.5 / 0.7 / 1 / 14 |
| n-epochs | 500 |
| batch-size | 1 |
| dropout 系列 | 0.2 / 0.2 / 0.2 / 0.2 |

| seed | GPU | checkpoint（models/ 下） |
|---|---|---|
| 1 | 1 | ...-angle_14-dp0.2_0.2_0.2_0.2-gpu1-checkpoint |
| 2 | 2 | ...-angle_14-dp0.2_0.2_0.2_0.2-gpu2-checkpoint |
| 3 | 3 | ...-angle_14-dp0.2_0.2_0.2_0.2-gpu3-checkpoint |

checkpoint 文件名由原始代码按超参+gpu 编号自动生成，不同 seed 靠 GPU 编号区分
（一 seed 一卡，互不覆盖）。

运行方式：`bash scripts/train_seed.sh <seed> <gpu>`（在 tmux 中执行，见
`docs/task3_experiments/` 的实验记录）。
