# 环境记录（任务 0 复检）

复检日期：2026-09-21。项目根目录：`/newSSD/MrLiu/hwb_project/TiRGN`（TiRGN 官方仓库原生代码，未改动模型逻辑）。

## 硬件 / 驱动

| 项 | 值 |
|---|---|
| GPU | 4 x NVIDIA GeForce RTX 4090 24GB |
| 驱动 | 535.104.05 |
| 驱动支持 CUDA | 12.2 |

## Python 环境

conda 环境名：`tirgn4090`（创建方式见 `RUN_4090.md` / `environment_4090.yml`）。

| 包 | 版本 |
|---|---|
| python | 3.9.25 |
| torch | 2.0.1+cu118 |
| torch CUDA | 11.8 |
| dgl | 1.1.3+cu118 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| pandas | 1.5.3 |

注意：直接 `import dgl` 报 `libcusparse.so.11: cannot open shared object file`，
需 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（该库位于环境 lib 目录下）。

## 数据（ICEWS14）

位置：`data/ICEWS14/`。文件日期 2026-07-15（任务 0 期间获取，来源为仓库配套数据）。

| 文件 | 行数 | md5 |
|---|---|---|
| train.txt | 63685 | 308f9cabb89956e0840f6673350d3cc5 |
| valid.txt | 13823 | 1f0f44766f0b2d6473408156b0ede81e |
| test.txt | 13222 | 225b6536b8ed324c22d60233c6868d15 |
| entity2id.txt | - | f5586922ab32894d24775c2187e2a172 |
| relation2id.txt | - | 1669c31b657e2a2d050d27bac7b66f29 |
| stat.txt | - | a734a34105bcb48c18b3f1eade654a76 |

统计：entities 7128，relations 230，train edges 63685；时间戳数 365。

## 历史矩阵预处理

`python get_history.py --dataset ICEWS14` 已执行，`data/ICEWS14/history/` 下
tail_history_*.npz 与 rel_history_*.npz 均已生成（覆盖全部时间戳）。

## 随机性说明

官方代码未暴露 `--seed` 参数；仅 `rgcn/knowledge_graph.py:15` 固定
`np.random.seed(123)`，torch / cuda 随机性未固定。多 seed 实验（任务 3）需要
在外部注入 seed，且必须写进 config。
