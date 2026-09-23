# 服务器实验任务：TiRGN 基线复现 + 无 LLM 的候选证据原型（Phase 0-2）

## 科研上下文（压缩版）
我在做时序知识图谱（TKG）外推方向，backbone 是 TiRGN（官方仓库 https://github.com/Liyyy2122/TiRGN）现以安装到当前工作目录下。总体研究线是 RCEV：先用 TiRGN 产出 Top-K 未来候选实体，再为每个候选构造与查询关系 r 显式对齐的时间证据链，最终用冻结 LLM 做候选验证和可校准重排。

文献结论（已完成检索，不需要你再查）：ExE-LLM（ACL 2026 Findings）、AnRe（ACL 2025）、STK-Adapter（ACL 2026）已经覆盖了 candidate scoring、多源时间/结构/语义检索、LLM 分数融合。本次实验完全不加 LLM，目的是验证最小假设：关系对齐的候选条件证据（candidate-conditioned evidence）是否比 query-only 证据更能提升 TiRGN 排名。如果这个假设不成立，整个 LLM 路线需要重新评估。

## 服务器环境（我已确认的信息）
- GPU: {RTX 4090 24GB}
- CUDA: {NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2  }
- Python 环境: {conda tirgn4090 }
- 网络: {是否能访问 GitHub / Google Drive / HuggingFace}自行尝试

## 任务 0：环境与数据 （已完成你复检）
1. 把 TiRGN 官方仓库 clone 到 {项目根目录}/TiRGN，保持原生代码，不要重写其结构。
2. 按仓库 README 创建环境并安装依赖，记录 python / torch / cuda / dgl 等全部版本到 {项目根目录}/environment.md。
3. 获取 ICEWS14（必须）和 ICEWS18（可选）数据，优先用 README 给的链接，链接失效再用官方 ICEWS 抽取脚本。记录数据来源、获取日期、文件校验和。
4. 跑 get_history.py 预处理，确认 tail/relation 历史矩阵正常生成。

## 任务 1：TiRGN 基线复现
1. 按 README 默认超参在 ICEWS14 上完成训练和测试。
2. 通过标准：filtered MRR 与论文/README 报告值相差在 1 个点以内。若不达标，先排查评测实现（filtered ranking、subject/object 双向预测合并、负例处理），记录排查过程，不要靠调参硬凑数字。
3. 交付：训练日志、测试输出（含 MRR / Hits@1/3/10）、checkpoint 路径、完整 config、random seed。汇总到 {项目根目录}/results/baseline_icews14.md。
4. 先估算单 epoch 时间再开长训练；长任务用 tmux 跑，避免会话中断。

## 任务 2：候选缓存 + 无 LLM 证据检索器原型（RCEV-NoLLM）
1. 挂载点：main.py 中 model.predict() 得到 final_score 之后。先实现一个 dump 模式：对验证集和测试集每个查询保存 Top-K 候选（默认 K=50，含 ground truth 标记和原始分数）到磁盘，格式自定但写清楚。后续证据实验全部基于这份缓存离线迭代，不重复训练 TiRGN。
2. 在 TiRGN/src/ 下新建 evidence/ 目录实现检索器，不改 model.py / rrgcn.py / decoder.py 的原始逻辑：
   - E1 直接证据：历史 (s,r,o,t') 出现次数、最近一次时间间隔；
   - E2 关系对齐路径：以 r 为规则头的一跳/二跳时间路径（TLogic 风格），统计支持度和路径置信度；
   - E3 候选端历史：候选 o 的近期活动频率、与 s 的历史交互；
   - E4 对比特征：候选在 Top-K 内原始排名、与最优候选的路径证据差。
3. 打分与融合：纯统计打分（加权或逻辑回归），权重只在验证集上拟合。融合方式 z' = z + λ·ρ·e，λ 固定，ρ 为证据覆盖率导出的可靠性。硬性约束：只重排 Top-K 内部顺序，Top-K 外实体保持原位，保证标准 filtered MRR / Hits@K 依然有效。

## 任务 3：Phase-2 对比实验（全部无 LLM）
在 ICEWS14 上跑以下配置，每个配置 3 个 seed，报告均值±标准差：
| 配置 | 证据来源 |
|---|---|
| A. TiRGN | 无 |
| B. TiRGN + Recent | query-only 短期历史 |
| C. TiRGN + Long | query-only 长期历史 |
| D. TiRGN + TemporalEvidence | query-only 时间证据 |
| E. TiRGN + CandidateEvidence | 候选条件证据（通用，ExE 风格特征） |
| F. TiRGN + RCEV | 关系对齐路径 + 对比特征（本研究最小假设） |

## 必须交付的结果
1. results/ 目录下每个配置每次运行的完整日志；
2. results/results_summary.md，主表包含：filtered MRR、Hits@1/3/10（均值±std）；
3. 诊断指标（F 配置相对 A）：MRR | gold∈Top-K、有益翻转率、有害翻转率、按 subject 历史事件数分桶（0-2 / 3-9 / 10+）的 MRR 提升；
4. 证据消融：E1/E2/E3/E4 各特征的贡献；
5. 每个配置的运行时间和显存占用；
6. 若某个假设不成立（如 F 不优于 E，或 F 的 MRR 提升全部来自 gold 已在前 3 的情况），如实报告并给出你的分析，不要美化结果。

## 规则
- 本阶段不做任何 LLM 相关的事（不下载 LLM 权重、不做 prompt）；
- 不修改 TiRGN 原始模型逻辑，不 git push，不删除任何已有文件；
- 所有超参改动必须写进 config 文件，不允许只改命令行；
- 每完成一个任务先停下来汇报结果，等我确认后再进入下一个任务；
- 遇到与 README 不符的报错，先记录完整堆栈再修。
