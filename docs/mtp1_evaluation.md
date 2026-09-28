# MTP-1：共享卷积历史与容差评估

2026-09-28，Qwen3.5-0.8B，BF16 / FP32 GDN state，RTX 3060 Ti。
验收采用数学语义、原 trial 端点选择与数值/质量预算。
MTP-1 表示每轮最多一个候选，target 验证 anchor + candidate。

## 卷积状态优化

每个请求保存初始卷积历史加 trial raw tokens，接受后选择端点对应历史窗口，
直接写回正式 slot，不生成中间卷积快照、不重跑 target。Recurrent 保存每个端点。
设历史宽度 W、验证长度 L（含 anchor）、通道数 C，conv 存储为 C×(W+L)。
本模型 W=3；MTP-1 每请求所有 18 层 conv 存储相较逐 token 快照节省 216 KiB，
整体状态显存仍由 FP32 recurrent 端点主导。预算检查按实际布局计算。
非 pool 后端保留快照表示。历史窗口、变长、slot 重排和未使用 slot 不变检查通过。
本次优化阶段 128 项单元测试通过；最新源码清理后结果见 [当前进度](speculative_decoding_progress.md)。

## 数值与执行

5 类固定输入、B1/B4、每请求 128 token，25 个请求：

- 1,892 次原 trial 接受端点检查零失败；真实 MTP 执行、EOS/生命周期检查通过。
- Greedy 指标中的 1,854 个候选接受 1,308 个，接受率 70.55%，每轮平均输出
  1.706 token；无重放、无资源回退。
- 17/25 请求与普通 target 全序列相同；8 个不同，作为跨路径诊断记录。
- 同输入 teacher-forced，S=2、rolling，5×128=640 个 query：5 次 argmax
  翻转（0.781%），平均 TV 0.008659、平均 KL 0.00061625 nats。
  每类平均 TV 0.000524–0.011231，最大单位置 TV 0.050168。
  翻转处普通 top1/top2 margin 为 0、0.0625 或 0.125。
- 这组概率比较以 temperature=1 的完整 softmax 计算；不是自由生成轨迹的
  token 错误率，且 diagnostic 两边 LM head 都逐行投影。

12 道固定算术题采用 chat template、temperature=0、正常 EOS、最多 64 token，
每次最多 4 个并发请求。普通与 MTP-1 输出全部相同，均答对 6/12。没有观察到
额外退化，但样本小且模型基线较弱，不能推广成通用质量无损的证明。
自然语言自由生成已保存解码文本，确有措辞和内容变化。

当前证据足以继续受控性能/后端工作；状态协议仍严格，完整质量仍持续监测。
不更改默认 `packed_guarded`，显式 `packed` 用于实际 MTP 测量。

## 性能与下一阶段

性能对照使用同一个已加载模型、B=1、prefix off、temperature=0、两边 eager，
输入截断至最多 509 token，每次生成 128 token。每模式每输入预热 1 次，
测量 3 次，轮换执行顺序；普通 baseline 关闭 MTP feature 记录。
不同输出的耗时比仅表示当前两条路径的实际运行时间比，不表示同轨迹加速。

公平复测结果：

| 输入 | 普通耗时 / MTP-1 耗时 | 候选接受率 | MTP-1 decode token/s |
|---|---:|---:|---:|
| natural_en | 1.068 | 64.9% | 79.3 |
| natural_zh | 1.026 | 58.8% | 76.2 |
| repository_code | 1.247 | 96.9% | 93.7 |
| long_records | 1.256 | 96.9% | 93.3 |
| low_match | 1.030 | 59.5% | 77.1 |

这些全部存在与该普通 baseline 的输出差异。3 次重复仅为探索性测量，
2.6%–3.0% 的小收益尚不视为稳定收益；未比较普通 CUDA graph 模式。

两类输入 K=2/4 探索已完成（相同计时条件）：

| 输入 | MTP-1 耗时比 | MTP-2 耗时比 | MTP-4 耗时比 |
|---|---:|---:|---:|
| natural_en | 1.068 | 1.196 | 1.276 |
| low_match | 1.030 | 1.082 | 1.040 |

候选越多不一定越快：low_match 的 K=4 接受率降至 28.4%，收益低于 K=2。
下一步建议先补五类输入 K=1/2/4 与 B1/B4 扫描、分项测量及普通 decode graph
对照，再根据成本优化批量 proposer、图执行和候选长度选择。当前性能脚本仅支持
B1，MTP 配置强制 eager，B4 与图路径对照需扩展入口。完整任务见
[实施计划](speculative_decoding_plan.md)。当前不统一将候选数设为 4。

原始产物：

- `logs/validate/mtp1_compact.json`：自由生成、接受率、端点验证。
- `logs/validate/mtp1_probability_*.json`：共同输入概率与 layer/state 对照。
- `logs/validate/mtp1_quality.json`、`mtp1_outputs.json`：客观任务和解码文本。
- `logs/bench/mtp1_compact_fair.json`：公平 baseline 性能复测。
- `logs/bench/mtp_k_sweep.json`：下一阶段候选长度扫描。
