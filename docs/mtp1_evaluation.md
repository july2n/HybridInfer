# MTP-1：共享卷积历史与容差评估

2026-09-28，Qwen3.5-0.8B，BF16 / FP32 GDN state，RTX 3060 Ti。
用户已允许浮点路径差异，不再将逐位对齐作为后续工作的阻塞门槛。
MTP-1 表示每轮最多一个候选，target 验证 anchor + candidate。

## 卷积状态优化

原方案为每个验证 token 保存 `[channels, kernel_size-1]` 快照。
新方案每个请求只保存初始历史加 trial raw tokens，使用选中端点的历史窗口
直接写回正式 slot，不生成中间快照、不重跑 target。Recurrent 仍保留每个端点。

设历史宽度 W、实际验证长度 L（包含 anchor），存储由 C×W×L 变为 C×(W+L)。
W=3 时，MTP-1（L=2）减少 16.7% 的 conv 端点存储；MTP-4（L=5）减少
46.7%。这里减少的是卷积存储，整体状态仍由 FP32 recurrent snapshots 主导。
对本模型 MTP-1 每请求所有 18 层合计节省 216 KiB，不宣称整体显存减少 16.7%。
预算检查已使用新的实际布局。非 pool 后端保留原快照表示。

全部历史窗口与旧表示精确对比通过，包含变长、slot 重排与未使用 slot 不变。
完整 128 项单元测试通过。

## 数值与执行

5 类固定输入、B1/B4、每请求 128 token，25 个请求：

- 1,892 次原 trial 接受端点检查零失败；真实 MTP 执行、EOS/生命周期检查通过。
- Greedy 指标中的 1,854 个候选接受 1,308 个，接受率 70.55%，每轮平均输出
  1.706 token；无重放、无资源回退。
- 17/25 请求与普通 target 全序列相同；8 个不同，不把它们改写为严格通过。
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
初轮 baseline 仍记录 feature 的产物只保留作历史，不采用其加速数值。

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

后续从单候选评估推进到 K=2/4 的收益扫描，确定实际候选长度与优化优先级，
继续共享 target verifier；不再为追求不同浮点路径逐位等价而逐行串行化 GEMM。
EAGLE-3/P-EAGLE 及 DFlash 家族仍需各自 checkpoint、特征与 draft-cache 契约。

已直接完成下一阶段的两类输入 K=2/4 探索（相同计时条件）：

| 输入 | MTP-1 耗时比 | MTP-2 耗时比 | MTP-4 耗时比 |
|---|---:|---:|---:|
| natural_en | 1.068 | 1.196 | 1.276 |
| low_match | 1.030 | 1.082 | 1.040 |

候选越多不一定越快：low_match 的 K=4 接受率降至 28.4%，收益低于 K=2。
下一步优先做候选长度选择、批量 proposer 和图执行，让真实后端收益更稳定，
再推广到其余输入与 B4；目前不统一把默认候选数设为 4。

原始产物：

- `logs/validate/mtp1_compact.json`：自由生成、接受率、端点验证。
- `logs/validate/mtp1_probability_*.json`：共同输入概率与 layer/state 对照。
- `logs/validate/mtp1_quality.json`、`mtp1_outputs.json`：客观任务和解码文本。
- `logs/bench/mtp1_compact_fair.json`：公平 baseline 性能复测。
- `logs/bench/mtp_k_sweep.json`：下一阶段候选长度扫描。
