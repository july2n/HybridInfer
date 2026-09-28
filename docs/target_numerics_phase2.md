# Target 数值定位：第二轮实验

日期：2026-09-28。模型 Qwen3.5-0.8B，BF16，FP32 recurrent state，TP=1，
eager、prefix cache off。全程 teacher forcing；只改 benchmark，没有改生产 kernel。
128 次 target query 对应生成 token #2–#129，token 编号从 1 开始。

## 1. Norm 已从类别定位到具体调用、具体操作

在 `repository_code`、S=5、reset 对照中，将卷积、BV/FMA、GEMM、attention
匹配普通路径，但保留原始 packed norm。捕获 offset=90 的块。
61 次 GemmaRMSNorm 调用以同一输入直接比较 packed 与逐 token 执行，只有
`model.layers.9.post_attention_layernorm` 有输出差异，发生在 query row 3，
即生成 token #95 对应的预测；其余 60 次调用同输入输出精确一致。

保存输入、residual、weight、eps、两种输出，并逐阶段重放：

| 操作 | packed 与单 token 的最大绝对差 |
|---|---:|
| BF16 residual add、转 FP32、square | 0 |
| mean reduction | 2.3283064365386963e-10 |
| rsqrt（不同 mean 为输入） | 1.9073486328125e-6 |
| FP32 normalize/scale | 1.9073486328125e-6 |
| BF16 cast | 0.00048828125 |

对相同操作数单独重新执行 `mean` 仍有差异，单独执行 square 和 rsqrt 没有。
所以这次局部误差首先来自 **mean 的 shape-dependent reduction**，不是 rsqrt
在相同输入上的不稳定，也不是残差输入已经不同。

最终只有 row 3、feature 1019 的 BF16 norm 输出不同：

- packed FP32：-0.0651855394244194 → BF16 -0.06494140625。
- 单 token FP32：-0.065185546875 → BF16 -0.0654296875。

单 token FP32 值恰落在两相邻 BF16 值的 midpoint；一个 FP32 ULP 级的
reduction 差异改变最终舍入。微实验精确复现捕获的实际输出，非近似公式替代。

在四项匹配控制基础上，**仅逐 token 执行这一处 layer 9 post-attention norm**，
reset 和 rolling 都恢复 128/128 logits、全部层 hidden/residual、块末 conv/recurrent
及有效 prefix KV 精确一致。这是当前样本、当前 S 的充分干预，不代表其它
样本的所有 norm 都只需改这一层。

## 2. 只控制 mean，验证归因与推广性

新增 `norm_mean`：GemmaRMSNorm 仅将 square 后的 mean 按 token 分组执行，
residual add、rsqrt、scale、cast 都继续 batched。配合卷积、recurrent、GEMM、
attention 四项控制，代码样本 reset/rolling 完全一致。

扩展输入（S=5，rolling）：

| 输入 | 原始 native 首次翻转 / 次数 | 联合控制 + norm_mean |
|---|---|---|
| natural_en | #33 / 1 | 128/128 精确一致 |
| natural_zh | #50 / 2 | 128/128 精确一致 |
| repository_code | #19 / 4 | 128/128 精确一致 |
| low_match | #70 / 2 | 128/128 精确一致 |
| long_records | 无 / 0 | 128/128 精确一致 |

原始 long_records 的 128 行 logits 全都有数值差异，尽管没有 argmax 翻转。
控制组“精确一致”同时包含全部层、块末 GDN 状态及有效 KV，不只指 token。

扩展 query count（repository_code，rolling）：

| S（含 anchor） | 原始 native 首次翻转 / 次数 | 联合控制 + norm_mean |
|---|---|---|
| 1 | 无 / 0 | 128/128 精确一致 |
| 2 | #19 / 3 | 128/128 精确一致 |
| 3 | #19 / 3 | 128/128 精确一致 |
| 5 | #19 / 4 | 128/128 精确一致 |
| 9 | #78 / 2 | 128/128 精确一致 |

S=1 原始 native 仍是 0/128 logits 精确一致；没有翻转不能证明 backend 相同。
这些是输入扫描和 S 扫描，不是五种输入与五种 S 的完整笛卡尔积。

## 3. 状态移植：递归、卷积历史和 KV 有交互

从同一个原始 native rolling trajectory 的块起点做反事实：输入、packed
算术都固定，分别将 GDN、KV、全部历史、仅 conv、仅 recurrent 恢复为 ordinary。
各反事实完成后恢复原始 rolling 终点再继续，检查原轨迹全部 logits/layer/state/KV
指标与无探针运行完全相同，避免实验本身污染历史。

表中列出块内发生翻转的输出 token；“无”不表示 logits 精确一致：

| 块内输出 | 原始 rolling | 恢复 GDN | 恢复 KV | 恢复全部 | 仅 conv | 仅 recurrent |
|---|---|---|---|---|---|---|
| #17–21 | #19 | 无 | 无 | 无 | 无 | #19 |
| #62–66 | #63 | 无 | 无 | 无 | 无 | 无 |
| #92–96 | 无 | 无 | 无 | 无 | 无 | 无 |
| #107–111 | 无 | #111 | #111 | 无 | #111 | 无 |
| #117–121 | #120、121 | 无 | #121 | #120、121 | #121 | 无 |

由此可以判定：

- #19 不能解释为“单独 recurrent state 差异造成”：恢复 recurrent 不消除，
  恢复 conv 或 KV 则消除。至少有卷积输入历史和 KV 历史的贡献。
- #120–121 恢复 recurrent 能消除，支持 recurrent 历史在这些位置的贡献。
- #107–111 混合状态会制造翻转，#117–121 恢复全部历史仍有翻转。
  当前块局部误差与历史误差能抵消或叠加，不能按单因素结果给出误差占比。
- 没有证据支持“误差必须单调增长”或“全序列只有一个误差源”。

这证明当前状态选择正确并不等于数值历史等价；也说明不能只用 recurrent
error 曲线增大来断言所有翻转都来自 GDN 递归放大。

## 4. vLLM 版本与执行语义必须单独固定

对保存的相同 norm 操作数，执行固定 commit
`a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb` 的原始
`vllm/ir/ops/layernorm.py` native 函数体（装饰器只做身份注册，Torch 操作共用）。
源码 hash、commit、工作树未修改检查记入结果。

该 pinned native fused norm 是先 FP32 residual add，再以未舍入的 FP32 值
计算 variance；本地则先 BF16 add，再转 FP32。此操作数上 norm 输出最大差
0.0625，RMS 约 0.002647。这个不同维度的实现差异不能由 matching mean shape
修复，因为要先确定以哪一种 residual 算术为标准。

**这里测的是 pinned native 公式，未测它的完整 CUDA dispatch。** pinned 的
`vllm_c` provider 要求 input/weight dtype 相同，不能把 FP32 `(1+weight)`
直接传入 BF16 custom kernel来代替真实 GemmaRMSNorm 调度。

已安装的完整 engine baseline 是 vLLM 0.19.0，其 GemmaRMSNorm BF16 native
公式却使用 BF16 residual add；其 CUDA forward 会 torch.compile static 函数。
因此不能把 pinned commit 的 FP32-add native 公式差异直接用来解释此前
0.19.0 完整 engine 的生成分歧。两套参考需分别报告版本、公式和执行 provider。

随后对安装版 0.19.0 的原始 `_forward_static_with_residual` 函数做实际
`torch.compile` 重放（当前 Torch 2.10.0+cu128，配置以运行环境为准）：

| 相同操作数对照 | 最大输出差 |
|---|---:|
| 0.19.0 eager vs 本地 eager | 0 |
| 0.19.0 默认 compiled vs eager | 0.0625 |
| 默认 compiled packed vs compiled 单 token | 0 |
| 默认 compiled vs pinned FP32-add native 公式 | 0 |
| compiled + emulate_precision_casts，单 token vs eager 单 token | 0 |
| compiled + emulate_precision_casts，packed vs eager packed | 0.00048828125 |
| compiled + emulate_precision_casts，packed vs 单 token | 0 |

Torch Inductor 配置源码说明，默认会消去融合的低精度算子之间的
downcast/upcast，`emulate_precision_casts` 用于保留这些中间舍入。
上述结果支持：此 norm 的默认融合消去了 eager 中 BF16 residual add 的
中间精度截断，使其输出等于未舍入 FP32 residual-add 公式；保留精度截断后，
单 token 恢复 eager 值，而 compiled reduction 仍与 eager packed reduction 有
一个 BF16 元素的差异。

这进一步收缩了与 0.19.0 对齐的候选来源：**本地 eager 与 vLLM compiled norm
并非同一数值运算路径**，尽管 BF16 Python 表达式相同。此结论来自实际静态
函数编译与捕获的 checkpoint 操作数，不是完整 engine gate；不能单凭它宣布
norm 是完整模型唯一差异来源。保留 `target_norm_stages_compiler.json` 作为证据。

## 5. 产物、验证与下一步

- 脚本：`benchmarks/diagnose_target_numerics.py`，支持 `norm:正则` 定点替换、
  `--audit-norm-offsets` 捕获操作数、`--counterfactual-offsets` 五类状态移植。
- 微实验：`benchmarks/diagnose_norm_operands.py`，重放各阶段、固定源码公式。
- 原始结果：`logs/validate/target_norm_audit.json`、`target_norm_single_call.json`、
  `target_norm_mean_only.json`、`target_state_transplant_split.json`、
  `target_numerics_expanded_*.json`、`target_numerics_shape_*.json`。
- 输入与 S 扫描汇总：`logs/validate/target_numerics_phase2_summary.json`。
- Norm 中间运算与编译配置对照：`logs/validate/target_norm_stages_compiler.json`。
- 独立控制 conv/recurrent/KV 的恢复检查、未来 KV 槽不被覆盖检查已通过；
  反事实探针保留原始轨迹指标的检查通过，脚本编译与 diff whitespace 检查通过。

当前没有 B=4 推广测试，没有重跑完整真实 MTP strict gate，也没有速度结论。
逐 token GEMM/mean 是定位用控制，不是待合入的加速方案。下一步应先固定
vLLM 参考版本与 norm provider，做其真实 norm 调度的共同操作数对照，再设计
统一 reduction/kernel 路径。随后 B=4 和 MTP 自由生成验收，最后测性能。

本轮共完成 24 个 128-query target 对照（含 norm 捕获、定点替换和三轮状态
探针），另外完成共同操作数的 norm 微实验。所有结果是逐元素精确数值比较，
不另声称已检查 IEEE 正负零存储位。首轮方案及结果见
`docs/target_numerics_experiments.md`。
