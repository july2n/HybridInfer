# Single-token target 与 packed verification 数值误差定位

第二轮已完成定点 norm 微实验、状态移植、输入与 query count 扫描，以及
vLLM norm 编译配置对照，详见 [第二轮结果](target_numerics_phase2.md)。

## 问题与证据边界

当前候选机制：两条 target 执行路径的局部舍入误差，经 GDN 递归状态及
attention KV 历史传递，在小 logit margin 位置导致 greedy 翻转。
首次分歧较晚、某些输入 128 token 一致，与此机制相容，但不能证明它。
状态端点选择和公共验证 kernel 的既有通过记录，不能替代 target 数值等价实验。

本实验隔离 target，不调用 MTP proposer 或 rejection sampler。所有 packed 输入
来自 ordinary target 的固定轨迹。这不是改用 n-gram：配置只用于分配实验状态池。
生产实现和默认验证模式不变。

## 固定条件

- 同一 checkpoint、prompt token IDs、BF16 权重、FP32 recurrent state、TP=1。
- eager、prefix cache 关闭；B=1 首先定位，之后 B=4 验证推广性。
- 每次同一路径 prefill；prefill 后从相同 conv/recurrent/KV 状态开始。
- teacher forcing：所有路径消费完全相同的 token，发生 argmax 翻转仍继续喂普通路径 token。
- 两条路径均逐行执行 LM head，先排除词表投影 GEMM shape 的干扰。
  LM head packed/逐行投影须另设独立实验，不混入上述对照。
- query count S=K+1，包括 anchor。扫描 S=1、2、3、5、9；S=1 仍可能触发
  native verification backend，并不等于 ordinary decode backend。
- 比较语义：输入生成 token #1 的 target logits 预测生成 token #2；报告中的
  `output_token_number` 从 1 起。prefill 生成的 #1 不在 target 实验行中。

## 实验 A：局部偏差与历史偏差分开

1. `reset`：每个 query 块开始时，packed 分支恢复 ordinary 分支当前的完整
   conv/recurrent/KV prefix 状态。只测本块运算差异，去除跨块累计。
2. `rolling`：ordinary 与 packed 各自保留状态，但消费相同 token。
   测相同上下文下历史数值差异的传播。对照二者误差曲线与翻转位置。

每块保存有效 prefix KV，不比较未来未计算槽位。记录各 GDN 层块末 conv 和
recurrent 状态、全部有效 KV；每个 token 记录所有 decoder layer 的 hidden 和
residual 两条流、最终 logits。不能只比较 hidden+residual，抵消会掩盖误差。
初始 harness 的状态误差时间分辨率为块，不声称已经测到块内每 token 状态；
需要时用 native recurrent/conv endpoint snapshots 细化。

## 实验 B：算术与 shape 干预

| 变体 | 仅在 packed 分支更改的因素 |
|---|---|
| native | 原始 packed target |
| conv | ordinary 卷积的乘法/累加、SiLU 前舍入和 SiLU 表达式 |
| bv | recurrent BV=8，保留 FMA |
| fma | recurrent 关闭 FMA，保留 BV=32 |
| recurrent | BV=8 且关闭 FMA |
| gemm | 全部 dense projection 按单 token 调用，包括 GDN/MLP/attention projection |
| attention | 相同 Q/K/V，逐 token 调用 decode attention backend |
| pointwise | norm/activation 按 token 分组调用，保持每 token 的 head 分组 |
| conv+recurrent+gemm | 三类候选因素同时匹配 |
| conv+recurrent+gemm+attention | 再统一 attention backend |

先筛选单因素和联合控制，再完成 conv/recurrent/gemm 的 2³ 全因子组合，检查
交互项；BV 与 FMA 另做 2×2。单因素改善不能排除其他因素，联合控制改善
也不能据此给各因素分配百分比。所有 monkeypatch 只在 benchmark 作用域生效并恢复。
逐 token GEMM 是诊断控制，有明显性能代价，不是最终优化方案。

若联合控制仍不逐位一致，继续查 norm/activation 的 shape、张量 stride、
attention 其他算术差异，以及 packed recurrent 与 decode 的循环/编译差异。
BV/FMA 参数一致本身并不保证这两个不同 kernel 完全相同。
`gemma_norm`、`gated_norm`、`activation` 可以分别替代 `pointwise`，细分残留来源。

## 实验 C：相同操作数的微实验

整模型存在上游输入差异，不能直接把某算子输出差异归因于算子本身。
在最早差异层截取 ordinary 输入，并分别执行两种实现：

- 相同 raw QKV、conv history、weight 的两种 convolution。
- 相同 conv 输出、a/b、A_log、bias、初始 recurrent state 的 BV/FMA 2×2。
- 相同 hidden 输入的 packed/单行 GEMM；单独测 LM head。
- 相同 Q/K/V 和有效 prefix KV 的 varlen/decode attention。

先比较输出和最终状态，再将这些操作数接回下游模型，观察是否消除翻转。
这样区分“算子自身舍入差异”和“接收了不同上游输入”。

## 实验 D：状态移植的因果验证

选择 rolling 首次翻转之前的块，以及 reset/rolling 差距开始增长的块。
保持 packed 算术和输入不变，从块开始的快照做四个反事实：

| GDN 状态来源 | KV 来源 | 目的 |
|---|---|---|
| packed | packed | 原始 rolling 对照 |
| ordinary | packed | 移除 GDN 历史误差（含 conv 与 recurrent） |
| packed | ordinary | 移除 attention 历史误差 |
| ordinary | ordinary | 移除全部历史误差，即 reset 对照 |

若恢复 GDN 后误差/翻转消失，支持 GDN 历史贡献；若只恢复 KV 有效，支持
attention 历史贡献；若均无效而联合干预有效，支持当前块局部执行差异。
两者有交互时继续拆分 conv/recurrent，并对 GDN 层分组移植。不得用“recurrent
state 误差变大”单独作为因果证据。

## 统计、判定与输出

记录 bitwise equality、max absolute error、RMS、relative L2、finite；不把
“无 token 翻转”等同于“数值一致”，也不预设状态误差单调增长。

对 ordinary 胜者 a 与 packed 胜者 b，定义 gap=z_ref[a]-z_ref[b]，
direction=(z_packed[b]-z_ref[b])-(z_packed[a]-z_ref[a])。
若 a≠b，应满足 direction≥gap，等号时检查 argmax tie rule。
packed 未翻转时用 ordinary runner-up 测 direction；翻转时一定使用实际 b，
不能假定 b 原本就是 top2。严格正 margin 大于 2×logit max_abs 是保住
argmax 的充分条件；低 margin 只是易感条件，不是误差来源。

记录首个 layer 差异、首个 logit 差异、首个 winner 翻转，三者须分别报告。
128 token 内无翻转是观察窗口内未发生，不能推断更长输出也一致。
保持所有干预的 teacher-token SHA256 相同，否则实验无效。

建议输出 JSON、token/block CSV 和静态图：token 横轴上的 recurrent error、
logit error、margin、翻转标记；各干预首次翻转与总翻转数对照。误差图使用
对数纵轴需单独标注零误差，避免把零截断成“很小的误差”。

## 执行顺序

1. `repository_code`（已知早期分歧）筛选，先 S=5、128 target queries。
2. 补 `natural_en`、`natural_zh`、`low_match` 和 `long_records` 对照。
3. 全因子和 S 扫描定位 shape/interaction；微实验、状态移植验证因果。
4. 选定修复后重跑原始真实 MTP、B=1/4、128-token strict gate，随后测吞吐与延迟。
   teacher forcing 结果不替代自由生成和端点恢复验收。

首轮命令（Python 路径按环境调整）：

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python -u \
  benchmarks/diagnose_target_numerics.py --case repository_code --tokens 128 --block 5 \
  --json-out logs/validate/target_numerics_code.json
```

`diagnose_target_numerics.py` 当前实现 A、B 的整模型对照和上述统计；C/D 是
下一阶段的定位实验，不能将其列为已经执行。与 vLLM 的最终对齐须固定其
commit/完整 engine 版本和模型执行配置；本实验首先回答本地两条 target
路径为何不同，不声称它们必然都与 vLLM 数值相同。

## 已执行的首轮结果（2026-09-28）

`repository_code`，B=1，S=5，128 次 target 预测（生成 token #2–#129）。
同一 ordinary 轨迹、共同 prefill，所有变体的 teacher-token hash 一致。
这些是 target teacher-forcing 对照，不是 MTP 自由生成验收或速度测试。

| 干预 | reset 首次翻转 / 次数 | rolling 首次翻转 / 次数 |
|---|---|---|
| native | #120 / 2 | #19 / 4 |
| conv | #111 / 2 | 无 / 0 |
| BV=8，仅此一项 | #120 / 2 | #121 / 2 |
| 禁用 FMA，仅此一项 | #121 / 1 | #19 / 5 |
| BV=8 + 禁用 FMA | #121 / 1 | #19 / 5 |
| 单 token GEMM | #121 / 1 | #111 / 3 |
| conv + recurrent + GEMM | #111 / 2 | #111 / 2 |
| attention，仅此一项 | 无 / 0 | #120 / 2 |
| conv + recurrent + GEMM + attention | 无 / 0 | #111 / 2 |
| 上述四项 + pointwise | 无 / 0 | 无 / 0 |

重要区别：仅改 conv 的 rolling 虽无翻转，但 128 行 logits 全部仍有差异。
四项联合控制的 reset 有 126/128 行 logits 逐位一致，rolling 为 93/128；
首个非一致预测均在 #95，最早不同的 decoder layer 为 index 9（第 10 层）。
仅四项控制不足以恢复严格一致。

再加入 pointwise 后，两种状态模式的 **128/128 logits、每 token 的全部层
hidden/residual、每块末全部 conv/recurrent 状态、有效 prefix KV 均逐位一致**。
这为“差异来自数值执行路径”提供了可干预的证据，同时说明最初只列举
conv/BV/FMA/GEMM 的候选清单不完整：attention 与 norm/activation shape 也需控制。
pointwise 类别内仍需细分，单因素表现也存在交互，不能据此宣布某个 kernel
是唯一根因。历史来源究竟主要是 GDN 还是 KV，仍需实验 D 的状态移植确认。

进一步的 rolling 细分对照已完成（`target_numerics_pointwise_split.json`）：
在四项控制基础上，仅增加 `gemma_norm` 就得到 128/128 行 logits 和全部层、
块末状态、有效 KV 逐位一致。仅增加 `gated_norm` 或 `activation` 则各有
2 次翻转、93/128 行 logits 一致，与四项控制相同。因此该样本的残留已收缩
到 **GemmaRMSNorm 类别的 token shape 执行差异**；尚未区分 input norm、
post-attention norm、q/k norm、final norm 中的具体调用，也未将内部 reduction
与 pointwise 运算各自的舍入分离。下一项微实验应从 #95 / layer index 9
前后的 norm 调用捕获相同输入，比较 packed 与单 token 输出，并逐调用替换。

原始 native rolling 的 #19：reference margin=0.125，direction=0.125；
两个候选在 packed logits 中形成平局，较小 token ID 的 argmax 规则改变胜者。
因此“接近零 margin”应包含量化后的 ties，而不能只查严格交叉。

原始 JSON：`logs/validate/target_numerics_code.json`、
`target_numerics_attention.json`、`target_numerics_pointwise.json`；
汇总与图使用 `benchmarks/report_target_numerics.py`，生成 `*_summary.json`、
`*_tokens.csv` 和 `*_errors.png`。原始 JSON 包含全部层误差与块末状态误差。
共完成 23 个 128-query 对照，四组原始结果 teacher-token hash 相同；
全部 logits finite，全部翻转符合 direction≥gap。脚本编译、异常路径 override
恢复和 `git diff --check` 通过。这里的 `equal` 对 finite tensor 使用
`torch.equal`，是逐元素精确数值相等；没有另查 IEEE 正负零的存储位差异。
生产路径未采用逐 token 干预，原真实 MTP strict gate 失败结论仍然有效。
