# Qwen3.5 Dense 与混合架构验收

本次验收限定为本地 `Qwen3.5-0.8B`、纯文本、BF16、单张 RTX 3060 Ti。
环境为 PyTorch 2.10.0+cu128、Transformers 5.3.0、FlashInfer 0.6.6、
FlashAttention 2.8.3。其他尺寸、多 GPU、视觉、MoE、MTP 不属于本次验收范围。

## Dense 审核

| 项目 | 检查与修正 |
| --- | --- |
| 配置 | 只接受 Dense text config；检查混合层数量与类型；旧配置按 full_attention_interval 补全 layer_types；不支持的 RoPE 类型明确报错 |
| 权重 | model.language_model 映射到 model；MLP gate/up 分片必须齐全；embedding/head 共享存储允许单份权重；只跳过已知 vision/MTP 分支；缺失或未知文本参数报错 |
| RMSNorm | 归一化和 `(1 + weight)` 乘法保持 FP32，最后转回 BF16；残差先按 BF16 相加再归一化，与参考执行顺序一致 |
| GDN gated norm | 保留 checkpoint 的 FP32 norm weight，避免乘法前将权重截断到 BF16 |
| GDN state dtype | 遵循 mamba_ssm_dtype；本地配置为 FP32，避免在 scheduler chunk 边界额外截断到 BF16；保留显式 BF16 配置及原有 debug override |
| RoPE | 保持 FP32 频率缓存；cos/sin 转为输入 dtype；旋转保留 BF16 中间舍入；文本位置的三路 MRoPE 相同；只旋转前 64/256 维 |
| Gated GQA | q_proj 按每头 query/gate 拆分；Q/K norm 与 partial RoPE；attention 输出乘 sigmoid(gate) 后执行 o_proj |
| Dense MLP | gate/up 权重打包；SiLU 结果先按 BF16 舍入再相乘，去除会改变中间舍入的编译融合 |

上述配置、权重损坏检查、Norm、RoPE、attention 和 MLP 对照在
`tests/test_dense_reference.py` 中覆盖。单层参考使用 Transformers 的实现。

## Transformers 对齐方法

```bash
PYTHONPATH=src:.runtime-deps python -u benchmarks/validate_dense_reference.py \
  --model models/Qwen3.5-0.8B \
  --reference logs/validate/dense_native_reference_v2.pt \
  --json-out logs/validate/dense_native_final.json
```

参考和引擎在不同进程执行，以适配 8 GB 显存。参考独立读取 safetensors，
执行 Transformers text model、eager attention、PyTorch GDN 与 cached decode，
不复用引擎加载器或 Triton/FlashInfer 内核。模型在 BF16 dtype 下构造，
避免整体 `.to(BF16)` 意外截断 RoPE 的 FP32 inverse-frequency buffer。

覆盖 3 个中文/英文/代码聊天提示，以及 1、63、64、65、127、129、257 token
的边界输入；每例生成 16 token，共比较 160 个完整词表 logits 向量。

- Teacher forcing：引擎每步输入参考 continuation，使所有 logits 比较拥有相同历史。
- Free generation：另起请求自由 greedy 生成，独立核对 token 序列。
- 每步记录 RMSE、相对 RMSE、max_abs、实际竞争 token 和 BF16 near-tie。
- position 0 的单 token 输入另外对每个 decoder 注入相同参考输入，区分局部与累计漂移。

验收脚本预设逐步相对 RMSE 不超过 2%、max_abs 不超过 1.0；token 必须一致，
或实际竞争 token 在两侧均处于一个 BF16 ULP 范围。失败时返回非零状态并保留 JSON。

### 当前 Dense 结果：严格 logits 门槛尚未通过

10/10 案例的 free greedy 生成完全一致，160/160 步 teacher-forced argmax 一致。
但逐步 logits 最大相对 RMSE 为 **2.7637%**，最大绝对误差 **0.40625**，
部分案例超过预设的 2% 门槛，
因此不能将本阶段写成“严格 logits 验收通过”。没有上调门槛掩盖该结果。

position 0 的逐层同输入诊断中，6 个 full-attention decoder 输出逐元素相同；
18 个 GDN decoder 的局部相对 RMSE 最大约 **0.8187%**。这说明该诊断的差异
集中在 GDN 路径，整模型会累计漂移；该结论限于被检查的输入，不能推广为任意上下文的保证。

原生 Transformers fallback 的 BF16 Q/K norm、BF16 标量权重、FP32 recurrent
state，与引擎的 FP32 Q/K reduction、FP32 checkpoint 标量有不同
精度规则。可用 `--reference-profile kernel` 做独立 PyTorch 精度对照；它修改
参考的 Q/K norm 和标量精度，并非未修改的 Transformers fallback。
该对照也未达到所有步骤 2% 门槛，不能替代原生结果作为通过证据。

GDN 单独数学检查使用独立逐 token 递推公式，对比 packed-varlen prefill、
最终递归状态和 FlashInfer decode，并验证未参与计算的 state slot 完全不变。
这些检查均在 2% 相对 RMSE 门槛内通过。

## 混合架构验收方法

```bash
PYTHONPATH=src:.runtime-deps python -u benchmarks/validate_engine_qwen35.py \
  --model models/Qwen3.5-0.8B \
  --json-out logs/validate/dense_hybrid_fp32_final.json
```

矩阵分别运行同步 eager、异步 eager、decode CUDA Graph、decode + piecewise
prefill CUDA Graph。所有模式均使用生产 scheduler、请求 slot、KV cache 和
GDN state pool，仅将 sampler 替换为确定性 argmax 以便对照。

| 场景 | 验收内容 |
| --- | --- |
| 请求边界与变长批次 | 卷积历史不能跨请求；packed prefill 的 cu_seqlens、slot 映射正确 |
| Batched decode | batch 1/2/3/4/5/8 与单请求对照；覆盖图命中及 eager fallback |
| 分块 prefill | 13 种长度 63..1024，分别按 64/128/256 分块；39 组与完整 prefill 对照 |
| Mixed batch | 254/255/256 token 的 A decode 与 128 token 的 B prefill，共 6 组；B 两种输入不影响 A；未调度请求的完整状态必须保持不变 |
| 抢占重算 | 调用真实 scheduler.preempt，下一轮通过 engine teardown 回收 slot；从位置 0 完整/分块重放已生成历史；65/257 token × 两种预算，共 4 组 |
| Slot 复用 | B 复用 A 的 slot 与 B 在新引擎运行结果一致 |
| 异步执行 | 队列深度、晚到请求、D2H 存储、长 decode 与同步对照 |

分块/mixed/抢占检查完整 GDN conv、recurrent 和按逻辑 token 排列的 KV 快照。
相同历史时采用既有相对 RMSE 上界：conv 5%、recurrent 2%、KV 5%；
逐元素 allclose/max_abs 同时记录。首个 near-tie 导致历史分歧后，后续状态
保留为诊断而不冒充相同输入对照。

Near-tie 诊断保存 top-8；存在完整 logits 时直接读取实际竞争 token 的分数，
避免 top-2 漏掉三方并列。不存在竞争 token 的分数时不予放行；无关 runner-up
的接近也不能掩盖实际 winner 的明确分歧。

此前强制 BF16 state 时，128-token prompt 按 64 分块在第二个生成 token
出现非 near-tie 分歧（完整 prefill token 149066，分块 token 19588），
使 w11 为 38/39。完整 chunk scan 在内核内部保留 FP32 累积；外部分块
若将最终状态存入 BF16 pool，会在边界多一次截断。按照本地配置恢复 FP32
state 后，四种模式的 w11 均为 **39/39**。最终矩阵 **46/46 单元通过数值
容差验收**（11 个跨模式用例 × 4 模式，加 2 个本地生命周期用例）；
每种模式的 mixed 为 **6/6**、抢占重算为 **4/4**。仍存在已明确记录的
near-tie token 差异，因此 `overall_pass=true`、`overall_exact_pass=false`。

FP32 FlashInfer decode 的 DLPack 路径要求输入没有 autograd 依赖。
CUDA Graph 捕获入口现使用 `torch.inference_mode()`，覆盖预热和捕获；
修正前图模式在初始化期间报错，修正后 B/C 全部验收通过。结果复用已通过的
BASELINE/A 输出，与重新运行的 B/C 输出一起汇总，没有把旧的启动失败当成通过。

FP32 recurrent pool 比 BF16 多一倍显存。引擎在估算 KV cache 容量前分配
GDN 状态并 warmup，实际状态占用已纳入显存预算；不声称此次更改提升性能。

## 回归检查与范围

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v
```

本次在 GPU 上执行 **42 项测试，全部通过，无跳过**。
JSON、参考 logits 和执行日志保存在 `logs/validate/`，该目录按仓库规则不提交。
混合模型 prefix cache 默认关闭；现已补充可选的 GDN 状态快照与联合 KV 恢复。
新增实现与验收结果见 [Prefix caching](prefix_caching.md)。

## GDN 内核优化后的独立复验

以上 Dense 数字为此前实现的验收记录。新增 packed 卷积和 FP32 indexed
recurrent decode 后，复用同一原生 Transformers 参考，160/160
teacher-forced argmax 和 10/10 free generation 案例仍一致；最大相对 RMSE
为 2.7451%，max_abs 为 0.44140625。**严格 2% logits 门槛仍未通过**。
完整实现、独立内核验证及性能记录见 [GDN 内核优化](gdn_kernel_optimization.md)。
