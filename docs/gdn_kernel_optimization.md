# GDN packed 卷积与 FP32 状态池优化

## 实现范围

普通 prefill 与 piecewise prefill 共用 `forward_dense_pre` 和
`forward_core_from_dense`，消除了两套卷积、delta-rule 与历史更新代码。

`gdn_kernels.py` 新增 packed causal depthwise convolution：

- 输入为 `[total_tokens, channels]`，用 GPU `cu_seqlens` 区分请求。
- 每个请求通过 `state_indices` 读取自己的卷积历史，无需 padded batch、
  逐请求 Python 数据复制、拼接和历史 gather/scatter。
- 卷积输出先舍入到输入 BF16/FP16，再执行 SiLU，保留原先的中间舍入。
- 第二个 kernel 更新最近 `kernel_size - 1` 个原始 QKV。短 chunk 会保留
  必要的旧历史；更新和卷积分开启动，防止不同 token tiles 读写历史竞争。
- Decode 使用同一内核，每请求只处理一个 token，不需要构造 `cat` 输入。

FP32 recurrent decode 新增按 slot 直接读写的 Triton kernel。每个程序负责
一个请求、一个 value head 和一组 value rows，读取 `[slot, head, V, K]`，
进行 Q/K normalization、gate、delta-rule 更新与输出计算，然后原地写回。
不同 value rows 不共享写入位置；GQA 在 kernel 内映射 key head，无需重复
Q/K。没有将 recurrent state 降为 BF16。

当前生产选择：FP32 state 且 K=V=128 时默认使用 `pool`；其他布局或 BF16
state 使用现有 FlashInfer 路径。显式设置以下环境变量，可在引擎初始化前
选择旧 recurrent decode 后端（packed 卷积仍生效）：

```bash
HYBRIDINFER_GDN_DECODE_BACKEND=flashinfer
```

该变量在层构造时读取；图捕获后修改环境变量不会切换已捕获的路径。
本地 FlashInfer 0.6.6 的 `initial_state_indices` 直接状态池接口仅支持 BF16
fast path，不能用于本地模型的 FP32 state，因而本版独立实现 FP32 kernel。

请求 slots 必须互不重复。生产 scheduler 已保证同一请求不会重复提交；
decode 图捕获也改为使用不同的 dummy slots，避免预热时并发写入同一状态。
状态恢复、prefix 快照和异步发布机制保持原有契约。

## 普通 prefill 与 piecewise prefill 如何共用实现

共用的是 `GatedDeltaNet` 的两个函数和同一组权重、状态池；两条路径的
区别是 dense 部分通过 eager 调用还是 CUDA Graph replay 执行。

`src/hybridinfer/layers/gated_delta_net.py` 中，普通 prefill 的入口直接组合
两个函数：

```python
def _forward_prefill(self, hidden_states):
    # hidden_states: [1, total_tokens, hidden_size]
    return self.forward_core_from_dense(
        self.forward_dense_pre(hidden_states.squeeze(0))
    )
```

`forward_dense_pre` 接收 `[total_tokens, hidden_size]`，执行四个线性投影，
返回 `(raw_qkv, z, b, a)`，四个张量的第一维都是 token 维度。这一步不
读取请求边界或修改历史状态，因此可以按固定 token bucket 捕获。

`forward_core_from_dense` 接收这四个投影结果，从当前 `Context` 获取
`cu_seqlens_q`、`state_indices` 和 prefill chunk 元数据，执行以下计算：

1. packed 卷积读取各请求对应 slot 的历史，并更新最近的原始 QKV。
2. 拆分 Q/K/V，计算 beta、decay gates，并处理 GQA。
3. `chunk_gated_delta_rule` 按请求边界扫描，直接读取和更新 recurrent
   state pool 中对应的 slots。
4. 执行带 z gate 的 normalization 和输出投影，返回
   `[1, total_tokens, hidden_size]`。

`src/hybridinfer/models/qwen3_5.py` 的 decoder layer 把 piecewise 调用
接到相同函数上：`forward_piecewise_pre` 先完成 input norm，再调用
`linear_attn.forward_dense_pre`；`forward_attention_core` 调用
`linear_attn.forward_core_from_dense`，并把输出压回二维。后续
`forward_output` 处理 residual、post-attention norm 和 MLP。

```text
普通 prefill：
  input norm → forward_dense_pre → forward_core_from_dense → residual/norm/MLP
                eager             eager                     eager

piecewise prefill：
  [input norm → forward_dense_pre] → forward_core_from_dense → [residual/norm/MLP]
             pre graph                      eager                    post graph
```

`src/hybridinfer/engine/cuda_graph.py` 的 `run_prefill` 选择大于等于真实
token 数的最小 bucket，将 embedding 写入固定地址的 hidden buffer，
清零 padding 尾部，然后逐层执行 pre replay、动态 core、post replay。
pre graph 的投影结果先切成真实的 `num_tokens` 行，再传给 core；因此
padding 不参与卷积和 recurrent 更新，也不会推进请求历史。core 输出
复制到 post graph 的固定输入 buffer，post 输出直接供下一层 pre 使用。
找不到对应 bucket 时回退到普通模型 forward。

core 当前整体在 graph 外执行，包括 normalization 和输出投影；没有
将整个 `forward_core_from_dense` 捕获。这样请求数、长度、状态 slot 和
chunk 边界可以随 batch 改变，静态投影及 MLP 则复用已捕获图。两条路径
都只在 core 中更新历史一次，因此 chunked prefill 和 prefix restore
沿用相同的状态推进逻辑；浮点运算结果仍按数值容差验证。

## 独立验证与性能测量

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v
PYTHONPATH=src:.runtime-deps python benchmarks/bench_gdn_state_pool.py \
    --json-out logs/bench/gdn_state_pool.json
```

新增测试使用独立 PyTorch 卷积和递推公式，检查 BF16/FP16、短历史、
非整齐 channel 数、63/64/65/129 等边界、slot 重排、多步 decode、GQA、
未参与 slots 完全不变和卷积 CUDA Graph 重放。FP32 decode 还与 FlashInfer
比较输出及完整最终状态。单算子 benchmark 不加载模型，将 GPU event 时间
与同步 host wall 时间分开报告。

2026-09-27，RTX 3060 Ti，BF16 QKV、FP32 recurrent，单算子 GPU 时间：

| 操作 | 输入规模 | 新旧 GPU 时间比（旧/新） |
|---|---|---:|
| 卷积 | 512 tokens，单请求 | 5.31× |
| 卷积 | 4 请求，各 128 tokens | 5.49× |
| 卷积 | 64/128/256/512 tokens | 4.40× |
| 卷积 | 8 请求，各 1 token | 6.27× |
| recurrent decode | batch 1 | 2.51× |
| recurrent decode | batch 3 | 3.78× |
| recurrent decode | batch 8 | 2.96× |
| recurrent decode | batch 16 | 3.01× |

卷积对照包括原有 padding、拼接和历史读写；decode 对照包括 gather、
FlashInfer 和 scatter。以上是单算子结果，不代表整模型等比例加速。

## 整模型验收

```bash
PYTHONPATH=src:.runtime-deps python benchmarks/validate_engine_qwen35.py \
    --model models/Qwen3.5-0.8B --cases w11,w12 \
    --gpu-memory-utilization 0.75 --regression-decode-steps 3 \
    --json-out logs/validate/gdn_pool_engine.json
PYTHONPATH=src:.runtime-deps python benchmarks/validate_prefix_cache.py \
    --model models/Qwen3.5-0.8B --graphs \
    --json-out logs/validate/gdn_pool_prefix.json
```

整模型沿用既有数值门槛，完整记录 KV、conv/recurrent state 和 token
对照，未降低标准。原生 Transformers 的严格 logits 验收仍需独立报告，
不能用内部模式之间通过对照来替代。

本次 GPU 完整测试集 **62/62 通过，无跳过**。W11/W12 四种模式都通过既有
数值容差验收：各模式 W11 为 39/39，W12 为 6/6；C 的 W11 仍有已记录的
BF16 near-tie，不能宣称所有模式 token 完全一致。Prefix 验证的重复请求、
不同后缀、并发分支、快照淘汰和 KV 淘汰五个场景全部通过，生成 tokens 一致，
4 次恢复和 8 次保存均逐元素一致。

复用独立原生 Transformers 参考进行 teacher forcing/free generation：

```bash
PYTHONPATH=src:.runtime-deps python benchmarks/validate_dense_reference.py \
    --phase engine --model models/Qwen3.5-0.8B \
    --reference logs/validate/dense_native_reference_v2.pt \
    --json-out logs/validate/gdn_pool_dense_reference.json
```

10/10 free generation 案例、160/160 teacher-forced argmax 一致。最大逐步
相对 RMSE 为 **2.7451%**（优化前 2.7637%），max_abs 为 **0.44140625**
（优化前 0.40625）。仍超过预设的 2% 相对 RMSE 门槛，因此该脚本按设计
返回非零状态；本次没有上调门槛，也不将此项写为通过。


## 端到端性能

使用与上个提交 `0cd0d12` 相同的模型、预算、bucket 和测量次数：

```bash
PYTHONPATH=src:.runtime-deps python benchmarks/bench_piecewise_prefill_paths.py \
    --model models/Qwen3.5-0.8B --prefill-graph-buckets 128,256,512 \
    --gpu-memory-utilization 0.75 --rounds 10 --warmup-rounds 2 \
    --mixed-rounds 3 --mixed-warmup-rounds 1 --decode-tokens 8 \
    --json-out logs/bench/gdn_pool_engine.json
python benchmarks/compare_piecewise_bench.py \
    --before logs/bench/graph_optimization_final.json \
    --after logs/bench/gdn_pool_engine.json \
    --json-out logs/bench/gdn_pool_engine_comparison.json
```

P3（decode graph + piecewise prefill）的 median wall time：

| 工作负载 | 优化前 | 优化后 | 延迟下降 |
|---|---:|---:|---:|
| 单请求 512 tokens | 42.95 ms | 36.85 ms | 14.20% |
| 4 请求，各 128 tokens | 43.45 ms | 36.90 ms | 15.07% |
| 变长，总计 480 tokens | 47.30 ms | 37.60 ms | 20.51% |
| chunked 变长，总计 960 tokens | 86.20 ms | 71.90 ms | 16.59% |
| 7 个 decode 请求 + 延后到达的 prefill | 125.30 ms | 90.70 ms | 27.61% |

P2（decode graph + eager prefill）对应下降约 14.05%、15.20%、20.30%、
16.55%、27.82%，说明收益来自两条 prefill 路径共用的 GDN 改进及 recurrent
state 直接访问，不能只归因于 piecewise graph。这里只报告此次硬件与
工作负载的结果；纯 prefill 每项 10 次、混合请求 3 次，不能推广为所有模型
或负载的保证。

混合请求 P3 的峰值分配由 4926.4 MiB 降到 4842.6 MiB，但初始化后的
总分配增加 12 MiB；KV 池容量由运行时显存估算决定，不能把整个引擎的
分配差异直接当作 GDN 临时 buffer 的差异。所有原始 JSON 保存在不提交的
`logs/bench/` 与 `logs/validate/`。
