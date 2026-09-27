# CUDA Graph buffer 与调度优化

## 实现

Piecewise prefill 仍按每层的 pre graph、eager attention core、post graph
顺序执行。静态算子和 BF16 舍入路径保持原实现，没有引入 compiler fusion。

原先每层都要将 hidden/residual 复制到 pre 输入，随后再将 residual 复制到
post 输入。现在整批共用三个固定 buffer：

- `hidden`：第一层输入，随后由每层 post graph 写回，下一层 pre 直接读取。
- `residual`：pre/post graph 都直接读写该 buffer。
- `attention`：eager attention 的输出复制到这里，作为 post graph 输入。

每批只复制一次 embedding 输出，并只清零 hidden/attention 的 padding
区间。Attention core 仅接收真实 token 行，padding 不参与 KV/GDN 状态更新。
相同输出形状、dtype、device 的投影结果共用输出 buffer，不再按层各保存
一份。投影输出仍在 graph 内复制到固定地址，避免依赖共享 graph pool 的
临时 tensor 生命周期。

这依赖当前 runner 的顺序重放：一个 pre 的输出被 eager attention 消费后，
下一个 pre 才能覆盖它。不同请求 batch 可以异步排队，但 forward 在同一
compute stream 上执行；该 manager 不支持多个 stream 同时重放共享 buffers。

Decode 图补齐了 1–16 的所有 batch sizes；更大尺寸仍使用 16 的倍数及
`max_bs`，最大为 512。仍要求精确尺寸命中，不向上 padding，因此不会让
虚拟请求更新真实 GDN slot。

Scheduler 只有成功预留 GDN checkpoint 后，才为保存位置额外切分 prefill。
当快照池全部被引用，或该前缀已经有快照时，保留原 token 预算对应的 chunk，
避免无用 forward。Prefix lookup 仍要求 KV 与 GDN 同时有效。

## 验证与复现

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v

PYTHONPATH=src:.runtime-deps python benchmarks/validate_prefix_cache.py \
    --model models/Qwen3.5-0.8B --graphs \
    --json-out logs/validate/prefix_cache_graph_optimization.json

PYTHONPATH=src:.runtime-deps python benchmarks/validate_engine_qwen35.py \
    --model models/Qwen3.5-0.8B --cases w11,w12 \
    --gpu-memory-utilization 0.75 --regression-decode-steps 3 \
    --json-out logs/validate/piecewise_shared_buffers.json
```

在 RTX 3060 Ti、本地 Qwen3.5-0.8B、TP=1 上，58 项测试全部通过，无跳过。
新增真实图测试覆盖 bucket 缩小/增大、padding、eager fallback、重新 capture
和共享 buffer 生命周期；新增 decode 测试确认 batch size=3 实际重放图。
调度测试覆盖快照池全被引用和已有快照但 KV 缺失时不额外切分。

W11 包含 39 个单请求分块对照，W12 包含 6 个混合 batch 对照，在
BASELINE/A/B/C 四种模式下均按现有数值验收标准通过。C 的 W11 有 BF16
近似并列 token 差异，分类为 `NUMERIC_TIE`，不宣称所有模式逐 token 一致。

性能命令如下，在优化前后使用相同参数分别保存 JSON：

```bash
PYTHONPATH=src:.runtime-deps python benchmarks/bench_piecewise_prefill_paths.py \
    --model models/Qwen3.5-0.8B --prefill-graph-buckets 128,256,512 \
    --gpu-memory-utilization 0.75 --rounds 10 --warmup-rounds 2 \
    --mixed-rounds 3 --mixed-warmup-rounds 1 --decode-tokens 8 \
    --json-out logs/bench/graph_optimization_final.json

python benchmarks/compare_piecewise_bench.py \
    --before logs/bench/piecewise_before.json \
    --after logs/bench/graph_optimization_final.json \
    --json-out logs/bench/graph_optimization_comparison.json
```

对照工具验证模型和配置一致，报告各工作负载的 median wall time、P2 eager
对照变化、P3 piecewise 变化和显存变化。数值采集与性能测量分开进行。

## 本次性能结果

基线为提交 `7c192f5`，对照为本次最终实现；每种模式独立进程，纯 prefill
每项测量 10 次、混合请求测量 3 次，下面使用 median wall time。

| 工作负载 | 原 P3 | 优化后 P3 | 延迟下降 |
|---|---:|---:|---:|
| 单请求 512 tokens | 43.50 ms | 42.95 ms | 1.26% |
| 4 请求，各 128 tokens | 44.10 ms | 43.45 ms | 1.47% |
| 变长，总计 480 tokens | 47.60 ms | 47.30 ms | 0.63% |
| chunked 变长，总计 960 tokens | 86.70 ms | 86.20 ms | 0.58% |
| 7 个 decode 请求 + 延后到达的 prefill | 154.80 ms | 125.30 ms | 19.06% |

混合请求收益主要来自补齐小 batch decode 图：P2 eager-prefill 模式的同一
混合工作负载也由 151.50 ms 降到 123.30 ms。不能将这 19% 归因于
piecewise prefill 本身。纯 prefill 的改善较小，初次仅修改 buffer 的测量中，
单请求 512-token 场景还出现过 1.26% 回退；这些数字不代表普遍或统计显著
的延迟收益。最终 P3 在本次工作负载中仍比 P2 慢约 0.6%–1.9%。

P3 初始化后的已分配显存由 4967.0 MiB 降到 4797.5 MiB，减少 169.5 MiB；
混合工作负载的峰值分配减少约 168.6 MiB。Allocator reserved memory 基本
相同，因此这不等于给其他进程立即释放了 169.5 MiB 显存。P2 补齐小 batch
图后初始化分配增加约 32.5 MiB；启动捕获耗时尚未单独验收。

最终 prefix cache 验证的五个场景全部通过，生成 tokens 一致，4 次恢复和
8 次保存均逐元素一致。完整结果存于不提交的 `logs/bench/`、
`logs/validate/` 目录。
