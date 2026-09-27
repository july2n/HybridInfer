# Hybrid prefix caching

启用方式：

```python
from hybridinfer.engine.llm_engine import LLMEngine

engine = LLMEngine(
    "models/Qwen3.5-0.8B",
    enable_prefix_cache=True,
    prefix_cache_num_snapshots=8,
)
```

默认关闭。`prefix_cache_num_snapshots` 是整个引擎可保存的 GDN 快照数量，
每份包括所有 GDN 层的卷积和 recurrent state。状态 dtype 与运行池一致，
不做压缩。本地 0.8B、TP=1、BF16 conv/FP32 recurrent 的每份快照约
18.63 MiB，8 份约 149.06 MiB；该池在 KV 容量估算前分配。

## 保存与恢复

- Full attention 沿用分页 KV 池；GDN 运行状态沿用每请求固定 slot。
- 保存块对齐的 prefill chunk 终点。scheduler 会切分 prompt 尾部，
  在成功预留快照后，使最后一个可复用的完整块终点成为独立计算边界。
  无法预留或已存在快照时不为保存边界额外切分。
- 完全重复的 prompt 也至少计算一个 token，以取得采样 logits。
  prompt 恰好块对齐时，会从前一个完整块终点恢复。
- 命中长度必须同时具备连续有效的 KV 前缀及所有 GDN 层的终点快照。
  GDN 快照命中后复制到请求运行 slot，各分支独立更新。
- 快照保存发生在 forward 后、采样完成事件之前；CPU 在 batch 输出完成后
  发布索引。待写快照不可命中，读写引用在 batch 完成前不可淘汰。
- 快照池淘汰未引用的最久未使用条目；全被引用时跳过本次保存。
  KV 与快照独立淘汰，联合查找会拒绝不完整的缓存。
- 关闭 prefix caching 时不会读取或发布 KV 前缀缓存。

本版只新增 prefill checkpoint；decode 不新增 GDN checkpoint。抢占后可以
从仍有效的联合前缀恢复，其余 tokens 重算。尚未提供 CPU/磁盘持久化、
任意 token 边界快照及多 GPU TP 专项验收。调度切分会影响冷请求的 prefill
调用次数，此次验证不用于吞吐或延迟结论。

## 验证

CPU 生命周期测试覆盖发布时机、并发引用、快照与 KV 分别淘汰、容量和
调度预算失败时释放引用、完全相同 prompt、sub-block 预算，以及实际状态
复制的独立性。

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v
PYTHONPATH=src:.runtime-deps python benchmarks/validate_prefix_cache.py \
    --model models/Qwen3.5-0.8B --json-out logs/validate/prefix_cache.json
PYTHONPATH=src:.runtime-deps python benchmarks/validate_prefix_cache.py \
    --model models/Qwen3.5-0.8B --graphs \
    --json-out logs/validate/prefix_cache_graphs.json
```

GPU 验证比较冷请求和缓存请求的生成 tokens、每一步完整 logits、所有层
GDN 状态及完整 attention KV，并在每次保存和恢复后检查逐元素一致。
五个场景为完全重复、共享前缀不同后缀、并发分支、快照淘汰、KV 淘汰。
并发场景额外检查共享快照内容没有被分支更新。

2026-09-27，在 RTX 3060 Ti、本地 Qwen3.5-0.8B、TP=1 上：

- 完整测试集 53 项全部通过，无跳过；其中 11 项为新增 prefix 回归。
- eager 和 CUDA Graph 两种模式的五个 GPU 场景均通过，生成 tokens 完全一致。
- 两种模式都完成 4 次恢复、8 次保存，复制逐元素一致。
- 复用场景跳过每请求 512 tokens；淘汰场景命中数为零并安全重算。

跨 batch 形状的 BF16 结果不要求逐元素一致。验证沿用已有的相对 RMSE
门槛：conv 5%、recurrent 2%、KV 5%，同时记录最大绝对误差；logits 使用
`rtol=0.03, atol=0.25`，生成 tokens 要求精确一致。冷/热单请求路径实际
无误差；并发路径的误差记录在 JSON 中。JSON 保存在不提交的
`logs/validate/` 目录。全量 tensor 捕获会同步 CUDA，只用于正确性验收。

CUDA Graph 与调度的后续优化见 [优化记录](cuda_graph_optimization.md)。

GDN 核心后续统一为 packed 卷积和 FP32 indexed decode；联合缓存的五个
GPU 场景已再次通过，详见 [GDN 内核优化](gdn_kernel_optimization.md)。
