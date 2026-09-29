# 正式分段编译配置

默认关闭，通过 `enable_piecewise_compile=True` 开启：

```python
engine = LLMEngine(
    'models/Qwen3.5-0.8B',
    enable_piecewise_compile=True,
    enforce_eager=False,
    use_prefill_cudagraph=True,
)
```

只编译现有 prefill CUDA Graph 的 `forward_piecewise_pre` 和 `forward_output`。
前者包含输入 Norm/投影，后者包含 residual/Norm/MLP。Attention/GDN 核心仍
在图外执行，普通 decode 图、LM head、采样和 KV/GDN 生命周期管理不变。
混合批次沿 prefill 分段图路径执行。超出图 bucket 的批次使用现有 eager 回退。

`CudaGraphManager` 缓存每层 pre/post 的编译 callable，在图捕获前预热。
使用 Inductor、fullgraph=True、dynamic=True，并关闭 Inductor 内部 CUDA Graph；
动态 token 维度允许多个图 bucket 复用编译代码。编译失败会报错，不静默
退回 eager。清理/重新捕获时同现有图和 callable 一起释放引用。

开启时必须启用 prefill CUDA Graph：`enforce_eager=True` 或
`use_prefill_cudagraph=False` 会被拒绝。该配置不启用完整模型编译或投机图。

## 验证

vllm_env 下124项单元测试通过，27项因环境/硬件条件跳过。前缀缓存验证
通过，覆盖128/256 token buckets、重复前缀、不同后缀、并发分叉及淘汰回退。

```bash
PYTHONPATH=src python benchmarks/validate_prefix_cache.py --graphs --compile-segments --json-out logs/compile/prefix_piecewise_compiled.json
```

此前临时实验的同输入同 KV/GDN 初态对照（结果在
`logs/compile/production_piecewise_eager_comparison.json`）：

|批次|概率平均TV|argmax翻转|
|---|---:|---:|
|首次prefill|0.000457|0/1|
|分块prefill|0.012478|0/1|
|decode|0|0/1|
|混合|0.010351|0/2|

这是少量批次的数值诊断，包含分段图 padding 引起的 GEMM 形状差异，不能
单独归因于 compiler，也不能证明通用生成质量无损。编译融合可以改变BF16
中间舍入，因此配置保持默认关闭。此轮未测正式执行路径的端到端性能。
