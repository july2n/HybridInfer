# torch.compile 分段实验

环境：vllm_env，PyTorch 2.10.0+cu128，NVIDIA GeForce RTX 3060 Ti，Qwen3.5-0.8B 真实权重。

生产执行路径未修改。实验使用随机 hidden states（seed=42），选择第0层 GDN
及第3层 full attention，覆盖 pre（有/无 residual）、post 和 MRoPE。
所有实验使用 fullgraph=True、dynamic=False，并关闭 Inductor 内部 CUDA Graph。
每个 case 重置 Dynamo，统计 backend 调用次数；编译后再分别捕获手动 CUDA Graph。
测试范围不包含 attention/GDN 动态核心、TP>1、端到端生成和跨形状重编译。

21/21 case 完整编译且可捕获 CUDA Graph，每项 backend 调用一次。
pre 输出逐位相同；post 的 relative RMSE 约0.0034–0.0037；MRoPE 最大
relative RMSE 约0.0012。这只是数值诊断，没有据此宣布端到端质量验收通过。
首次调用时间包含 tracing/编译，但缓存未清空，不能视为冷编译时间。

## CUDA Graph 下的局部性能

表中为 eager graph 时间 / compiled graph 时间，大于1表示编译更快。
5轮、每轮30次重放，取中位数。只有片段执行，不包含分段输入输出复制、
完整图分发和缓存管理开销；不可作为端到端加速比。跨轮 GPU 频率变化
明显，性能结论采用同一轮对照。

|组件|1 token|128 tokens|512 tokens|
|---|---:|---:|---:|
|linear_attention.pre.residual|1.00x|1.00x|1.01x|
|linear_attention.pre.no_residual|1.00x|1.00x|1.00x|
|linear_attention.post|1.02x|1.07x|1.07x|
|full_attention.pre.residual|1.00x|1.00x|1.00x|
|full_attention.pre.no_residual|1.00x|1.00x|1.00x|
|full_attention.post|1.03x|1.07x|1.08x|
|mrope|4.58x|5.32x|3.20x|

pre 片段主要由已有 Triton norm 和 GEMM 组成，本次编译基本没有额外收益。
post 局部约1.02–1.08x；MRoPE 局部约3.20–5.32x，是后续接入的优先候选，
但存在中间 BF16 舍入差异，需端到端概率和状态对照后决定是否启用。

复现：

```bash
PYTHONPATH=src /home/lang/anaconda3/envs/vllm_env/bin/python benchmarks/experiment_torch_compile.py
```

原始产物（logs 默认被 git 忽略）：
`logs/compile/segments.json`、`logs/compile/segments_with_graphs.json`。
