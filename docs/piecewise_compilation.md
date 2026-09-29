# Compiler 与 nested CUDA Graph wrapper

默认图策略为 `FULL_AND_PIECEWISE`，编译默认关闭：

```python
engine = LLMEngine(
    'models/Qwen3.5-0.8B',
    enable_piecewise_compile=True,
    cudagraph_mode='FULL_AND_PIECEWISE',
)
```

## 组件与执行链

- `PiecewiseBackend`：通过 torch.compile/Inductor 编译模型的静态 pre/post
  片段，fullgraph=True、dynamic=True，关闭 Inductor 内部 CUDA Graph。
- `SplitGraph`：以模型已有的 attention/GDN 核心为分界，组织静态片段与动态
  核心；FULL 与 PIECEWISE 使用相同的静态 callable。
- 内层 `CUDAGraphWrapper(PIECEWISE)`：包装每层 pre/post callable，拥有
  片段图与输出；动态 attention/GDN 核心留在图外。
- 外层 `CUDAGraphWrapper(FULL)`：包装整个 SplitGraph 模型执行计划，拥有
  decode 完整图。LM head 与采样仍在外层图之外。
- `CUDAGraphDispatcher`：每个批次选择一个 runtime mode，通过 Context
  传递给两层 wrapper。ModelRunner 不再自行分支选择 full/piecewise 图。

```text
ModelRunner -> dispatcher -> FULL wrapper -> SplitGraph
                                           ├─ PIECEWISE wrapper -> compiled pre
                                           ├─ attention/GDN core
                                           └─ PIECEWISE wrapper -> compiled post
```

这里使用模型原生的显式分图边界，不是 vLLM 的整模型 Dynamo FX 图自动切分。
开启编译时，FULL 与 PIECEWISE 共用静态 callable；关闭编译时，FULL/NONE
直接调用原始模型，PIECEWISE 使用未编译的片段。
两层 wrapper 的运行语义遵循 vLLM nested-wrapper 设计；暂未引入此前数值
验证失败的整模型 Inductor 优化路径。

FULL 模式：外层捕获/重放；内层模式不匹配，直接执行 callable，不捕获或
重放片段图。因此完整图可以包含所有片段与动态核心，并且没有嵌套捕获冲突。
PIECEWISE 模式：外层直接执行执行计划，内层重放各片段图。
NONE 模式：两层均直接执行；启用编译时仍调用 compiled 片段。

预热由 manager 执行，捕获由 wrapper 执行。输入缓冲区准备、图 buckets 和
图池由 manager 协调。图、输出和编译 callable 在 clear/recapture 时清理。
GDN decode 抽出消费预投影结果的核心，与 native decode 共用状态更新 kernel。

## 图策略

支持 `NONE`、`FULL`、`PIECEWISE`、`FULL_AND_PIECEWISE`。

默认策略：

- 单 token 纯 decode 且命中精确请求数 bucket：FULL。
- prefill/mixed：PIECEWISE，按总 token 数选 bucket。
- decode 未命中完整图 bucket：可回退 PIECEWISE。
- 无合适图 bucket：NONE；不对 GDN 请求行做任意 padding。
- spec/draft 验证仍走现有独立路径，尚未接入图捕获。

统一解码需要请求都是 decode，query 长度一致；不能只检查 max_query_len。
当前完整图能力只覆盖单 token decode，尚不覆盖多 token 投机验证。
PIECEWISE 单独模式也允许普通 decode 走片段图。

`enforce_eager=True` 禁用 CUDA Graph；不再禁止显式开启 compiler。
`use_prefill_cudagraph=False` 禁用内层图，也不再禁止 compiler。
编译失败仍明确报错，不静默回退。GDN/full-attention pre 使用独立编译入口，
减少层类型切换耗尽 Dynamo 重编译额度的风险。

## 验证

vllm_env 下129项测试全部通过（包含实际GPU测试）。GPU 测试覆盖 mode 隔离、bucket miss、单 token prefill 分类、投机回退、
跨 bucket 重放、recapture、共享缓冲区和 feature 快照生命周期。
真实 Qwen3.5 前缀缓存验证覆盖开启/关闭编译下的保存/恢复、并发分叉和淘汰，
两种设置均通过。FULL_AND_PIECEWISE、PIECEWISE、NONE+编译的同状态
对照均完成，当前小样本没有 argmax 翻转，概率平均TV最大约0.015。
原始产物位于 logs/compile/nested_*.json（默认被Git忽略）。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python benchmarks/validate_prefix_cache.py --graphs --compile-segments
PYTHONPATH=src python benchmarks/validate_nested_graphs.py --compile-segments
PYTHONPATH=src python benchmarks/validate_nested_graphs.py --compile-segments --mode PIECEWISE
PYTHONPATH=src python benchmarks/validate_nested_graphs.py --compile-segments --mode NONE
```

同输入同 KV/GDN 初态对照为局部数值诊断，不作为通用质量验收。静态融合及
padding 的 GEMM 形状变化仍可能导致 BF16 偏差；compile 默认关闭。这次重构
没有测量端到端性能，也没有声明生成质量预算。

参考：https://docs.vllm.ai/en/latest/design/cuda_graphs/#nested-wrapper-design
