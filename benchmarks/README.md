# 投机采样验收与测量

当前目标：数学语义一致、正确选择 speculative endpoint、控制数值误差。
不同浮点执行路径的 bit-for-bit 或 greedy 自由生成一致性仅作诊断。
使用 CUDA 环境，以下命令设置 `PYTHONPATH=src:.runtime-deps`。

## 入口与职责

| 入口 | 验收范围 | 失败条件 |
|---|---|---|
| `python -m unittest discover -s tests -v` | 行映射、接受/补偿算法、EOS/边界、事务、端点、缓存生命周期、kernel 数学检查 | 算法/状态错误或局部已声明数值契约违约 |
| `validate_spec_batch.py` | 变长真实 target、prefix 冷/热、长前缀、GPU 历史与原 trial 端点 | 历史/长度/端点错误、非有限状态或没有覆盖验证路径 |
| `validate_mtp.py` | 真实 MTP 权重、draft 生命周期、B1/B4 生成、原 trial 端点、随机执行 smoke | 执行/长度/历史/端点错误；跨路径 token 匹配不决定 passed |
| `diagnose_target_numerics.py` | 同输入同初始状态的 reset/rolling 数值、层与状态偏差、概率 TV/KL、near-tie | 非有限结果或显式数值预算超限 |
| `validate_mtp_quality.py` | 12 个固定算术任务，普通 target 与 MTP-1 正确率比较 | 正确答案数量下降超过显式质量预算 |
| `bench_spec_decode.py` | 真实 n-gram/MTP 候选、预热、配对轮换计时、接受率、内存与回退 | 执行或输出长度错误；耗时比与输出差异分别报告 |
| `validate_vllm_spec_alignment.py` / `validate_vllm_random_rejection.py` | 固定源码的元数据与同概率接受算法 | 数学/索引/拒绝采样契约不一致 |
| `validate_vllm_gdn_recurrent.py` / `validate_vllm_spec_conv.py` / `validate_target_norm.py` | 固定参考 kernel 的共同操作数检查 | 超过该检查声明的容差；conv 历史复制与 norm 固定布局不变性仍精确检查 |
| `validate_vllm_model_reference.py` | 可选完整 vLLM 生成参考 | 引擎执行/配置/输入/长度错误；跨路径 token 差异仅记录 |

`spec_validation.py` 共享端点检查及数值汇总。端点 oracle 来自**同一次原始 trial**，
不重新运行 target 或用缩短后的 replay 代替。`passed` 只代表脚本声明的验收范围，
不能用执行通过代替数值预算或质量验收。

源码统一使用批量验证入口（B=1 也走同一路径）。Native 缺失原 trial 端点时
报错并恢复事务；`sequential`/`packed_guarded` 提交普通单步 anchor。
Conv 窗口 oracle 使用独立 Torch 公式。

## 常用命令

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v

PYTHONPATH=src:.runtime-deps python benchmarks/validate_spec_batch.py \
  --modes packed packed_guarded --batch-sizes 3 4 --long-prefix

PYTHONPATH=src:.runtime-deps python benchmarks/validate_mtp.py \
  --draft-tokens 1 --modes baseline packed --batch-sizes 1 4

PYTHONPATH=src:.runtime-deps python benchmarks/diagnose_target_numerics.py \
  --case repository_code --block 2 --tokens 128 --variants native \
  --state-modes reset rolling --max-mean-tv 0.02 --max-tv 0.1 --max-flip-rate 0.02

PYTHONPATH=src:.runtime-deps python benchmarks/validate_mtp_quality.py --max-correct-drop 0

PYTHONPATH=src:.runtime-deps python benchmarks/bench_spec_decode.py \
  --method mtp --draft-sweep 1 2 4 --suite natural --enforce-eager \
  --warmups 1 --repeats 3
```

上述数值预算是探索性命令示例，不是所有模型/输入的统一门槛；须在运行前确定
适用范围。未提供预算时，数值脚本报告 `budget_checked=false`、`budget_passed=null`，
不能宣称误差已通过验收。Softmax TV/KL 按 temperature=1 计算；teacher forcing
翻转率和自由生成首处分歧是不同指标。诊断脚本两边 LM head 都逐行投影，
生产 packed head 的差异由自由生成/质量脚本另行覆盖。

MTP 验证与数值诊断默认从本地模型及公共自然输入生成 fixtures，不依赖已有日志。
可显式传 `--vllm-baseline` 或 `--fixtures` 使用已有固定输入；
`--check-vllm-forward` 可额外核对本地固定源码的 MTP 操作顺序。
公共接受算法的固定 vLLM 源码对照仍依赖对应 checkout。

性能仅计时，不运行 intrusive 端点/数值探针；baseline 不记录 MTP feature。
输出不同仍提供 `measured_decode_time_ratio`，不能把它当成相同轨迹的加速证明。
小样本质量 smoke 不能证明通用质量无损。

性能入口当前只支持 B1，MTP 要求 eager、关闭 prefix cache；B4 和普通 decode graph
对照需扩展测量入口。下一步任务见 [实施计划](../docs/speculative_decoding_plan.md)。

## 分段编译

正式配置与数值限制见 [分段编译说明](../docs/piecewise_compilation.md)。

```bash
PYTHONPATH=src python benchmarks/validate_prefix_cache.py --graphs --compile-segments
PYTHONPATH=src python benchmarks/experiment_torch_compile.py
```

前者验证正式路径的前缀缓存；后者测量静态片段的编译兼容性、数值偏差和
局部性能。局部加速不代表端到端收益。实验及历史完整 forward 结果见
[片段实验](../docs/torch_compile_experiment.md)和
[完整 forward 实验](../docs/torch_compile_full_forward_experiment.md)。

## Nested CUDA Graph wrapper

默认 `cudagraph_mode=FULL_AND_PIECEWISE`：单 token decode 优先外层 FULL，
prefill/mixed 使用内层 PIECEWISE。编译与图模式独立配置，详见
[执行链与限制](../docs/piecewise_compilation.md)。

```bash
PYTHONPATH=src python benchmarks/validate_nested_graphs.py --compile-segments
PYTHONPATH=src python benchmarks/validate_nested_graphs.py --compile-segments --mode NONE
```

该脚本在同一输入及 KV/GDN 初态下比较原始 eager 和选定图路径的 logits、
概率与状态，并沿 eager 历史推进；仅作局部数值诊断。
