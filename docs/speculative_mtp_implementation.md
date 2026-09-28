# 公共投机链路与真实 MTP 实施（2026-09-28）

本次执行公共 target 验证、设备草稿/特征接口与共享拒绝采样、真实 MTP 三项工作。最终目标仍是 EAGLE-3、P-EAGLE、DFlash、DFlash2、DSpark、MTP；n-gram 作为协议测试后端保留。实现已经可运行，完整生成的严格一致性门槛仍未通过，不能将本次交付表述为六种方法全部适配或整模型已与 vLLM 对齐。

## 实现

1. **原验证端点恢复。** `spec_decode` 的 GDN 改用多词元 recurrent Triton kernel，保留每个输入后的 conv/recurrent 状态。部分接受、EOS 或输出预算截断均选择 `cu_q[row] + actual_output_length - 1`，最后输出保持为未计算 anchor。保留原 trial 已写入的有效 target KV，不重跑较短的 target。单请求和批量 native 路径均支持端点选择；仅兼容旧测试 stub 的路径保留重放。
2. **设备接口与共享拒绝采样。** `DeviceDraftProposal` 持有设备候选及实际草稿概率 q；CPU 调度边界仍将候选转成 tuple，这不是全设备调度。模型可导出指定 decoder 输入边界的 `hidden + residual` 和最终 normalized hidden。共享 sampler 支持混合 greedy/random 请求、点质量草稿及实际概率草稿，首拒绝补偿分布为归一化 `max(p-q, 0)`，全接受产生 target bonus，再统一处理停止条件和提交端点。随机接受、补偿和 bonus 使用独立、按逻辑位置确定的随机域；与 vLLM 的 RNG 实现不同，不保证同 seed 的随机 token 完全一致。
3. **真实 Qwen3.5 MTP。** 独立模块完整加载本地 checkpoint 的 15 个张量（MLP 两分片合并后 14 个参数），共享 target embedding/LM head；按 vLLM 的 embedding norm、hidden norm、`[embedding, hidden]` 拼接、FC、full-attention decoder、final norm 顺序运行。使用独立草稿 KV，目标位置 t 的 feature 与确认 token t+1 配对，RoPE 位置保持 t（与 vLLM 一致，不额外加一）；新一轮重算新确认的 target feature 区间，丢弃上轮草稿 hidden 对该区间的替代。超出有效长度的草稿 KV 尾部不进入后续注意力。请求结束或抢占时释放草稿缓存所有权。

状态快照预算默认 256 MiB，KV 容量计算预留该预算，调度同时限制实际验证输入数；因此 `max_draft_tokens` 是上限，内存受限时各行 K 可以不同。运行中真正发生 OOM 时保留 ordinary anchor 回退，随机请求的回退也使用随机 sampler。

## vLLM 源码差分与完整模型对比

源码参考固定为 `a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`，差分脚本检查 revision、源文件未修改及 SHA256，并执行原定义。完整 vLLM 模型基线是环境安装的 **0.19.0**，不是该固定提交的构建。

- `validate_vllm_gdn_recurrent.py`：直接执行原多词元 recurrent kernel，核对输出及所有输入状态端点，包括不同 GQA 布局和之前接受数驱动的 initial-state 索引。对齐融合乘加、BV=32、warp/stage 设置后要求按位一致。
- `validate_vllm_spec_conv.py`：直接执行原投机卷积，检查 BF16/FP16 输出、rolling history 与各接受端点窗口。vLLM 的低精度乘法、FP32 累加和 SiLU 顺序与原 ordinary 卷积不同；新 spec 分支显式使用 vLLM 顺序，普通执行保持既有行为。
- `validate_vllm_random_rejection.py`：200 批、1000 请求，与原随机接受 kernel 逐项一致，覆盖概率/点质量 q、零概率、阈值相等、零草稿和变长请求。输入 uniform、recovered、bonus 相同；这不检验两者随机数生成器是否相同。单元测试另以实际采样函数检查两类 q 的目标边际分布。
- `validate_mtp.py`：真实权重运行、原 MTP forward 顺序检查、原 trial 状态端点检查、槽位释放、随机输出长度，以及同 prompt IDs 的普通 HybridInfer/vLLM 生成对比。原 MTP forward 对比复用本项目算子，只证明调用顺序；完整 vLLM 比对独立承担整模型门槛。

最新完整结果保存在 `logs/validate/mtp_alignment_final_current.json`，详细差异位置在其 `cases` 字段。端点和执行通过不能覆盖 `hybrid_token_passed` 或 `vllm_token_passed` 的失败。首次逐层浮点分歧仍需定位；不能仅以近似数值容差消除逐 token 严格失败。


## 最终复测结果

123 项 CPU/CUDA 单元测试全部通过；固定 vLLM 元数据 CPU/GPU 各 104 布局、新旧 greedy kernel 112 批/931 请求重新对比通过；GDN recurrent 72 布局/1386 端点、投机卷积 BF16/FP16 8 布局/140 端点均按位一致；随机接受原 kernel 200 批/1000 请求通过。

真实 MTP 使用 5 类自然输入、batch=1/4、每请求 128 输出词元，并重复生成以检查槽位释放。最新原 trial 检查共 1523 个请求端点，零失败；packed 的 25 个请求提出 4124 个草稿词元、接受 1672 个（约 40.5%），状态重跑和显存回退均为 0。另两个随机请求各输出 32 个词元。下面的通过数要求整个 128-token 序列逐 token 相同：

| 路径 | 对普通 HybridInfer | 对安装版 vLLM 0.19.0 普通基线 |
|---|---:|---:|
| baseline | 25/25 | 7/25 |
| packed | 5/25 | 6/25 |
| packed_guarded | 25/25 | 7/25 |

补充最终代码回归：控制接受位置/EOS/上下文截断/前缀复用的 36 场景、84 个原 trial 端点全部通过；native n-gram batch=3/8、decode CUDA Graph、509-token 长前缀与 prefix 开关的 4 场景、51 批/291 请求端点恢复全部通过，但普通基线 token/state 对比仍产生 6 条严格失败，`spec_batch_final_current.json` 保留失败，整体退出码为 1。

因此 `completed=true`、`execution_passed=true`，但 `hybrid_token_passed=false`、`vllm_token_passed=false`、`passed=false`，验收脚本退出码 1。保守路径与本项目普通解码保持一致；不能将 native 提升为默认，也不能宣称整模型数值路径已全部对齐。接受率不是加速比，本次诊断带同步检查，不作为性能基准。

## 使用与限制

```python
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.sampling_params import SamplingParams

engine = LLMEngine(
    'models/Qwen3.5-0.8B', enforce_eager=True, enable_prefix_cache=False,
    max_num_seqs=4, max_model_len=1024, gpu_memory_utilization=0.6,
    speculative=SpeculativeConfig(enabled=True, method='mtp', max_draft_tokens=4),
)
try:
    output = engine.generate(['请解释投机解码'], SamplingParams(temperature=0, max_tokens=64))
finally:
    engine.exit()
```

默认 `packed_guarded` 用 ordinary batched anchor 保持本项目普通 decode 行为，不提交多词元 MTP 草稿，也不承诺加速。显式设 `verification_mode='packed'` 才使用 native 多接受及共享随机拒绝采样；该路径尚未通过严格生成一致性门槛。真实 MTP proposer 当前生成 greedy 草稿，随机 target 将其 q 当作点质量，不能声称已实现概率 MTP 草稿。

初版 MTP 限于单 GPU、单个 MTP 层、共享 embedding/head、eager 和关闭前缀缓存；不满足条件明确报错。EAGLE-3/P-EAGLE、DFlash/DFlash2/DSpark 模型尚未实现，配置不会静默退回 n-gram。下一步先定位普通 target 与 vLLM、native 与普通 target 的首次数值分歧，再接入其他真实草稿模型、图执行、前缀缓存和批量 proposer 优化。

复测入口：

```bash
PYTHONPATH=src:.runtime-deps python -m unittest discover -s tests -v
PYTHONPATH=src:.runtime-deps python benchmarks/validate_vllm_spec_alignment.py
PYTHONPATH=src:.runtime-deps python benchmarks/validate_vllm_gdn_recurrent.py
PYTHONPATH=src:.runtime-deps python benchmarks/validate_vllm_spec_conv.py
PYTHONPATH=src:.runtime-deps python benchmarks/validate_vllm_random_rejection.py
PYTHONPATH=src:.runtime-deps python benchmarks/validate_mtp.py
```

`validate_mtp.py` 对任何严格 token 差异返回非零退出码；不存在仅打印失败而返回成功的验收路径。
