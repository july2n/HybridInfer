# 公共投机链路与真实 MTP 实现

更新：2026-09-29。当前验收目标为数学语义、原 trial 端点选择与数值/质量控制。
当前结果见 [进度](speculative_decoding_progress.md)，测量见 [MTP 评估](mtp1_evaluation.md)。

## Target 与提交

所有请求统一走批量入口。Native 对 `[anchor,d1,...,dK_i]` 执行一次因果前向，
投影所有有效预测行。GDN 使用多词元 recurrent scan；recurrent 保存每个输入端点，
conv 保存初始历史加 trial raw tokens，接受后选择历史窗口。
输出 m 个 token 时，选择原 trial 零基端点 m-1，提交正式状态和 computed length，
保留 trial 已写入的有效 KV；correction/bonus 仍未计算。部分接受/EOS/预算截断
不重跑 target，端点缺失报错并恢复事务。

状态预算默认 256 MiB，调度按实际存储布局裁剪 K_i；资源不足回退普通 anchor。
请求在事务和独立输出句柄完成前保持 in_flight；异常恢复状态/历史/长度，释放试算资源。

## 设备接口与随机拒绝采样

`DeviceDraftProposal` 保存设备候选及可选实际 q；调度边界仍转 CPU tuple。
目标模型可按需导出 decoder 输入边界的 hidden+residual 与最终 normalized hidden。
共享 sampler 支持混合 greedy/random 请求、点质量与实际概率 q。
随机接受使用 min(1,p(d)/q(d))，首拒绝从归一化 max(p-q,0) 采样，全接受输出 target bonus。
接受、补偿和 bonus 使用独立的逻辑位置随机域。与普通路径或 vLLM 同 seed 的 token
相同不是随机验收要求；边际分布与实际条件 q 正确是要求。

## Qwen3.5 MTP

独立模型加载 checkpoint 全部 15 个张量，MLP 分片合并为 14 个参数，共享 target
embedding/head。前向为 embedding norm、hidden norm、[embedding,hidden] 拼接、FC、
full-attention decoder 和 final norm；使用独立 draft KV。
目标位置 t 的 feature 与确认 token t+1 配对，RoPE 位置为 t。
新一轮根据新确认 target feature 重算对应区间，覆盖此前的 draft hidden 条件。
无效 draft KV 尾部不进入后续注意力；结束或抢占时释放缓存所有权。

当前 proposer 按请求、按候选串行生成草稿。`mtp_draft_sampling="greedy"` 为默认模式，q 为点质量；`"random"` 按请求 temperature 对 draft logits 做 FP32 softmax，并使用独立 `draft` 随机域采样，返回实际条件分布 q。temperature=0 的请求在 random 模式下仍生成 greedy 候选及 one-hot q，支持同批混合 greedy/random target。

DraftContext 携带请求 temperature/seed；随机键由 seed、逻辑 token 位置与 domain 决定，相同已提交历史的重试和请求重排不会改变草稿随机流。未指定 seed 时使用进程初始 seed 与请求 ID；这是当前进程内复现规则。实际 q 为 `[候选总数, vocab_size]` FP32 张量，额外空间为候选数 × 词表大小 × 4 字节。random 草稿要求显式 `verification_mode="packed"`，与 n-gram 或 anchor-only 模式组合时配置报错。

## 配置与限制

```python
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.sampling_params import SamplingParams

engine = LLMEngine(
    'models/Qwen3.5-0.8B', enforce_eager=True, enable_prefix_cache=False,
    max_num_seqs=4, max_model_len=1024, gpu_memory_utilization=0.6,
    speculative=SpeculativeConfig(
        enabled=True, method='mtp', max_draft_tokens=1,
        verification_mode='packed',
    ),
)
try:
    outputs = engine.generate(['请解释投机解码'],
                              SamplingParams(temperature=0, max_tokens=64))
finally:
    engine.exit()
```

该示例显式启用 native 测量路径；K=1 是示例值，最佳 K 需按负载测量。
投机默认关闭，开启后默认 `packed_guarded`；它先试算 packed 再提交普通 batch
单步 anchor，候选接受数为零，不承诺加速。`sequential` 直接执行普通单步 anchor。

MTP 限于单 GPU、单个预测层、共享 embedding/head、eager、关闭 prefix cache，
不满足配置明确报错。批量 proposer、投机图执行、MTP prefix 协作尚未实现。
EAGLE-3 的初始线性适配见 [实现说明](eagle3_implementation.md)；P-EAGLE、DFlash/DFlash2/DSpark 尚未实现，不会静默退回 n-gram。

## 概率 MTP 使用示例

```python
speculative = SpeculativeConfig(
    enabled=True, method="mtp", max_draft_tokens=2,
    verification_mode="packed", mtp_draft_sampling="random",
)
params = SamplingParams(temperature=0.8, seed=42, max_tokens=64)
```

引擎仍需 `enforce_eager=True`、`enable_prefix_cache=False`。随机模式不要求与普通 target 同 seed 逐 token 一致，也不据此宣称通用质量无损。

## 验收与测量

`validate_spec_batch.py`、`validate_mtp.py` 的 passed 表示各自声明的协议/执行与
原 trial 端点范围；跨路径 token 匹配独立记录，不决定 passed。
数值预算由 `diagnose_target_numerics.py` 显式检查，质量由 `validate_mtp_quality.py`
检查。性能用无 intrusive 探针的 `bench_spec_decode.py`，baseline 不记录 MTP feature。
命令、预算与范围见 [验收指南](../benchmarks/README.md)。
