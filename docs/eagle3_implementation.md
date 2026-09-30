# EAGLE-3 初始适配

2026-09-29。已接入 Llama EAGLE-3 草稿架构与现有投机验证链路，初始范围是
Qwen3.5 Dense target、单 GPU、eager、prefix cache 关闭、线性候选。
本地尚无与 Qwen3.5-0.8B 配套训练的 EAGLE-3 checkpoint；当前验证使用明确标记的随机权重夹具，不能用于质量或性能结论。

## 配置

```python
SpeculativeConfig(
    enabled=True,
    method="eagle3",
    draft_model="/path/to/matching-eagle3-checkpoint",
    max_draft_tokens=2,
    verification_mode="packed",
)
```

引擎设置 `enforce_eager=True`、`enable_prefix_cache=False`。默认草稿为 greedy，
target 可为 greedy 或 random；随机 target 使用点质量 q 的共享拒绝采样。
当前 `mtp_draft_sampling="random"` 仅适用于 MTP，不用于 EAGLE-3。
投机验证默认仍为 `packed_guarded`，要执行实际多 token 接受需显式设置 `packed`。

## 模型和特征契约

- 从 draft `config.json` 的 `eagle_config.eagle_aux_hidden_state_layer_ids` 读取特征层序；未提供时使用 vLLM Qwen3.5 的默认 `(2, L//2, L-3)`。可显式配置 `eagle3_feature_layers`，但不能覆盖 checkpoint 已声明的层序。
- 层号表示 decoder 输入边界的 hidden+residual，与 HF hidden-state tuple 的边界对应；不是随意选择三层 final-norm 特征。
- 特征按 checkpoint 顺序拼接，可选整体 `input_norm` / 各段 `fc_norm`，再经 FC 投影到 draft hidden size。正式位置保存投影特征，后续确认历史覆盖此前 trial 特征。
- 草稿首层分别归一化 token embedding 与 feature，拼接后输入双宽 Q/K/V 投影；后续层为普通单宽 attention+MLP。支持标准 RMSNorm、SiLU、GQA、unscaled full/partial RoPE，以及 `norm_before_residual` / `norm_output` 反馈选项。
- 每请求、每 draft 层有独立 paged KV。位置 t 的 target feature 与 token t+1 配对，draft RoPE 位置为 t；递归阶段使用 draft 自身的反馈特征。新一轮覆盖未确认尾部，结束和抢占释放所有权。
- 独立 lm_head 支持裁剪词表；checkpoint `d2t` 保存的是 offset，实际 target ID 为 `arange(draft_vocab_size)+d2t`。校验映射范围与唯一性，映射以外 logits 为负无穷。

## 权重格式与范围

初始 loader 支持本地 `config.json` 与单文件/分片 safetensors，支持 `midlayer.*` 或
`layers.0.*` 命名、可选 `model.` 前缀、独立 q/k/v 与 gate/up 投影。
必需权重完整检查，不允许用随机参数补缺失权重。若缺少 embedding 且形状兼容，显式共享 target embedding；lm_head 必须提供。
缺失/重复/多余权重、形状不符、非法映射、特征层序冲突均报错。
hidden size 和 vocab size 的形状兼容只是必要条件；使用者仍需选择为该 target 与 tokenizer 配套训练的权重。

尚未支持树候选、P-EAGLE 并行草稿、投机 CUDA Graph、批量 proposer、量化 draft、
scaled RoPE、滑动窗口、非标准归一化布局，以及 `.bin` checkpoint。
不同 EAGLE-3 checkpoint 结构需要按明确契约扩展，不会静默退回 MTP/n-gram。

## 验证

```bash
PYTHONPATH=src conda run --no-capture-output -n vllm_env python benchmarks/validate_mtp.py \
  --method eagle3 --draft-model /path/to/matching-eagle3-checkpoint \
  --modes packed packed_guarded --draft-tokens 2 \
  --prompt-tokens 64 --output-tokens 16 --batch-sizes 1 4 \
  --json-out logs/validate/eagle3_protocol.json
```

无训练权重时的协议夹具生成入口为 `benchmarks/make_eagle3_test_fixture.py`，输出路径默认带 `UNTRAINED` 标记。
136 项单元/GPU 测试通过，覆盖权重、词表、特征顺序、残差/反馈归一化、partial RoPE 的整段/分块/decode 私有 KV 一致性。
真实 Qwen3.5 target + 未训练草稿的 5 类输入、B1/B4、packed/guarded 与混合 greedy/random 请求已完成：410 次原 trial 端点检查零失败，输出长度与请求释放通过。
该夹具的候选接受数为 0，主要覆盖候选拒绝路径，不能证明训练后接受率、质量或加速收益。产物：`logs/validate/eagle3_untrained_protocol.json`。

参考：[vLLM Llama EAGLE-3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama_eagle3.py)、[EAGLE 官方模型](https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py)。

共享生命周期改动后的真实 MTP 回归通过，263 次端点检查零失败，产物为 `logs/validate/eagle3_mtp_regression.json`。
