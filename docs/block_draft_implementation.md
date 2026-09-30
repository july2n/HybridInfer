# Qwen3.5-2B DFlash / DSpark 草稿适配

当前实现支持单卡 eager、文本输入、greedy 块草稿。目标模型为 `models/Qwen3.5-2B`。公开草稿权重固定在 `scripts/download_block_drafts.py` 中的两个 revision；运行下载器会核对发布方 LFS SHA-256 并写入各模型目录的 `download_manifest.json`。草稿不单独存 token embedding；DFlash 也复用 target LM head，DSpark 则加载 50,000 词表的独立 head、Markov head、偏移映射和 confidence head。

草稿层从 target 的指定边界提取特征，经 FC 和 RMSNorm 投影一次。DFlash 的 `target_layer_ids` 是输出层编号，因此转换为输入边界 `i+1`；DSpark 的 `aux_hidden_state_layer_ids` 已经是输入边界编号。每层从同一组投影特征计算 context K/V；块内 query 是最后一个已知 token 加 mask token；query 数量跟随本轮候选数量，而非总是填满配置的最大 block。DFlash 从 mask 位置生成候选（最多 `block_size-1`），全注意力层的块内 query 非因果；DSpark 可从 anchor 位置开始生成（该权重最多 `block_size`），sliding 层使用因果窗口。DSpark 的候选先经 Markov 偏置，再把 draft 词表 ID 映射到 target 词表 ID。confidence 概率目前只作诊断保存，尚未用于自适应验证长度。

草稿 context K/V 是私有缓存，只写入已提交 target 特征。验证、接受/拒绝、GDN/KV 状态恢复以及请求释放继续复用现有投机执行路径。当前没有为草稿接入 `torch.compile` 或 CUDA Graph，也没有概率草稿采样；温度大于零时仍用已有的点质量候选拒绝采样路径。当前不支持 prefix cache、张量并行、多模态输入或不同形状/布局的草稿权重。

## 下载与验证

以下命令使用 `vllm_env`，并假设仓库根目录为当前目录：

```bash
HF_HUB_DISABLE_XET=1 conda run --no-capture-output -n vllm_env \
  python scripts/download_block_drafts.py

PYTHONPATH=src:.runtime-deps conda run --no-capture-output -n vllm_env \
  python benchmarks/validate_mtp.py --model models/Qwen3.5-2B \
  --method dflash --draft-model models/Qwen3.5-2B-DFlash \
  --prompt-tokens 48 --output-tokens 12 --draft-tokens 3 \
  --batch-sizes 1 --modes baseline packed --gpu-memory-utilization .85 \
  --json-out logs/validate/dflash_2b_real.json

PYTHONPATH=src:.runtime-deps conda run --no-capture-output -n vllm_env \
  python benchmarks/validate_mtp.py --model models/Qwen3.5-2B \
  --method dspark --draft-model models/Qwen3.5-2B-DSpark \
  --prompt-tokens 48 --output-tokens 12 --draft-tokens 4 \
  --batch-sizes 1 --modes baseline packed packed_guarded \
  --gpu-memory-utilization .98 --state-snapshot-budget-mb 128 \
  --json-out logs/validate/dspark_2b_real.json
```

`validate_block_draft_forward.py` 可接收显式提供的 z-lab DFlash 参考源码，把真实权重骨干与参考模型在合成 target 特征上逐层计算后比较；它不会从模型目录执行远程 Python 文件。端到端脚本核对同配置 eager target 的输出、请求历史、原 trial 接受端点和资源生命周期。8 GiB RTX 3060 Ti 上 DSpark 额外权重较大，需要降低快照预算，并给 target KV 留出足够显存。数值/速度收益应另测，不能由协议验证推断。

## 当前验证记录

2026-09-30，RTX 3060 Ti 8 GiB、`vllm_env`：两份真实草稿 checkpoint 的骨干前向在两组不同 context/query 长度下与 z-lab DFlash 参考实现一致（BF16 最大绝对差 0）。Qwen3.5-2B target 的真实端到端 B=1 验证中，DFlash `packed` 执行 25 轮、70 个候选、接受 25 个，42 次原 trial 端点检查通过；DSpark `packed` 执行 25 轮、86 个候选、接受 28 个，44 次端点检查通过。两者在这组 B=1 自由生成测试中都与 eager target 输出一致。

DSpark B=2 测试执行 86 个请求轮次，127 次原 trial 端点检查通过，没有历史/状态错误；`repository_code` 的两个请求在第 11 个输出 token 与独立 eager 自由生成不同。该差异不改变端点契约的通过结果，但说明 B=2 尚不能声称逐 token 与 eager 一致。`packed_guarded` 在本配置下逐轮选择普通 anchor 回退，其数值检查没有报错；性能和质量仍待专门测量。
