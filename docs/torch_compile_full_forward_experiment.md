# 完整 forward 编译实验

日期：2026-09-29。环境：vllm_env，PyTorch 2.10.0+cu128，RTX 3060 Ti，
Qwen3.5-0.8B 真实权重。生产代码未修改。

范围是完整 backbone + runner 所选位置的 LM head 投影；包含 attention、GDN、
KV 写入和 GDN 状态更新。调度、输入准备、采样、缓存分配与快照管理仍在 Python。
不覆盖投机解码、TP>1、CUDA Graph 或完整生成质量/性能验收。
每个案例独立 reset Dynamo，fullgraph=True、dynamic=False；Inductor 内部
CUDA Graph 关闭。计算前复制已分配 KV 页及所有 GDN 状态，用同一状态执行
原始和编译路径，记录首次及恢复输入状态后的第二次调用。之后恢复原始执行
完成后的状态，确保后续案例沿原始轨迹推进。

## 捕获断点与实验适配

未经包装时，首次/分块 prefill 和混合批次可由 Dynamo 完整捕获。
decode 在 FlashAttention 的 pybind `fwd_kvcache` 扩展处报 Unsupported。
实验脚本以带 fake 实现、mutates_args=() 的 custom op 包装只读 decode attention
调用；KV 写入仍由前置 store_kvcache 执行。没有用 compiler.disable 掩盖断点。

包含 runner.compute_logits 的直接 Inductor 实验还触发 LM head 处的内部
weakref/storage 错误，随后 dispatcher 状态异常。改为同一个 tied Parameter
并未解决。最终使用 no_grad、标准 backend 和等价纯 tensor logits 投影（先选
prefill 的末行，再 F.linear），避免在图中进入 compute_logits 的 inference_mode
装饰器，编译可以继续。由于同时调整了多项条件，尚未证明错误的单一根因。

## 最终对照

|批次|query长度|Dynamo eager backend|Inductor backend|
|---|---|---|---|
|首次 prefill|128|单图，logits/KV/GDN逐位一致|执行成功，概率TV 0.000323|
|分块 prefill|72|单图，logits/KV/GDN逐位一致|执行成功，但概率TV 0.771809，argmax翻转|
|decode|1|单图，logits/KV/GDN逐位一致|执行成功，但概率TV 0.296724|
|混合|16,1|单图，logits/KV/GDN逐位一致|编译形成单图，但数值检查 AssertionError|

Dynamo eager backend 每项 unique_graphs=1；捕获算子数 prefill/mixed 为2118，
decode 为1517。Inductor 的前三项也为单图。TV 在 temperature=1 下计算，
每项仅1或2行，不能当作通用质量评估。status=ok 只表示编译/执行及基本有限值
检查完成，不表示数值质量通过。

结论：当前可以完整捕获（decode 需算子包装），但完整 Inductor 路径不能启用。
分块 prefill/decode 的巨大偏差在恢复状态后的第二次调用仍存在，不是仅首次
调用的编译预热现象。Dynamo eager backend 保持状态和输出一致，问题范围可
收窄至 Inductor 优化执行及缓存/递归状态副作用处理，尚未定位具体算子原因。
因此没有测量或宣传完整 Inductor 的性能收益。

后续应先隔离 GDN 状态更新与 paged attention 的读写顺序/alias，逐层对照，
必要时把这些核心操作注册为显式声明输入与 mutation 的 custom op；解决后再
验证跨批次 guards、重编译和端到端质量。

本轮完整 forward 临时脚本未纳入正式代码；以下保留历史结果与限制。

原始结果保存在 logs/compile（默认被 Git 忽略）。完整 backbone 的隔离实验
另见 full_model_inductor.json；该实验同样存在大量状态和输出偏差。
