# 投机解码当前进度

更新：2026-09-29。验收目标是数学语义、原 trial 端点选择和数值/质量控制。
跨浮点路径的 greedy 序列相同率仅作诊断。下一阶段建议集中于 MTP 性能测量和优化。

## 已实现

- Qwen3.5 Dense、单 GPU、线性候选、变长批次，B1/多请求统一验证入口。
- Native 一次多词元因果 target 前向，全部有效行投影；GPU 贪心/随机拒绝采样、
  correction/bonus、EOS/长度截断与正式历史/长度提交。
- 私有 GDN trial slot、recurrent 扫描与逐位置端点、conv 共享扩展历史。
  部分接受直接选择原 trial 状态并保留有效 KV，无 target 重跑。
- 独立异步输出句柄、完成事件、KV 容量预留/回收、异常事务恢复与资源不足回退。
- 设备草稿与目标特征接口；真实 MTP 权重、独立 draft KV、特征历史和结束/抢占释放。
  当前 MTP 逐请求、逐候选生成 greedy 草稿，随机 target 使用点质量 q。

投机默认关闭，开启后默认 `packed_guarded`：试算 packed 后提交普通 batch 形状的
单步 anchor，候选接受数为零，不承诺加速。显式 `packed` 执行多词元接受；
`sequential` 是普通单步参考。n-gram 为公共链路测试后端。

## 当前验证证据

| 最新清理后产物 | 结果与范围 |
|---|---|
| `logs/validate/src_cleanup_tests_final.log` | 124 项单元测试通过 |
| `logs/validate/src_cleanup_batch.json` | 变长真实 target 协议与原 trial 端点检查通过 |
| `logs/validate/src_cleanup_mtp.json` | MTP 执行通过；493 次端点检查零失败 |
| `logs/validate/src_cleanup_numerics.json` | repository_code rolling：平均 TV 0.010773、最大 TV 0.045765、翻转 1/128；本次显式预算通过 |
| `logs/validate/src_cleanup_quality.json` | 12 道算术题普通与 MTP 均正确 6 道 |

上述范围有限，不能推广到全部输入、K 或后端。
固定源码差分范围见 [vLLM 参考与后端契约](vllm_speculative_alignment.md)。
MTP-1 较完整评估有 1,892 次端点检查零失败，接受率 70.55%，每轮平均输出
1.706 token。5 类输入 B1/B4 的 25 请求中 17 个与普通 target 全序列相同。
同输入 640 query 平均 TV 0.008659、平均 KL 0.00061625 nats，argmax 翻转 5 次。
条件见 [MTP 评估](mtp1_evaluation.md)，不同产物不累加为统一验收数字。

## 性能证据与限制

公平 B1 eager/eager 对照中，MTP-1 普通耗时/MTP 耗时约 1.03–1.26。
条件为 prefix off、greedy、128 输出 token、预热 1 次、计时 3 次；五类输入均有
输出差异，仅表示实际路径耗时比。小幅收益尚需重复测量，普通 decode graph 未对照。
两类输入 K=2/4 探索已完成：英文 K=4 耗时比 1.276；low_match K=2 为 1.082、
K=4 为 1.040，后者接受率降至 28.4%。最佳 K 随负载变化。

性能入口目前只支持 B1；MTP 强制 eager、关闭 prefix cache。
批量 proposer、投机图执行、MTP prefix 协作、概率 MTP 草稿和自适应 K 尚未实现。
EAGLE-3、P-EAGLE、DFlash、DFlash2、DSpark 尚未实现。

## 下一步建议

补齐五类自然输入 K=1/2/4、B1/B4 扫描和耗时分解，建立普通 decode graph 对照；
依据测量优化批量 proposer、元数据/缓冲区及图执行，再实现候选长度选择和低收益回退。
B4 与图路径对照需先扩展测量入口。质量和复杂系统覆盖伴随推进。
任务顺序及交付标准见 [实施计划](speculative_decoding_plan.md)。
