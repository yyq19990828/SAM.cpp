# 历史 v3 图像验收流程

仅作历史归档；当前应用 benchmark 不要求 campaign、质量档位或质量通过凭据。

## v3 图像验收

ranked precision 工具支持显式 v3 campaign，与原有图像／视频验证器独立。
`export_precision_outputs.py` 和 `validate_precision_regression.py run`
传入 `--policy-version 3 --quality-tier balanced`，也可预先选择 `high-fidelity`
或 `compact`。评估器从运行记录读取固定策略，拒绝 v2／v3 混比或档位身份改变。
数据集、标准化输入和 ranked 张量载荷仍为版本 2；v3 配置、campaign 和判定使用
独立版本与门槛哈希。

质量报告给出 `task_quality_status`、独立的绝对／缓存增量结果和 `diagnostics`。
缺少完整、独立的算术与回归证据时，`qualification_status` 保持 `NOT_RUN`；
已知质量失败则为 `FAIL`。质量通过本身不认证整个模型。性能报告按工作负载分开
给出 `measurement_status`、`non_regression_status` 和 `benefit_status`。
`NOT_DEMONSTRATED` 表示未获得收益标签，不表示任务质量失败。

最终推理前须冻结新 campaign，在 `evaluation_history` 中声明全部历史 precision
数据集及 SHA-256。冻结器排除已使用的图像 ID／内容，也不会开启旧 reserve。
历史完整性由实验者声明，工具无法自动发现清单外的实验。旧评估图像只用于开发，
原始报告继续保留。除最终图像数量要求外，开发质量检查必须通过。
v3 只做质量验收时可省略性能样例／基线；压缩缓存仍须指定完全匹配的 `cache_baseline`。

完整标准、selection 文件示例、命令及本次验证范围见[质量档位](quantization-qualification_zh.md#v3-质量档位与可选收益)
和 [v3 计划](../plans/20261010-101413-precision-acceptance-v3.md)。Linux precision
工具目前接入 CPU／CUDA，不代表新增 Metal 接入或视频验收。质量预算不能用来豁免
实现层的算术错误。
