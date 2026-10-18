# PeMS08 历史实验记录

来源：同级 `ISY5002-Traffic-Recognition` 的 `flow_10epoch_batch8_seed0`。

整个实验被用户中止，`run_config.json` 保留 `stopped_by_user`。FNN、GRU、LSTM、MTGNN 已完成完整测试，3 个朴素基线也有指标；GWNET 只训练了 5 轮，AGCRN/STGCN 尚未开始该次正式训练。

`comparison.csv` 是旧实验原表。`reference_comparison.csv` 是与新仓库 flow/seed0 结果的分组并列表，**协议不同，不能合并排名**。原始权重、预测数组和数据文件没有复制到这里，仍保留在旧项目本地。

数据来源对应同一 PeMS08 基准，但新仓库历史训练数据未附带、未记录哈希，因此未验证逐值一致性。详见 [对比与优化建议](../../../docs/PEMS08_IMPORTED_COMPARISON.md)。

`source_snapshot/` 只用于查明旧实验实际执行了哪些参数；文件的目录依赖仍是旧工程结构，不能直接作为新仓库训练入口运行。
