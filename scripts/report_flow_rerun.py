"""Write a progress/final report from recorded results; --watch waits for the queue."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def report(tag):
    folder = ROOT / "results/reruns" / tag
    state = json.loads((folder / "status.json").read_text())
    records = []
    for model, status in state["models"].items():
        if status == "completed":
            run = ROOT / "results/forecast/PEMSD8/traffic_flow" / model / f"seed0_{tag}"
            record = json.loads((run / "run.json").read_text())
            records.append((run, record))
    lines = ["# PeMSD8 流量实验：各模型默认配置复现（2026-10-13）", "",
             f"状态：**{state['status']}**。已完成 {len(records)}/{len(state['models'])} 个模型。", "",
             "## 实验设置", "",
             "使用新仓库锁定的 LibCity 版本 `5a6391d41944e937f2c15e9be85ab7f40ac8b23e`，"
             "交通流量单目标，170 个节点，5 分钟采样，12 步输入预测 12 步。"
             "窗口构造后按 7:1:2 划分，batch 64，最多 100 轮，早停耐心 30，seed 0。"
             f"本轮设备：`{state['device']}`；原参考实验设备为 CUDA。", "",
             "各模型保留其 LibCity 默认输入特征、优化器、学习率和调度器；"
             "MultiHeadSTGCN 使用新仓库的定型配置，MTGNN 使用 step_size1=627。"
             "这是按原仓库公共协议复现的对照，不是所有模型输入/优化器完全相同的结构消融。", "",
             "数据来自已校验的 ASTGCN PeMS08 原始文件，转换为 PEMSD8；流量的 float32 转换无精度损失。"
             "原子格式全量数值核对与 FNN 测试目标时间对齐已通过，见 `dataset_verification.json`。"
             "历史参考实验没有原始数据文件/哈希，因此仍不能证明与其数据逐值一致。", "",
             "## 对比范围与下一组受控实验", "",
             "本轮比较的是各模型默认配置的整体表现。输入、优化器、训练损失和多步训练策略不同，"
             "不能把误差差异全部归因于网络结构。", "",
             "| 模型 | 实际输入 |", "|---|---|",
             "| FNN / STGCN / GWNET / AGCRN | 流量，1 通道 |",
             "| GRU / LSTM / MTGNN | 流量 + 时刻，2 通道 |",
             "| MultiHeadSTGCN | 流量 + 时刻 + 星期，9 通道 |", "",
             "建议补充一张统一流量输入的对照表：保持本轮数据、划分、窗口、batch、选模与指标，"
             "关闭所有外部/日历特征。现有 1 通道模型的结果可在数据指纹和协议一致时复用，"
             "只需补跑 GRU、LSTM、MTGNN、MultiHeadSTGCN 四项。"
             "再与本轮同模型做配对比较，衡量时间特征的收益。", "",
             "统一输入只控制输入信息。若进一步声称结构贡献，应对照相同损失/训练策略，"
             "或为各模型提供相同的验证集调参预算，并增加匹配种子的重复实验。"
             "固定图与学习图的使用情况也应列明。", "",
             "实施命令与比较口径见 [统一输入对照方案](PEMS08_CONTROLLED_COMPARISON.md)。"
             "该组补充实验尚未启动。", "",
             "## 本轮已完成结果（默认配置）", "",
             "MAE/RMSE/MAPE 使用新仓库的逐步指标平均方式，零值阈值为 1e-4。"
             "为复现原口径，保留 LibCity 最后批次补齐：3,567 个测试窗口补到 3,584 个。", "",
             "| 模型 | 完成轮数 / 最佳轮次 | 输入通道 | 平均 MAE | 平均 RMSE | MAPE | 30 分钟 MAE | 60 分钟 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    first = None
    for run, record in records:
        p = record["data_provenance"]
        with np.load(run / "test_indices.npz") as archive:
            indices = archive["first_target_indices"]
        identity = (p["raw_files_sha256"], p["node_order"], p["scaler"], indices)
        if first is None:
            first = identity
        elif identity[:3] != first[:3] or not np.array_equal(indices, first[3]):
            raise RuntimeError("Completed runs use different source data/scaler/test order; no combined ranking.")
        scores = record["per_target"]["traffic_flow"]
        lines.append(f"| {record['model']} | {record['epochs_completed']} / {record['best_epoch']} | "
                     f"{record['effective']['input_channels']} | {scores['masked_MAE']['_avg']:.4f} | "
                     f"{scores['masked_RMSE']['_avg']:.4f} | {100*scores['masked_MAPE']['_avg']:.2f}% | "
                     f"{scores['masked_MAE']['@30min']:.4f} | {scores['masked_MAE']['@60min']:.4f} |")
    lines += ["", "## 运行状态", "", "| 模型 | 状态 |", "|---|---|"]
    for model, status in state["models"].items():
        lines.append(f"| {model} | {status} |")
    lines += ["", "## 与原仓库参考记录并列核对", "",
              "原记录采用相同公共协议，但设备不同、历史数据指纹缺失；"
              "以下差异属于复现诊断，不能单独归因于结构优化。", "",
              "| 模型 | 本轮 MAE | 原 seed0 MAE | 相对差异 |", "|---|---:|---:|---:|"]
    for run, record in records:
        reference = run.parent / "seed0/run.json"
        if reference.is_file():
            old = json.loads(reference.read_text())["per_target"]["traffic_flow"]["masked_MAE"]["_avg"]
            new = record["per_target"]["traffic_flow"]["masked_MAE"]["_avg"]
            lines.append(f"| {record['model']} | {new:.4f} | {old:.4f} | {(new/old-1)*100:+.2f}% |")
    lines += ["", "## 记录与限制", "",
              "- 每个运行目录保留 `resolved_config.json`、`data_provenance.json`、`training_history.csv`、"
              "`best_model.pt`、`test_indices.npz`，以及 LibCity 的预测数组和 checkpoint。大文件仅保留本地。",
              "- 8 个模型经过一次实际道路邻接下的 GPU 前向/反向检查，该检查不产生 benchmark 分数。",
              "- GWNET 的二维 kernel Conv1d 在进程内替换为 Conv2d；原第三方源码不变，兼容项在运行记录中标明。",
              "- 本轮只跑 seed 0；仍需多种子重复才能判断小差异是否稳定。",
              "- 与旧项目 6:2:2、10 轮实验的数值保持分开，见 [历史实验核对](PEMS08_IMPORTED_COMPARISON.md)。", ""]
    path = ROOT / "docs/PEMS08_RERUN_20261013.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    if state["status"] == "completed":
        import sys
        sys.path.insert(0, str(ROOT))
        from src.forecast.run_benchmark import write_summary
        write_summary([r for _, r in records], "PEMSD8", ("traffic_flow",))
    return state["status"], len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="aligned_20261013_mps")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        state = json.loads((ROOT / "results/reruns" / args.tag / "status.json").read_text())
        signature = json.dumps(state, sort_keys=True)
        if signature != previous:
            status, count = report(args.tag)
            print(f"Report updated: {status}, {count} completed", flush=True)
            previous = signature
            if status in ("completed", "failed", "interrupted"):
                return
        if not args.watch:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
