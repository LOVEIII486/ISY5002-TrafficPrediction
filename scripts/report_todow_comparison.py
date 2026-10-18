"""Pair each model's default-input run with its unified 9-channel run and write the report.

Also re-checks the seed-0 top four over seeds 0-4 (mean ± std and paired differences).
Reads only recorded results; never trains. Safe to re-run while the queue is in progress.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "results/forecast/PEMSD8/traffic_flow"
MODELS = ["FNN", "STGCN", "GRU", "LSTM", "MTGNN", "AGCRN", "GWNET", "MultiHeadSTGCN"]
# MultiHeadSTGCN's default configuration already uses flow + time of day + day of week.
REUSE_DEFAULT = {"MultiHeadSTGCN"}
GRAPH = {"FNN": "无", "GRU": "无", "LSTM": "无", "STGCN": "道路邻接", "MultiHeadSTGCN": "道路邻接（对称化）",
         "MTGNN": "学习图", "GWNET": "学习图（`aptonly`）", "AGCRN": "学习节点嵌入"}
OURS = "MultiHeadSTGCN"
TOP = ["GWNET", "MTGNN", "AGCRN", OURS]  # Seed-0 top four under unified input.
SEEDS = [0, 1, 2, 3, 4]
NOISE = 0.005  # Same-config same-seed rerun difference.


def load(model, tag, seed=0):
    run = FLOW / model / f"seed{seed}_{tag}"
    status = run / "status.json"
    if not status.is_file() or json.loads(status.read_text())["status"] != "completed":
        return None
    record = json.loads((run / "run.json").read_text())
    with np.load(run / "test_indices.npz") as archive:
        indices = archive["first_target_indices"]
    return run, record, indices


def load_cuda(model, seed):
    """Master's earlier CUDA record; it has no status file, provenance or test order to check."""
    path = FLOW / model / f"seed{seed}" / "run.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    return record if record["device"].startswith("cuda") else None


def flow(record, metric="masked_MAE"):
    return record["per_target"]["traffic_flow"][metric]["_avg"]


def is_cuda(record):
    return record["device"].startswith("cuda")


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else float("nan")


def paired(series, model):
    """Relative MAE difference to OURS on the seeds both have, with a t-based 95% CI."""
    common = sorted(set(series[model]) & set(series[OURS]))
    diffs = np.array([flow(series[model][s]) / flow(series[OURS][s]) - 1 for s in common])
    if len(diffs) < 2:
        return common, diffs, None, None
    mean, sd = mean_sd(diffs)
    half = stats.t.ppf(.975, len(diffs) - 1) * sd / np.sqrt(len(diffs))
    return common, diffs, mean - half, mean + half


def verdict(diffs, low, high):
    if low is None:
        return "种子不足"
    if high < 0:
        return "优于 MultiHeadSTGCN（显著）" if diffs.mean() < -NOISE else "显著，但幅度小于 0.5% 噪声底线"
    if low > 0:
        return "劣于 MultiHeadSTGCN（显著）" if diffs.mean() > NOISE else "显著，但幅度小于 0.5% 噪声底线"
    return "不显著（95% CI 跨 0）"


def multi_seed_lines(args, seed_runs):
    series = {m: {s: run[1] for s, run in seed_runs[m].items() if run} for m in TOP}
    mps_ours = dict(series[OURS])
    cuda = {s: r for s in SEEDS if (r := load_cuda(OURS, s))}
    for seed, record in cuda.items():  # Seeds not yet rerun on mps reuse master's CUDA record.
        series[OURS].setdefault(seed, record)
    mixed = any(is_cuda(r) for r in series[OURS].values())
    ours_label = f"{OURS}（配对基准）" if mixed else OURS
    done = "、".join(f"{m} {len(mps_ours if m == OURS else series[m])}/{len(SEEDS)}" for m in TOP)

    lines = ["", "## 多种子复核（seed 0–4）", "",
             f"seed 0 的前 4 名补跑 seed 1–4，确认排名是否稳定。mps 已完成：{done}。", "",
             "```bash",
             f"python scripts/run_flow_rerun.py --tag {args.tag} --device mps --seed N \\",
             "  --models GWNET MTGNN AGCRN \\",
             "  --set load_external=true --set add_time_in_day=true --set add_day_in_week=true",
             f"python scripts/run_flow_rerun.py --tag {args.default_tag} --device mps --seed N --models {OURS}",
             "```", "",
             "每个种子的队列状态写在 `results/reruns/<tag>_seed<N>/`（seed 0 仍在 `results/reruns/<tag>/`）。"
             f"{OURS} 不加 `--set`，与 seed 0 复用的默认配置运行（9 通道）完全一致。"]
    if mixed:
        cuda_seeds = "、".join(str(s) for s, r in sorted(series[OURS].items()) if is_cuda(r))
        lines.append(f"mps 尚未补跑的种子（{cuda_seeds}）暂用 master 上的 CUDA 记录，表中以 † 标出，见下方说明。")

    def row(rank, label, records):
        cells = [f"{flow(records[s]):.4f}{'†' if is_cuda(records[s]) else ''}" if s in records else "—" for s in SEEDS]
        maes = [flow(r) for r in records.values()]
        rmses = [flow(r, "masked_RMSE") for r in records.values()]
        mae_m, mae_sd = mean_sd(maes)
        rmse_m, rmse_sd = mean_sd(rmses)
        spread = (lambda sd: f" ± {sd:.4f}" if len(maes) > 1 else "")
        return (f"| {rank} | {label} | {' | '.join(cells)} | **{mae_m:.4f}**{spread(mae_sd)} | {len(maes)} | "
                f"{rmse_m:.4f}{spread(rmse_sd)} |")

    ranked = sorted((m for m in TOP if series[m]), key=lambda m: mean_sd([flow(r) for r in series[m].values()])[0])
    lines += ["", "### 各种子 MAE 与均值 ± 标准差", "",
              "| 名次 | 模型 | " + " | ".join(f"seed {s}" for s in SEEDS) + " | MAE 均值 ± 标准差 | n | RMSE 均值 ± 标准差 |",
              "|---:|---|" + "---:|" * len(SEEDS) + "---:|---:|---:|"]
    lines += [row(rank, ours_label if m == OURS else m, series[m]) for rank, m in enumerate(ranked, 1)]
    if mixed and mps_ours:
        lines.append(row("—", f"{OURS}（仅 mps）", mps_ours))
    if cuda:
        lines.append(row("—", f"{OURS}（CUDA，master 记录）", cuda))
    lines += ["", "名次按 MAE 均值排列；标准差为样本标准差（n−1）。"]

    lines += ["", "### 相对 MultiHeadSTGCN 的配对差（相同种子）", "",
              "| 模型 | 配对种子 | 逐种子相对差 | 均值 ± 标准差 | 95% CI | 优于 MultiHeadSTGCN | 结论 |",
              "|---|---|---|---:|---|---:|---|"]
    results = {}
    for model in TOP[:-1]:
        common, diffs, low, high = paired(series, model)
        results[model] = (common, diffs, low, high)
        if not len(diffs):
            lines.append(f"| {model} | — | — | — | — | — | 种子不足 |")
            continue
        per_seed = " / ".join(f"{d*100:+.2f}%{'†' if is_cuda(series[OURS][s]) else ''}" for s, d in zip(common, diffs))
        mean, sd = mean_sd(diffs)
        ci = f"{low*100:+.2f}% ~ {high*100:+.2f}%" if low is not None else "—"
        spread = f"{mean*100:+.2f}% ± {sd*100:.2f}%" if len(diffs) > 1 else f"{mean*100:+.2f}%"
        lines.append(f"| {model} | {','.join(map(str, common))} | {per_seed} | {spread} | {ci} | "
                     f"{int((diffs < 0).sum())}/{len(diffs)} | {verdict(diffs, low, high)} |")
    lines += ["", "相对差 = 同一种子下 MAE(模型) / MAE(MultiHeadSTGCN) − 1，负值表示模型更好；"
              "95% CI 用自由度 n−1 的 t 分布。"]

    lines += ["", "### 多种子结论", ""]
    for model in TOP[:-1]:
        common, diffs, low, high = results[model]
        if low is None:
            lines.append(f"- **{model}**：配对种子不足，暂无结论。")
            continue
        seed0 = diffs[common.index(0)] if 0 in common else None
        claim = ("seed 0 显示更好" if seed0 < -NOISE else "seed 0 在噪声范围内持平" if seed0 <= NOISE
                 else "seed 0 显示更差") if seed0 is not None else "无 seed 0"
        result = verdict(diffs, low, high)
        if seed0 is not None and seed0 < -NOISE:
            if result.startswith("优于"):
                holds = "**仍然成立**"
            elif diffs.mean() < -NOISE:
                holds = "方向一致但**不显著**，不能下“更好”的结论"
            elif abs(diffs.mean()) <= NOISE:
                holds = "**不成立**：多种子均值落在 ±0.5% 噪声底线内，与 MultiHeadSTGCN 持平"
            else:
                holds = f"**不成立**（{result}）"
        elif seed0 is not None and abs(seed0) <= NOISE:
            holds = "仍为持平" if result.startswith("不显著") else f"多种子下变为：{result}"
        else:
            holds = result
        shown = f"（{seed0*100:+.2f}%）" if seed0 is not None else ""
        lines.append(f"- **{model}**：{claim}{shown}；{len(diffs)} 个种子配对均值 "
                     f"{diffs.mean()*100:+.2f}%（95% CI {low*100:+.2f}% ~ {high*100:+.2f}%），"
                     f"{int((diffs < 0).sum())}/{len(diffs)} 个种子优于 MultiHeadSTGCN"
                     f"（符号检验双侧 p = {stats.binomtest(int((diffs < 0).sum()), len(diffs)).pvalue:.3f}）→ {holds}。")

    complete = [s for s in SEEDS if all(s in series[m] for m in TOP)]
    if complete:
        places = []
        for seed in complete:
            order = sorted(TOP, key=lambda m: flow(series[m][seed]))
            places.append(f"seed {seed} 第 {order.index(OURS) + 1}")
        lines.append(f"- **排名**：按均值为 {' < '.join(ranked)}（MAE 由低到高）。"
                     f"MultiHeadSTGCN 在各种子内的名次：{'、'.join(places)}。")
    relative = [mean_sd([flow(r) for r in series[m].values()]) for m in TOP if len(series[m]) > 1]
    if relative:
        ratios = [sd / mean for mean, sd in relative]
        lines.append(f"- **种子间波动**：各模型 MAE 的标准差为均值的 {min(ratios)*100:.2f}%–{max(ratios)*100:.2f}%，"
                     "可与 0.5% 的同种子重跑噪声底线对照。")
    late = [f"{m} seed {s}（{r['best_epoch']}/{r['epochs_completed']}）" for m in TOP for s, r in sorted(series[m].items())
            if not is_cuda(r) and r["best_epoch"] is not None and r["best_epoch"] >= .9 * r["effective"]["max_epoch"]]
    if late:
        lines.append(f"- **最佳轮次接近 100 轮上限**：{'、'.join(late)}。这些运行加预算可能还会下降，"
                     "即这些模型的均值可能偏高。")
    if 0 in mps_ours and 0 in cuda:
        gap = flow(mps_ours[0]) / flow(cuda[0]) - 1
        device = (f"同一 seed 0 在 CUDA 与 mps 上分别为 {flow(cuda[0]):.4f} / {flow(mps_ours[0]):.4f}"
                  f"（mps 相对 {gap*100:+.2f}%）")
    else:
        device = "无同种子的跨设备对照"
    if mixed:
        lines.append(f"- **设备差异（†）**：master 上的 CUDA 记录与 mps 运行的有效配置相同（9 通道、batch 64、最多 100 轮、"
                     "耐心 30、lr 0.001），但设备和代码版本不同，且记录里没有划分比例、数据哈希、测试窗口顺序和最佳轮次，"
                     f"无法像 mps 运行那样逐项核对；{device}。mps 补跑完成后重新运行本脚本即自动改用 mps 记录。")
    elif cuda:
        cuda_mean, _ = mean_sd([flow(r) for r in cuda.values()])
        ours_mean, _ = mean_sd([flow(r) for r in series[OURS].values()])
        lines.append(f"- **设备差异**：MultiHeadSTGCN 全部使用 mps 运行；master 上 CUDA 记录（n={len(cuda)}）的均值为 "
                     f"{cuda_mean:.4f}，mps 均值 {ours_mean:.4f}（{(ours_mean/cuda_mean-1)*100:+.2f}%），仅作参考，不参与配对；"
                     f"{device}。")
    return lines


def identity(entry):
    p = entry[1]["data_provenance"]
    return json.dumps([p["raw_files_sha256"], p["node_order"], p["scaler"], p["adjacency_sha256"]], sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-tag", default="aligned_20261013_mps")
    parser.add_argument("--tag", default="todow_20261015_mps")
    parser.add_argument("--out", default="docs/PEMS08_TODOW_20261015.md")
    args = parser.parse_args()

    rows, entries = [], []
    for model in MODELS:
        default = load(model, args.default_tag)
        unified = default if model in REUSE_DEFAULT else load(model, args.tag)
        rows.append((model, default, unified))
        entries += [e for e in (default, unified) if e]
    seed_runs = {m: {s: load(m, args.default_tag if m in REUSE_DEFAULT else args.tag, s) for s in SEEDS} for m in TOP}
    entries += [run for runs in seed_runs.values() for run in runs.values() if run]
    if entries:
        first = entries[0]
        for entry in entries[1:]:
            # Adjacency differs by bidir_adj_mx, so only data, nodes, scaler and test order must match.
            a, b = json.loads(identity(first)), json.loads(identity(entry))
            if a[:3] != b[:3] or not np.array_equal(first[2], entry[2]):
                raise RuntimeError(f"{entry[0]} uses different data/scaler/test order; no combined table.")
    done = sum(1 for _, _, u in rows if u)

    def mae(entry, horizon="_avg"):
        return entry[1]["per_target"]["traffic_flow"]["masked_MAE"][horizon]

    lines = ["# PeMSD8 流量实验：统一 9 通道输入对照（2026-10-15）", "",
             f"已完成 {done}/{len(MODELS)} 个模型。由 `scripts/report_todow_comparison.py` 根据运行记录生成。"
             "前 4 名的 seed 0–4 复核见文末「多种子复核」。", "",
             "## 实验设置", "",
             f"与 [默认配置复现](PEMS08_RERUN_20261013.md)（`{args.default_tag}`）相同的数据、7:1:2 划分、"
             "12 步预测 12 步、batch 64、最多 100 轮、早停耐心 30、seed 0、MPS。"
             "唯一改动：所有模型的输入统一为 **流量 + 一天内时刻 + 星期 one-hot = 9 通道**，"
             "即 MultiHeadSTGCN 定型配置所用的输入。", "",
             "```bash",
             f"python scripts/run_flow_rerun.py --tag {args.tag} --device mps \\",
             "  --models GWNET MTGNN AGCRN STGCN GRU LSTM FNN \\",
             "  --set load_external=true --set add_time_in_day=true --set add_day_in_week=true",
             "```", "",
             "MultiHeadSTGCN 的默认配置本来就是 9 通道，直接复用默认配置那次运行。"
             "日历特征在预测时刻提前已知；STGCN/GRU/LSTM 滚动预测时从未来窗口取的只是日历通道，不含未来流量。", "",
             "## 结果", "",
             "| 模型 | 默认通道 | 默认 MAE | 9 通道 MAE | 变化 | 9 通道 RMSE | MAPE | 30 分钟 MAE | 60 分钟 MAE | 轮数 / 最佳 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, default, unified in rows:
        d_ch = default[1]["effective"]["input_channels"] if default else "—"
        d_mae = f"{mae(default):.4f}" if default else "—"
        if unified:
            s = unified[1]["per_target"]["traffic_flow"]
            change = "（复用）" if model in REUSE_DEFAULT else (f"{(mae(unified)/mae(default)-1)*100:+.2f}%" if default else "—")
            lines.append(f"| {model} | {d_ch} | {d_mae} | **{mae(unified):.4f}** | {change} | "
                         f"{s['masked_RMSE']['_avg']:.4f} | {100*s['masked_MAPE']['_avg']:.2f}% | "
                         f"{mae(unified, '@30min'):.4f} | {mae(unified, '@60min'):.4f} | "
                         f"{unified[1]['epochs_completed']} / {unified[1]['best_epoch']} |")
        else:
            lines.append(f"| {model} | {d_ch} | {d_mae} | 未完成 | — | — | — | — | — | — |")

    finished = sorted([(mae(u), m) for m, _, u in rows if u])
    if finished:
        lines += ["", "## 统一输入下的排名（12 步平均 MAE）", "", "| 名次 | 模型 | MAE | 相对 MultiHeadSTGCN | 图信息 |",
                  "|---:|---|---:|---:|---|"]
        ours = next((v for v, m in finished if m == "MultiHeadSTGCN"), None)
        for rank, (value, model) in enumerate(finished, 1):
            rel = f"{(value/ours-1)*100:+.2f}%" if ours else "—"
            lines.append(f"| {rank} | {model} | {value:.4f} | {rel} | {GRAPH[model]} |")

    ours = next((u for m, _, u in rows if m == "MultiHeadSTGCN" and u), None)
    if ours and done == len(MODELS):
        noise = 0.005
        rel = {m: mae(u) / mae(ours) - 1 for m, _, u in rows if m != "MultiHeadSTGCN"}
        better = [f"{m}（{v*100:+.2f}%）" for m, v in sorted(rel.items(), key=lambda kv: kv[1]) if v < -noise]
        tied = [f"{m}（{v*100:+.2f}%）" for m, v in rel.items() if abs(v) <= noise]
        gain = {m: mae(u) / mae(d) - 1 for m, d, u in rows if m not in REUSE_DEFAULT}
        helped = [f"{m} {v*100:+.1f}%" for m, v in sorted(gain.items(), key=lambda kv: kv[1]) if v < -noise]
        hurt = [f"{m} {v*100:+.1f}%" for m, v in sorted(gain.items(), key=lambda kv: kv[1]) if v > noise]
        unconverged = [m for m, _, u in rows if u and u[1]["best_epoch"] == u[1]["epochs_completed"]
                       == u[1]["effective"]["max_epoch"]]
        lines += ["", "## 要点（seed 0）", "",
                  f"- **统一输入后优于 MultiHeadSTGCN、且差距超过 0.5% 噪声底线**：{'、'.join(better) or '无'}。",
                  f"- **与 MultiHeadSTGCN 在噪声范围内持平**：{'、'.join(tied) or '无'}。",
                  f"- **日历特征有收益的模型**（同模型配对）：{'、'.join(helped) or '无'}。",
                  f"- **日历特征反而变差的模型**：{'、'.join(hurt) or '无'}。"]
        if any(m in ("GRU", "LSTM") for m in gain if gain[m] > noise):
            lines.append("  LibCity 的 RNN 不做空间建模，把 170 个节点 × 通道拼成一个向量输入（9 通道时为 1,530 维，"
                         "隐藏层 64）；星期 one-hot 对所有节点相同，等于多出 1,190 维冗余输入。训练损失和验证损失"
                         "从第一轮起都更高，属于默认超参（lr 0.01、多步衰减）下的优化困难，不是过拟合，"
                         "也不能说明日历特征本身有害。")
        if unconverged:
            lines.append(f"- 最佳验证轮次等于第 100 轮、仍未收敛：{'、'.join(unconverged)}，加预算可能继续下降。")

    lines += ["", "## 怎么读这张表", "",
              "- 本表只控制**输入信息**。优化器、学习率、损失（MultiHeadSTGCN 的 `scale_aware_loss`）、"
              "多步策略（STGCN 单步滚动、MultiHeadSTGCN 直接多步、MTGNN 课程学习）仍按各模型默认，"
              "差异不能全部归因于网络结构。",
              "- 噪声底线约 0.5%（同配置同种子重跑的 MAE 差）。以上各表只有 seed 0，小于 0.5% 的差异不作结论；"
              "前 4 名的多种子复核见下一节。",
              "- 「变化」列是同一模型加日历特征前后的配对差，可直接说明日历特征对该模型的收益。",
              "- 所有 mps 运行（含多种子）的原始数据哈希、节点顺序、scaler 和测试窗口顺序已逐一核对一致"
              "（脚本不一致时会拒绝生成）。"]
    lines += multi_seed_lines(args, seed_runs) + [""]
    out = ROOT / args.out
    temporary = out.with_suffix(".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(out)
    print(f"{out}: {done}/{len(MODELS)} completed")


if __name__ == "__main__":
    main()
