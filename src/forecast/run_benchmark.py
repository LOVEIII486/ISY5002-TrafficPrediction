"""基准实验入口。

    # 最快的一次完整训练+评估（单模型、单目标、一个 epoch）
    python -m src.forecast.run_benchmark --targets traffic_flow --models FNN --max-epoch 1

    # 三个基线模型的默认对比
    python -m src.forecast.run_benchmark --dataset PEMSD8

    # 多种子
    python -m src.forecast.run_benchmark --models STGCN --seeds 0,1,2

    # 自研模型（三目标联合）与其消融
    python -m src.forecast.run_benchmark --models MultiHeadSTGCN --targets all
    python -m src.forecast.run_benchmark --models MultiHeadSTGCN --targets traffic_flow \
        --set direct_multi_step=false
"""

from __future__ import annotations

import argparse
import csv
import json

from . import datasets, paths, train
from . import metrics as metrics_module
from .models import registry
from .protocols import default_protocol

DEFAULT_MODELS = "FNN,STGCN,MTGNN"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="交通预测基准实验")
    parser.add_argument("--dataset", default="PEMSD8", choices=sorted(datasets.REGISTRY))
    parser.add_argument("--targets", default="", help="逗号分隔；留空用默认目标；all 表示全部目标联合")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="逗号分隔或 all；已注册：" + ",".join(
        registry.available()))
    parser.add_argument("--seeds", default="0", help="逗号分隔，如 0,1,2")
    parser.add_argument("--max-epoch", type=int, default=None, help="覆盖协议的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖模型超参，可重复；值按 JSON 解析，如 --set direct_multi_step=false",
    )
    parser.add_argument("--tag", default="", help="产物目录后缀；不填则由 --set 自动生成")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def resolve_targets(dataset: datasets.Dataset, text: str) -> tuple[str, ...]:
    """把命令行的目标写法翻译成目标元组。"""
    if not text:
        return (dataset.default_target,)
    if text == "all":
        return dataset.targets
    return tuple(part.strip() for part in text.split(",") if part.strip())


def brief(series: dict) -> str:
    """一行摘要。"""
    return (
        f"MAE {series['masked_MAE']['_avg']:.4f}"
        f"  RMSE {series['masked_RMSE']['_avg']:.4f}"
        f"  MAPE {100 * series['masked_MAPE']['_avg']:.2f}%"
    )


def print_table(records: list[dict], dataset_name: str, targets: tuple[str, ...]) -> None:
    """并排打印本次结果与论文表 5 的对应数值。"""
    print(f"\n{dataset_name} / {'+'.join(targets)}   （12 步平均；MAPE 为百分数）")
    header = (
        f"{'模型':<17}{'目标':<20}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}"
        f"{'论文MAE':>10}{'论文RMSE':>10}{'论文MAPE':>10}"
    )
    print(header)
    print("-" * 100)
    for record in records:
        for target, series in record["per_target"].items():
            line = (
                f"{record['model']:<17}{target:<20}"
                f"{series['masked_MAE']['_avg']:>10.4f}"
                f"{series['masked_RMSE']['_avg']:>10.4f}"
                f"{100 * series['masked_MAPE']['_avg']:>10.2f}"
            )
            reference = datasets.reference(record["dataset"], (target,), record["model"])
            if reference:
                line += f"{reference[0]:>10.2f}{reference[2]:>10.2f}{reference[1]:>10.2f}"
            else:
                line += f"{'—':>10}{'—':>10}{'—':>10}"
            print(line)


def write_summary(records: list[dict], dataset_name: str, targets: tuple[str, ...]) -> None:
    """把本次全部运行写成 CSV 与聚合后的 JSON，每个目标一行。"""
    out_dir = paths.RESULTS_ROOT / dataset_name / "+".join(targets)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        for target, series in record["per_target"].items():
            row = {
                "model": record["model"],
                "source": record["source"],
                "target": target,
                "seed": record["seed"],
                "epochs": record["effective"]["max_epoch"],
                "train_seconds": record["train_seconds"],
                "exp_id": record["exp_id"],
            }
            for key, values in series.items():
                for horizon, value in values.items():
                    row[f"{key}{horizon}"] = value
            rows.append(row)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {"runs": records, "aggregate": metrics_module.aggregate(records)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n产物：{out_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = datasets.resolve(args.dataset)
    targets = resolve_targets(dataset, args.targets)
    models = registry.available() if args.models == "all" else [
        name.strip() for name in args.models.split(",") if name.strip()
    ]
    seeds = [int(value) for value in args.seeds.split(",")]

    extra_args = {}
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到：{item}")
        try:
            extra_args[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            extra_args[key.strip()] = raw  # 不是 JSON 就当字符串（如 head_mode=flatten）

    protocol = default_protocol(*targets)
    # 覆盖项进目录名，否则消融变体会覆盖基准的同名产物。
    tag = args.tag or "_".join(f"{key}-{value}" for key, value in sorted(extra_args.items()))
    records = []
    for model in models:
        for seed in seeds:
            print(f"\n=== {model} / {'+'.join(targets)} / seed {seed} ===", flush=True)
            record = train.run(
                args.dataset,
                targets,
                model,
                protocol,
                seed=seed,
                gpu=not args.cpu,
                max_epoch=args.max_epoch,
                batch_size=args.batch_size,
                extra_args=extra_args or None,
                tag=tag,
            )
            for target, series in record["per_target"].items():
                print(f"    {target:<20}{brief(series)}  用时 {record['train_seconds']:.0f}s")
            records.append(record)

    print_table(records, args.dataset, targets)
    write_summary(records, args.dataset, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
