"""按场景组自助重采样，给 mAP@0.5 及模型间差值算置信区间。

    python -m src.detection.bootstrap --data datasets/miotcd_yolo_v4 --split test --imgsz 736 \
        --weights yolo/yolo26m.pt results/detection/finetune/m736_mio_v4/weights/best.pt \
                  results/detection/finetune/m736_joint_v4/weights/best.pt

同一场景组（大致同一机位）的图高度相关，按单张图重采样会低估不确定性，所以每次有放回地抽**场景组**，
组内图片整组带入。所有模型共用同一批重采样，差值是配对的。

只算 IoU 0.5：每张图每类先用 evaluate.greedy_match 做一次匹配，重采样时只拼接结果再算 AP，口径与
evaluate.py 的 AP50 相同。某次重采样里某类没有真值时，该类不计入那次的 mAP（摩托车最常遇到）。

区间是百分位法 95% 区间，只反映「换一批场景组」带来的抽样波动，不包括训练随机性（每个配置只训练了一次），
场景组数少时也只是近似。产物 `results/detection/finetune/eval/<数据集>/<划分>/bootstrap.json`。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate, paths
from .run_detect import VEHICLE_CLASSES


def per_image_matches(images: list[str], truth: dict, predictions: dict, sizes: dict) -> dict[str, list[tuple]]:
    """{图片: [每类一个 (置信度数组, 是否 TP 数组, 真值数)]}，IoU 0.5。"""
    table = {}
    for image in images:
        rows = []
        for cls in range(len(VEHICLE_CLASSES)):
            gt = evaluate.to_xyxy(truth[image]["xywhn"][truth[image]["cls"] == cls], *sizes[image])
            pred = predictions[image]
            mask = pred["cls"] == cls
            conf, hits = evaluate.greedy_match(pred["conf"][mask], pred["xyxy"][mask], gt, 0.5)
            rows.append((conf, hits, len(gt)))
        table[image] = rows
    return table


def ap50(images: list[str], table: dict) -> tuple[float, list[float | None]]:
    """在一组（可重复的）图片上算 mAP@0.5 与各类 AP@0.5。"""
    per_class = []
    for cls in range(len(VEHICLE_CLASSES)):
        confs = [table[i][cls][0] for i in images]
        hits = [table[i][cls][1] for i in images]
        n_gt = sum(table[i][cls][2] for i in images)
        if n_gt == 0:
            per_class.append(None)
            continue
        conf = np.concatenate(confs)
        if len(conf) == 0:
            per_class.append(0.0)
            continue
        order = np.argsort(-conf, kind="stable")
        hit = np.concatenate(hits)[order]
        tp, fp = np.cumsum(hit), np.cumsum(~hit)
        per_class.append(evaluate.average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1)))
    present = [v for v in per_class if v is not None]
    return (float(np.mean(present)) if present else float("nan")), per_class


def interval(values: list[float]) -> list[float]:
    clean = np.array([v for v in values if v is not None and not np.isnan(v)])
    return [round(float(np.percentile(clean, 2.5)), 4), round(float(np.percentile(clean, 97.5)), 4)] if len(clean) else [None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="按场景组自助重采样算 mAP@0.5 的置信区间")
    parser.add_argument("--weights", nargs="+", type=lambda value: paths.ROOT / value, required=True)
    parser.add_argument("--data", type=lambda value: paths.ROOT / value, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir: Path = args.data
    split = pd.read_csv(data_dir / "split.csv", dtype={"image": str})
    split = split[split["split"] == args.split].reset_index(drop=True)
    evaluate.check_frozen(data_dir, args.split, split)
    if "scene_group" not in split.columns:
        raise SystemExit("split.csv 没有 scene_group 列，无法按组重采样")
    images = sorted(split["image"])
    truth = evaluate.load_ground_truth(data_dir, args.split, images)
    members: dict[int, list[str]] = defaultdict(list)
    for image, group in zip(split["image"], split["scene_group"]):
        members[int(group)].append(image)
    groups = sorted(members)

    eval_dir = paths.FINETUNE_RESULTS / "eval" / data_dir.name / args.split
    labels, tables = [], {}
    for weights in args.weights:
        label = f"{evaluate.weights_label(weights)}@{args.imgsz}"
        print(f"推理 {label} · {len(images)} 张", flush=True)
        predictions, sizes = evaluate.predict(weights, data_dir, args.split, eval_dir, split, args.imgsz, examples=0)
        tables[label] = per_image_matches(images, truth, predictions, sizes)
        labels.append(label)

    rng = np.random.default_rng(args.seed)
    samples: dict[str, list[float]] = defaultdict(list)
    per_class_samples: dict[tuple[str, str], list] = defaultdict(list)
    for _ in range(args.resamples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        resample = [image for g in drawn for image in members[int(g)]]
        scores = {}
        for label in labels:
            score, per_class = ap50(resample, tables[label])
            scores[label] = score
            samples[label].append(score)
            for name, value in zip(VEHICLE_CLASSES, per_class):
                per_class_samples[(label, name)].append(value)
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                samples[f"{b} - {a}"].append(scores[b] - scores[a])

    report = {
        "data": data_dir.name,
        "split": args.split,
        "imgsz": args.imgsz,
        "resamples": args.resamples,
        "seed": args.seed,
        "unit": "scene_group",
        "n_groups": len(groups),
        "n_images": len(images),
        "note": "percentile 95% CI over scene-group resamples; excludes training randomness (single run per config)",
        "models": {},
        "differences": {},
    }
    for label in labels:
        point, per_class = ap50(images, tables[label])
        report["models"][label] = {
            "mAP50": round(point, 4),
            "mAP50_ci95": interval(samples[label]),
            "AP50": {name: (None if v is None else round(v, 4)) for name, v in zip(VEHICLE_CLASSES, per_class)},
            "AP50_ci95": {name: interval(per_class_samples[(label, name)]) for name in VEHICLE_CLASSES},
        }
    for key, values in samples.items():
        if " - " in key:
            b, a = key.split(" - ")
            point = report["models"][b]["mAP50"] - report["models"][a]["mAP50"]
            arr = np.array(values)
            report["differences"][key] = {
                "mAP50_diff": round(point, 4),
                "ci95": interval(values),
                "share_of_resamples_above_0": round(float((arr > 0).mean()), 4),
            }
    (eval_dir / "bootstrap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{args.split}：{len(groups)} 个场景组、{len(images)} 张，按组重采样 {args.resamples} 次")
    for label, m in report["models"].items():
        print(f"  {label:24s} mAP50 {m['mAP50']:.3f}  95% 区间 {m['mAP50_ci95']}")
    for key, d in report["differences"].items():
        print(f"  {key:44s} 差 {d['mAP50_diff']:+.3f}  95% 区间 {d['ci95']}  （>0 的比例 {d['share_of_resamples_above_0']:.0%}）")
    print(f"产物 → {eval_dir / 'bootstrap.json'}")


if __name__ == "__main__":
    main()
