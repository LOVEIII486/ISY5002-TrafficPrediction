"""微调 YOLO。

    python -m src.detection.train --data datasets/miotcd_yolo_v2/data.yaml --imgsz 736 --batch 8 --epochs 30 --name m736_mio_v2
    python -m src.detection.train --data datasets/joint_yolo_v2/data.yaml  --imgsz 736 --batch 8 --epochs 30 --name m736_joint_v2

产物在 `results/detection/finetune/<name>/`，权重是其中的 `weights/best.pt`（按 data.yaml 里 val 划分的 mAP 挑选，
v2 数据里 val 指向 dev，测试集不参与）。同名目录已存在时拒绝运行，避免覆盖历史结果；确需重跑加 --overwrite。
`provenance.json` 记录训练清单、起点权重与配置的哈希，供复现与核对。

类名就是 car / motorcycle / bus / truck，接入检测管线不需要写 classes.yaml：

    python -m src.detection.run_detect --weights results/detection/finetune/<name>/weights/best.pt

关于「继续训练」：--base 指向已有微调权重时，ultralytics 只加载权重、重新开始一个训练阶段（新的优化器状态、
学习率计划与 mosaic 阶段），不是断点续训。报告里应写成「以某权重初始化的第二阶段微调」。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from . import paths
from .frozen import check_data_yaml
from .run_detect import pick_device


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def train_lists(data_yaml: Path) -> dict[str, str]:
    """data.yaml 引用的清单文件（train / val 为 .txt 时）及其哈希。"""
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    found = {}
    for key in ("train", "val"):
        entry = Path(str(config.get(key, "")))
        entry = entry if entry.is_absolute() else root / entry
        if entry.suffix == ".txt" and entry.is_file():
            # 按行计数：仓库路径里可能有空格，不能按空白切分
            found[key] = f"{entry} sha256:{sha256_16(entry)} lines:{len(entry.read_text().splitlines())}"
        else:
            found[key] = f"{entry} (directory scan)"
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="微调 YOLO")
    parser.add_argument("--base", type=lambda value: paths.ROOT / value, default=paths.WEIGHTS, help="起点权重：默认 COCO 预训练；也可指向已有微调权重做第二阶段微调")
    parser.add_argument("--data", type=lambda value: paths.ROOT / value, default=paths.DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=1280, help="与检测管线的推理尺寸一致")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0, help="早停耐心；默认 0 关闭早停，对照实验跑满相同轮数")
    parser.add_argument("--fraction", type=float, default=1.0, help="只用训练集的这一比例做试跑（取清单**前段**，不是随机抽样）；正式训练保持 1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", required=True, help="实验名，即结果目录名；要唯一")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名结果目录")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"缺少 {args.data}")
    check_data_yaml(args.data)
    run_dir = paths.FINETUNE_RESULTS / args.name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} 已存在。换一个 --name，或加 --overwrite 覆盖")
        shutil.rmtree(run_dir)

    # 训练开始前先落一份来源记录，训练中断也能知道用了什么数据和起点
    provenance = {
        "base_weights": f"{args.base} sha256:{sha256_16(args.base)}",
        "data_yaml": f"{args.data} sha256:{sha256_16(args.data)}",
        "lists": train_lists(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "fraction": args.fraction,
        "seed": args.seed,
        "device": str(pick_device()),
        "checkpoint_selection": "ultralytics fitness on data.yaml val split = mAP@0.5:0.95",
    }
    provenance_path = paths.FINETUNE_RESULTS / f"{args.name}.provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    from ultralytics import YOLO

    model = YOLO(str(args.base))
    # 增强：除下面两项外沿用 ultralytics 默认值，即 mosaic=1.0（最后 close_mosaic=10 轮关闭）、scale=0.5、
    # translate=0.1、HSV 抖动。监控画面里没有倒置的车，所以关掉上下翻转。对比实验之间保持同一套增强与轮数。
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        seed=args.seed,
        deterministic=True,
        device=pick_device(),
        project=str(paths.FINETUNE_RESULTS),
        name=args.name,
        exist_ok=False,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
    )
    for name in ("best.pt", "last.pt"):
        weights = run_dir / "weights" / name
        if weights.is_file():
            provenance[name] = f"sha256:{sha256_16(weights)}"
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        provenance["epochs_completed"] = max(0, len(results_csv.read_text().splitlines()) - 1)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path.unlink(missing_ok=True)
    print(f"权重 → {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
