"""在 LTA 标注集上微调 YOLO。

    python -m src.detection.train                                  # 默认 yolo26m、1280、30 轮
    python -m src.detection.train --epochs 100 --batch 8 --name m1280_human   # 5070 上正式训练

产物在 `results/detection/finetune/<name>/`，权重是其中的 `weights/best.pt`。
类名就是 car / motorcycle / bus / truck，接入检测管线不需要写 classes.yaml：

    python -m src.detection.run_detect --weights results/detection/finetune/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse

from . import paths
from .run_detect import pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="微调 YOLO")
    parser.add_argument("--base", type=lambda value: paths.ROOT / value, default=paths.WEIGHTS, help="起点权重（COCO 预训练）")
    parser.add_argument("--data", type=lambda value: paths.ROOT / value, default=paths.DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=1280, help="与检测管线的推理尺寸一致")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15, help="验证集 mAP 连续这么多轮不涨就停")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="m1280_pseudo")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"缺少 {args.data}，先运行 python -m src.detection.prelabel（或导入人工标注）")

    from ultralytics import YOLO

    model = YOLO(str(args.base))
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        device=pick_device(),
        project=str(paths.FINETUNE_RESULTS),
        name=args.name,
        exist_ok=True,
        # 数据少，左右翻转之外不做激进增强；LTA 画面里没有倒置的车，不做上下翻转。
        fliplr=0.5,
        flipud=0.0,
        plots=True,
    )
    print(f"权重 → {paths.FINETUNE_RESULTS / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
