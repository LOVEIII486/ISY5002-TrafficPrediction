"""用教师模型给抽中的画面生成预标注，整理成 YOLO 训练格式。

    python -m src.detection.prelabel                  # 默认教师 yolo26x、原图分辨率 1920
    python -m src.detection.prelabel --conf 0.4

产物 `datasets/lta_yolo/`：

    images/{train,val}/*.jpg     抽中的画面（从 datasets/lta/ 复制）
    labels/{train,val}/*.txt     每行 `类别 cx cy w h`，坐标归一化到 0–1
    data.yaml                    ultralytics 训练配置，类别顺序同 run_detect.VEHICLE_CLASSES
    prelabel.json                教师模型、阈值与各类框数

**这些标签是模型猜的，不是真值。** 它们有两个用途：

1. 人工修正的起点。整个目录可直接导入 CVAT（Ultralytics YOLO Detection 1.0）或
   Label Studio / Roboflow，改完导出回同一结构即可；
2. 人工标注完成之前，先当伪标签跑通微调流程（大模型教中模型，即知识蒸馏）。
   此时在 val 上算出的指标只表示「学生学得像不像老师」，不能当准确率引用。

零检出的画面也写一个空的 .txt：ultralytics 把它当作纯背景样本，而不是缺标签。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

import pandas as pd

from . import paths
from .run_detect import VEHICLE_CLASSES, load_class_map, pick_device


def write_data_yaml(split: pd.DataFrame) -> None:
    """写清单 train.txt / val.txt 与 data.yaml。训练只读清单，不扫目录。"""
    import yaml

    for name in ("train", "val"):
        images = sorted(split[split["split"] == name]["image"])
        listing = "\n".join(str(paths.LABELED / "images" / name / image) for image in images) + "\n"
        (paths.LABELED / f"{name}.txt").write_text(listing, encoding="utf-8")
    config = {
        "path": str(paths.LABELED),
        "train": "train.txt",
        "val": "val.txt",
        "names": dict(enumerate(VEHICLE_CLASSES)),
    }
    paths.DATA_YAML.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="教师模型预标注")
    parser.add_argument("--teacher", type=lambda value: paths.ROOT / value, default=paths.TEACHER_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1920, help="LTA 原图宽 1920，远处车辆只有 6–10 像素，不缩小")
    parser.add_argument("--conf", type=float, default=0.3, help="伪标签阈值，略高于检测管线的 0.25 以压低误检")
    parser.add_argument("--overwrite", action="store_true", help="清空已有 images/ labels/ 后重建（会丢掉人工修正）")
    args = parser.parse_args()

    if not paths.SPLIT_FILE.is_file():
        raise SystemExit(f"缺少抽样清单 {paths.SPLIT_FILE}，先运行：python -m src.detection.sample")
    split = pd.read_csv(paths.SPLIT_FILE, dtype={"camera_id": str})

    from ultralytics import YOLO

    # 标签目录里可能已经有人工修正过的标签，默认绝不覆盖；确认要重建时整体清空，杜绝旧样本残留。
    existing = list((paths.LABELED / "labels").glob("*/*.txt"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"{paths.LABELED / 'labels'} 下已有 {len(existing)} 个标签文件（可能含人工修正）。\n"
            "确认要用教师预测整体重建时加 --overwrite（会清空 images/ 与 labels/）。"
        )
    for sub in ("images", "labels"):
        shutil.rmtree(paths.LABELED / sub, ignore_errors=True)

    teacher = YOLO(str(args.teacher))
    class_map = load_class_map(teacher.names)
    to_index = {name: index for index, name in enumerate(VEHICLE_CLASSES)}
    device = pick_device()
    print(f"教师 {args.teacher.name} · imgsz={args.imgsz} · conf={args.conf} · device={device} · {len(split)} 张")

    counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    for index, row in enumerate(split.itertuples(), start=1):
        source = paths.LTA_DATA / row.file
        if not source.is_file():
            raise SystemExit(f"图片不存在：{source}\n先运行 python -m src.detection.sample 下载")
        image_dir = paths.LABELED / "images" / row.split
        label_dir = paths.LABELED / "labels" / row.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dir / row.image)

        result = teacher.predict(str(source), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        lines = []
        for box in result.boxes:
            name = class_map.get(int(box.cls.item()))
            if name is None:
                continue
            cx, cy, w, h = (float(value) for value in box.xywhn[0])
            lines.append(f"{to_index[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[row.split][name] += 1
        (label_dir / row.image).with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if index % 40 == 0 or index == len(split):
            print(f"    {index}/{len(split)}", flush=True)

    write_data_yaml(split)
    report = {
        "teacher": args.teacher.name,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "labels_are": "pseudo (model predictions, not human-verified)",
        "images": split.groupby("split").size().to_dict(),
        "boxes": {name: dict(counter) for name, counter in counts.items()},
    }
    (paths.LABELED / "prelabel.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["boxes"], ensure_ascii=False))
    print(f"完成 → {paths.LABELED}")


if __name__ == "__main__":
    main()
