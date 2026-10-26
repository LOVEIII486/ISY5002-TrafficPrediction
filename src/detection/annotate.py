"""LTA 验证图的人工修正：准备 Label Studio 任务、提供本地图片服务、把导出结果收回成冻结的人工修正验证子集。

    python -m src.detection.annotate prepare      # 生成任务 JSON 与标注界面配置
    python -m src.detection.annotate serve        # 本机只读图片服务（Label Studio 从这里加载图片）
    python -m src.detection.annotate collect <Label Studio 导出的 JSON>   # 转回 YOLO，写 datasets/lta_human_v1/
    python -m src.detection.annotate collect <export.json> --annotations-only --excluded <csv>
        # 只写标签/清单，不复制图片；用于仓库和 Hugging Face

起点是 week2 的 80 张验证图（从未参与训练）：
- 预标注 = 教师 yolo26x 的伪标签（置信度 ≥ 0.3，即 prelabel.py 的产物），以四个类别出现；
- 候选框 = 教师置信度 0.1–0.3、且不与预标注重叠（IoU ≤ 0.5）的框，以 `candidate` 类出现，多为远处小车与摩托车；
  标注者必须把每个候选改成四类之一或删除，collect 时仍有 candidate 会拒绝收回。

顺序：每个相机 10 张里先排 5 张（白天、夜间都有），前 40 张本身就是均衡的一半，时间不够标到第 40 张也能用。

**偏差说明（报告要写）**：预标注来自教师，联合训练模型又是学教师的伪标签，标注者若只「微调现有框」，
验证子集会偏向教师的判断（框的范围、漏标的车）。标注时必须逐车检查、补全漏标，规范见 docs/LTA_ANNOTATION_GUIDE.md。
结果写成「教师模型预标注 + 人工逐张修正」，不要写成「从零人工标注」；
报告里称它为 human-corrected validation subset，不要称为独立 test set。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .evaluate import iou_matrix, to_xyxy
from .run_detect import VEHICLE_CLASSES, load_class_map, pick_device

WORK = paths.FINETUNE_RESULTS.parent / "annotation"
HUMAN = paths.ROOT / "datasets" / "lta_human_v1"
PORT = 8777  # 8765 常被其他本机程序占用
CANDIDATE = "candidate"
CANDIDATE_CONF = (0.10, 0.30)
LABEL_CONFIG = f"""<View>
  <Header value="逐车检查：改正类别与框、补全漏标；candidate 必须改成四类之一或删除"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" opacity="0.15">
    <Label value="car" background="#2a78d6" hotkey="1"/>
    <Label value="motorcycle" background="#eb6834" hotkey="2"/>
    <Label value="bus" background="#1baf7a" hotkey="3"/>
    <Label value="truck" background="#eda100" hotkey="4"/>
    <Label value="{CANDIDATE}" background="#e34948" hotkey="5"/>
  </RectangleLabels>
</View>
"""


def val_rows() -> pd.DataFrame:
    split = pd.read_csv(paths.SPLIT_FILE, dtype={"camera_id": str})
    rows = split[split["split"] == "val"].copy()
    # 每个相机内：白天、夜间交替排，前 5 张进第一批
    ordered = []
    for _camera, group in rows.groupby("camera_id"):
        day = group[group["period"] == "day"].sort_values("captured_at_utc").to_dict("records")
        night = group[group["period"] == "night"].sort_values("captured_at_utc").to_dict("records")
        mixed = [x for pair in zip(day, night) for x in pair] + day[len(night):] + night[len(day):]
        for rank, row in enumerate(mixed):
            ordered.append({**row, "batch": "A" if rank < 5 else "B", "rank": rank})
    frame = pd.DataFrame(ordered).sort_values(["batch", "rank", "camera_id"]).reset_index(drop=True)
    frame["order"] = np.arange(1, len(frame) + 1)
    return frame


def yolo_to_ls(cls_name: str, cx: float, cy: float, w: float, h: float, width: int, height: int, score: float | None = None) -> dict:
    cx, cy, w, h = (float(v) for v in (cx, cy, w, h))  # numpy float32 不能直接写 JSON
    result = {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": width,
        "original_height": height,
        "value": {"x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100, "width": w * 100, "height": h * 100, "rotation": 0, "rectanglelabels": [cls_name]},
    }
    if score is not None:
        result["score"] = round(float(score), 4)
    return result


def prepare(args) -> None:
    from PIL import Image
    from ultralytics import YOLO

    WORK.mkdir(parents=True, exist_ok=True)
    rows = val_rows()
    teacher = YOLO(str(paths.TEACHER_WEIGHTS))
    class_map = load_class_map(teacher.names)
    tasks, n_pre, n_cand = [], 0, 0
    for row in rows.itertuples():
        image_path = paths.LABELED / "images" / "val" / row.image
        with Image.open(image_path) as image:
            width, height = image.size
        results = []
        label_file = (paths.LABELED / "labels" / "val" / row.image).with_suffix(".txt")
        pre = np.loadtxt(label_file, ndmin=2) if label_file.stat().st_size else np.zeros((0, 5))
        for c, cx, cy, w, h in pre:
            results.append(yolo_to_ls(VEHICLE_CLASSES[int(c)], cx, cy, w, h, width, height))
        n_pre += len(pre)
        low = teacher.predict(str(image_path), imgsz=1920, conf=CANDIDATE_CONF[0], classes=sorted(class_map), device=pick_device(), verbose=False)[0].boxes
        keep = (low.conf < CANDIDATE_CONF[1]).cpu().numpy()
        cand_xyxy = low.xyxy.cpu().numpy()[keep]
        cand_conf = low.conf.cpu().numpy()[keep]
        if len(cand_xyxy) and len(pre):
            overlap = iou_matrix(cand_xyxy, to_xyxy(pre[:, 1:5], width, height)).max(axis=1)
            cand_xyxy, cand_conf = cand_xyxy[overlap <= 0.5], cand_conf[overlap <= 0.5]
        for (x1, y1, x2, y2), score in zip(cand_xyxy, cand_conf):
            results.append(yolo_to_ls(CANDIDATE, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height, width, height, score))
        n_cand += len(cand_xyxy)
        tasks.append(
            {
                "data": {
                    "image": f"http://localhost:{PORT}/{row.image}",
                    "image_name": row.image,
                    "camera_id": row.camera_id,
                    "period": row.period,
                    "captured_at_utc": row.captured_at_utc,
                    "batch": row.batch,
                    "order": int(row.order),
                },
                "predictions": [{"model_version": "yolo26x-prelabel+candidates", "result": results}],
            }
        )
    (WORK / "lta_val_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    (WORK / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    rows[["order", "batch", "image", "camera_id", "period", "captured_at_utc"]].to_csv(WORK / "order.csv", index=False)
    print(f"{len(tasks)} 个任务：预标注 {n_pre} 个框，候选 {n_cand} 个 → {WORK}")


def serve(args) -> None:
    """只读、只听 localhost 的图片服务，并允许跨域（Label Studio 在另一个端口）。"""
    import functools
    import http.server

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def do_PUT(self):  # noqa: N802
            self.send_error(405)

        do_POST = do_DELETE = do_PUT

    directory = paths.LABELED / "images" / "val"
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=str(directory)))
    print(f"图片服务 http://localhost:{PORT}/ ← {directory}（Ctrl+C 结束）", flush=True)
    server.serve_forever()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(args) -> None:
    from PIL import Image

    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    by_name, unresolved = {}, []
    for task in export:
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        name = task["data"]["image_name"]
        boxes = []
        for item in sorted(annotations, key=lambda a: a.get("updated_at", ""))[-1]["result"]:
            if item.get("type") != "rectanglelabels":
                continue
            value = item["value"]
            label = value["rectanglelabels"][0]
            if label == CANDIDATE:
                unresolved.append(name)
                continue
            x, y, w, h = (value[k] / 100 for k in ("x", "y", "width", "height"))
            boxes.append((VEHICLE_CLASSES.index(label), x + w / 2, y + h / 2, w, h))
        by_name[name] = (task["data"], boxes)
    if unresolved:
        raise SystemExit(f"{len(set(unresolved))} 张图里还有未处理的 candidate 框，例如 {sorted(set(unresolved))[:3]}；改成四类之一或删除后再导出")
    if not by_name:
        raise SystemExit("导出里没有任何已提交的标注")

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} 已存在（冻结集不覆盖）；确认要重建加 --overwrite")
        shutil.rmtree(out)
    (out / "labels" / "val").mkdir(parents=True)
    if args.annotations_only:
        (out / "annotations").mkdir(parents=True)
    else:
        (out / "images" / "val").mkdir(parents=True)

    rows: list[dict] = []
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    jsonl_rows: list[dict] = []
    counts = defaultdict(int)
    annotation_id = 0
    for image_id, (name, (data, boxes)) in enumerate(sorted(by_name.items()), start=1):
        if args.annotations_only:
            source = paths.LTA_DATA / "week2" / data["camera_id"] / name
        else:
            source = paths.LABELED / "images" / "val" / name
        if not source.is_file():
            raise SystemExit(f"找不到源图：{source}")
        with Image.open(source) as image:
            width, height = image.size
        if args.annotations_only:
            sha256 = _file_sha256(source)
        else:
            shutil.copy2(source, out / "images" / "val" / name)
            sha256 = _file_sha256(source)

        yolo_lines, jsonl_boxes = [], []
        for cls, cx, cy, w, h in boxes:
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            x1 = max(0.0, (cx - w / 2) * width)
            y1 = max(0.0, (cy - h / 2) * height)
            bw = min(width - x1, w * width)
            bh = min(height - y1, h * height)
            annotation_id += 1
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            jsonl_boxes.append({
                "category": VEHICLE_CLASSES[cls],
                "category_id": cls + 1,
                "bbox_xywh": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
            })
            counts[VEHICLE_CLASSES[cls]] += 1

        label_path = (out / "labels" / "val" / name).with_suffix(".txt")
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        image_path = (
            f"week2/{data['camera_id']}/{name}"
            if args.annotations_only
            else str(out / "images" / "val" / name)
        )
        rows.append({
            "image": name,
            "image_path": image_path,
            "split": "val",
            "camera_id": data["camera_id"],
            "period": data["period"],
            "captured_at_utc": data["captured_at_utc"],
            "batch": data["batch"],
            "order": data["order"],
            "width": width,
            "height": height,
            "sha256": sha256,
            "n_boxes": len(boxes),
        })
        coco_images.append({"id": image_id, "file_name": image_path, "width": width, "height": height, "sha256": sha256})
        jsonl_rows.append({
            "image": name,
            "image_path": image_path,
            "camera_id": data["camera_id"],
            "period": data["period"],
            "captured_at_utc": data["captured_at_utc"],
            "width": width,
            "height": height,
            "sha256": sha256,
            "boxes": jsonl_boxes,
        })

    split = pd.DataFrame(rows)
    split.to_csv(out / "split.csv", index=False)
    split.to_csv(out / "images.csv", index=False)
    (out / "val.txt").write_text("\n".join(r["image_path"] for r in rows) + "\n", encoding="utf-8")

    if args.annotations_only:
        coco = {
            "info": {
                "description": "LTA week-2 human-corrected vehicle boxes",
                "version": "1.0",
                "year": 2026,
            },
            "licenses": [
                {"id": 1, "name": "Singapore Open Data Licence 1.0", "url": "https://data.gov.sg/open-data-licence"},
                {"id": 2, "name": "CC BY 4.0 (annotations)", "url": "https://creativecommons.org/licenses/by/4.0/"},
            ],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {"id": index + 1, "name": name, "supercategory": "vehicle"}
                for index, name in enumerate(VEHICLE_CLASSES)
            ],
        }
        (out / "annotations" / "instances.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "annotations" / "annotations.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in jsonl_rows) + "\n", encoding="utf-8"
        )

    if args.excluded:
        excluded = pd.read_csv(args.excluded, encoding="utf-8-sig")
        excluded["reason"] = "too_dense_or_dark"
        excluded.to_csv(out / "excluded.csv", index=False)

    digest = hashlib.sha256()
    for label in sorted((out / "labels" / "val").glob("*.txt")):
        digest.update(label.name.encode() + b"\0" + label.read_bytes())
    report = {
        "name": "lta_human_v1",
        "version": "1.0",
        "labels_are": "human-corrected from yolo26x pre-labels (teacher-assisted, not annotated from scratch)",
        "annotator": args.annotator,
        "source_images": "LTA week2 validation frames; images are not redistributed in this package",
        "image_source": "https://huggingface.co/datasets/LOVEIII486/isy5002-lta-traffic-images",
        "license": {
            "annotations": "CC-BY-4.0",
            "images": "Singapore Open Data Licence 1.0",
        },
        "classes": list(VEHICLE_CLASSES),
        "images": len(rows),
        "boxes": dict(counts),
        "images_per_batch": split["batch"].value_counts().to_dict(),
        "images_per_camera": split["camera_id"].value_counts().to_dict(),
        "images_per_period": split["period"].value_counts().to_dict(),
        "excluded_images": int(len(pd.read_csv(out / "excluded.csv"))) if (out / "excluded.csv").is_file() else 0,
        "manifest_sha256_16": {"val": hashlib.sha256((out / "val.txt").read_bytes()).hexdigest()[:16]},
        "labels_sha256_16": {"val": digest.hexdigest()[:16]},
        "export_file_sha256_16": hashlib.sha256(Path(args.export).read_bytes()).hexdigest()[:16],
    }
    (out / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"收回 {len(rows)} 张、{sum(counts.values())} 个框 {dict(counts)} → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LTA 验证图人工修正")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("serve")
    p = sub.add_parser("collect")
    p.add_argument("export", help="Label Studio 导出的 JSON（Export → JSON）")
    p.add_argument("--annotator", required=True, help="标注人姓名，写进 dataset.json")
    p.add_argument("--out", default=str(HUMAN), help="输出目录")
    p.add_argument("--annotations-only", action="store_true", help="只写标注与清单，不复制图片（用于仓库/Hugging Face）")
    p.add_argument("--excluded", help="删除的图片 CSV（可选）")
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "serve": serve, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
