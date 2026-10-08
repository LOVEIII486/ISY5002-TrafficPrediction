"""对 LTA 相机图跑 YOLO，产物写入 results/detection/。

    python -m src.detection.run_detect
    python -m src.detection.run_detect --limit 64 --imgsz 1280
    python -m src.detection.run_detect --weights results/<你的实验>/weights/last.pt

产物三份：`frames.csv`（处理过的画面）、`detections.csv`（逐目标结果）、`cameras.csv`
（相机清单与坐标）。支持断点续跑，已在 frames.csv 里的图片会被跳过，中途 Ctrl+C 后
原样重跑即可。换了权重或推理参数则拒绝续跑，见 `_guard_config`。

换用自己微调的权重时，**通常还要写 `classes.yaml`**（模型的类名 -> car/bus/truck/motorcycle），
否则类名对不上会静默地一个框都不输出。见同目录 README.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import manifest, paths

# 管线统计的车辆类；proposal 要求的就是这四类。自建数据集的类名未必叫这些，
# 靠 classes.yaml 映射过来。
VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

DETECTION_COLUMNS = ("camera_id", "captured_at_utc", "image", "cls", "conf", "x1", "y1", "x2", "y2")
FRAME_COLUMNS = ("image", "camera_id", "captured_at_utc")


def load_class_map(model_names: dict) -> dict[int, str]:
    """模型类索引 -> 管线统计的车辆类。只保留映射得上的类。

    按类名匹配、大小写不敏感，所以 COCO 权重开箱即用。自建数据集的类名靠
    classes.yaml 显式映射（左键是权重里的类名，右值必须是 VEHICLE_CLASSES 之一）。
    """
    aliases: dict[str, str] = {}
    if paths.CLASSES_FILE.is_file():
        import yaml

        raw = yaml.safe_load(paths.CLASSES_FILE.read_text(encoding="utf-8")) or {}
        aliases = {str(key).lower(): str(value).lower() for key, value in raw.items()}

    mapping = {}
    for index, name in model_names.items():
        target = aliases.get(str(name).lower(), str(name).lower())
        if target in VEHICLE_CLASSES:
            mapping[int(index)] = target
    return mapping


def ensure_weights(weights: Path) -> Path:
    """权重缺失时从 Ultralytics 官方发布下载。只对默认权重生效，自训的权重不在官方发布里。"""
    if weights.is_file():
        return weights
    if weights != paths.WEIGHTS:
        raise SystemExit(
            f"权重不存在：{weights}\n"
            "自训权重没有自动下载的地方，检查路径是否写对。"
        )
    from ultralytics.utils.downloads import attempt_download_asset

    paths.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载权重 {paths.WEIGHTS.name} → {paths.WEIGHTS.parent}")
    attempt_download_asset(str(paths.WEIGHTS))
    if not paths.WEIGHTS.is_file():
        raise SystemExit(f"权重下载失败：{paths.WEIGHTS}")
    return paths.WEIGHTS


def _previous_config() -> dict | None:
    if not paths.RUN_INFO.is_file():
        return None
    return json.loads(paths.RUN_INFO.read_text(encoding="utf-8"))


def _already_done() -> set[str]:
    if not paths.FRAMES.is_file():
        return set()
    with paths.FRAMES.open(encoding="utf-8", newline="") as handle:
        return {row["image"] for row in csv.DictReader(handle)}


def _differs(previous: object, current: object) -> bool:
    """类名表按集合比：旧产物记的是元组顺序，新代码记的是排序后的列表，直接比会把
    沿用旧产物的人误判成「换了配置」而拦下来。"""
    if isinstance(current, list) and isinstance(previous, list):
        return sorted(previous) != sorted(current)
    return previous != current


def _guard_config(
    previous: dict | None, weights: Path, class_map: dict[int, str], imgsz: int, confidence: float
) -> None:
    """换配置重跑不能续跑，那样会把两套配置的结果混进同一个 CSV，而且不留痕迹。

    换权重也算换配置：两个模型同名类的含义未必一样，混进同一份 CSV 后下游分不出来。
    """
    if previous is None:
        return
    current = {
        "model": weights.name,
        "classes": sorted(set(class_map.values())),
        "imgsz": imgsz,
        "confidence": confidence,
    }
    changed = {key: value for key, value in current.items() if _differs(previous.get(key), value)}
    if not changed:
        return
    detail = "".join(
        f"    {key}: 已有 {previous.get(key)!r}，本次 {value!r}\n" for key, value in changed.items()
    )
    raise SystemExit(
        "已有结果是别的配置跑的，续跑会把两套配置混在一起：\n"
        + detail
        + "换个配置请先清空产物目录：\n"
        f"    rm {paths.DETECTIONS} {paths.FRAMES} {paths.RUN_INFO}"
    )


def _box_rows(camera_id: str, captured_at: str, image: str, name: str, box, shape):
    """坐标一律归一化到 0–1，换个分辨率展示不必重算。"""
    height, width = shape
    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
    return (
        camera_id,
        captured_at,
        image,
        name,
        round(float(box.conf.item()), 4),
        round(x1 / width, 5),
        round(y1 / height, 5),
        round(x2 / width, 5),
        round(y2 / height, 5),
    )


def detect(weights: Path, limit: int | None, imgsz: int, confidence: float, batch: int) -> None:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            f"缺少依赖（{error.name}）。先执行：\n    pip install -r requirements.txt"
        ) from None

    weights = ensure_weights(weights)
    model = YOLO(str(weights))
    class_map = load_class_map(model.names)
    if not class_map:
        names = sorted(str(name) for name in model.names.values())
        shown = "、".join(names[:10]) + (f" …（共 {len(names)} 个）" if len(names) > 10 else "")
        raise SystemExit(
            f"权重的类名一个都映射不到车辆类：{shown}\n"
            f"管线只统计 {list(VEHICLE_CLASSES)}。微调过的权重请把映射写进\n"
            f"    {paths.CLASSES_FILE.relative_to(paths.ROOT)}\n"
            "格式（左边是权重里的类名，右边必须是上面四类之一）：\n"
            "    Truck: truck\n"
            "    Bus: bus\n"
            "    Car: car\n"
            "不写的话每个框都会被丢掉，产出空的 detections.csv 而不报错。"
        )

    _guard_config(_previous_config(), weights, class_map, imgsz, confidence)

    frame = manifest.frames()
    done = _already_done()
    pending = frame[~frame["image_path"].map(lambda path: Path(path).name).isin(done)]
    if limit:
        pending = pending.head(limit)

    # 相机清单与检测无关，但下游只认产物、不 import 本模块，所以一并落盘。
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    manifest.cameras().to_csv(paths.CAMERAS, index=False)

    print(f"[1/3] 画面 {len(frame)} 张，已完成 {len(done)}，本次待处理 {len(pending)}")
    if pending.empty:
        print("      没有待处理的图片，结束")
        return

    # 必须显式指定 device：ultralytics 默认把模型放在 CPU 上，实测慢 20 倍以上。
    device = 0 if torch.cuda.is_available() else "cpu"
    print(
        f"[2/3] 权重 {weights.name} · 车辆类 {sorted(set(class_map.values()))} · "
        f"device={device} · imgsz={imgsz} · conf={confidence}"
    )

    started = time.time()
    processed = 0
    hits = 0
    records = pending.to_dict("records")

    # 用文件大小判断是否要写表头：追加模式下 tell() 的行为依平台而定，不可靠。
    fresh_detections = not paths.DETECTIONS.exists() or paths.DETECTIONS.stat().st_size == 0
    fresh_frames = not paths.FRAMES.exists() or paths.FRAMES.stat().st_size == 0

    with (
        paths.DETECTIONS.open("a", encoding="utf-8", newline="") as handle,
        paths.FRAMES.open("a", encoding="utf-8", newline="") as frames_handle,
    ):
        writer = csv.writer(handle)
        frame_writer = csv.writer(frames_handle)
        if fresh_detections:
            writer.writerow(DETECTION_COLUMNS)
        if fresh_frames:
            frame_writer.writerow(FRAME_COLUMNS)

        for offset in range(0, len(records), batch):
            chunk = records[offset : offset + batch]
            results = model.predict(
                [record["image_path"] for record in chunk],
                imgsz=imgsz,
                conf=confidence,
                device=device,
                verbose=False,
            )
            for record, result in zip(chunk, results):
                image = Path(record["image_path"]).name
                captured_at = record["captured_at"].isoformat()
                # 先把帧记下来：这一帧可能一个目标都没有，仍算处理过。
                frame_writer.writerow((image, record["camera_id"], captured_at))
                for box in result.boxes:
                    label = class_map.get(int(box.cls.item()))
                    if label is None:
                        continue
                    writer.writerow(
                        _box_rows(record["camera_id"], captured_at, image, label, box, result.orig_shape)
                    )
                    hits += 1
            handle.flush()
            frames_handle.flush()

            processed += len(chunk)
            elapsed = time.time() - started
            rate = processed / elapsed
            remaining = (len(records) - processed) / rate
            print(
                f"      {processed}/{len(records)}  {rate:.2f} 张/秒  已检出 {hits}  "
                f"剩余约 {remaining/60:.1f} 分钟",
                flush=True,
            )

    elapsed = time.time() - started
    paths.RUN_INFO.write_text(
        json.dumps(
            {
                "model": weights.name,
                "model_path": str(weights),
                "imgsz": imgsz,
                "confidence": confidence,
                "device": str(device),
                # 记映射结果而不是模型原始类名：下游只认这四类，换了权重也得能对上。
                "classes": sorted(set(class_map.values())),
                "class_map": {str(index): name for index, name in sorted(class_map.items())},
                "frames_total": int(len(frame)),
                "frames_done": int(len(done) + processed),
                "detections_this_run": hits,
                "seconds": round(elapsed, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[3/3] 完成：{processed} 张 / {hits} 个目标 / {elapsed/60:.1f} 分钟 -> {paths.DETECTIONS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LTA 相机图的目标检测")
    parser.add_argument(
        "--weights",
        type=Path,
        default=paths.WEIGHTS,
        help="权重路径。换用自己微调的模型时，通常还要写 src/detection/classes.yaml",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张，用于试跑")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    detect(args.weights, args.limit, args.imgsz, args.conf, args.batch)


if __name__ == "__main__":
    main()
