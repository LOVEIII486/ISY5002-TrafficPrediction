"""检测页的数据访问：只读 results/detection/ 下的产物，不 import 检测模块。

按仓库约定，模块之间只通过产物通信。产物由 `python -m src.detection.run_detect` 生成；
没有产物时各函数返回空结果，页面显示空状态而不是报错。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "detection"
LTA_DATA = ROOT / "datasets" / "lta"

DETECTIONS = RESULTS / "detections.csv"
CAMERAS = RESULTS / "cameras.csv"
FRAMES = RESULTS / "frames.csv"
RUN_INFO = RESULTS / "run.json"

# 与 src/detection/run_detect.py 保持一致。各自持有一份，不跨模块 import。
VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

# 展示用色，顺序与 console.js 里的 CLASS_COLOR 一致。
TIMELINE_POINTS = 240


@lru_cache(maxsize=1)
def run_info() -> dict[str, Any]:
    if not RUN_INFO.is_file():
        return {}
    return json.loads(RUN_INFO.read_text(encoding="utf-8"))


def available() -> bool:
    return CAMERAS.is_file() and FRAMES.is_file()


@lru_cache(maxsize=1)
def cameras() -> pd.DataFrame:
    if not CAMERAS.is_file():
        return pd.DataFrame(columns=["camera_id", "road_segment", "direction", "frames", "lat", "lon"])
    return pd.read_csv(CAMERAS, dtype={"camera_id": str})


@lru_cache(maxsize=1)
def frames() -> pd.DataFrame:
    if not FRAMES.is_file():
        return pd.DataFrame(columns=["image", "camera_id", "captured_at_utc"])
    frame = pd.read_csv(FRAMES, dtype={"camera_id": str})
    frame["captured_at"] = pd.to_datetime(frame["captured_at_utc"], utc=True, format="ISO8601")
    return frame


@lru_cache(maxsize=1)
def detections() -> pd.DataFrame:
    if not DETECTIONS.is_file():
        return pd.DataFrame(columns=["camera_id", "captured_at_utc", "image", "cls", "conf"])
    return pd.read_csv(DETECTIONS, dtype={"camera_id": str})


@lru_cache(maxsize=1)
def counts() -> pd.DataFrame:
    """每个已处理画面一行的分车型计数。

    以 frames 为左表：处理过但一个目标都没检出的画面要留成 0，不能丢，那会在时间序列上
    留下假缺口。
    """
    frame = frames()
    if frame.empty:
        return frame
    tables = detections()
    if tables.empty:
        pivot = pd.DataFrame(index=frame["image"])
    else:
        pivot = tables.pivot_table(index="image", columns="cls", values="conf", aggfunc="count")
    joined = frame.set_index("image").join(pivot).fillna(0)
    for name in VEHICLE_CLASSES:
        if name not in joined:
            joined[name] = 0
    joined[[*VEHICLE_CLASSES]] = joined[list(VEHICLE_CLASSES)].astype(int)
    return joined.reset_index()


def _pack(rows: pd.DataFrame, label: str = "step") -> dict[str, Any]:
    return {
        "steps": [int(value) for value in rows[label]],
        "classes": {name: [int(v) for v in rows[name]] for name in VEHICLE_CLASSES},
    }


@lru_cache(maxsize=32)
def series(camera_id: str, points: int = TIMELINE_POINTS) -> dict[str, Any]:
    """单台相机的分车型计数序列，按时间等距抽样到 points 个点。"""
    table = counts()
    table = table[table["camera_id"] == camera_id].sort_values("captured_at")
    if table.empty:
        return {"camera": camera_id, "class_names": list(VEHICLE_CLASSES), "current": {"steps": [], "classes": {}}, "totals": {}}

    step = max(1, len(table) // points)
    sampled = table.iloc[::step].reset_index(drop=True)
    sampled["step"] = range(len(sampled))
    return {
        "camera": camera_id,
        "class_names": list(VEHICLE_CLASSES),
        "current": _pack(sampled),
        # 时间轴上的每个点都要能对应回一帧具体画面，前端才能点着看。
        "images": [str(value) for value in sampled["image"]],
        "timestamps": [value.isoformat() for value in sampled["captured_at"]],
        "totals": {name: int(table[name].sum()) for name in VEHICLE_CLASSES},
    }


@lru_cache(maxsize=1)
def rhythm(timezone: str = "Asia/Singapore") -> dict[str, Any]:
    """全岛分车型的日内节律：把所有相机按当地时刻的小时归并。

    相机的采样时刻互不对齐（每台约 11 分钟一次），所以直接按时刻求和会得到一堆错开的点，
    按小时归并才有可比性。
    """
    table = counts()
    if table.empty:
        return {"hours": [], "classes": {}}
    local = table["captured_at"].dt.tz_convert(timezone)
    table = table.assign(hour=local.dt.hour)
    grouped = table.groupby("hour")[list(VEHICLE_CLASSES)].mean()
    grouped = grouped.reindex(range(24), fill_value=0.0)
    return {
        "hours": list(range(24)),
        "classes": {name: [round(float(v), 2) for v in grouped[name]] for name in VEHICLE_CLASSES},
    }


@lru_cache(maxsize=1)
def _image_index() -> dict[str, Path]:
    """文件名 -> 绝对路径。数据集里混着 macOS 的 `._xxx.jpg` 资源叉，必须排除。"""
    index: dict[str, Path] = {}
    for path in LTA_DATA.rglob("*.jpg"):
        if not path.name.startswith("._"):
            index[path.name] = path
    return index


def image_path(name: str) -> Path | None:
    """按文件名取图片。用精确查表而不是拼路径，天然没有目录穿越问题。"""
    return _image_index().get(Path(name).name)


def frame_detail(image: str) -> dict[str, Any]:
    """单帧的检测框。"""
    tables = detections()
    rows = tables[tables["image"] == image]
    meta = frames()
    row = meta[meta["image"] == image]
    return {
        "image": image,
        "camera": str(row["camera_id"].iloc[0]) if not row.empty else "",
        "captured_at": row["captured_at_utc"].iloc[0] if not row.empty else "",
        "boxes": [
            {
                "cls": item.cls,
                "conf": round(float(item.conf), 3),
                "x1": float(item.x1),
                "y1": float(item.y1),
                "x2": float(item.x2),
                "y2": float(item.y2),
            }
            for item in rows.itertuples()
        ],
    }


@lru_cache(maxsize=1)
def latest_frames() -> list[dict[str, Any]]:
    """每台相机最近一次**有检出**的画面，供总览页的画面墙。

    取「最后一帧」的话八张全是夜间图，数据集止于 SGT 23:50，而夜间检测本就大面积失效，
    画面墙上会出现好几张写着「0 辆」的图，它们是漏检，不代表真的没车，看着像系统坏了。
    """
    table = counts()
    if table.empty:
        return []
    columns = list(VEHICLE_CLASSES)
    rows = []
    for camera_id, group in table.sort_values("captured_at").groupby("camera_id"):
        group = group.assign(total=group[columns].sum(axis=1))
        hits = group[group["total"] > 0]
        last = (hits if not hits.empty else group).iloc[-1]
        rows.append(
            {
                "camera": str(camera_id),
                "image": str(last["image"]),
                "url": f"/lta/{last['image']}",
                "captured_at": last["captured_at"].isoformat(),
                "detections": int(last["total"]),
            }
        )
    return sorted(rows, key=lambda row: row["camera"])


def coverage() -> dict[str, Any]:
    """已处理多少帧，用来在界面上如实说明数据完整度。"""
    info = run_info()
    return {
        "frames_done": int(len(frames())),
        "frames_total": int(info.get("frames_total", 0)),
        "imgsz": info.get("imgsz"),
        "confidence": info.get("confidence"),
        "model": info.get("model"),
    }
