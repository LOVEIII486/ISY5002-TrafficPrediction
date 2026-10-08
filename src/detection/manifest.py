"""LTA 数据集的观测记录、图片定位、相机元数据。

清单有两个来源。`manifest.csv` 是采集脚本的副产物，**不在本仓库、也不在 HF 数据集里**
（那边只有图片），所以缺清单时直接从文件名重建，文件名里编码了同样的东西：

    week1/2701/20260913T124023Z_2701_1240_97acd1b15491.jpg
    └─┬─┘└┬─┘ └─────┬──────┘
     week 相机   captured_at (UTC)

有清单时以清单为准：它的一行是一次**请求**而非一帧画面，只有 `status == downloaded`
的行对应真实文件，其余（stale / duplicate / missing / error）没有图片。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import paths

# 清单里的 image_path 是采集脚本当时的相对路径（形如 images/2701/xxx.jpg），
# 数据集搬到本仓库后目录层级变了，所以只取文件名，再到各周目录下找。
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# 采集配置里的相机 -> （路段, 朝向）。那份配置同样不在仓库里，而这 8 个相机是固定的，
# 直接内联，没有它 demo 的路段筛选会空掉。
CAMERA_META = {
    "2701": ("causeway", "Towards Johor"),
    "2702": ("causeway", "Towards BKE"),
    "2704": ("causeway", "Towards Checkpoint"),
    "4703": ("second_link", "Second Link view"),
    "4712": ("second_link", "Towards Tuas"),
    "4713": ("second_link", "Checkpoint view"),
    "4798": ("sentosa_gateway", "Towards Telok Blangah"),
    "4799": ("sentosa_gateway", "Towards Sentosa"),
}

IMAGE_NAME = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z_(\d+)_\d{4}_[0-9a-f]+$", re.I)


@lru_cache(maxsize=1)
def _image_index() -> dict[str, Path]:
    """文件名 -> 绝对路径。数据集里混着 macOS 的 `._xxx.jpg` 资源叉，必须排除。"""
    index: dict[str, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for path in paths.LTA_DATA.rglob(f"*{suffix}"):
            if path.name.startswith("._"):
                continue
            index[path.name] = path
    if not index:
        raise SystemExit(
            f"{paths.LTA_DATA} 下没有图片。数据集是外部获取的，不在版本控制里；\n"
            "    确认 datasets/lta/<week>/<camera_id>/*.jpg 存在。"
        )
    return index


@lru_cache(maxsize=1)
def camera_coordinates() -> dict[str, tuple[float, float]]:
    """相机经纬度。数据集本身不含坐标，取自仓库里那份 LTA 接口样本响应。"""
    if not paths.CAMERA_COORDINATES.is_file():
        return {}
    payload = json.loads(paths.CAMERA_COORDINATES.read_text(encoding="utf-8"))
    return {
        str(item["CameraID"]): (float(item["Latitude"]), float(item["Longitude"]))
        for item in payload["value"]
    }


def _from_manifest(manifest: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest, dtype=str)
    frame = frame[frame["status"] == "downloaded"].copy()

    index = _image_index()
    frame["image_path"] = frame["image_path"].map(lambda value: index.get(Path(value).name))
    missing = int(frame["image_path"].isna().sum())
    if missing:
        # 清单说下载成功、文件却不在，说明数据集不完整；这会让计数偏低，必须显式暴露。
        raise SystemExit(f"清单里有 {missing} 条 downloaded 记录找不到图片文件；数据集不完整。")
    return frame


def _from_filenames() -> pd.DataFrame:
    """按文件名重建。相机取自目录名，采集时刻取自文件名开头的时间戳。"""
    rows = []
    for name, path in _image_index().items():
        match = IMAGE_NAME.match(Path(name).stem)
        if match is None:
            continue
        year, month, day, hour, minute, second, camera = match.groups()
        road, direction = CAMERA_META.get(camera, ("", ""))
        rows.append(
            {
                "camera_id": camera,
                "road_segment": road,
                "direction": direction,
                "image_path": path,
                "captured_at_utc": f"{year}-{month}-{day}T{hour}:{minute}:{second}Z",
            }
        )
    if not rows:
        raise SystemExit(
            f"{paths.LTA_DATA} 下没有符合命名规律的图片。文件名要形如\n"
            "    20260913T124023Z_2701_1240_97acd1b15491.jpg\n"
            "数据集用 `python scripts/fetch_data.py lta` 获取。"
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def observations() -> pd.DataFrame:
    """已采集到的观测，列含 camera_id / road_segment / direction / captured_at / image_path。"""
    manifest = next(paths.LTA_DATA.rglob("manifest.csv"), None)
    frame = _from_manifest(manifest) if manifest is not None else _from_filenames()

    frame["captured_at"] = pd.to_datetime(frame["captured_at_utc"], utc=True, format="ISO8601")
    return frame.sort_values(["camera_id", "captured_at"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def frames() -> pd.DataFrame:
    """去重后的画面。采集窗口之间有重叠，同一帧会被两个窗口各记一次（README 记了 80 条），
    不去重会让那一时刻的计数翻倍。"""
    return observations().drop_duplicates(subset="image_path").reset_index(drop=True)


def cameras() -> pd.DataFrame:
    """相机清单：位置、方向、观测数与坐标。"""
    frame = observations()
    coordinates = camera_coordinates()
    rows = []
    for camera_id, group in frame.groupby("camera_id"):
        latitude, longitude = coordinates.get(str(camera_id), (float("nan"), float("nan")))
        rows.append(
            {
                "camera_id": str(camera_id),
                "road_segment": group["road_segment"].iloc[0],
                "direction": group["direction"].iloc[0],
                "frames": int(len(group)),
                "first": group["captured_at"].min(),
                "last": group["captured_at"].max(),
                "lat": latitude,
                "lon": longitude,
            }
        )
    return pd.DataFrame(rows).sort_values("camera_id").reset_index(drop=True)
