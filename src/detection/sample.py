"""从 LTA 数据集分层抽样出待标注的画面，只下载抽中的图片。

    python -m src.detection.sample                       # 默认：week1 训练 240 张，week2 验证 80 张
    python -m src.detection.sample --train-per-camera 40 --val-per-camera 15

划分按**周**切，不随机：相邻两帧只差 10 分钟、画面几乎一样，随机划分会让同一辆车
同时出现在训练集与验证集里，指标虚高。week1 与 week2 时间不重叠，相机集合相同。

每个相机内再按白天 / 夜间分层（夜间召回率最差，必须有足够样本），层内按时间等距
取帧，让样本铺满整周而不是挤在某一天。产物 `datasets/lta_yolo/split.csv` 记录每张
图属于哪一层、哪个划分，同一种子重跑结果相同。
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict

import pandas as pd

from . import paths

REPO_ID = "LOVEIII486/isy5002-lta-traffic-images"

# 数据集 README 说文件名中段的 HHMM 是新加坡时间，**实际是 UTC**：全部 15,316 个文件里
# 它都等于开头时间戳的 UTC 时分。白天 / 夜间必须按新加坡时间（UTC+8）判断。
NAME = re.compile(r"(?P<week>week\d+)/(?P<camera>\d+)/(?P<stamp>\d{8}T\d{6}Z)_")
SGT_OFFSET_HOURS = 8
DAY_HOURS = range(7, 19)  # 新加坡时间 07:00–18:59 算白天

# 夜间帧占比。夜间是检测器最弱的时段，比例略高于其自然占比的一半，又不至于喧宾夺主。
NIGHT_SHARE = 0.4


def period_of(stamp: str) -> str:
    """由 UTC 时间戳（20260913T124023Z）判断新加坡时间的白天 / 夜间。"""
    sgt_hour = (int(stamp[9:11]) + SGT_OFFSET_HOURS) % 24
    return "day" if sgt_hour in DAY_HOURS else "night"


def list_images() -> pd.DataFrame:
    """列出数据集里的全部图片（只读文件列表，不下载）。"""
    from huggingface_hub import list_repo_files

    rows = []
    for name in list_repo_files(REPO_ID, repo_type="dataset"):
        if not name.endswith(".jpg") or "/._" in name:  # 排除 macOS 资源叉
            continue
        match = NAME.search(name)
        if match is None:
            continue
        stamp = match["stamp"]
        rows.append(
            {
                "file": name,
                "image": name.rsplit("/", 1)[1],
                "week": match["week"],
                "camera_id": match["camera"],
                "captured_at_utc": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z",
                "period": period_of(stamp),
            }
        )
    return pd.DataFrame(rows)


def evenly(items: list, count: int, rng: random.Random) -> list:
    """从按时间排好的列表里等距取 count 个，起点随机。"""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    step = len(items) / count
    offset = rng.random() * step
    return [items[int(offset + i * step)] for i in range(count)]


def stratify(images: pd.DataFrame, week: str, per_camera: int, split: str, rng: random.Random) -> list[dict]:
    """某一周内按 (相机, 白天/夜间) 分层抽样。"""
    night = round(per_camera * NIGHT_SHARE)
    quota = {"day": per_camera - night, "night": night}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in images[images["week"] == week].sort_values("captured_at_utc").to_dict("records"):
        groups[(row["camera_id"], row["period"])].append(row)

    chosen = []
    for (camera_id, period), rows in sorted(groups.items()):
        for row in evenly(rows, quota[period], rng):
            chosen.append({**row, "split": split})
    return chosen


def download(files: list[str]) -> None:
    """只下载抽中的图片，落到与 fetch_data.py 相同的目录结构下。"""
    from huggingface_hub import hf_hub_download

    pending = [name for name in files if not (paths.LTA_DATA / name).is_file()]
    print(f"下载 {len(pending)} 张（已存在 {len(files) - len(pending)} 张）→ {paths.LTA_DATA}")
    for index, name in enumerate(pending, start=1):
        hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=paths.LTA_DATA)
        if index % 50 == 0 or index == len(pending):
            print(f"    {index}/{len(pending)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="分层抽样待标注画面")
    parser.add_argument("--train-week", default="week1")
    parser.add_argument("--val-week", default="week2")
    parser.add_argument("--train-per-camera", type=int, default=30)
    parser.add_argument("--val-per-camera", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true", help="只写清单，不下载图片")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images = list_images()
    chosen = stratify(images, args.train_week, args.train_per_camera, "train", rng)
    chosen += stratify(images, args.val_week, args.val_per_camera, "val", rng)
    split = pd.DataFrame(chosen)

    paths.LABELED.mkdir(parents=True, exist_ok=True)
    split.to_csv(paths.SPLIT_FILE, index=False)
    summary = split.groupby(["split", "period"]).size().unstack(fill_value=0)
    print(f"数据集共 {len(images)} 张，抽中 {len(split)} 张 → {paths.SPLIT_FILE}")
    print(summary.to_string())

    if not args.no_download:
        download(split["file"].tolist())


if __name__ == "__main__":
    main()
