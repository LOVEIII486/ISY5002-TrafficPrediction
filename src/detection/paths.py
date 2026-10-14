"""检测模块的路径。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 数据集按「周」分目录（datasets/lta/week1/...），新增周次不需要改这里。
LTA_DATA = ROOT / "datasets" / "lta"

RESULTS = ROOT / "results" / "detection"
DETECTIONS = RESULTS / "detections.csv"
CAMERAS = RESULTS / "cameras.csv"
RUN_INFO = RESULTS / "run.json"
# 已处理过的画面清单。不能只看 detections.csv：零检出的图不会在里面留下任何行，
# 于是续跑会重算它们，下游也无法把「处理过但没车」和「还没处理」区分开。
FRAMES = RESULTS / "frames.csv"

# 默认权重。换模型不必改这里，用 `run_detect.py --weights <路径>` 覆盖。
# yolo/ 不入版本控制（.gitignore 里有），换台机器要重新下。
WEIGHTS = ROOT / "yolo" / "yolo26m.pt"

# 模型的类名 -> 本管线统计的车辆类。自建数据集上微调过的权重几乎一定要写它，
# 否则类名对不上会静默地一个框都输出不了，见同目录 README.md。
CLASSES_FILE = Path(__file__).resolve().parent / "classes.yaml"

# 微调用的标注集（YOLO 格式）。datasets/ 不入版本控制，抽样清单 split.csv 可由
# sample.py 按同一种子重建。
LABELED = ROOT / "datasets" / "lta_yolo"
SPLIT_FILE = LABELED / "split.csv"
DATA_YAML = LABELED / "data.yaml"
# 预标注用的教师模型：最大号的 COCO 权重，配原图分辨率推理。
TEACHER_WEIGHTS = ROOT / "yolo" / "yolo26x.pt"
FINETUNE_RESULTS = RESULTS / "finetune"

# 相机坐标不在数据集里，取自仓库里那份 LTA 接口样本响应。
CAMERA_COORDINATES = ROOT / "docs" / "LTA" / "TrafficImages" / "Traffic-Imagesv2.json"
