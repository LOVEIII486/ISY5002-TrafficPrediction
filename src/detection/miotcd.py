"""把 MIO-TCD 定位子集转成 YOLO 训练目录：按场景组切 train / dev / test，版本化并冻结。

    python -m src.detection.miotcd                       # 生成 datasets/miotcd_yolo_<版本>/，已存在则拒绝
    python -m src.detection.miotcd --version v4 ...      # 改了划分参数就换新版本号，旧版本原样保留

输入 `datasets/miotcd/MIO-TCD-Localization.tar`（官方直链，3.7 GB）：
    MIO-TCD-Localization/train/<8 位编号>.jpg   110,000 张，720×480 与 342×228 两种尺寸（约 4:1）
    MIO-TCD-Localization/gt_train.csv           每行 `编号,类别,x1,y1,x2,y2`，零起始像素闭区间
    MIO-TCD-Localization/test/                  官方不公开标注，本脚本不用

输出 `datasets/miotcd_yolo_<版本>/`：
    images/<划分>/*.jpg        符号链接到原图
    labels/<划分>/*.txt        YOLO 格式
    <划分>.txt                 冻结清单：训练只读清单、评估核对清单摘要，都不扫目录
    split.csv / data.yaml / dataset.json（含清单与标签内容摘要、复核结果）

## 场景分组

MIO-TCD 不提供摄像头编号，而同一摄像头会贡献大量连续帧：11 万张里 9,627 组字节完全相同，
绝大多数图都能找到背景几乎一样的「邻居」。按编号切分（v1）在验证集里留下了 109 张训练图的原样副本。

四条规则任一成立，两张图即连边，再取连通分量作为场景组，整组只进一个划分：
    1. 文件 SHA-256 相同；
    2. 256 位差值哈希（dHash）汉明距离 ≤ GROUP_HAMMING —— 抓近乎相同的帧；
    3. 背景缩略图相关系数 ≥ THUMB_CORR —— 灰度、高斯模糊后缩到 16×12，小车在这个尺度几乎消失，
       剩下的主要是静态背景，抓「同一机位、车辆位置不同」的帧（v2 只有前两条，
       同机位相隔 3 秒的帧因车辆移动、dHash 距离 17 而被分进了不同划分）；
    4. 抠车背景相关系数 ≥ MASKED_CORR —— 先用标注框挖掉所有车辆、用周围背景填补，再比较缩略图。
       前景大车会遮住大片背景，只靠第 3 条时同机位的帧仍会漏并。

这是基于图像内容的近似分组，**不等于真实摄像头编号**：同一摄像头在光照剧变（白天 / 夜间、雨天）后
背景外观差异可能超过阈值，仍会被分成不同组。复核结果因此只能表述为「通过上述规则的重复检查」，
dataset.json 另报告阈值以下的残余相似度，报告里不能声称已按摄像头隔离。

## 划分

    train      训练；摩托车图优先纳入（富集）
    dev        选 checkpoint、比较方案；**分层开发集**：dev 池里的摩托车图全部纳入，其余随机补足
    test       冻结，所有决定做完后只评估一次；测试池内均匀抽样
    test_moto  测试池里剩余的摩托车图，作补充诊断集，与 test 不重叠但共享场景组，不是独立证据

分组分池时，含摩托车的组单独按 RARE_POOL_SHARE 配比（按去重后的摩托车图片数），其余组按 POOL_SHARE
（按去重后的图片数）。这改变了各场景进入各集合的机会，所以 dev / test **不是自然分布**，而是
「按稀有类分层、按组隔离后在池内抽样」；实际配比写在 dataset.json 里。超大的组（> BIG_GROUP 张）
一律放进训练池，避免单个机位占据测试集的大半。同一划分内字节相同的图只保留一张。

每个划分用独立的随机流（种子含划分名），改训练配额不会连带改变 dev / test 的抽样。

## 类别映射（11 类 -> 管线的 4 类）

    car        <- car, work_van
    truck      <- pickup_truck, single_unit_truck, articulated_truck
    bus        <- bus
    motorcycle <- motorcycle
    pedestrian / bicycle / non-motorized_vehicle 的框删除（不是机动车）
    motorized_vehicle（官方定义：太小、无法细分车型的机动车）无法映射也不能当背景，含它的整张图剔除，
    约 18%。**因此结果只适用于这个筛选后的四类子集，不是官方 MIO-TCD 基准成绩。**
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .run_detect import VEHICLE_CLASSES

TAR = paths.MIOTCD_RAW / "MIO-TCD-Localization.tar"
# tar 内所有成员都带 MIO-TCD-Localization/ 前缀
RAW_ROOT = paths.MIOTCD_RAW / "MIO-TCD-Localization"
RAW_TRAIN = RAW_ROOT / "train"
GT_CSV = RAW_ROOT / "gt_train.csv"

CLASS_MAP = {
    "car": "car",
    "work_van": "car",
    "pickup_truck": "truck",
    "single_unit_truck": "truck",
    "articulated_truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
}
DROP_BOX = {"pedestrian", "bicycle", "non-motorized_vehicle"}
DROP_IMAGE = {"motorized_vehicle"}
RARE_CLASS = "motorcycle"

# 特征算法版本：改了任何特征的计算方式就改这里，缓存随之失效。
FEATURE_VERSION = "f3-sha256-dhash256-thumb16x12blur2-maskedthumb16x12-truesize"
GROUP_HAMMING = 16
THUMB_CORR = 0.95
# 抠车背景：用标注框把所有车辆（含被剔除的类别）挖掉，再用周围背景插值填补后比较。前景大车会把普通缩略图的
# 相关系数拉到 0.90–0.95，同机位的帧因此漏并（v3 初版抽查的残余图对全部是同一机位）；抠车后它们大多在 0.96 以上。
# 阈值 0.93：降到 0.90 时开始出现占 17% 的巨型组（疑似把不同机位连在一起）；0.93 时最大组约 7%。
MASKED_CORR = 0.93
# 超过这么多张的场景组一律放进训练池（约为测试池目标的 1/10），保证 dev / test 由足够多的机位组成。
BIG_GROUP = 1500
POOL_SHARE = {"test": 0.15, "dev": 0.10, "train": 0.75}
# 去重后不重复的摩托车图约 1,100 张，分布极不均匀（单组最多上百张，过半的组不到 5 张）。
# 摩托车大户组（> RARE_DOMINANT 张）放进训练池：放在评估集里会一家独大（v3 的 dev 有 125/126 个摩托车框出自同一组）。
# 其余含摩托车的组按 RARE_POOL_SHARE 分池，dev / test 多分一些，好让评估集覆盖更多不同的组。
RARE_DOMINANT = 30
RARE_POOL_SHARE = {"test": 0.30, "dev": 0.20, "train": 0.50}
# dev 与 test_moto 抽样时每组最多取这么多张摩托车图，限制单组对摩托车 AP 的支配
RARE_PER_GROUP_CAP = 10
SPLITS = ("train", "dev", "test", "test_moto")
# test 与 test_moto 取自同一个测试池，两者都只用于评估，彼此相似不算泄漏，复核时不要求隔开。
SAME_POOL_PAIRS = {("test", "test_moto")}
# 每个字节里 1 的个数，用查表代替 unpackbits，比对时内存小 8 倍
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组 32 字节哈希的两两汉明距离，形状 (len(a), len(b))。"""
    return POPCOUNT[a[:, None, :] ^ b[None, :, :]].sum(axis=2, dtype=np.uint16)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def extract() -> None:
    """只在还没解压时解压 train/ 与 gt_train.csv。"""
    if RAW_TRAIN.is_dir() and GT_CSV.is_file():
        return
    if not TAR.is_file():
        raise SystemExit(
            f"缺少 {TAR}\n"
            "下载：curl -L -o datasets/miotcd/MIO-TCD-Localization.tar "
            "https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar"
        )
    print(f"解压 {TAR.name}（只取 train/ 与 gt_train.csv）…", flush=True)
    with tarfile.open(TAR) as archive:
        members = [m for m in archive if "/train/" in m.name or m.name.endswith("gt_train.csv")]
        if not members:
            raise SystemExit(f"{TAR.name} 里没有 train/ 成员，tar 布局与预期不符")
        archive.extractall(paths.MIOTCD_RAW, members=members, filter="data")


def load_boxes() -> tuple[dict[str, list[tuple[int, int, int, int, int]]], Counter]:
    """读官方 CSV，按映射规则得到 {编号: [(类别索引, x1, y1, x2, y2)]}；含 DROP_IMAGE 类的图不返回。"""
    to_index = {name: index for index, name in enumerate(VEHICLE_CLASSES)}
    per_image: dict[str, list] = defaultdict(list)
    rejected: set[str] = set()
    stats: Counter = Counter()
    with GT_CSV.open(newline="") as stream:
        for image_id, label, x1, y1, x2, y2 in csv.reader(stream):
            stats[label] += 1
            if label in DROP_IMAGE:
                rejected.add(image_id)
            elif label in DROP_BOX:
                per_image.setdefault(image_id, [])  # 图保留，框丢掉
            elif label in CLASS_MAP:
                per_image[image_id].append((to_index[CLASS_MAP[label]], int(x1), int(y1), int(x2), int(y2)))
            else:
                raise SystemExit(f"未知类别 {label!r}，请补充 CLASS_MAP")
    stats["_images_rejected_motorized"] = len(rejected)
    return {k: v for k, v in per_image.items() if k not in rejected}, stats


# ---------------------------------------------------------------- 图像特征

def _unit(vector: np.ndarray) -> np.ndarray:
    """零均值、单位范数，之后两向量的点积就是相关系数。"""
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


_ALL_BOXES: dict[str, list[tuple[int, int, int, int]]] = {}


def _init_worker(all_boxes: dict) -> None:
    global _ALL_BOXES
    _ALL_BOXES = all_boxes


def _features(stem: str) -> tuple[str, str, bytes, np.ndarray, np.ndarray]:
    """一张图的四种特征：文件 SHA-256、256 位 dHash、背景缩略图、抠车背景缩略图（后两者为 16×12 单位向量）。"""
    import cv2
    from PIL import Image

    path = RAW_TRAIN / f"{stem}.jpg"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as header:  # 只读文件头拿真实尺寸：数据集里混有 720×480 与 342×228 两种
        width, height = header.size
    gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)  # 解码时直接降到 1/4，快一个数量级
    pixels = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    dhash = np.packbits((pixels[:, 1:] > pixels[:, :-1]).ravel()).tobytes()
    thumb = cv2.resize(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2), (16, 12), interpolation=cv2.INTER_AREA).ravel()

    # 抠车：64×48 上把每个标注框（向外扩 1 格）挖掉，用归一化卷积（周围背景的加权平均）填补
    small = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = np.ones((48, 64), np.float32)
    for x1, y1, x2, y2 in _ALL_BOXES.get(stem, []):
        # 框是原图像素坐标，按这张图自己的宽高缩到 64×48（不能写死 720×480，也不要从降采样尺寸反推）
        left, top = int(x1 * 64 / width) - 1, int(y1 * 48 / height) - 1
        right, bottom = int(np.ceil((x2 + 1) * 64 / width)) + 1, int(np.ceil((y2 + 1) * 48 / height)) + 1
        mask[max(0, top) : min(48, bottom), max(0, left) : min(64, right)] = 0
    numerator = cv2.GaussianBlur(small * mask, (0, 0), 6)
    denominator = cv2.GaussianBlur(mask, (0, 0), 6)
    fallback = small[mask > 0].mean() if mask.any() else small.mean()
    filled = np.where(denominator > 1e-3, numerator / np.maximum(denominator, 1e-3), fallback)
    background = np.where(mask > 0, small, filled)
    masked = cv2.resize(cv2.GaussianBlur(background, (0, 0), 1.5), (16, 12), interpolation=cv2.INTER_AREA).ravel()
    return stem, digest, dhash, _unit(thumb), _unit(masked)


def _all_raw_boxes() -> dict[str, list[tuple[int, int, int, int]]]:
    """官方 CSV 里的全部框（所有 11 类），抠车用。"""
    boxes: dict[str, list] = defaultdict(list)
    with GT_CSV.open(newline="") as stream:
        for image_id, _label, x1, y1, x2, y2 in csv.reader(stream):
            boxes[image_id].append((int(x1), int(y1), int(x2), int(y2)))
    return dict(boxes)


def image_features(ids: list[str]) -> tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    """按 ids 顺序返回 (sha 字典, dHash (n,32) uint8, 背景缩略图 (n,192), 抠车背景缩略图 (n,192))。

    缓存绑定特征算法版本与标注文件摘要：任一变化都重新计算。
    """
    cache = RAW_ROOT / f"features_{FEATURE_VERSION}.npz"
    fingerprint = f"{FEATURE_VERSION}|{file_digest(GT_CSV)}|{len(ids)}|{ids[0]}|{ids[-1]}"
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if str(data["fingerprint"]) == fingerprint and list(data["ids"]) == ids:
            return dict(zip(ids, data["sha"].tolist())), data["dhash"], data["thumb"], data["masked"]
    print(f"计算 {len(ids)} 张图的特征（缓存到 {cache.name}）…", flush=True)
    with Pool(initializer=_init_worker, initargs=(_all_raw_boxes(),)) as pool:
        rows = pool.map(_features, ids, chunksize=500)
    sha = np.array([row[1] for row in rows])
    dhash = np.frombuffer(b"".join(row[2] for row in rows), dtype=np.uint8).reshape(len(ids), 32)
    thumb = np.stack([row[3] for row in rows]).astype(np.float32)
    masked = np.stack([row[4] for row in rows]).astype(np.float32)
    np.savez(cache, fingerprint=fingerprint, ids=np.array(ids), sha=sha, dhash=dhash, thumb=thumb, masked=masked)
    return dict(zip(ids, sha.tolist())), dhash, thumb, masked


# ---------------------------------------------------------------- 场景组

def _sha_edges(ids: list[str], sha: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, image_id in enumerate(ids):
        by_sha[sha[image_id]].append(index)
    pairs = [(members[0], other) for members in by_sha.values() for other in members[1:]]
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def _dhash_edges(dhash: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dHash 汉明距离 ≤ GROUP_HAMMING 的所有图对。

    分段鸽巢原理：256 位切成 GROUP_HAMMING+1 段，距离 ≤ 阈值的两张图至少有一段完全相同，只在同段桶内比对。
    """
    n = len(dhash)
    bits = np.unpackbits(dhash, axis=1)
    rows, cols = [], []
    for segment in np.array_split(np.arange(256), GROUP_HAMMING + 1):
        keys = np.ascontiguousarray(np.packbits(bits[:, segment], axis=1))
        keys = keys.view(np.dtype((np.void, keys.shape[1]))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        for start, end in zip(starts, np.r_[starts[1:], n]):
            members = order[start:end]
            if len(members) < 2:
                continue
            for chunk in range(0, len(members), 256):  # 大桶分块，内存可控、不漏对
                block = members[chunk : chunk + 256]
                a, b = np.nonzero(hamming(dhash[block], dhash[members]) <= GROUP_HAMMING)
                keep = block[a] < members[b]
                rows.append(block[a][keep])
                cols.append(members[b][keep])
    return (np.concatenate(rows), np.concatenate(cols)) if rows else (np.zeros(0, int), np.zeros(0, int))


def _thumb_edges(thumb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """背景缩略图相关系数 ≥ threshold 的所有图对（分块矩阵乘法，精确、不近似）。"""
    rows, cols = [], []
    for start in range(0, len(thumb), 2048):
        a, b = np.nonzero(thumb[start : start + 2048] @ thumb.T >= threshold)
        keep = (a + start) < b
        rows.append(a[keep] + start)
        cols.append(b[keep])
    return np.concatenate(rows), np.concatenate(cols)


def scene_groups(
    ids: list[str],
    sha: dict[str, str],
    dhash: np.ndarray,
    thumb: np.ndarray,
    masked: np.ndarray,
    thumb_corr: float = THUMB_CORR,
    masked_corr: float = MASKED_CORR,
) -> dict[str, int]:
    """四条规则连边后取连通分量。返回 {编号: 组号}；dhash、thumb、masked 与 ids 同序。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = [_sha_edges(ids, sha), _dhash_edges(dhash), _thumb_edges(thumb, thumb_corr), _thumb_edges(masked, masked_corr)]
    rows = np.concatenate([e[0] for e in edges])
    cols = np.concatenate([e[1] for e in edges])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(ids), len(ids)))
    _count, labels = connected_components(graph, directed=False)
    return {image_id: int(label) for image_id, label in zip(ids, labels)}


# ---------------------------------------------------------------- 分池与抽样

def has_class(boxes: dict, image_id: str, name: str) -> bool:
    return any(cls == VEHICLE_CLASSES.index(name) for cls, *_ in boxes[image_id])


def unique_by_sha(image_ids: list[str], sha: dict[str, str]) -> list[str]:
    """字节相同的图只留编号最小的一张。"""
    seen, kept = set(), []
    for image_id in sorted(image_ids):
        if sha[image_id] not in seen:
            seen.add(sha[image_id])
            kept.append(image_id)
    return kept


def assign_pools(groups: dict[str, int], boxes: dict, sha: dict[str, str], seed: int) -> dict[str, list[str]]:
    """场景组分池：超大组进训练池；含稀有类的组按去重后的稀有类图片数配比；其余组按去重后的图片数配比。"""
    members: dict[int, list[str]] = defaultdict(list)
    for image_id, group in groups.items():
        members[group].append(image_id)
    unique = {g: unique_by_sha(ids, sha) for g, ids in members.items()}
    rare = {g: sum(has_class(boxes, i, RARE_CLASS) for i in ids) for g, ids in unique.items()}
    order = sorted(members)
    random.Random(f"pools-{seed}").shuffle(order)

    pools: dict[str, list[str]] = {name: [] for name in POOL_SHARE}
    filled = {"rare": dict.fromkeys(POOL_SHARE, 0), "all": dict.fromkeys(POOL_SHARE, 0)}
    total_rare = max(1, sum(rare.values()))
    total_all = sum(len(ids) for ids in unique.values())
    def forced_train(group: int) -> bool:
        return len(unique[group]) > BIG_GROUP or rare[group] > RARE_DOMINANT

    for group in order:  # 超大组、摩托车大户组先进训练池，并计入训练池的配额
        if forced_train(group):
            pools["train"].extend(unique[group])
            filled["rare"]["train"] += rare[group]
            filled["all"]["train"] += len(unique[group])
    for stratum, share, total in (("rare", RARE_POOL_SHARE, total_rare), ("all", POOL_SHARE, total_all)):
        for group in order:
            if forced_train(group) or (rare[group] > 0) != (stratum == "rare"):
                continue
            target = max(share, key=lambda name: share[name] - filled[stratum][name] / total)
            pools[target].extend(unique[group])
            filled["rare"][target] += rare[group]
            filled["all"][target] += len(unique[group])
    return {name: sorted(ids) for name, ids in pools.items()}


def split_rng(seed: int, split: str) -> random.Random:
    """每个划分一条独立随机流：改一个划分的配额不影响其他划分。"""
    return random.Random(f"miotcd-{seed}-{split}")


def sample(pool: list[str], quota: int, rng: random.Random) -> list[str]:
    """无放回随机抽 quota 张；池不够就全取。"""
    pool = sorted(pool)
    return sorted(rng.sample(pool, quota)) if quota < len(pool) else pool


def cap_per_group(image_ids: list[str], groups: dict[str, int], cap: int, rng: random.Random) -> list[str]:
    """每个场景组最多保留 cap 张（组内随机）。"""
    by_group: dict[int, list[str]] = defaultdict(list)
    for image_id in sorted(image_ids):
        by_group[groups[image_id]].append(image_id)
    kept = []
    for group in sorted(by_group):
        kept += sample(by_group[group], cap, rng)
    return sorted(kept)


def sample_rare_first(
    pool: list[str], boxes: dict, quota: int, rng: random.Random, groups: dict[str, int] | None = None, per_group_cap: int | None = None
) -> list[str]:
    """稀有类图优先（最多占满配额），其余随机补足。训练集富集与分层开发集共用；
    给了 per_group_cap 时，稀有类图每组最多取这么多张（开发集用，限制单组支配）。"""
    rare = [i for i in pool if has_class(boxes, i, RARE_CLASS)]
    if per_group_cap is not None and groups is not None:
        rare = cap_per_group(rare, groups, per_group_cap, rng)
    if quota <= len(rare):
        return sample(rare, quota, rng)
    taken = set(rare)
    others = [i for i in pool if i not in taken and not has_class(boxes, i, RARE_CLASS)]
    return sorted(rare + sample(others, quota - len(rare), rng))


def class_diversity(image_ids: list[str], boxes: dict, groups: dict[str, int]) -> dict:
    """每类：框数、涉及的场景组数、框数最多的组占该类的比例。"""
    report = {}
    for index, name in enumerate(VEHICLE_CLASSES):
        per_group: Counter = Counter()
        for image_id in image_ids:
            per_group[groups[image_id]] += sum(cls == index for cls, *_ in boxes[image_id])
        per_group = Counter({g: n for g, n in per_group.items() if n})
        total = sum(per_group.values())
        report[name] = {
            "boxes": total,
            "groups": len(per_group),
            "largest_group_share": round(per_group.most_common(1)[0][1] / total, 3) if total else None,
        }
    return report


# ---------------------------------------------------------------- 写盘与复核

def write_split(out: Path, image_ids: list[str], split: str, boxes: dict) -> None:
    """写 labels/<split>/*.txt、images/<split>/ 符号链接与清单 <split>.txt。"""
    from PIL import Image

    image_dir, label_dir = out / "images" / split, out / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for image_id in image_ids:
        source = RAW_TRAIN / f"{image_id}.jpg"
        (image_dir / f"{image_id}.jpg").symlink_to(source)
        with Image.open(source) as image:
            width, height = image.size
        lines = []
        for cls, x1, y1, x2, y2 in boxes[image_id]:
            # 官方是零起始像素闭区间：真实宽 x2-x1+1，对应连续框 [x1, x2+1]。评估时不要再加 1。
            w, h = (x2 - x1 + 1) / width, (y2 - y1 + 1) / height
            cx, cy = (x1 + x2 + 1) / 2 / width, (y1 + y2 + 1) / 2 / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    listing = "\n".join(str(image_dir / f"{image_id}.jpg") for image_id in image_ids) + "\n"
    (out / f"{split}.txt").write_text(listing, encoding="utf-8")


def label_digest(out: Path, split: str) -> str:
    """一个划分全部标签内容的摘要（按文件名排序后拼接）。"""
    digest = hashlib.sha256()
    for label in sorted((out / "labels" / split).glob("*.txt")):
        digest.update(label.name.encode() + b"\0" + label.read_bytes())
    return digest.hexdigest()[:16]


def verify(
    out: Path,
    chosen: dict[str, list[str]],
    ids: list[str],
    sha: dict[str, str],
    dhash: np.ndarray,
    thumb: np.ndarray,
    masked: np.ndarray,
    thumb_corr: float,
    masked_corr: float,
) -> dict:
    """从磁盘重新读清单、split.csv、目录核对一致；再做跨划分的规则重复检查与残余相似度统计。"""
    index = {image_id: i for i, image_id in enumerate(ids)}
    split_csv = pd.read_csv(out / "split.csv", dtype={"image": str})
    report: dict = {"manifest_consistent": True, "cross_split": {}}
    for split, expected in chosen.items():
        listed = {Path(line).stem for line in (out / f"{split}.txt").read_text(encoding="utf-8").splitlines()}
        on_disk = {p.stem for p in (out / "images" / split).glob("*.jpg")}
        labels = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        in_csv = {Path(name).stem for name in split_csv[split_csv["split"] == split]["image"]}
        if not (listed == on_disk == labels == in_csv == set(expected)):
            report["manifest_consistent"] = False
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            ia = np.array([index[x] for x in chosen[a]])
            ib = np.array([index[x] for x in chosen[b]])
            nearest = min(int(hamming(dhash[ia[s : s + 256]], dhash[ib]).min()) for s in range(0, len(ia), 256))
            corr = np.concatenate([(thumb[ia[s : s + 2048]] @ thumb[ib].T).max(axis=1) for s in range(0, len(ia), 2048)])
            # 以 b 为基准：b 里每张图在 a 中最像的那张有多像（b 是更靠后的评估划分，问的是「它在 a 里见过没有」）
            masked_best = np.concatenate([(masked[ib[s : s + 2048]] @ masked[ia].T).max(axis=1) for s in range(0, len(ib), 2048)])
            report["cross_split"][f"{a}~{b}"] = {
                "same_file_hash": len({sha[x] for x in chosen[a]} & {sha[x] for x in chosen[b]}),
                "min_dhash_distance": nearest,
                "max_thumb_corr": round(float(corr.max()), 4),
                "max_masked_corr": round(float(masked_best.max()), 4),
                # 残余：阈值以下仍较相似的比例，可能是同一摄像头在光照变化后的画面，报告里要披露
                f"share_of_{b}_with_masked_corr_ge_0.90": round(float((masked_best >= 0.90).mean()), 4),
                "must_be_separate": (a, b) not in SAME_POOL_PAIRS,
            }
    report["rule_check_passed"] = report["manifest_consistent"] and all(
        v["same_file_hash"] == 0
        and v["min_dhash_distance"] > GROUP_HAMMING
        and v["max_thumb_corr"] < thumb_corr
        and v["max_masked_corr"] < masked_corr
        for v in report["cross_split"].values()
        if v["must_be_separate"]
    )
    return report


def write_joint(joint_dir: Path, mio_dir: Path, train_ids: list[str], lta_repeat: int) -> dict | None:
    """联合训练清单：MIO-TCD 训练图 + LTA 训练图 × lta_repeat；开发集用 MIO-TCD 的 dev 真值。

    LTA 图从 lta_yolo/train.txt 读取，不扫目录；重复列出的路径会被重复采样。
    """
    import yaml

    lta_list = paths.LABELED / "train.txt"
    if not lta_list.is_file():
        print(f"跳过联合训练清单：缺少 {lta_list}（先跑 sample.py 与 prelabel.py）")
        return None
    lta_images = [Path(line) for line in lta_list.read_text(encoding="utf-8").splitlines()]
    missing = [p for p in lta_images if not p.is_file() or not Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt").is_file()]
    if missing:
        raise SystemExit(f"LTA 清单里有 {len(missing)} 张图或标签不存在，例如 {missing[0]}")
    lines = [str(mio_dir / "images" / "train" / f"{i}.jpg") for i in train_ids] + [str(p) for p in lta_images] * lta_repeat
    joint_dir.mkdir(parents=True)
    (joint_dir / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (joint_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(joint_dir), "train": "train.txt", "val": str(mio_dir / "dev.txt"), "names": dict(enumerate(VEHICLE_CLASSES))},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    lta_boxes = sum(len(Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines()) for p in lta_images)
    info = {
        "miotcd_images": len(train_ids),
        "lta_images": len(lta_images),
        "lta_repeat": lta_repeat,
        "entries_per_epoch": len(lines),
        "lta_image_share": round(len(lta_images) * lta_repeat / len(lines), 4),
        "lta_boxes_per_epoch": lta_boxes * lta_repeat,
        "train_list_sha256_16": file_digest(joint_dir / "train.txt"),
        "lta_list_sha256_16": file_digest(lta_list),
        "lta_dir": str(paths.LABELED),
        "lta_labels_sha256_16": label_digest(paths.LABELED, "train"),
    }
    print(f"  joint: MIO-TCD {len(train_ids)} + LTA {len(lta_images)}×{lta_repeat}（图片占比 {info['lta_image_share']:.0%}）→ {joint_dir}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="转换 MIO-TCD 并按场景组切 train/dev/test")
    parser.add_argument("--version", default=paths.MIOTCD_VERSION, help="数据版本号；改了任何划分参数都要换新版本")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--dev", type=int, default=1000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--test-moto", type=int, default=500, help="稀有类诊断集上限")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thumb-corr", type=float, default=THUMB_CORR)
    parser.add_argument("--masked-corr", type=float, default=MASKED_CORR)
    parser.add_argument("--joint-lta-repeat", type=int, default=2, help="联合训练里 LTA 训练图重复几遍；0 不生成联合清单")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同版本目录（只在该版本还没用于任何训练时使用）")
    args = parser.parse_args()

    out = paths.ROOT / "datasets" / f"miotcd_yolo_{args.version}"
    joint_dir = paths.ROOT / "datasets" / f"joint_yolo_{args.version}"
    for directory in (out, joint_dir):
        if directory.exists():
            if not args.overwrite:
                raise SystemExit(f"{directory} 已存在。冻结的数据版本不覆盖；要改划分请换 --version")
            shutil.rmtree(directory)

    extract()
    boxes, stats = load_boxes()
    ids = sorted(boxes)
    sha, dhash, thumb, masked = image_features(ids)
    groups = scene_groups(ids, sha, dhash, thumb, masked, args.thumb_corr, args.masked_corr)
    group_sizes = Counter(groups.values())
    pools = assign_pools(groups, boxes, sha, args.seed)

    chosen = {
        "train": sample_rare_first(pools["train"], boxes, args.train, split_rng(args.seed, "train")),
        "dev": sample_rare_first(pools["dev"], boxes, args.dev, split_rng(args.seed, "dev"), groups, RARE_PER_GROUP_CAP),
        "test": sample(pools["test"], args.test, split_rng(args.seed, "test")),
    }
    in_test = set(chosen["test"])
    moto_left = [i for i in pools["test"] if i not in in_test and has_class(boxes, i, RARE_CLASS)]
    moto_rng = split_rng(args.seed, "test_moto")
    chosen["test_moto"] = sample(cap_per_group(moto_left, groups, RARE_PER_GROUP_CAP, moto_rng), args.test_moto, moto_rng)

    import yaml

    out.mkdir(parents=True)
    rows, class_counts = [], {}
    for split in SPLITS:
        write_split(out, chosen[split], split, boxes)
        class_counts[split] = dict(Counter(VEHICLE_CLASSES[c] for i in chosen[split] for c, *_ in boxes[i]))
        rows += [{"image": f"{i}.jpg", "split": split, "scene_group": groups[i], "n_boxes": len(boxes[i])} for i in chosen[split]]
    pd.DataFrame(rows).to_csv(out / "split.csv", index=False)
    (out / "data.yaml").write_text(
        yaml.safe_dump({"path": str(out), "train": "train.txt", "val": "dev.txt", "names": dict(enumerate(VEHICLE_CLASSES))}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    check = verify(out, chosen, ids, sha, dhash, thumb, masked, args.thumb_corr, args.masked_corr)
    joint = write_joint(joint_dir, out, chosen["train"], args.joint_lta_repeat) if args.joint_lta_repeat > 0 else None

    pool_rare = {name: sum(has_class(boxes, i, RARE_CLASS) for i in pool) for name, pool in pools.items()}
    report = {
        "source": "MIO-TCD Localization (Luo et al., IEEE TIP 2018), CC BY-NC-SA 4.0",
        "labels_are": "human-annotated ground truth",
        "scope": "filtered 4-class subset (images containing motorized_vehicle removed); not the official MIO-TCD benchmark",
        "version": args.version,
        "seed": args.seed,
        "class_map": CLASS_MAP,
        "dropped_boxes": sorted(DROP_BOX),
        "dropped_images_containing": sorted(DROP_IMAGE),
        "raw_label_counts": {k: v for k, v in stats.items() if not k.startswith("_")},
        "images_rejected_motorized": stats["_images_rejected_motorized"],
        "images_available": len(ids),
        "scene_grouping": {
            "rule": f"same SHA-256, or 256-bit dHash Hamming <= {GROUP_HAMMING}, or background-thumbnail correlation >= {args.thumb_corr}, or vehicle-masked background correlation >= {args.masked_corr}",
            "feature_version": FEATURE_VERSION,
            "note": "content-based approximation of camera/scene identity; not true camera IDs",
            "groups": len(group_sizes),
            "largest_groups": [size for _, size in group_sizes.most_common(5)],
            "big_group_to_train_threshold": BIG_GROUP,
            "rare_dominant_group_to_train_threshold": RARE_DOMINANT,
            "rare_per_group_cap_in_dev_and_test_moto": RARE_PER_GROUP_CAP,
        },
        "sampling": "groups stratified by rare class and isolated across pools; uniform sampling within pools; dev is rare-class stratified; NOT the natural distribution",
        "pool_share_target": POOL_SHARE,
        "rare_pool_share_target": RARE_POOL_SHARE,
        "pool_images_after_dedup": {name: len(pool) for name, pool in pools.items()},
        "pool_rare_images_after_dedup": pool_rare,
        "pool_rare_share_actual": {name: round(v / max(1, sum(pool_rare.values())), 4) for name, v in pool_rare.items()},
        "images": {split: len(chosen[split]) for split in SPLITS},
        "boxes": class_counts,
        "class_diversity": {split: class_diversity(chosen[split], boxes, groups) for split in SPLITS},
        "manifest_sha256_16": {split: file_digest(out / f"{split}.txt") for split in SPLITS},
        "labels_sha256_16": {split: label_digest(out, split) for split in SPLITS},
        "verification": check,
        "joint": joint,
    }
    (out / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"可用 {len(ids)} 张（剔除含 motorized_vehicle 的 {stats['_images_rejected_motorized']} 张），场景组 {len(group_sizes)} 个，最大 {report['scene_grouping']['largest_groups'][:3]}")
    print(f"  测试池 / 开发池 / 训练池 摩托车图（去重后）：{pool_rare['test']} / {pool_rare['dev']} / {pool_rare['train']}")
    for split in SPLITS:
        moto = report["class_diversity"][split][RARE_CLASS]
        print(f"  {split:9s} {len(chosen[split]):5d} 张  {class_counts[split]}  摩托车来自 {moto['groups']} 个组，最大组占 {moto['largest_group_share']}")
    for pair, value in check["cross_split"].items():
        note = "" if value["must_be_separate"] else "（同为测试池，不要求隔开）"
        residual = next(v for k, v in value.items() if k.startswith("share_of_"))
        print(f"  复核 {pair:18s} 相同文件 {value['same_file_hash']}，最小 dHash {value['min_dhash_distance']}，最大背景相关 {value['max_thumb_corr']:.3f}，最大抠车背景相关 {value['max_masked_corr']:.3f}，后者抠车相关≥0.90 占 {residual:.1%}{note}")
    if not check["rule_check_passed"]:
        raise SystemExit("复核失败：划分之间仍有规则判定的重复 / 近似图，或清单与目录不一致。不要用这份数据训练。")
    print(f"规则重复检查通过 → {out}")


if __name__ == "__main__":
    main()
