"""冻结数据版本的核对：训练与评估开始前确认实际读取的清单、标签和冻结记录（dataset.json）一致。

版本化目录（miotcd_yolo_<版本> / joint_yolo_<版本>）缺少冻结记录直接报错；其他目录（如 lta_yolo）没有冻结记录，
只打印提示后放行。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# lta_human_：人工修正的 LTA 测试集（annotate.py collect 生成），只用于评估，不用于训练
VERSIONED_PREFIXES = ("miotcd_yolo_", "joint_yolo_", "lta_human_")


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def label_digest(data_dir: Path, split: str) -> str:
    """一个划分全部标签内容的摘要（与 miotcd.label_digest 同一算法）。"""
    digest = hashlib.sha256()
    for label in sorted((data_dir / "labels" / split).glob("*.txt")):
        digest.update(label.name.encode() + b"\0" + label.read_bytes())
    return digest.hexdigest()[:16]


def _resolve(entry: str, root: Path) -> Path:
    path = Path(entry)
    return path if path.is_absolute() else root / path


def _load_meta(mio_dir: Path) -> dict:
    meta_path = mio_dir / "dataset.json"
    if not meta_path.is_file():
        raise SystemExit(f"{mio_dir} 是版本化数据目录，但缺少冻结记录 {meta_path.name}，拒绝使用")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _check_split(mio_dir: Path, meta: dict, split: str, listing: Path) -> None:
    """listing 必须正是 mio_dir/<split>.txt，且清单摘要、标签摘要与冻结记录一致。"""
    expected_listing = (mio_dir / f"{split}.txt").resolve()
    if listing.resolve() != expected_listing:
        raise SystemExit(f"配置引用的 {listing} 不是冻结的 {expected_listing}")
    if sha256_16(listing) != meta["manifest_sha256_16"][split]:
        raise SystemExit(f"{listing.name} 的摘要与冻结记录不符，拒绝使用")
    if label_digest(mio_dir, split) != meta["labels_sha256_16"][split]:
        raise SystemExit(f"{mio_dir.name}/labels/{split} 的标签内容与冻结记录不符，拒绝使用")


def check_data_yaml(data_yaml: Path) -> None:
    """训练前：核对 data.yaml 实际引用的 train / val 清单及其标签。"""
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    train, val = _resolve(str(config["train"]), root), _resolve(str(config["val"]), root)
    if not root.name.startswith(VERSIONED_PREFIXES):
        print(f"[frozen] {root.name} 不是版本化数据目录，没有冻结记录可核对")
        return
    if root.name.startswith("lta_human_"):
        raise SystemExit(f"{root.name} 是人工修正的测试集，只用于评估，不能用来训练")
    if root.name.startswith("miotcd_yolo_"):
        meta = _load_meta(root)
        _check_split(root, meta, "train", train)
        _check_split(root, meta, "dev", val)
        return
    # 联合训练：train 是联合清单，val 是对应 MIO-TCD 版本的 dev
    mio_dir = root.parent / root.name.replace("joint_yolo_", "miotcd_yolo_", 1)
    meta = _load_meta(mio_dir)
    joint = meta.get("joint")
    if not joint:
        raise SystemExit(f"{mio_dir.name}/dataset.json 没有联合训练的冻结记录")
    if train.resolve() != (root / "train.txt").resolve() or sha256_16(train) != joint["train_list_sha256_16"]:
        raise SystemExit(f"联合训练清单 {train} 与冻结记录不符，拒绝训练")
    _check_split(mio_dir, meta, "dev", val)
    # 联合清单里的 MIO-TCD 部分：标签与冻结的 train 标签一致
    if label_digest(mio_dir, "train") != meta["labels_sha256_16"]["train"]:
        raise SystemExit(f"{mio_dir.name}/labels/train 的标签内容与冻结记录不符，拒绝训练")
    lta_labels = joint.get("lta_labels_sha256_16")
    if lta_labels:
        lta_dir = Path(joint["lta_dir"])
        if label_digest(lta_dir, "train") != lta_labels:
            raise SystemExit(f"{lta_dir.name}/labels/train（LTA 伪标签）与生成联合清单时不一致，拒绝训练")


def check_eval_split(data_dir: Path, split: str) -> None:
    """评估前：版本化目录的 <split>.txt 与该划分的标签必须与冻结记录一致。"""
    if not data_dir.name.startswith(VERSIONED_PREFIXES):
        return
    meta = _load_meta(data_dir)
    if split not in meta.get("manifest_sha256_16", {}):
        raise SystemExit(f"{data_dir.name} 的冻结记录里没有划分 {split!r}")
    _check_split(data_dir, meta, split, data_dir / f"{split}.txt")
