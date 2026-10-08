"""产物目录与实验编号。本模块自己的约定，不与 src/ 共用。"""

from __future__ import annotations

from pathlib import Path

from . import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "forecast"


def run_dir(dataset: str, targets: tuple[str, ...], model: str, seed: int, tag: str = "") -> Path:
    """一次运行的产物目录；tag 区分同一配置下的消融变体，避免互相覆盖。"""
    name = f"seed{seed}" if not tag else f"seed{seed}_{tag}"
    return RESULTS_ROOT / dataset / "+".join(targets) / model / name


def exp_id(dataset: str, model_index: int, target_mask: int, seed: int, variant: int = 0) -> int:
    """给 LibCity 的实验编号。

    LibCity 以它为键组织缓存路径，重号会共用评估缓存。除了
    （数据集, 模型, 目标组合, 种子），消融变体也必须各占一个编号，否则并发跑同一配置的
    两个变体会互相踩坏缓存里的预测数组（实测会读到半写的 npz 而报 BadZipFile）。
    """
    return (
        datasets.BASE_EXP_ID[dataset]
        + model_index * 100000
        + target_mask * 10000
        + seed * 1000
        + variant
    )


def target_mask(dataset: str, targets: tuple[str, ...]) -> int:
    """把目标组合映射成一个小整数：每个目标占一个二进制位。"""
    meta = datasets.resolve(dataset)
    return sum(1 << meta.targets.index(target) for target in targets)
