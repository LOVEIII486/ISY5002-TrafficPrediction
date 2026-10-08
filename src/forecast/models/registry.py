"""模型注册表。

index 用于拼 exp_id，一旦分配不再更改，否则会与既有缓存冲突。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from .. import compat


@dataclass(frozen=True)
class ModelSpec:
    """一个可训练的模型。

    default_args 覆盖模型自带的个别默认值，只在有明确理由时使用（见 MTGNN 那条的注释）。
    自研模型也用它给出自己的超参：LibCity 会校验模型名是否在 task_config.json 的白名单内，
    自研模型借内置名字过校验（见 train.py），不显式给出就会继承借来那份配置的取值。
    """

    name: str
    index: int
    source: str  # "libcity" 或 "ours"
    loader: Callable[[], type]
    default_args: dict = field(default_factory=dict)


def _libcity(name: str) -> Callable[[], type]:
    return lambda: compat.load_model_class(name)


def _ours(module: str, class_name: str) -> Callable[[], type]:
    def load() -> type:
        return getattr(importlib.import_module(f".{module}", __package__), class_name)

    return load


# 三个基线：无空间建模、图卷积、图学习。
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("FNN", 0, "libcity", _libcity("FNN")),
    ModelSpec("STGCN", 1, "libcity", _libcity("STGCN")),
    ModelSpec(
        "MTGNN",
        2,
        "libcity",
        _libcity("MTGNN"),
        # 课程学习的进度以批次数计（每 step_size1 批前进一步）。自带值 2500 是按 batch 16
        # 调的：那时 100 epoch 共 78100 批，够走完 12 步所需的 30000 批；本协议用 batch 64，
        # 只有 19600 批，课程学习走到约 7/12 步就停了，却在 12 步上评估。按批次数比例
        # 196/781 缩放到 627，保持课程按 epoch 计的节奏不变。
        default_args={"step_size1": 627},
    ),
    ModelSpec(
        "MultiHeadSTGCN",
        3,
        "ours",
        _ours("multitask", "MultiHeadSTGCN"),
        # 借 FNN 的配置过白名单校验（见 train.py），所以这里要把取值逐项定死。
        # 定型的配置与依据见 notes/MODEL_DESIGN.md 与 notes/HANDOFF.md，摘要：
        #   时间特征（时刻 + 周几），论文的基线都不用星期特征，实测贡献 −6.5%；
        #   对称化邻接 + 谱卷积，谱算子要求拉普拉斯对称，配非对称邻接理论上不成立；
        #   塌缩输出头，实测比保留时间轴直接投影好 4.5%。
        default_args={
            "dataset_class": "TrafficStatePointDataset",
            "executor": "TrafficStateExecutor",
            "evaluator": "TrafficStateEvaluator",
            "load_external": True,
            "add_time_in_day": True,
            "add_day_in_week": True,
            "bidir_adj_mx": True,
            "spatial_mode": "cheb",
            "head_mode": "collapse",
            "learner": "rmsprop",
            "learning_rate": 0.001,
            "weight_decay": 0,
            "lr_decay": True,
            "lr_scheduler": "steplr",
            "lr_decay_ratio": 0.7,
            "step_size": 5,
        },
    ),
)
# GWNET 未登记：该 checkout 的实现把 4 维张量传给 conv1d，任何配置下都在前向崩溃。

REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in SPECS}


def available() -> list[str]:
    """全部已注册的模型名，按注册顺序。"""
    return [spec.name for spec in SPECS]


def resolve(name: str) -> ModelSpec:
    """按名字取模型条目。"""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"未注册的模型：{name}；已注册：{available()}") from None


def build(name: str, config, data_feature):
    """按名字构造模型。"""
    return resolve(name).loader()(config, data_feature)
