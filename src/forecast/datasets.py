"""公开基准数据集的注册表。

数据包放在 `third_party/LibCity/raw_data/<数据集名>/`，含 .dyna / .geo / .rel /
config.json。此处只记录元数据与需要覆盖的加载参数，实际加载以数据包自带的
config.json 为准。

REFERENCE 抄自 LibCity 基准论文（arXiv:2308.12899）表 5，用于把本次结果与文献并排打印。
该表的数字是 12 步平均，MAPE 为百分数。
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocols import Protocol


@dataclass(frozen=True)
class Dataset:
    """一个公开基准数据集。"""

    name: str
    targets: tuple[str, ...]
    default_target: str
    interval_seconds: int
    num_nodes: int
    num_edges: int
    adjacency: str  # "link" 直接用 .rel 的边；"dist" 由距离经高斯核生成
    coordinates: bool

    @property
    def interval_minutes(self) -> int:
        return self.interval_seconds // 60


PEMSD8 = Dataset(
    name="PEMSD8",
    targets=("traffic_flow", "traffic_occupancy", "traffic_speed"),
    default_target="traffic_flow",
    interval_seconds=300,
    num_nodes=170,
    num_edges=277,
    adjacency="link",
    coordinates=False,
)

METR_LA = Dataset(
    name="METR_LA",
    targets=("traffic_speed",),
    default_target="traffic_speed",
    interval_seconds=300,
    num_nodes=207,
    num_edges=11753,
    adjacency="dist",
    coordinates=True,
)

REGISTRY: dict[str, Dataset] = {dataset.name: dataset for dataset in (PEMSD8, METR_LA)}

# 各数据集用互不重叠的 exp_id 区间；重号会共用 LibCity 的评估缓存。
BASE_EXP_ID: dict[str, int] = {"PEMSD8": 1000000, "METR_LA": 2000000}

# (MAE, MAPE%, RMSE)，按 (数据集, 目标, 模型) 索引。
REFERENCE: dict[tuple[str, str, str], tuple[float, float, float]] = {
    ("PEMSD8", "traffic_flow", "FNN"): (20.74, 13.21, 32.03),
    ("PEMSD8", "traffic_flow", "STGCN"): (15.63, 10.50, 24.63),
    ("PEMSD8", "traffic_flow", "MTGNN"): (14.88, 9.93, 23.81),
    ("PEMSD8", "traffic_occupancy", "FNN"): (0.83, 15.41, 2.18),
    ("PEMSD8", "traffic_occupancy", "STGCN"): (0.71, 11.54, 1.89),
    ("PEMSD8", "traffic_occupancy", "MTGNN"): (0.66, 10.87, 1.85),
    ("PEMSD8", "traffic_speed", "FNN"): (1.89, 4.72, 5.08),
    ("PEMSD8", "traffic_speed", "STGCN"): (1.42, 3.32, 3.42),
    ("PEMSD8", "traffic_speed", "MTGNN"): (1.29, 2.89, 3.21),
    ("METR_LA", "traffic_speed", "STGCN"): (3.29, 8.80, 6.59),
    ("METR_LA", "traffic_speed", "MTGNN"): (3.02, 8.25, 6.09),
}


def resolve(name: str) -> Dataset:
    """按名字取数据集元数据。"""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"未注册的数据集：{name}；已注册：{sorted(REGISTRY)}") from None


def reference(dataset: str, targets: tuple[str, ...], model: str) -> tuple[float, float, float] | None:
    """查论文表 5 的对应数值；单目标实验才有。"""
    if len(targets) != 1:
        return None
    return REFERENCE.get((dataset, targets[0], model))


def build_other_args(
    dataset: Dataset,
    protocol: Protocol,
    *,
    exp_id: int,
    seed: int = 0,
    gpu: bool = True,
    gpu_id: int = 0,
) -> dict:
    """拼出传给 LibCity ConfigParser 的 other_args。

    必须显式覆盖的项：目标选择（data_col / output_dim）、time_intervals（LibCity 只读
    复数形式）。
    """
    unknown = [target for target in protocol.targets if target not in dataset.targets]
    if unknown:
        raise ValueError(f"{dataset.name} 没有这些目标：{unknown}；可选 {dataset.targets}")

    args = protocol.as_other_args()
    args.update(
        {
            "gpu": gpu,
            "gpu_id": gpu_id,
            "seed": seed,
            "exp_id": exp_id,
            "time_intervals": dataset.interval_seconds,
        }
    )
    return args


def check_data_feature(dataset: Dataset, data_feature: dict, output_dim: int) -> list[str]:
    """把加载结果与注册的元数据对一遍，返回不一致之处。"""
    problems = []
    if data_feature.get("num_nodes") != dataset.num_nodes:
        problems.append(
            f"节点数不符：注册 {dataset.num_nodes}，实际 {data_feature.get('num_nodes')}"
        )
    if data_feature.get("output_dim") != output_dim:
        problems.append(f"输出维度应为 {output_dim}，实际 {data_feature.get('output_dim')}")
    if data_feature.get("scaler") is None:
        problems.append("未生成 scaler，检查 scaler 是否被设为 standard")
    return problems
