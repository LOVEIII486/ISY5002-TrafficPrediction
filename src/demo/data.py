"""真实产物访问层：把 results/ 和 datasets/ 里的实验产物整理成接口要的形状。

一个 run 目录里堆着多次重跑留下的 npz，文件名只带时间戳，看不出对应哪一次。
判定办法是**重算指标再与 run.json 比对**，见 `resolve`。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
RESULTS = _ROOT / "results" / "forecast"
LIBRARY = _ROOT / "datasets" / "libcity"

DATASET = "PEMSD8"
INTERVAL_SECONDS = 300
HORIZONS = ("@15min", "@30min", "@60min")
BASELINES = ("FNN", "STGCN", "MTGNN")
OURS = "MultiHeadSTGCN"

# run.json 存的是数据原生单位（occupancy 是 0–1 的小数），而报告与论文用百分数，差 100 倍。
# HANDOFF 记的 0.673 就是这里的 0.0067×100；这个换算必须和报告一致，否则两边数字对不上。
DISPLAY_SCALE = {"traffic_occupancy": 100.0}

# 目录名：基线都用 seed0；自研模型在报告里用各自目标上的最优配置，不是基础配置。
BASE_TAG = "seed0"

# tag 与 HANDOFF 第 4 节一一对应。
BEST_TAG = {
    "traffic_flow": "seed0_todow-collapse",
    "traffic_occupancy": "seed0_best",
    "traffic_speed": "seed0_best",
}

# 报告里记录的值（occupancy 已按 ×100 口径）。启动自检用：对不上说明选错了 run 或
# 产物被覆盖，必须显式报错，不能静默展示与报告不一致的数字。
EXPECTED_MAE = {
    "traffic_flow": {"FNN": 21.251, "STGCN": 17.605, "MTGNN": 14.877, "MultiHeadSTGCN": 15.002},
    "traffic_occupancy": {"FNN": 1.400, "STGCN": 0.888, "MTGNN": 0.780, "MultiHeadSTGCN": 0.673},
    "traffic_speed": {"FNN": 1.544, "STGCN": 1.595, "MTGNN": 1.321, "MultiHeadSTGCN": 1.350},
}


def scale(target: str) -> float:
    return DISPLAY_SCALE.get(target, 1.0)


@dataclass(frozen=True)
class RunRef:
    """一个可用的实验产物目录。"""

    target: str
    model: str
    tag: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.target}/{self.model}/{self.tag}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _scan() -> dict[tuple[str, str, str], RunRef]:
    found: dict[tuple[str, str, str], RunRef] = {}
    for run_json in sorted(RESULTS.rglob("seed*/run.json")):
        meta = _read_json(run_json)
        target = "+".join(meta["targets"])
        ref = RunRef(target=target, model=meta["model"], tag=run_json.parent.name, path=run_json.parent)
        found[(target, ref.model, ref.tag)] = ref
    if not found:
        raise SystemExit(f"没有在 {RESULTS} 下找到任何 run.json；先跑 src.forecast.run_benchmark")
    return found


def discover() -> dict[tuple[str, str, str], RunRef]:
    """扫描全部 run.json，按 (目标, 模型, tag) 建索引；结果缓存，见 `run_for`。"""
    return _scan()


def run_for(target: str, model: str, tag: str | None = None) -> RunRef:
    """tag 省略时取该模型的代表配置：基线是 seed0，自研是各自目标上的最优。"""
    if tag is None:
        tag = BEST_TAG.get(target, BASE_TAG) if model == OURS else BASE_TAG
    key = (target, model, tag)
    index = discover()
    ref = index.get(key)
    # 产物目录在服务运行期间会被改名或新增，缓存里的路径可能已经指不到东西；重扫一次再判死。
    if ref is None or not (ref.path / "run.json").is_file():
        _scan.cache_clear()
        index = discover()
        ref = index.get(key)
    if ref is None or not (ref.path / "run.json").is_file():
        available = sorted(k[2] for k in index if k[0] == target and k[1] == model)
        raise SystemExit(f"没有这个产物：{target}/{model}/{tag}；该目标该模型下有：{available}")
    return ref


def metrics_of(run: RunRef, target: str) -> dict[str, Any]:
    """从 run.json 取该目标的三个时间窗与平均值，按展示口径换算过。

    展示层只用得到 MAPE（百分数，谁都看得懂），但 MAE 一并给出，它是绝对量，做自检和
    对账时比百分比好用。
    """
    meta = _read_json(run.path / "run.json")
    per_target = meta["per_target"][target]
    value = scale(target)
    return {
        **{horizon: per_target["masked_MAE"][horizon] * value for horizon in HORIZONS},
        "avg": per_target["masked_MAE"]["_avg"] * value,
        # MAPE 在 run.json 里是 0–1 的小数，展示一律换成百分数（0.0997 -> 9.97）。
        "mape": {horizon: per_target["masked_MAPE"][horizon] * 100 for horizon in HORIZONS},
        "mape_avg": per_target["masked_MAPE"]["_avg"] * 100,
    }


def deployed_engine(target: str) -> dict[str, Any]:
    """对外呈现的「预测引擎」：系统部署了哪个模型、它准到什么程度。

    不暴露 MAE / 参数量这类训练期术语，只给百分比误差。MAE 仍然保留在各方法的返回值里，
    供自检与对账使用。
    """
    run = run_for(target, OURS)
    values = metrics_of(run, target)
    return {
        "model": run.model,
        "tag": run.tag,
        "target": target,
        "mape_avg": round(values["mape_avg"], 2),
        "horizons": [
            {"label": label, "mape": round(values["mape"][horizon], 2)}
            for label, horizon in zip(("+15 分钟", "+30 分钟", "+60 分钟"), HORIZONS)
        ],
    }


def _candidates(run: RunRef) -> list[Path]:
    """候选 npz 按文件名里的时间戳升序；时间戳是唯一可靠的先后依据。"""
    cache = run.path / "libcity" / "evaluate_cache"
    return sorted(cache.glob("*_predictions.npz"), key=lambda path: path.name[:19])


@lru_cache(maxsize=64)
def resolve(run: RunRef) -> Path:
    """定位与该 run 的 run.json 对应的那个 npz。

    目录里堆着多次重跑的产物，文件名只有时间戳，光看名字分不出哪次是哪次。所以从最新
    往回逐个重算 MAE 并与 run.json 比对，只有真正对应的那个数组能对上。
    """
    from src.forecast import metrics as forecast_metrics
    from src.forecast.protocols import default_protocol

    meta = _read_json(run.path / "run.json")
    targets = tuple(meta["targets"])
    protocol = default_protocol(*targets)
    expected = meta["per_target"][targets[0]]["masked_MAE"]["_avg"]
    tolerance = max(1e-9, abs(expected) * 1e-5)

    candidates = _candidates(run)
    if not candidates:
        raise SystemExit(f"{run.label} 没有预测数组，无法取曲线；该目录只有 run.json")

    for candidate in reversed(candidates):
        recomputed = forecast_metrics.from_predictions(
            candidate, protocol, meta["interval_minutes"] * 60, targets
        )[targets[0]]["masked_MAE"]["_avg"]
        if abs(recomputed - expected) <= tolerance:
            return candidate

    raise SystemExit(
        f"{run.label} 里没有任何 npz 与 run.json 的 MAE({expected:.6f}) 对得上"
        f"（{len(candidates)} 个候选）。产物可能被覆盖，请重跑该实验。"
    )


def _load(run: RunRef) -> tuple[np.ndarray, np.ndarray]:
    """解压整个 npz（单份约 30 MB）。只在推导小表时调用，**不要缓存整份数组**，
    四个模型各留一份就是 120 MB 起，而且每次都要重新解压，缓存小表才是有效的。"""
    data = np.load(resolve(run))
    return data["prediction"], data["truth"]


@lru_cache(maxsize=8)
def load_matrix(target: str = "traffic_flow", model: str = OURS, tag: str | None = None) -> np.ndarray:
    """(测试样本, 节点) 的真实观测负荷，供网络图按时刻着色；约 2.4 MB 一张。"""
    _, truth = _load(run_for(target, model, tag))
    return np.asarray(truth[:, 0, :, 0], dtype=np.float32)


@lru_cache(maxsize=32)
def _node_pair(run: RunRef, node: int) -> tuple[np.ndarray, np.ndarray]:
    """某节点的预测与真值两条序列，形状 (测试样本, 12)。"""
    prediction, truth = _load(run)
    return prediction[:, :, node, 0], truth[:, :, node, 0]


def sample_count(target: str = "traffic_flow", model: str = OURS) -> int:
    return int(load_matrix(target, model).shape[0])


def node_series(node: int, target: str, models: tuple[str, ...], sample: int) -> dict[str, Any]:
    """某节点在某个测试样本上的历史、预测与真值。

    历史取自前 12 个样本的第一步观测值，npz 里没有输入窗口，但 truth[i][0] 就是 i+1
    时刻的观测，串起来正好是 t 时刻之前的 12 个点。
    """
    if not models:
        raise SystemExit("至少要选一个模型")
    sample = int(np.clip(sample, 12, sample_count(target, models[0]) - 1))
    node = int(np.clip(node, 0, 169))

    history: list[float] = []
    truth: list[float] = []
    future: dict[str, list[float]] = {}
    for model in models:
        predicted, actual = _node_pair(run_for(target, model), node)
        if not future:
            history = [float(value) for value in actual[sample - 12 : sample, 0]]
            truth = [float(value) for value in actual[sample]]
        future[model] = [float(value) for value in predicted[sample]]

    # 数组是 float32，直接序列化会带出 63.99998474121094 这样的尾数，展示前先收干净。
    factor = scale(target)
    return {
        "node": node,
        "sample": sample,
        "history": [round(value * factor, 3) for value in history],
        "truth": [round(value * factor, 3) for value in truth],
        "models": {
            name: [round(value * factor, 3) for value in values] for name, values in future.items()
        },
    }


@lru_cache(maxsize=4)
def graph() -> dict[str, Any]:
    """PEMSD8 的真实拓扑，以及由站间距离还原出的近似平面布局。

    `.geo` 是空的（这份数据不带坐标），位置只能从边上的 `cost` 反推：`cost` 是相邻传感器
    之间的实际距离，两两最短路就近似路网里程，对它做 MDS 能把路网展回二维。

    不能用 `spring_layout`，它只看拓扑、不看边长，会把一条条走廊揉成一团，
    看起来就是一堆糊在一起的点和线。
    """
    import networkx as nx
    from sklearn.manifold import MDS

    origin, destination, cost = [], [], []
    with (LIBRARY / DATASET / f"{DATASET}.rel").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            _, _, src, dst, weight = line.strip().split(",")
            origin.append(int(src))
            destination.append(int(dst))
            cost.append(float(weight))

    edges = nx.DiGraph()
    edges.add_nodes_from(range(170))
    edges.add_weighted_edges_from(zip(origin, destination, cost))

    undirected = edges.to_undirected()
    lengths = dict(nx.all_pairs_dijkstra_path_length(undirected, weight="weight"))
    distances = np.array([[lengths[i][j] for j in range(170)] for i in range(170)])

    # init 显式给 classical_mds：默认值在 sklearn 1.10 会变，不写会报警告。
    embedding = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        n_init=1,
        max_iter=600,
        random_state=7,
        normalized_stress=True,
    ).fit_transform(distances)
    span = float(np.abs(embedding).max())
    if span > 0:
        embedding = embedding / span

    return {
        "nodes": [
            {"id": node, "x": float(embedding[node, 0]), "y": float(embedding[node, 1])}
            for node in range(170)
        ],
        # 保留全部 277 条有向边，不做对称化，方向正是「改动 4 用有向拓扑」的切入点。
        "edges": [[int(src), int(dst)] for src, dst in edges.edges()],
    }


def self_check() -> list[str]:
    """把展示值与报告记录逐个对照，返回不一致的描述。空列表表示全部吻合。

    容差放宽到 1%：run.json 里的 MAE 只存了 4 位小数，occupancy 乘 100 之后精度只剩 0.01。
    """
    problems = []
    for target, expected in EXPECTED_MAE.items():
        for model, want in expected.items():
            got = metrics_of(run_for(target, model), target)["avg"]
            if abs(got - want) > max(0.005, abs(want) * 0.01):
                problems.append(f"{target}/{model}: 展示 {got:.4f}，报告记录 {want}")
    return problems
