"""指标：从执行器保存的预测数组按目标分别重算，口径与 LibCity 的 masked_* 一致。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .protocols import Protocol

MASKED_KEYS = ("masked_MAE", "masked_RMSE", "masked_MAPE")


# LibCity 的 masked_* 把 |真值| < 1e-4 一并视为零值（逆变换后的极小残值），须照此屏蔽，
# 否则这些点进入 MAPE 会把结果放大几个数量级。
_ZERO_TOLERANCE = 1e-4


def _per_step_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, ...]:
    """逐预测步计算 MAE / RMSE / MAPE，零值与无效值不参与。"""
    mae, rmse, mape = [], [], []
    for step_pred, step_true in zip(np.moveaxis(prediction, 1, 0), np.moveaxis(truth, 1, 0)):
        mask = (
            np.isfinite(step_pred)
            & np.isfinite(step_true)
            & (np.abs(step_true) >= _ZERO_TOLERANCE)
        )
        if not mask.any():
            continue
        error = step_pred[mask] - step_true[mask]
        mae.append(np.abs(error).mean())
        rmse.append(np.sqrt((error**2).mean()))
        mape.append(np.abs(error / step_true[mask]).mean())
    return np.array(mae), np.array(rmse), np.array(mape)


def _summarize_series(series: np.ndarray, protocol: Protocol, interval_seconds: int) -> dict:
    """把逐预测步的序列压成 12 步平均与 15 / 30 / 60 分钟三个点。"""
    summary = {"_avg": float(series.mean())}
    for step in protocol.horizon_steps:
        summary[f"@{step * interval_seconds // 60}min"] = float(series[step - 1])
    return summary


def from_predictions(
    npz_path: Path, protocol: Protocol, interval_seconds: int, targets: tuple[str, ...]
) -> dict[str, dict]:
    """从预测数组按目标分别算指标；LibCity 的评估器会把多个目标的列混在一起算。"""
    arrays = np.load(npz_path)
    prediction, truth = arrays["prediction"], arrays["truth"]
    result: dict[str, dict] = {}
    for index, name in enumerate(targets):
        mae, rmse, mape = _per_step_metrics(prediction[..., index], truth[..., index])
        result[name] = {
            "masked_MAE": _summarize_series(mae, protocol, interval_seconds),
            "masked_RMSE": _summarize_series(rmse, protocol, interval_seconds),
            "masked_MAPE": _summarize_series(mape, protocol, interval_seconds),
        }
    return result


def flatten(per_target: dict[str, dict]) -> dict[str, float]:
    """摊平成 CSV 用的一行。"""
    row: dict[str, float] = {}
    for name, series in per_target.items():
        for key, values in series.items():
            for horizon, value in values.items():
                row[f"{name}|{key}{horizon}"] = value
    return row


def aggregate(records: list[dict]) -> dict:
    """对同一组配置的多个种子求均值与标准差。"""
    rows = [record for record in records if record.get("per_target")]
    if not rows:
        return {}
    keys = set()
    for record in rows:
        keys.update(flatten(record["per_target"]))
    out: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = np.array(
            [flatten(record["per_target"]).get(key, np.nan) for record in rows], dtype=float
        )
        values = values[np.isfinite(values)]
        out[key] = {
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return out
