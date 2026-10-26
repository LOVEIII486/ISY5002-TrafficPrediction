"""交通监控与预测工作台的服务端：静态文件 + /api/* 的 JSON，只用标准库。

    python -m src.demo.web.server
    python -m src.demo.web.server --port 8600

两个数据域各读自己的产物：检测侧 `results/detection/`（见 `src/demo/detection.py`），
预测侧 `results/forecast/`（见 `src/demo/data.py`）。服务端只做取数与整形，不产出数据。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

# 以 -m 或脚本方式启动时，仓库根都不在 sys.path 上，src.* 无法 import。
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.demo import congestion, data, detection, filtering  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parent
# 浏览器切换相机、刷新页面时可能主动断开连接；这不是服务端错误。
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
DEFAULT_PORT = 8600

DEFAULT_LANG = "en"

# 服务端生成的界面文字（数据来源、模块状态、路段/方向、预见期、错误）都在这里。
# 前端自己的文案在 assets/i18n.js；两边用同一套取值：en / zh。
MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "sources.detection.detail": "data.gov.sg real-time traffic camera API · 10-minute sampling · "
        "8 camera points. This workbench uses 15,316 frames captured 2026-09-13 → 09-29.",
        "sources.detection.link.hf": "Dataset · Hugging Face",
        "sources.forecast.detail": "San Bernardino / Riverside, California, USA · 2016-07-01 → 08-31 · "
        "170 detectors · one sample every 5 minutes. Raw measurements come from Caltrans PeMS; "
        "this system uses the public preprocessed LibCity release.",
        "sources.forecast.link.libcity": "Dataset · LibCity Google Drive",
        "sources.forecast.link.caltrans": "Raw source · Caltrans PeMS",
        "overview.domainNote": (
            "Note: the road-network data on this page and the camera images on the live monitoring "
            "page <b>do not come from the same source</b>. Camera images only give instantaneous "
            "vehicle counts, which are not enough to train a time-series forecasting model, so "
            "forecasting is trained and evaluated separately on the public PEMSD8 road-network "
            "dataset. The monitoring point IDs below belong to PEMSD8 and are unrelated to the "
            "camera IDs on the monitoring page."
        ),
        "service.detection.name": "Traffic detection",
        "service.detection.detail": "{cameras} points · {frames} frames processed",
        "service.detection.hint": "YOLO {model} · {imgsz} px",
        "service.forecast.name": "Traffic forecasting",
        "service.forecast.detail": "{dataset} · 170 nodes · next 12 steps (60 minutes)",
        "service.forecast.hint": "Engine {model} · mean error {mape:.1f}%",
        "service.data.name": "Data ingestion",
        "service.data.detail": "data.gov.sg live traffic images · 8 cameras",
        "service.data.detailWarnings": "{count} item(s) to verify",
        "service.data.hint": "Capture range {start} → {end}",
        "segment.causeway": "Causeway",
        "segment.second_link": "Second Link",
        "segment.sentosa_gateway": "Sentosa Gateway",
        "direction.towards_johor": "Towards Johor",
        "direction.towards_bke": "Towards BKE",
        "direction.towards_checkpoint": "Towards Checkpoint",
        "direction.second_link_view": "Second Link view",
        "direction.towards_tuas": "Towards Tuas",
        "direction.checkpoint_view": "Checkpoint view",
        "direction.towards_telok_blangah": "Towards Telok Blangah",
        "direction.towards_sentosa": "Towards Sentosa",
        "horizon.15": "+15 min",
        "horizon.30": "+30 min",
        "horizon.60": "+60 min",
        "error.noRuns": "No run.json found under {path}. Run: python -m src.forecast.run_benchmark",
        "error.missingRun": "No such artifact: {target}/{model}/{tag}; available for this target "
        "and model: {available}",
        "error.noPredictions": "{label} has no prediction array; that directory only contains run.json",
        "error.noMatchingPredictions": "{label} has no npz matching the MAE in run.json "
        "({expected:.6f}) ({count} candidates). The artifacts may have been overwritten; "
        "rerun the experiment.",
        "error.noModels": "Select at least one model",
        "error.noFrames": "No processed frames yet. Run: python -m src.detection.run_detect",
        "error.unknownEndpoint": "Unknown endpoint {path}; known: {known}",
        "error.internal": "Internal error: {error}",
        "warning.metricsMismatch": "{target}/{model}: showing {got:.4f}, report records {want}",
    },
    "zh": {
        "sources.detection.detail": "data.gov.sg 实时交通图像接口 · 10 分钟采样 · 8 个点位。"
        "本系统处理的是 2026-09-13 → 09-29 的 15,316 帧。",
        "sources.detection.link.hf": "数据集 · Hugging Face",
        "sources.forecast.detail": "美国加州圣贝纳迪诺 / 河滨 · 2016-07-01 → 08-31 · 170 个检测器 · "
        "每 5 分钟一个采样。原始测量来自 Caltrans PeMS，本系统用的是 LibCity 整理好的公开版本。",
        "sources.forecast.link.libcity": "数据集 · LibCity Google Drive",
        "sources.forecast.link.caltrans": "原始来源 · Caltrans PeMS",
        "overview.domainNote": (
            "注意：本页路网数据与实时监控页的相机图像<b>不是同一来源</b>。相机图像只能给出瞬时"
            "车辆计数，不足以训练时序预测模型，因此预测部分使用 PEMSD8 这一公开路网数据集独立"
            "训练与评估；下面的监测点编号属于 PEMSD8，与监控页的相机编号无关。"
        ),
        "service.detection.name": "交通检测",
        "service.detection.detail": "{cameras} 个监测点位 · 已处理 {frames} 帧",
        "service.detection.hint": "YOLO {model} · {imgsz} px",
        "service.forecast.name": "交通预测",
        "service.forecast.detail": "{dataset} · 170 个节点 · 未来 12 步（60 分钟）",
        "service.forecast.hint": "预测引擎 {model} · 平均误差 {mape:.1f}%",
        "service.data.name": "数据接入",
        "service.data.detail": "data.gov.sg 实时交通图像 · 8 台相机",
        "service.data.detailWarnings": "{count} 项待核对",
        "service.data.hint": "采集区间 {start} → {end}",
        "segment.causeway": "新柔长堤",
        "segment.second_link": "第二通道",
        "segment.sentosa_gateway": "圣淘沙",
        "direction.towards_johor": "往柔佛",
        "direction.towards_bke": "往 BKE",
        "direction.towards_checkpoint": "往关卡",
        "direction.second_link_view": "第二通道全景",
        "direction.towards_tuas": "往大士",
        "direction.checkpoint_view": "关卡全景",
        "direction.towards_telok_blangah": "往直落布兰雅",
        "direction.towards_sentosa": "往圣淘沙",
        "horizon.15": "+15 分钟",
        "horizon.30": "+30 分钟",
        "horizon.60": "+60 分钟",
        "error.noRuns": "没有在 {path} 下找到任何 run.json；先跑 src.forecast.run_benchmark",
        "error.missingRun": "没有这个产物：{target}/{model}/{tag}；该目标该模型下有：{available}",
        "error.noPredictions": "{label} 没有预测数组，无法取曲线；该目录只有 run.json",
        "error.noMatchingPredictions": "{label} 里没有任何 npz 与 run.json 的 MAE"
        "({expected:.6f}) 对得上（{count} 个候选）。产物可能被覆盖，请重跑该实验。",
        "error.noModels": "至少要选一个模型",
        "error.noFrames": "还没有任何已处理的画面；先跑 python -m src.detection.run_detect",
        "error.unknownEndpoint": "未知接口 {path}；已知：{known}",
        "error.internal": "内部错误：{error}",
        "warning.metricsMismatch": "{target}/{model}: 展示 {got:.4f}，报告记录 {want}",
    },
}

# 两页用的是两个互不相干的数据集，必须在界面上写明来源，否则「同一个系统」的观感会误导。
SOURCES = {
    "detection": {
        "name": "LTA Traffic Camera Images (Singapore)",
        "detail_key": "sources.detection.detail",
        "links": [
            {
                "label_key": "sources.detection.link.hf",
                "url": "https://huggingface.co/datasets/LOVEIII486/isy5002-lta-traffic-images",
            },
        ],
    },
    "forecast": {
        # 原始测量出自 Caltrans，但 PEMSD8 作为研究基准是由 LibCity 以预处理好的形式公开分发的
        # （见 LibCity README 与 scripts/fetch_data.py），不需要注册。别把「Caltrans 门户要注册」
        # 写成「这个数据集要注册」，两者不是一回事。
        "name": "PEMSD8 · Caltrans PeMS District 8",
        "detail_key": "sources.forecast.detail",
        "links": [
            {
                "label_key": "sources.forecast.link.libcity",
                "url": "https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing",
            },
            {"label_key": "sources.forecast.link.caltrans", "url": "https://pems.dot.ca.gov/"},
        ],
    },
}

SEGMENTS = {
    "causeway": "segment.causeway",
    "second_link": "segment.second_link",
    "sentosa_gateway": "segment.sentosa_gateway",
}

DIRECTIONS = {
    "Towards Johor": "direction.towards_johor",
    "Towards BKE": "direction.towards_bke",
    "Towards Checkpoint": "direction.towards_checkpoint",
    "Second Link view": "direction.second_link_view",
    "Towards Tuas": "direction.towards_tuas",
    "Checkpoint view": "direction.checkpoint_view",
    "Towards Telok Blangah": "direction.towards_telok_blangah",
    "Towards Sentosa": "direction.towards_sentosa",
}

HORIZON_KEYS = ("horizon.15", "horizon.30", "horizon.60")


def _lang(query: dict[str, str]) -> str:
    value = (query.get("lang") or "").strip().lower()
    return "zh" if value in {"zh", "zh-cn", "cn"} else DEFAULT_LANG


def _t(lang: str, key: str, **values: Any) -> str:
    table = MESSAGES.get(lang) or MESSAGES[DEFAULT_LANG]
    template = table.get(key) or MESSAGES[DEFAULT_LANG].get(key) or key
    return template.format(**values)


def _lookup(lang: str, mapping: dict[str, str], value: str) -> str:
    """未知取值原样返回：以后多一个方向时，界面显示原文而不是一个 key。"""
    key = mapping.get(value)
    return _t(lang, key) if key else value


def _sources(lang: str) -> dict[str, Any]:
    return {
        key: {
            "name": item["name"],
            "detail": _t(lang, item["detail_key"]),
            "links": [
                {"label": _t(lang, link["label_key"]), "url": link["url"]}
                for link in item["links"]
            ],
        }
        for key, item in SOURCES.items()
    }


def _engine(target: str, lang: str) -> dict[str, Any]:
    """data.py 只负责取数；预见期标签属于展示层，在这里换语言。"""
    engine = data.deployed_engine(target)
    for item, key in zip(engine["horizons"], HORIZON_KEYS):
        item["label"] = _t(lang, key)
    return engine

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _overview(query: dict[str, str]) -> dict[str, Any]:
    """总览：一句话态势 + 跨两域的注意力列表 + 两域健康。

    两个数据域始终分开：注意力条目带域名，健康卡各占一张，不合成单一风险分。
    """
    lang = _lang(query)
    risk = data.speed_risk(None, 60)
    reliability = detection.reliability()
    engine = _engine("traffic_speed", lang)
    warnings = [_t(lang, f"warning.{item['code']}", **item) for item in data.self_check()]

    return {
        "time": risk["time"],
        "forecast": {
            "sample": risk["sample"],
            "summary": risk["summary"],
            "thresholds": risk["thresholds"],
            "time": risk["time"],
        },
        "detection": reliability["overall"],
        "health": _health(reliability, engine, risk),
        "attention": _attention(risk, reliability, warnings),
        "engine": engine,
        "dataset": data.DATASET,
        "cameras": reliability["overall"]["cameras"],
        "nodes": risk["summary"]["total"],
        "warnings": warnings,
        "sources": _sources(lang),
        "domain_note": _t(lang, "overview.domainNote"),
    }


def _health(
    reliability: dict[str, Any],
    engine: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    cameras = reliability["cameras"]
    flagged = sorted(
        (
            (camera_id, info)
            for camera_id, info in cameras.items()
            if info["health"] in ("warn", "watch")
        ),
        key=lambda pair: (-pair[1]["zero_rate"], pair[0]),
    )
    return {
        "detection": {
            **reliability["overall"],
            "attention": len(flagged),
            "attention_ids": [camera_id for camera_id, _ in flagged],
        },
        "forecast": {
            "engine": engine["model"],
            "mape_avg": engine["mape_avg"],
            "nodes": risk["summary"]["total"],
            "sample": risk["sample"],
            "time": risk["time"],
            "severity_counts": risk["summary"]["severity_counts"],
        },
    }


def _attention(
    risk: dict[str, Any],
    reliability: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """语言中立的注意力条目：前端按 code 取文案，避免在服务端复制 20 条产品文案。"""
    items: list[dict[str, Any]] = []

    for row in risk["nodes"]:
        if row["severity"] == "normal":
            break
        if len(items) >= 3:
            break
        items.append({
            "domain": "forecast",
            "severity": row["severity"],
            "code": "attention.forecast.congestion",
            "node": row["node"],
            "sample": risk["sample"],
            "values": {
                "now": row["now"],
                "worst": row["worst"],
                "minutes": row["worst_at_min"],
            },
            "view": "forecast",
        })

    cameras = sorted(
        (
            (camera_id, info)
            for camera_id, info in reliability["cameras"].items()
            if info["health"] in ("warn", "watch")
        ),
        key=lambda pair: (-pair[1]["zero_rate"], pair[0]),
    )
    for camera_id, info in cameras:
        if len(items) >= 5:
            break
        if "lowDetections" in info["issues"]:
            code = "attention.detection.lowDetections"
        elif "missingFrames" in info["issues"]:
            code = "attention.detection.missingFrames"
        else:
            code = "attention.detection.lowConfidence"
        items.append({
            "domain": "camera",
            "severity": info["health"],
            "code": code,
            "camera": camera_id,
            "values": {
                "zeroRate": info["zero_rate"],
                "completeness": info["completeness"],
                "confidence": info["confidence"],
            },
            "view": "monitoring",
        })

    if warnings and len(items) < 5:
        items.append({
            "domain": "system",
            "severity": "warn",
            "code": "attention.system.metricsMismatch",
            "values": {"count": len(warnings)},
            "view": "overview",
        })
    return items


def _forecast_risk(query: dict[str, str]) -> dict[str, Any]:
    return data.speed_risk(_int(query.get("sample"), None), _int(query.get("horizon"), 60))


def _forecast_detail(query: dict[str, str]) -> dict[str, Any]:
    return data.speed_detail(
        _int(query.get("node"), 0),
        _int(query.get("sample"), None),
        _int(query.get("horizon"), 60),
    )


def _forecast_replay(_: dict[str, str]) -> dict[str, Any]:
    return data.replay_range()


def _forecast_graph(query: dict[str, str]) -> dict[str, Any]:
    """路网图只画一件事：所选回放时刻、所选预见期下的预测严重度。"""
    risk = data.speed_risk(_int(query.get("sample"), None), _int(query.get("horizon"), 60))
    topology = data.graph()
    rows = {row["node"]: row for row in risk["nodes"]}
    return {
        "sample": risk["sample"],
        "horizon": risk["horizon"],
        "time": risk["time"],
        "thresholds": risk["thresholds"],
        "summary": risk["summary"],
        "nodes": [
            {
                **node,
                "severity": rows[node["id"]]["severity"],
                "status": rows[node["id"]]["status"],
                "now": rows[node["id"]]["now"],
                "worst": rows[node["id"]]["worst"],
                "worst_at_min": rows[node["id"]]["worst_at_min"],
            }
            for node in topology["nodes"]
        ],
        "edges": topology["edges"],
    }


def _forecast_engine(query: dict[str, str]) -> dict[str, Any]:
    """部署中的预测引擎与它的准确度。面向值班人员，只给百分比误差。"""
    return _engine(query.get("target", "traffic_flow"), _lang(query))


# ---- 检测页：读 results/detection/ 的产物；没有产物时返回空状态 ----


def _cameras(query: dict[str, str]) -> dict[str, Any]:
    lang = _lang(query)
    frame = detection.cameras()
    counts = detection.counts()
    totals = (
        counts.groupby("camera_id")[list(detection.VEHICLE_CLASSES)].sum().sum(axis=1)
        if not counts.empty
        else {}
    )
    reliability = detection.reliability()["cameras"]
    rows = []
    for row in frame.itertuples():
        stats = reliability.get(row.camera_id, {})
        first = getattr(row, "first", None)
        first_iso = None if first is None or pd.isna(first) else pd.Timestamp(first).isoformat()
        rows.append(
            {
                "id": row.camera_id,
                "name": f"{_lookup(lang, SEGMENTS, row.road_segment)} · "
                f"{_lookup(lang, DIRECTIONS, row.direction)}",
                "lat": None if pd.isna(row.lat) else round(float(row.lat), 6),
                "lon": None if pd.isna(row.lon) else round(float(row.lon), 6),
                "total": int(totals.get(row.camera_id, 0)),
                "frames": int(row.frames),
                "zero_rate": stats.get("zero_rate"),
                "confidence": stats.get("confidence"),
                "completeness": stats.get("completeness"),
                "health": stats.get("health", "ok"),
                "issues": stats.get("issues", []),
                "first": first_iso,
                "last": stats.get("last"),
                # Removal counts from the camera filter; None when the filter has not been run.
                "filter": filtering.summary().get(row.camera_id),
            }
        )
    return {"cameras": rows, "coverage": detection.coverage()}


def _frame(query: dict[str, str]) -> dict[str, Any]:
    image = query.get("image")
    if not image:
        frame = detection.frames()
        camera = query.get("camera")
        if camera:
            frame = frame[frame["camera_id"] == camera]
        if frame.empty:
            raise SystemExit(_t(_lang(query), "error.noFrames"))
        frame = frame.sort_values("captured_at")
        image = None
        moment = query.get("time")
        if moment:
            try:
                target = pd.to_datetime(moment, utc=True)
                offset = (frame["captured_at"] - target).abs()
                image = str(frame.loc[offset.idxmin(), "image"])
            except (TypeError, ValueError):
                image = None
        if image is None:
            image = str(frame["image"].iloc[-1])
    detail = filtering.annotate(detection.frame_detail(str(image)))
    detail["congestion"] = congestion.frame_level(str(image))
    detail["url"] = f"/lta/{image}"
    return detail


def _detection_series(query: dict[str, str]) -> dict[str, Any]:
    return detection.series(query.get("camera", ""))


def _congestion_heatmap(query: dict[str, str]) -> dict[str, Any]:
    """Weekday x hour share of congested frames for one camera; empty when not available."""
    return congestion.heatmap(query.get("camera", "")) or {}


ROUTES = {
    "/api/overview": _overview,
    "/api/forecast/risk": _forecast_risk,
    "/api/forecast/detail": _forecast_detail,
    "/api/forecast/replay": _forecast_replay,
    "/api/forecast/graph": _forecast_graph,
    "/api/forecast/engine": _forecast_engine,
    "/api/detection/cameras": _cameras,
    "/api/detection/frame": _frame,
    "/api/detection/series": _detection_series,
    "/api/congestion/heatmap": _congestion_heatmap,
}


class WorkbenchHTTPServer(ThreadingHTTPServer):
    """HTTP server that treats client disconnects as normal, not as server errors."""

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, _CLIENT_GONE):
            return
        if isinstance(error, OSError) and getattr(error, "winerror", None) in (10053, 10054):
            return
        super().handle_error(request, client_address)


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "TrafficWorkbench/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        lang = _lang(query)
        try:
            if parsed.path in ROUTES:
                self._send_json(HTTPStatus.OK, ROUTES[parsed.path](query))
            elif parsed.path.startswith("/lta/"):
                self._send_image(unquote(parsed.path[len("/lta/") :]))
            elif parsed.path.startswith("/api/"):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": _t(lang, "error.unknownEndpoint",
                                 path=parsed.path, known=sorted(ROUTES))},
                )
            else:
                self._send_static(parsed.path)
        except data.ArtifactError as error:
            # 产物类错误带机器可读的 code，才能按当前语言给出可照做的修复动作。
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": _t(lang, f"error.{error.code}", **error.context)},
            )
        except SystemExit as error:
            # 服务端自己抛的 SystemExit 已经按 lang 生成过文字，原样端给前端。
            self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
        except _CLIENT_GONE:
            # 浏览器在图片/JSON 写完前切走（切相机、刷新、关闭页面）。
            # 客户端已经不在了，不要再尝试返回 500；否则会打印二次 traceback。
            self.close_connection = True
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": _t(lang, "error.internal", error=repr(error))},
            )

    def _send_static(self, path: str) -> None:
        relative = "index.html" if path in ("", "/") else unquote(path).lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if not target.is_file() or WEB_ROOT.resolve() not in target.parents:
            self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        self._send_bytes(
            HTTPStatus.OK,
            target.read_bytes(),
            _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
        )

    def _send_image(self, name: str) -> None:
        """相机图按文件名查表取用，文件名里带内容哈希，所以可以无条件长缓存。"""
        target = detection.image_path(name)
        if target is None:
            self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        self._send_bytes(
            HTTPStatus.OK,
            target.read_bytes(),
            "image/jpeg",
            cache="public, max-age=31536000, immutable",
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(
        self, status: HTTPStatus, payload: bytes, content_type: str, cache: str = "no-store"
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.send_header("cache-control", cache)
            self.end_headers()
            self.wfile.write(payload)
        except _CLIENT_GONE:
            # 写响应时客户端已经断开（WinError 10053/10054）。静默收尾即可。
            self.close_connection = True

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="交通监控与预测工作台")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    problems = data.self_check()
    if problems:
        print("展示值与报告记录不一致，先处理再启动：")
        for problem in problems:
            print(f"  - {_t('zh', 'warning.' + problem['code'], **problem)}")
        raise SystemExit(1)

    # 预热：路网布局要跑 spring_layout，曲线要解压几十 MB 的 npz。放在这里一次性付掉，
    # 否则第一个打开页面的人会看到一段没有反馈的空白。
    print("预热预测风险与回放时间 ...", flush=True)
    data.graph()
    data.deployed_engine("traffic_speed")
    data.speed_risk(data.latest_sample(), 60)
    data.replay_range()

    server = WorkbenchHTTPServer((args.host, args.port), ConsoleHandler)
    # flush：重定向到文件时 stdout 是块缓冲的，不刷新就看不到启动信息。
    print(f"工作台已启动 -> http://{args.host}:{args.port}", flush=True)
    print(f"已核对 {len(data.discover())} 个实验产物，展示值与报告记录一致", flush=True)
    print("按 Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
