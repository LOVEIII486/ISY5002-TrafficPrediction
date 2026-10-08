"""交通监控与预测系统的控制台服务端：静态文件 + /api/* 的 JSON，只用标准库。

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

from src.demo import data, detection  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8600

# 两页用的是两个互不相干的数据集，必须在界面上写明来源，否则「同一个系统」的观感会误导。
SOURCES = {
    "detection": {
        "name": "LTA Traffic Camera Images (Singapore)",
        "detail": "data.gov.sg 实时交通图像接口 · 10 分钟采样 · 8 个点位。"
        "本系统处理的是其中 week1（2026-09-13 → 09-20，7,321 帧）。",
        "links": [
            {
                "label": "数据集 · Hugging Face",
                "url": "https://huggingface.co/datasets/LOVEIII486/isy5002-lta-traffic-images",
            },
        ],
    },
    "forecast": {
        # 原始测量出自 Caltrans，但 PEMSD8 作为研究基准是由 LibCity 以预处理好的形式公开分发的
        # （见 LibCity README 与 scripts/fetch_data.py），不需要注册。别把「Caltrans 门户要注册」
        # 写成「这个数据集要注册」，两者不是一回事。
        "name": "PEMSD8 · Caltrans PeMS District 8",
        "detail": "美国加州圣贝纳迪诺 / 河滨 · 2016-07-01 → 08-31 · 170 个检测器 · 每 5 分钟一个采样。"
        "原始测量来自 Caltrans PeMS，本系统用的是 LibCity 整理好的公开版本。",
        "links": [
            {
                "label": "数据集 · LibCity Google Drive",
                "url": "https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing",
            },
            {"label": "原始来源 · Caltrans PeMS", "url": "https://pems.dot.ca.gov/"},
        ],
    },
}

# 预测页必须讲清楚数据域不同，否则会把加州路网的预测读成监控页相机点位的预测。
DOMAIN_NOTE = (
    "注意：本页路网数据与实时监控页的相机图像<b>不是同一来源</b>。相机图像只能给出瞬时"
    "车辆计数，不足以训练时序预测模型，因此预测部分使用 PEMSD8 这一公开路网数据集独立"
    "训练与评估；下面的监测点编号属于 PEMSD8，与监控页的相机编号无关。"
)

# 网络图带不动全部 3584 个测试样本（约 6 MB JSON），按等间隔抽样到这么多帧。
GRAPH_MOMENTS = 120

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _numbers(values: Any, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _overview(_: dict[str, str]) -> dict[str, Any]:
    """系统态势：检测与预测两侧的关键状态，供总览页。

    全站只讲一个「部署中的引擎」，早先总览按 MAE 取最优（MTGNN）、预测页按自研模型
    （MultiHeadSTGCN），同一个系统两页说的模型不一样，是明显的自相矛盾。
    """
    engine = data.deployed_engine("traffic_flow")

    counts = detection.counts()
    if counts.empty:
        detected_vehicles, camera_count, confidence, frames_done = 0, 0, 0.0, 0
    else:
        detected_vehicles = int(counts[list(detection.VEHICLE_CLASSES)].to_numpy().sum())
        camera_count = int(counts["camera_id"].nunique())
        confidence = round(float(detection.detections()["conf"].mean()), 3)
        frames_done = int(len(detection.frames()))

    coverage = detection.coverage()
    complete = coverage["frames_total"] and frames_done >= coverage["frames_total"]
    warnings = data.self_check()

    return {
        "cameras": camera_count,
        "detected_vehicles": detected_vehicles,
        "confidence": confidence,
        "frames_done": frames_done,
        "frames_total": coverage["frames_total"],
        "nodes": data.sample_count() and 170,
        "targets": ["traffic_flow", "traffic_occupancy", "traffic_speed"],
        "models": ["FNN", "STGCN", "MTGNN", data.OURS],
        "engine": engine,
        "dataset": data.DATASET,
        "warnings": warnings,
        "sources": SOURCES,
        "domain_note": DOMAIN_NOTE,
        "services": _services(camera_count, frames_done, coverage, complete, warnings, engine),
    }


def _services(
    camera_count: int,
    frames_done: int,
    coverage: dict[str, Any],
    complete: bool,
    warnings: list[str],
    engine: dict[str, Any],
) -> list[dict[str, str]]:
    """模块状态表。状态色只是提示，文字才是判据。"""
    return [
        {
            "name": "交通检测",
            "state": "on" if camera_count else "warn",
            "detail": f"{camera_count} 个监测点位 · 已处理 {frames_done:,} 帧"
            + ("" if complete else f" / {coverage.get('frames_total', 0):,}"),
            "hint": f"YOLO {coverage.get('model') or '—'} · {coverage.get('imgsz') or '—'} px",
        },
        {
            "name": "交通预测",
            "state": "on",
            "detail": f"{data.DATASET} · 170 个节点 · 未来 12 步（60 分钟）",
            "hint": f"预测引擎 {engine['model']} · 平均误差 {engine['mape_avg']:.1f}%",
        },
        {
            "name": "数据接入",
            "state": "warn" if warnings else "on",
            "detail": "data.gov.sg 实时交通图像 · 8 台相机" if not warnings else f"{len(warnings)} 项待核对",
            "hint": "采集区间 2026-09-13 → 2026-09-20",
        },
    ]


def _forecast_graph(query: dict[str, str]) -> dict[str, Any]:
    """路网着色要跟着所选指标走，否则选了「速度」图上画的还是流量。"""
    topology = data.graph()
    loads = data.load_matrix(query.get("target", "traffic_flow"))
    total = loads.shape[0]
    stride = max(1, total // GRAPH_MOMENTS)
    indices = list(range(0, total, stride))
    return {
        "nodes": topology["nodes"],
        "edges": topology["edges"],
        "samples": indices,
        "load": [_numbers(loads[index]) for index in indices],
    }


def _forecast_series(query: dict[str, str]) -> dict[str, Any]:
    node = int(query.get("node", 42))
    target = query.get("target", "traffic_flow")
    models = tuple(name for name in query.get("models", "").split(",") if name)
    sample = int(query.get("sample", data.sample_count(target) - 1))
    return data.node_series(node, target, models or (data.OURS,), sample)


def _forecast_engine(query: dict[str, str]) -> dict[str, Any]:
    """部署中的预测引擎与它的准确度。面向值班人员，只给百分比误差。"""
    return data.deployed_engine(query.get("target", "traffic_flow"))


def _latest(_: dict[str, str]) -> dict[str, Any]:
    return {"frames": detection.latest_frames()}


# ---- 检测页：读 results/detection/ 的产物；没有产物时返回空状态 ----


def _cameras(_: dict[str, str]) -> dict[str, Any]:
    frame = detection.cameras()
    counts = detection.counts()
    totals = (
        counts.groupby("camera_id")[list(detection.VEHICLE_CLASSES)].sum().sum(axis=1)
        if not counts.empty
        else {}
    )
    rows = []
    for row in frame.itertuples():
        segments = {"causeway": "新柔长堤", "second_link": "第二通道", "sentosa_gateway": "圣淘沙"}
        rows.append(
            {
                "id": row.camera_id,
                "name": f"{segments.get(row.road_segment, row.road_segment)} · {row.direction}",
                "lat": None if pd.isna(row.lat) else round(float(row.lat), 6),
                "lon": None if pd.isna(row.lon) else round(float(row.lon), 6),
                "total": int(totals.get(row.camera_id, 0)),
                "frames": int(row.frames),
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
            raise SystemExit("还没有任何已处理的画面；先跑 python -m src.detection.run_detect")
        image = str(frame.sort_values("captured_at")["image"].iloc[-1])
    detail = detection.frame_detail(str(image))
    detail["url"] = f"/lta/{image}"
    return detail


def _detection_series(query: dict[str, str]) -> dict[str, Any]:
    return detection.series(query.get("camera", ""))


def _rhythm(_: dict[str, str]) -> dict[str, Any]:
    return detection.rhythm()


ROUTES = {
    "/api/overview": _overview,
    "/api/forecast/graph": _forecast_graph,
    "/api/forecast/series": _forecast_series,
    "/api/forecast/engine": _forecast_engine,
    "/api/detection/cameras": _cameras,
    "/api/detection/frame": _frame,
    "/api/detection/series": _detection_series,
    "/api/detection/rhythm": _rhythm,
    "/api/detection/latest": _latest,
}


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "TrafficConsole/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        try:
            if parsed.path in ROUTES:
                self._send_json(HTTPStatus.OK, ROUTES[parsed.path](query))
            elif parsed.path.startswith("/lta/"):
                self._send_image(unquote(parsed.path[len("/lta/") :]))
            elif parsed.path.startswith("/api/"):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"未知接口 {parsed.path}；已知：{sorted(ROUTES)}"},
                )
            else:
                self._send_static(parsed.path)
        except SystemExit as error:
            # data.py 用 SystemExit 报「产物缺失/对不上」，那正是要原样端给前端的信息。
            self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(error)})

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
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", cache)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="交通预测可视化控制台")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    problems = data.self_check()
    if problems:
        print("展示值与报告记录不一致，先处理再启动：")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    # 预热：路网布局要跑 spring_layout，曲线要解压几十 MB 的 npz。放在这里一次性付掉，
    # 否则第一个打开页面的人会看到一段没有反馈的空白。
    print("预热产物缓存 ...", flush=True)
    data.graph()
    data.load_matrix()
    data.deployed_engine("traffic_flow")

    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    # flush：重定向到文件时 stdout 是块缓冲的，不刷新就看不到启动信息。
    print(f"控制台已启动 -> http://{args.host}:{args.port}", flush=True)
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
