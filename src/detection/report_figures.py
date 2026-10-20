"""报告用图：正式实验（v4）的测试集对比、跨域取舍、开发集与测试集一致性、检测效果对比。

    python -m src.detection.report_figures

前提：已用 evaluate.py 在 miotcd_yolo_v4 的 dev / test 与 lta_yolo 的 val 上评估过三个模型
（官方预训练 yolo26m、m736_mio_v4、m736_joint_v4）。产物在 results/detection/finetune/figures/v4/。

配色固定跟随模型（不随排名变）：官方预训练橙、只用 MIO-TCD 蓝、联合训练青。青色在浅底上对比度不足 3:1，
所以所有柱子都直接标数值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .run_detect import VEHICLE_CLASSES, load_class_map, pick_device

OUT = paths.FINETUNE_RESULTS / "figures" / "v4"
EVAL = paths.FINETUNE_RESULTS / "eval"
MODELS = [  # (评估表里的名字前缀, 图例名, 颜色, 权重)
    ("yolo26m", "官方预训练（未微调）", "#eb6834", paths.WEIGHTS),
    ("m736_mio_v4", "只用 MIO-TCD 微调", "#2a78d6", paths.FINETUNE_RESULTS / "m736_mio_v4" / "weights" / "best.pt"),
    ("m736_joint_v4", "MIO-TCD + LTA 联合微调", "#1baf7a", paths.FINETUNE_RESULTS / "m736_joint_v4" / "weights" / "best.pt"),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_ZH = {"car": "小汽车", "motorcycle": "摩托车", "bus": "公交车", "truck": "货车"}
CLASS_BGR = {"car": (214, 120, 42), "motorcycle": (52, 104, 235), "bus": (122, 175, 27), "truck": (0, 161, 237)}


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Arial Unicode MS"], "axes.unicode_minus": False})
    return plt


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def table(data: str, split: str, imgsz: int) -> pd.DataFrame:
    frame = pd.read_csv(EVAL / data / split / "comparison.csv")
    frame = frame[frame["period"] == "all"].set_index("model")
    return frame.loc[[f"{key}@{imgsz}" for key, *_ in MODELS]]


def bars(ax, groups: list[str], values: list[list[float]], fmt: str = "{:.2f}", ymax: float | None = 1.0):
    """分组柱状图：values[i] 是第 i 个模型在各组上的值。"""
    width = 0.8 / len(MODELS)
    x = np.arange(len(groups))
    for i, ((_, name, color, _), row) in enumerate(zip(MODELS, values)):
        rects = ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, row, width - 0.03, color=color, label=name)
        for rect, value in zip(rects, row):
            ax.text(rect.get_x() + rect.get_width() / 2, value + (ymax or max(map(max, values))) * 0.012, fmt.format(value), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, groups, color=INK)
    if ymax:
        ax.set_ylim(0, ymax * 1.12)
        ax.set_yticks(np.arange(0, ymax + 1e-9, 0.2))
    style(ax)


def fig_test(plt):
    t = table("miotcd_yolo_v4", "test", 736)
    metrics = [("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95")] + [(f"AP50_{c}", f"{CLASS_ZH[c]} AP") for c in VEHICLE_CLASSES]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [4.2, 1]}, facecolor=SURFACE)
    bars(ax1, [label for _, label in metrics], [[t.loc[f"{key}@736", m] for m, _ in metrics] for key, *_ in MODELS])
    ax1.set_title("冻结测试集（MIO-TCD v4，2,000 张，人工标注真值）· 越高越好", color=INK, loc="left")
    ax1.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    bars(ax2, ["计数误差"], [[t.loc[f"{key}@736", "count_MAE"]] for key, *_ in MODELS], ymax=None)
    ax2.set_title("每图计数误差（辆，越低越好）", color=INK, loc="left")
    fig.text(0.01, 0.01, "单次训练、单次评估；测试集摩托车仅 33 个框，该列波动大。筛选后的四类子集，非官方基准。", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig1_test_metrics.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_cross_domain(plt):
    mio = table("miotcd_yolo_v4", "test", 736)["mAP50"]
    lta = table("lta_yolo", "val", 960)["mAP50"]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    bars(ax, ["MIO-TCD 测试集（人工真值）", "LTA 新加坡验证集（与教师预测的一致性）"], [[mio.iloc[i], lta.iloc[i]] for i in range(len(MODELS))])
    ax.set_ylabel("mAP@0.5", color=INK2)
    ax.set_title("跨域取舍：只用北美数据微调会丢掉新加坡场景，联合微调两边兼顾", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=9)
    fig.text(0.01, 0.01, "右组的「答案」是 yolo26x 生成的伪标签，只衡量一致程度，不是准确率。", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig2_cross_domain.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_dev_vs_test(plt):
    dev = table("miotcd_yolo_v4", "dev", 736)["mAP50"]
    test = table("miotcd_yolo_v4", "test", 736)["mAP50"]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    bars(ax, ["开发集（选模型用）", "测试集（只评估一次）"], [[dev.iloc[i], test.iloc[i]] for i in range(len(MODELS))])
    ax.set_ylabel("mAP@0.5", color=INK2)
    ax.set_title("两个微调模型在 MIO-TCD 上的差距方向不一致：属于噪声范围", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_dev_vs_test.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def detect(model, class_map: dict[int, str], image_path: Path, imgsz: int):
    import cv2

    result = model.predict(str(image_path), imgsz=imgsz, conf=0.25, classes=sorted(class_map), device=pick_device(), verbose=False)[0]
    image = result.orig_img.copy()
    counts = dict.fromkeys(VEHICLE_CLASSES, 0)
    thickness = max(2, image.shape[1] // 640)
    for cls, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
        name = class_map[int(cls)]
        counts[name] += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), CLASS_BGR[name], thickness)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), counts


def ground_truth(image_path: Path, label_path: Path):
    import cv2

    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    counts = dict.fromkeys(VEHICLE_CLASSES, 0)
    for line in label_path.read_text().splitlines():
        c, cx, cy, w, h = map(float, line.split())
        name = VEHICLE_CLASSES[int(c)]
        counts[name] += 1
        cv2.rectangle(image, (int((cx - w / 2) * width), int((cy - h / 2) * height)), (int((cx + w / 2) * width), int((cy + h / 2) * height)), CLASS_BGR[name], 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), counts


def panel_figure(plt, panels, title: str, out: Path):
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.6), facecolor=SURFACE)
    for ax, (name, image, counts) in zip(axes, panels):
        ax.imshow(image)
        ax.axis("off")
        detail = "  ".join(f"{CLASS_ZH[n]} {counts[n]}" for n in VEHICLE_CLASSES)
        ax.set_title(f"{name}：共 {sum(counts.values())} 辆\n{detail}", color=INK, fontsize=10.5, loc="left")
    handles = [plt.Line2D([], [], color=np.array(CLASS_BGR[n][::-1]) / 255, linewidth=3, label=CLASS_ZH[n]) for n in VEHICLE_CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(title, color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_examples(plt):
    from ultralytics import YOLO

    loaded = [(name, YOLO(str(weights))) for _, name, _, weights in MODELS]
    maps = [(name, model, load_class_map(model.names)) for name, model in loaded]

    # LTA：白天、夜间各挑伪标签车辆最多的一张验证图
    lta = pd.read_csv(paths.SPLIT_FILE, dtype={"camera_id": str})
    lta = lta[lta["split"] == "val"].copy()
    lta["n"] = [len((paths.LABELED / "labels" / "val" / f"{Path(i).stem}.txt").read_text().splitlines()) for i in lta["image"]]
    for period, group in lta.groupby("period"):
        row = group.sort_values("n").iloc[-1]
        image_path = paths.LABELED / "images" / "val" / row["image"]
        panels = [(name, *detect(model, cmap, image_path, 1280)) for name, model, cmap in maps]
        when = "白天" if period == "day" else "夜间"
        panel_figure(plt, panels, f"LTA 相机 {row['camera_id']} · {when} · 输入 1280 · 阈值 0.25", OUT / f"fig4_lta_{period}.jpg")

    # MIO-TCD 测试集：含摩托车且框最多的一张，对照人工标注
    mio_dir = paths.ROOT / "datasets" / "miotcd_yolo_v4"
    split = pd.read_csv(mio_dir / "split.csv", dtype={"image": str})
    test = split[split["split"] == "test"]
    labels = mio_dir / "labels" / "test"
    with_moto = [i for i in test["image"] if any(line.startswith("1 ") for line in (labels / f"{Path(i).stem}.txt").read_text().splitlines())]
    pick = max(with_moto, key=lambda i: len((labels / f"{Path(i).stem}.txt").read_text().splitlines()))
    image_path = mio_dir / "images" / "test" / pick
    panels = [("人工标注（真值）", *ground_truth(image_path, labels / f"{Path(pick).stem}.txt"))]
    panels += [(name, *detect(model, cmap, image_path, 736)) for name, model, cmap in maps]
    panel_figure(plt, panels, f"MIO-TCD 测试图 {pick} · 输入 736 · 阈值 0.25", OUT / "fig5_miotcd_test_example.jpg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = setup()
    fig_test(plt)
    fig_cross_domain(plt)
    fig_dev_vs_test(plt)
    fig_examples(plt)
    print("\n".join(str(p) for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
