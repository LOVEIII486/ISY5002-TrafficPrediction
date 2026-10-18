"""人工抽查：评估划分里的图在训练集中最相似的图是不是同一机位。生成抽查图与待填写的记录表。

    python -m src.detection.audit_split                          # 默认 v 当前版本、test、40 张、种子 42
    python -m src.detection.audit_split --split dev --n 40

场景分组只能近似摄像头身份，残余的同机位重叠要靠人工抽查估计。每对图左边是评估图、右边是训练集中
「背景缩略图 / 抠车背景缩略图」相关系数最高的那张，下方放大两张图右下角的时间戳，便于判断是否近邻帧。

产物 `results/detection/finetune/audit/<数据版本>_<划分>/`：
    sheet_*.jpg     抽查图（每张 20 对）
    audit.csv       每对一行；same_camera、test_time、train_time 等列由人工填写
    audit.json      抽样参数（版本、划分、种子、数量）

抽样结果只是本次抽查的观察值，样本小时比例的区间很宽，报告里要写清楚样本量与判定人。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import miotcd, paths


def tile(image_path: Path, label: str, size=(240, 160)):
    """缩略图 + 放大的右下角时间戳条。"""
    import cv2

    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    stamp = image[int(height * 0.88) :, int(width * 0.62) :]
    stamp = cv2.resize(stamp, (size[0], 40), interpolation=cv2.INTER_CUBIC)  # 固定高度：两种原图尺寸裁出来差一两像素
    thumb = cv2.resize(image, size)
    cv2.rectangle(thumb, (0, 0), (size[0], 16), (0, 0, 0), -1)
    cv2.putText(thumb, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return cv2.vconcat([thumb, stamp])


def main() -> None:
    parser = argparse.ArgumentParser(description="抽查评估划分与训练集的同机位重叠")
    parser.add_argument("--version", default=paths.MIOTCD_VERSION)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import cv2

    data_dir = paths.ROOT / "datasets" / f"miotcd_yolo_{args.version}"
    listing = lambda split: [Path(line).stem for line in (data_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()]
    train, target = listing("train"), listing(args.split)
    ids = sorted(miotcd.load_boxes()[0])
    _sha, _dhash, thumb, masked = miotcd.image_features(ids)
    index = {image_id: i for i, image_id in enumerate(ids)}
    picked = sorted(random.Random(args.seed).sample(target, min(args.n, len(target))))
    a, b = [index[i] for i in picked], [index[i] for i in train]
    similarity = np.maximum(thumb[a] @ thumb[b].T, masked[a] @ masked[b].T)
    best, nearest = similarity.max(axis=1), similarity.argmax(axis=1)

    out = paths.FINETUNE_RESULTS / "audit" / f"{args.version}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "audit.csv"
    if csv_path.is_file():
        # 已有记录（可能已填写判定）：参数或配对不一致就拒绝，免得新图片配上旧判定
        existing = pd.read_csv(csv_path, dtype=str)
        previous = json.loads((out / "audit.json").read_text(encoding="utf-8")) if (out / "audit.json").is_file() else {}
        same_pairs = list(existing[f"{args.split}_image"]) == picked and list(existing["train_image"]) == [train[k] for k in nearest]
        if previous != vars(args) or not same_pairs:
            raise SystemExit(
                f"{csv_path} 已存在，但参数或图片配对与本次不同（之前 {previous}，本次 {vars(args)}）。\n"
                "换个 --split，或先把旧记录挪走再重跑。"
            )
    rows, tiles = [], []
    for k, image_id in enumerate(picked):
        partner = train[nearest[k]]
        pair = cv2.hconcat([tile(miotcd.RAW_TRAIN / f"{image_id}.jpg", f"#{k + 1} {args.split} {image_id}"), tile(miotcd.RAW_TRAIN / f"{partner}.jpg", f"train {partner}  r={best[k]:.2f}")])
        tiles.append(cv2.copyMakeBorder(pair, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(90, 90, 90)))
        rows.append({"pair": k + 1, f"{args.split}_image": image_id, "train_image": partner, "max_corr": round(float(best[k]), 4), "same_camera": "", f"{args.split}_time": "", "train_time": "", "judged_by": "", "note": ""})
    for sheet in range(0, len(tiles), 20):
        chunk = tiles[sheet : sheet + 20]
        chunk += [np.full_like(tiles[0], 255)] * (-len(chunk) % 4)
        cv2.imwrite(str(out / f"sheet_{sheet // 20 + 1}.jpg"), cv2.vconcat([cv2.hconcat(chunk[i : i + 4]) for i in range(0, len(chunk), 4)]))
    if csv_path.is_file():
        print(f"{csv_path} 已存在且配对一致，保留已填写的判定；抽查图已重新生成")
    else:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    (out / "audit.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"抽查图与记录表 → {out}")


if __name__ == "__main__":
    main()
