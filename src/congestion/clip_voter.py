"""Voter C for congestion: zero-shot CLIP classification of each camera's detection zones.

    python -m src.congestion.clip_voter                       # all frames, default crop
    python -m src.congestion.clip_voter --limit 20            # quick trial: 20 frames per camera
    python -m src.congestion.clip_voter --crop context mask zones \
        --images-from results/congestion/labeling/congestion_labels.json --out results/congestion/clip_trials
    python -m src.congestion.clip_voter --model ViT-L-14 --pretrained laion2b_s32b_b82k

A pretrained vision-language model, used without training on our data:
1. Crop the frame (--crop), pad it to a square and give it to CLIP:
   - context (default): bounding box of all detection zones, grown by 10%, unmasked;
   - mask: the zones only, everything else painted CLIP's mean colour;
   - zones: each zone cropped and masked on its own; probabilities averaged by zone area.
   context was chosen on the practice labels (accuracy 76.7% vs 65.9% zones, 59.6% mask).
2. Compare the crop with three groups of text prompts (free-flowing / moderate / congested),
   averaging the wordings within each group.
3. Softmax gives three probabilities; clip_score = p_moderate * 0.5 + p_congested. With several
   crops, clip_score uses the area-weighted probabilities, clip_max the most congested zone and
   clip_z{k} each zone.

zone_contrast (grey-level standard deviation inside the zones) is recorded to spot frames hidden by
rain, fog or glare; it does not vote. Weights (ViT-B-16, about 600 MB) are downloaded from Hugging Face
on the first run; set HF_ENDPOINT to use a mirror.

Outputs (in --out; with several crops the file names get the crop as a suffix, e.g. voter_clip_zones.csv):
- voter_clip.csv: image, camera_id, clip_free, clip_moderate, clip_congested, clip_score, clip_max,
  zone_contrast, and clip_z{k} per zone in "zones" mode;
- clip_run.json: model, prompts and settings, so the run can be reported exactly;
- clip_examples/<camera>.jpg (with --examples): the lowest- and highest-scoring crops per camera.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .common import image_index, load_frames, read_image, zone_polygons

CROPS = ("context", "mask", "zones")
FILL = (123, 117, 104)  # CLIP's mean colour (RGB, 0-255): masked pixels are ~0 after normalisation
CONTEXT_MARGIN = 0.1

PROMPTS = {
    "free": [
        "a traffic camera photo of a road with light, free-flowing traffic",
        "a traffic camera photo of an almost empty road with only a few cars",
        "a highway with sparse traffic moving freely",
    ],
    "moderate": [
        "a traffic camera photo of a road with moderate traffic",
        "a busy road with many vehicles moving steadily",
        "a highway with steady, fairly heavy traffic",
    ],
    "congested": [
        "a traffic camera photo of a congested road with a traffic jam",
        "bumper-to-bumper traffic with vehicles queuing on the road",
        "a road packed with slow or stationary vehicles in a long queue",
    ],
}
CLASSES = list(PROMPTS)


def square(crop: np.ndarray, fill) -> Image.Image:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    out = np.empty((side, side, 3), np.uint8)
    out[:] = fill
    oy, ox = (side - ch) // 2, (side - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    return Image.fromarray(out)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class Cropper:
    """Turns a frame into the square crop(s) CLIP judges, with one weight per crop (see step 1)."""

    def __init__(self, polys: list[np.ndarray], mode: str, grow: float):
        self.polys, self.mode, self.grow = polys, mode, grow
        self._cache: dict[tuple[int, int], dict] = {}

    def _geometry(self, h: int, w: int) -> dict:
        if (h, w) in self._cache:
            return self._cache[(h, w)]
        r = int(round(self.grow * w))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        zones, grown = [], []
        for poly in self.polys:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(poly * [w, h]).astype(np.int32)], 1)
            zones.append(m > 0)
            grown.append((cv2.dilate(m, kernel) if r else m) > 0)
        union = np.logical_or.reduce(zones)
        geo = {"union": union}
        if self.mode == "context":
            x1, y1, x2, y2 = bbox(union)
            mx, my = int((x2 - x1) * CONTEXT_MARGIN), int((y2 - y1) * CONTEXT_MARGIN)
            geo["parts"] = [((max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)), None, 1.0)]
        elif self.mode == "mask":
            g = np.logical_or.reduce(grown)
            geo["parts"] = [(bbox(g), g, 1.0)]
        else:
            geo["parts"] = [(bbox(g), g, float(z.sum())) for z, g in zip(zones, grown)]
        self._cache[(h, w)] = geo
        return geo

    def __call__(self, path: Path) -> tuple[list[Image.Image], np.ndarray, float]:
        img = read_image(path)
        if img is None:
            return [], np.zeros(0), float("nan")
        geo = self._geometry(*img.shape[:2])
        contrast = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[geo["union"]].std())
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crops, weights = [], []
        for (x1, y1, x2, y2), mask, weight in geo["parts"]:
            crop = rgb[y1:y2, x1:x2].copy()
            if mask is None:  # context: pad with the crop's own mean colour
                crops.append(square(crop, crop.reshape(-1, 3).mean(0).astype(np.uint8)))
            else:
                crop[~mask[y1:y2, x1:x2]] = FILL
                crops.append(square(crop, FILL))
            weights.append(weight)
        return crops, np.array(weights) / sum(weights), contrast


def text_features(model, tokenizer, device, torch):
    feats = []
    with torch.no_grad():
        for name in CLASSES:
            t = model.encode_text(tokenizer(PROMPTS[name]).to(device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            mean = t.mean(0)
            feats.append(mean / mean.norm())
    return torch.stack(feats)  # (3, d)


def save_examples(path: Path, crops: list[Image.Image], scores: list[float], size: int = 160) -> None:
    tiles = []
    for im, s in zip(crops, scores):
        t = np.array(im.resize((size, size)))[:, :, ::-1].copy()
        cv2.putText(t, f"{s:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    half = len(tiles) // 2
    grid = np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])])
    cv2.imencode(".jpg", grid)[1].tofile(str(path))


def read_subset(path: Path) -> set[str]:
    """Image names from a labels export (congestion_labels.json) or any CSV with an `image` column."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {item["image"] for item in data["items"]}
    return set(pd.read_csv(path)["image"])


def run(mode, args, model, preprocess, text, scale, device, torch, frames, index, zones, subset):
    suffix = "" if len(args.crop) == 1 else f"_{mode}"
    examples_dir = args.out / f"clip_examples{suffix}"
    if args.examples:
        examples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for cam in sorted(frames["camera_id"].unique()):
        if cam not in zones:
            print(f"{cam}: no detection zone in the config, skipped")
            continue
        cropper = Cropper(zones[cam], mode, args.grow)
        names = [n for n in frames.loc[frames["camera_id"] == cam, "image"] if n in index and (subset is None or n in subset)]
        if args.limit:
            names = names[: args.limit]
        cam_rows = []
        for start in range(0, len(names), args.batch):
            chunk = names[start:start + args.batch]
            done = [(n, *out) for n, out in zip(chunk, pool.map(lambda n: cropper(index[n]), chunk)) if out[0]]
            if not done:
                continue
            flat = [c for _, crops, _, _ in done for c in crops]
            feats = []
            for s in range(0, len(flat), args.batch):
                batch = torch.stack([preprocess(c) for c in flat[s:s + args.batch]]).to(device)
                with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
                    feats.append(model.encode_image(batch).float())
            img = torch.cat(feats)
            img = img / img.norm(dim=-1, keepdim=True)
            prob = (scale * img @ text.T).softmax(dim=-1).cpu().numpy()
            i = 0
            for n, crops, weights, contrast in done:
                pz = prob[i:i + len(crops)]
                i += len(crops)
                p = (pz * weights[:, None]).sum(0)
                sz = 0.5 * pz[:, 1] + pz[:, 2]  # score of each crop (one per zone in "zones" mode)
                row = {"image": n, "camera_id": cam, "clip_free": round(float(p[0]), 4),
                       "clip_moderate": round(float(p[1]), 4), "clip_congested": round(float(p[2]), 4),
                       "clip_score": round(float(0.5 * p[1] + p[2]), 4), "clip_max": round(float(sz.max()), 4),
                       "zone_contrast": round(contrast, 2)}
                if mode == "zones":
                    row.update({f"clip_z{k}": round(float(v), 4) for k, v in enumerate(sz, start=1)})
                cam_rows.append(row)
        rows += cam_rows
        if not cam_rows:
            continue
        scores = np.array([r["clip_score"] for r in cam_rows])
        if args.examples and len(cam_rows) >= 2 * args.examples:
            order = np.argsort(scores)
            pick = list(order[: args.examples]) + list(order[-args.examples:])
            shown = []
            for k in pick:  # the largest crop of each picked frame
                crops, weights, _ = cropper(index[cam_rows[k]["image"]])
                shown.append(crops[int(np.argmax(weights))])
            save_examples(examples_dir / f"{cam}.jpg", shown, [scores[k] for k in pick])
        print(f"[{mode}] {cam}: {len(cam_rows)} frames, median score {np.median(scores):.3f}, "
              f"lowest zone contrast {min(r['zone_contrast'] for r in cam_rows):.1f}")
    pool.shutdown()

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / f"voter_clip{suffix}.csv", index=False)
    crop_text = {"context": f"bounding box of all detection zones grown by {CONTEXT_MARGIN:.0%}, unmasked, padded to square",
                 "mask": f"detection zones grown by {args.grow:.1%} of the image width, everything else painted CLIP's "
                         f"mean colour {list(FILL)}; bounding box of the grown zones, padded to square",
                 "zones": f"each detection zone on its own (grown by {args.grow:.1%}, outside painted {list(FILL)}), "
                          "padded to square; probabilities averaged with zone areas as weights"}[mode]
    (args.out / f"clip_run{suffix}.json").write_text(json.dumps(
        {"model": args.model, "pretrained": args.pretrained, "device": device, "prompts": PROMPTS,
         "crop_mode": mode, "crop": crop_text, "score": "p_moderate * 0.5 + p_congested", "frames": len(rows),
         "subset": str(args.images_from) if args.images_from else None}, indent=2), encoding="utf-8")
    print(f"[{mode}] wrote {len(rows)} frames -> {args.out / f'voter_clip{suffix}.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Congestion voter C: zero-shot CLIP on the detection zones")
    ap.add_argument("--images", type=Path, default=Path("datasets/lta"))
    ap.add_argument("--frames", type=Path, default=Path("results/detection/frames.csv"))
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/congestion"))
    ap.add_argument("--crop", nargs="+", choices=CROPS, default=["context"], help="one or more crop modes")
    ap.add_argument("--images-from", type=Path, default=None,
                    help="only these frames: a labels export (.json) or a CSV with an image column")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--grow", type=float, default=0.01, help="grow the zones by this share of the image width")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="threads for reading images")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames per camera (trial run)")
    ap.add_argument("--examples", type=int, default=4, help="lowest / highest crops saved per camera (0 = none)")
    args = ap.parse_args()

    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained or None)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    text = text_features(model, tokenizer, device, torch)
    scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    print(f"{args.model} / {args.pretrained} on {device}")

    frames = load_frames(args.frames)
    index = image_index(args.images)
    zones = zone_polygons(args.config)
    subset = read_subset(args.images_from) if args.images_from else None
    if subset is not None:
        print(f"Only the {len(subset)} frames listed in {args.images_from}")
    for mode in args.crop:
        run(mode, args, model, preprocess, text, scale, device, torch, frames, index, zones, subset)


if __name__ == "__main__":
    main()
