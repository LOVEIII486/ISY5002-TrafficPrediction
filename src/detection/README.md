# Vehicle detection on LTA traffic-camera images

This module runs YOLO on Singapore LTA traffic-camera stills, counts vehicles
and writes the detection artifacts consumed by the workbench. It does not depend
on the forecasting module.

The CSV columns and values are the module's public contract. The workbench reads
the files under `results/detection/` and does not import this package.

## Outputs

| File | Contents |
|---|---|
| `detections.csv` | One row per detected vehicle: camera, UTC time, image, class, confidence and box |
| `frames.csv` | One row per processed image, including frames with no detections |
| `cameras.csv` | Camera IDs and the camera list used by the workbench |
| `run.json` | Weights, class mapping and inference settings for the run |

## Quick start

From the repository root:

```bash
python scripts/fetch_data.py lta
python scripts/fetch_data.py weights
python -m src.detection.run_detect --limit 8
python -m src.detection.run_detect
```

The default detector is the COCO-pretrained `yolo26m`. The `--limit` run writes
a small preview; the full run processes the complete LTA collection.

## Using fine-tuned weights

Pass the checkpoint explicitly:

```bash
python -m src.detection.run_detect \
    --weights results/detection/finetune/<run>/weights/best.pt
```

If the checkpoint uses different class names, map them in
`src/detection/classes.yaml`. The pipeline retains only `car`, `motorcycle`,
`bus` and `truck`. For example:

```yaml
Truck: truck
Bus: bus
Car: car
Van: car
```

Run `--limit 8` after changing weights or settings. The console reports the
mapped classes and the number of detections.

Changing weights or inference settings requires a fresh output directory because
the three CSVs belong to one configuration. Remove the previous
`detections.csv`, `frames.csv` and `run.json` before a new full run.

## Fine-tuning and evaluation

The fine-tuning workflow uses the LTA images plus MIO-TCD, a public
human-labeled vehicle detection benchmark. A typical sequence is:

```bash
python -m src.detection.sample
python -m src.detection.prelabel
python -m src.detection.train --data datasets/miotcd_yolo_v4/data.yaml \
    --imgsz 736 --batch 8 --epochs 30 --name m736_mio_v4
python -m src.detection.train --data datasets/joint_yolo_v4/data.yaml \
    --imgsz 736 --batch 8 --epochs 30 --name m736_joint_v4
python -m src.detection.evaluate --data datasets/miotcd_yolo_v4 \
    --split test --imgsz 736 --weights yolo/yolo26m.pt \
    results/detection/finetune/m736_mio_v4/weights/best.pt \
    results/detection/finetune/m736_joint_v4/weights/best.pt
```

`sample.py` creates a week-1 training split and a week-2 validation split.
`prelabel.py` creates teacher pseudo-labels for the LTA images. `miotcd.py`
builds the MIO-TCD scene-group splits, and `evaluate.py` scores every model with
the same four-class evaluator.

Metrics computed against pseudo-labels measure agreement with the teacher, not
accuracy against human labels. The human-corrected LTA subset is described in
`datasets/lta_human_v1/`.

## Human-corrected annotations

`datasets/lta_human_v1/` contains 63 human-corrected images and 1,197 vehicle
boxes in YOLO, COCO and JSONL formats. The package includes image paths,
SHA-256 hashes and the list of excluded frames. See
[`datasets/lta_human_v1/README.md`](../../datasets/lta_human_v1/README.md).

The annotations can be regenerated from a Label Studio export with:

```bash
python -m src.detection.annotate collect <export.json> \
    --annotator "<name>" --annotations-only \
    --excluded <deleted-images.csv> --out datasets/lta_human_v1
```

## Related documentation

- Evaluation protocol and results: `docs/DETECTION_REPORT.en.md`
- Workbench: `src/demo/`
- Dataset card: `datasets/lta_human_v1/README.md`
