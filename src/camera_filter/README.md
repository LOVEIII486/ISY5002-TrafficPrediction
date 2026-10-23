# Camera filter (`camera_filter`)

Removes fixed false positives from the detector output using a per-camera, human-calibrated
**road region** and **exclusion set**. The model is not retrained and the detection module
(`src/detection/`) is not modified: the filter only reads `results/detection/detections.csv` and
writes to `results/camera_filter/`. When these outputs exist, the demo's camera monitoring page
draws the road region, detection zones and removed boxes.

It is a hybrid approach: the learned detector recognises vehicles, and camera-specific rules
calibrated by a person remove its systematic false positives in each fixed scene.

## Files

| Path | Purpose |
|---|---|
| `camera_filter.json` | Calibration for the 8 cameras: road regions, detection zones (for congestion analysis) and exclusions |
| `core.py` | `CameraFilter`: decides `keep` / `off_road` / `exclude:k` for every box |
| `apply.py` | Filters an existing `detections.csv` into `detections_filtered.csv` with the same columns |
| `evaluate.py` | Compares raw and filtered output on the human-labelled test set (63 images from task A) |
| `detector.py` | `FilteredDetector`: YOLO + camera filter, used like a model |
| `calibrator/` | Builds the calibration page (detection points + suggested drafts) |

## Usage (from the repository root)

```bash
python -m src.camera_filter.apply                                               # a few seconds
python -m src.camera_filter.evaluate --labels lta_annotation_export_final.json  # ground truth
python src/camera_filter/calibrator/build_calibrator.py --detections results/detection \
    --images <folder with the 80 test images> --out camera_calibrator.html
```

```python
from src.camera_filter import CameraFilter, FilteredDetector

kept = CameraFilter.from_json("src/camera_filter/camera_filter.json").filter(detections_df)

det = FilteredDetector("results/detection/finetune/m736_joint_v4/weights/best.pt",
                       "src/camera_filter/camera_filter.json")
boxes = det.predict("20260924T040558Z_2701_0405_5bc30fa4fa4e.jpg")  # camera id and time come from the file name
```

## Rules

- Each box is tested at its **bottom-centre** (normalised 0-1 coordinates); points on the image border are pulled 0.001 inwards.
- For a camera with a road region, a box whose point lies outside every road polygon is removed.
- An exclusion removes a box when its time window matches (all day / day / night; day is 07:00-19:00 Singapore time)
  and either `point` (the point lies in the exclusion box) or `box` (the whole detection box lies inside it).

## Calibration

1. The calibration page shows two weeks of detections as points on each camera's background. The 80 human-labelled
   test images are excluded, so the exclusion set is not drawn from test errors.
2. Each camera opens with a suggested road region and detection zone generated from the point density; a person
   adjusts the vertices and confirms.
3. Candidate fixed false positives (the same box recurring in many frames) are listed for a person to accept or reject;
   queues at the same position are real vehicles and are left alone.
4. The page exports `camera_filter.json`.

## Results (63 human-labelled test images, class-agnostic, IoU >= 0.5, 2,000 bootstrap resamples)

| | Raw | Filtered |
|---|---|---|
| Precision | 0.816 | 0.833 (95% CI of change: +1.0 to +2.9 points) |
| Recall | 0.547 | 0.543 |
| False positives | 148 | 130 |

Across all 279,476 detections from the two weeks, 2.0% are removed. The filter raises precision but does not make
vehicle counts more accurate: the detector already misses many vehicles (recall about 0.55), so removing false
positives lowers the counts slightly further (per-image count MAE 6.92 -> 7.00).
