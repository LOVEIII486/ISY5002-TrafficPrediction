# Camera filter

The camera filter removes recurring false positives from the detector output
using a per-camera road region and exclusion set. It does not retrain or modify
the detector. It reads `results/detection/detections.csv` and writes filtered
detections to `results/camera_filter/`.

The approach combines a learned detector with manually calibrated, fixed-scene
rules: the detector recognises vehicles, and the camera-specific configuration
removes systematic errors such as road signs, traffic lights and lane markings.

## Files

| Path | Purpose |
|---|---|
| `camera_filter.json` | Calibration for the eight cameras: road regions, detection zones and exclusions |
| `core.py` | `CameraFilter`: returns `keep`, `off_road` or `exclude:k` for each box |
| `apply.py` | Filters an existing `detections.csv` into `detections_filtered.csv` |
| `evaluate.py` | Compares raw and filtered detections on the human-corrected subset |
| `detector.py` | `FilteredDetector`: detector plus camera filter |
| `calibrator/` | Browser-based calibration page and draft-polygon tools |

## Usage

From the repository root:

```bash
python -m src.camera_filter.apply
python -m src.camera_filter.evaluate --labels <label-export.json>
```

Build the calibration page with:

```bash
python src/camera_filter/calibrator/build_calibrator.py \
    --detections results/detection \
    --images <folder with validation images> \
    --out camera_calibrator.html
```

The Python API is available as well:

```python
from src.camera_filter import CameraFilter, FilteredDetector

kept = CameraFilter.from_json(
    "src/camera_filter/camera_filter.json"
).filter(detections_df)

detector = FilteredDetector(
    "results/detection/finetune/m736_joint_v4/weights/best.pt",
    "src/camera_filter/camera_filter.json",
)
boxes = detector.predict("20260924T040558Z_2701_0405_5bc30fa4fa4e.jpg")
```

## Rules

- Each box is tested at its bottom-center point in normalized coordinates.
- A box outside every road polygon is removed.
- An exclusion applies when its time window matches (all day, day or night) and
  either the bottom-center point lies inside the exclusion box or the whole
  detection box lies inside it.
- Day is 07:00–19:00 Singapore time.

## Calibration

1. The calibration page displays two weeks of detection points over each
   camera's background.
2. A suggested road region and detection zone are generated from the point
   density; the annotator adjusts the vertices.
3. Candidate fixed false positives are listed for acceptance or rejection.
4. The page exports `camera_filter.json`.

The human-corrected validation images are excluded from calibration. See
`datasets/lta_human_v1/` for the evaluation subset and
`docs/DETECTION_REPORT.en.md` for the evaluation protocol.
