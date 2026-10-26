# Congestion analysis

This module rates LTA traffic-camera frames as **free-flowing**, **moderate** or
**congested** inside the calibrated detection zones. It reads the detection
artifacts and the camera calibration, and writes frame-level levels and reports
to `results/congestion/`.

The level describes the visible traffic state in the camera view. It is separate
from the speed-based risk indicator used on the PEMSD8 forecasting view.

## Voters

| Voter | Method | Output |
|---|---|---|
| A | Vehicle count and box occupancy from the filtered detector | 0–1 score |
| B | Hourly median background and Canny-edge difference, ignoring glare | 0–1 score |
| C | OpenCLIP ViT-B/16 zero-shot classification with prompt groups | 0–1 score |

A multinomial logistic regression combines the three scores. The combination
weights and decision thresholds are selected on practice labels; the test labels
are used once for evaluation.

## Pipeline

From the repository root:

```bash
python -m src.congestion.occupancy
python -m src.congestion.background
python -m src.congestion.clip_voter

python -m src.congestion.make_label_tasks
# Label frames with labeling/label_tool.html and export congestion_labels.json
# into results/congestion/labeling/.

python -m src.congestion.ensemble_eval fit
python -m src.congestion.ensemble_eval test
python -m src.congestion.ensemble_eval apply
python -m src.congestion.report
```

`clip_voter.py` requires `open_clip_torch`; the other commands use the packages
listed in `requirements.txt`.

## Outputs

| Path | Contents |
|---|---|
| `results/congestion/voter_occupancy.csv` | Voter A scores |
| `results/congestion/voter_background.csv` | Voter B scores |
| `results/congestion/voter_clip.csv` | Voter C scores |
| `results/congestion/ensemble_config.json` | Frozen combination model |
| `results/congestion/congestion_levels.csv` | Final level for every frame |
| `results/congestion/report/` | Heatmaps, timeline and summary table |

## Human labels

The label set contains 600 frames, sampled per camera across five time strata.
Each camera contributes 25 practice frames and 50 test frames. The annotation
tool outlines the detection zones and hides the timestamp; the annotator judges
only the outlined road and uses the more congested road when two roads are
visible. Labels are stored in
`results/congestion/labeling/congestion_labels.json`.

## Files

| File | Purpose |
|---|---|
| `common.py` | Frame list, image index, image loading and zone masks |
| `occupancy.py` | Voter A |
| `background.py` | Voter B |
| `clip_voter.py` | Voter C |
| `make_label_tasks.py` | Sampling and labeling-page generation |
| `ensemble_eval.py` | Fit, test and apply the combination model |
| `report.py` | Heatmaps, timeline and summary table |
