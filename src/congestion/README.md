# Congestion analysis (`congestion`)

Rates every frame of the 8 LTA cameras (15,316 frames, 13–29 Sep 2026) as **free-flowing / moderate /
congested**, judging only the detection zones calibrated in `src/camera_filter/camera_filter.json`.
Three independent voters are combined and evaluated against human labels. The detection module is
not modified; inputs are `results/detection/` and `datasets/lta/`, outputs go to `results/congestion/`.

## Voters

| | Method | Type | Score |
|---|---|---|---|
| A | Vehicle count and box occupancy from the joint detector (`m736_joint_v4`) after the camera filter | trained detector | count, 0–1 |
| B | Hourly median background, Canny-edge difference, glare ignored | classical vision | 0–1 |
| C | open_clip ViT-B-16 zero-shot, three prompt groups, zone bounding box | pretrained vision-language model | 0–1 |

## Pipeline (from the repository root)

```bash
pip install open_clip_torch                     # needed by clip_voter only

python -m src.congestion.occupancy              # A, ~15 s
python -m src.congestion.background             # B, ~2 min
python -m src.congestion.clip_voter             # C, ~7 min on an RTX 4070

python -m src.congestion.make_label_tasks       # 600 frames to label -> labeling/label_tool.html
#   label in the browser, export congestion_labels.json into results/congestion/labeling/
python -m src.congestion.clip_voter --crop context mask zones \
    --images-from results/congestion/labeling/congestion_labels.json --out results/congestion/clip_trials

python -m src.congestion.ensemble_eval fit      # practice labels only -> ensemble_config.json
python -m src.congestion.ensemble_eval test     # test labels, once
python -m src.congestion.ensemble_eval apply    # every frame -> congestion_levels.csv
python -m src.congestion.report                 # heatmaps, timeline, summary table
```

The demo's camera monitoring page (`python -m src.demo.web.server`) shows each frame's level and the
weekday × hour heatmap once `congestion_levels.csv` exists.

## Human labels

- 600 frames, time-stratified per camera (weekday peak / off-peak, weekend day, evening, late night),
  never selected by any voter's score: 25 practice + 50 test per camera.
- Rule: judge only the outlined road; when the outlines cover two roads, label the more congested one.
- Practice: 107 free / 47 moderate / 46 congested (1 skipped). Test: 171 / 127 / 102 (none skipped).

## Method selection (practice labels, 5-fold cross-validation × 20)

Features chosen: A = vehicle count of the most congested zone, B = all zones, C = `context` crop.

| Method | Accuracy | Weighted κ |
|---|---|---|
| A | 80.0% | 0.745 |
| B | 65.3% | 0.495 |
| C | 76.7% | 0.727 |
| Median vote | 81.3% | 0.773 |
| **Stacking (logistic regression on A, B, C)** | **81.5%** | **0.780** |

Stacking was frozen as the primary method before the test labels were used.

## Test results (400 frames, evaluated once, 95% bootstrap intervals)

| Method | Accuracy | Weighted κ | Congested precision / recall |
|---|---|---|---|
| Always "free" | 42.8% | 0 | – |
| A | 72.3% [67.7, 76.5] | 0.671 | 0.86 / 0.78 |
| B | 62.3% [57.7, 67.0] | 0.527 | 0.62 / 0.83 |
| C | 74.0% [69.8, 78.2] | 0.703 | 0.88 / 0.89 |
| Median vote | 73.0% [68.8, 77.5] | 0.696 | 0.84 / 0.88 |
| **Stacking** | **74.3% [69.8, 78.5]** | **0.704** | **0.88 / 0.84** |

- Stacking − A: +2.0 points [−1.8, +5.5], not significant.
- Most errors are moderate frames rated free (69 of 127).
- Population-weighted accuracy (correcting for the peak-hour over-sampling): 75.8%.

## Share of frames rated congested (all 15,316 frames)

2701 48% · 2702 41% · 4703 26% · 4713 22% · 2704 5% · 4712 2% · 4798 0.2% · 4799 0%.
Heatmaps, the timeline and the per-camera summary are produced by `report.py`.

## Limitations

- One annotator; "congested" is a visual judgement of the outlined road.
- 50 test frames per camera; each weekday appears only twice in the two weeks.
- Only the detection zones are judged, not the whole road.

## Files

| File | Purpose |
|---|---|
| `common.py` | Frame list, image index, image reading (non-ASCII paths), zones |
| `occupancy.py` | Voter A |
| `background.py` | Voter B |
| `clip_voter.py` | Voter C |
| `make_label_tasks.py`, `label_tool_template.html` | Sampling and the labelling page (3 levels, skip with replacement, undo) |
| `ensemble_eval.py` | `fit` / `test` / `apply` |
| `report.py` | Heatmaps, timeline, summary table |
