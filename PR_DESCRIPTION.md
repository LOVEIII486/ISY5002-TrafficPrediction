## Camera filter and congestion analysis

Two modules on top of the joint detector (`m736_joint_v4`), plus optional layers in the demo's camera monitoring page. No retraining; `src/detection/` is unchanged. The modules read `results/detection/` and `datasets/lta/` and write to `results/camera_filter/` and `results/congestion/`.

### `src/camera_filter/` — per-camera false-positive filter
- Human-calibrated road region, exclusion set and detection zones for all 8 cameras (`camera_filter.json`), drawn in a calibration page that plots two weeks of detections; the 80 human-labelled test images are excluded from calibration.
- `FilteredDetector` wraps YOLO + filter and is used like a model.
- 63 human-labelled test images: precision 0.816 → 0.833 (95% CI of change +1.0 to +2.9 points); 2.0% of 279,476 boxes removed.

### `src/congestion/` — congestion level of every frame (free / moderate / congested)
- Three voters inside the detection zones: A detector vehicle count / occupancy, B background difference, C zero-shot CLIP.
- 600 human labels, time-stratified, never selected by any voter's score: 25 practice + 50 test per camera.
- Method and features chosen on practice labels only (5-fold CV × 20); stacking (logistic regression on A, B, C) frozen as primary before the test labels were used.
- Test (400 frames, evaluated once): stacking 74.3% accuracy [69.8, 78.5], weighted κ 0.704, congested precision 0.88 / recall 0.84; A alone 72.3%, C alone 74.0%, always-free baseline 42.8%.
- `report.py`: weekday × hour heatmaps, two-week timeline and per-camera summary. Congested share of all frames: 2701 48%, 2702 41%, 4703 26%, 4713 22%, 2704 5%, 4712 2%, 4798 0.2%, 4799 0%.

### `src/demo/` — optional layers on the camera monitoring page
- Detection frame: a "Camera filter" switch. When it is on, the area outside the road region is dimmed, detection zones are outlined, known false-positive spots are dotted, and boxes the filter removed are drawn red and dashed with the reason. The frame header shows the frame's congestion level.
- Reliability card: share of the camera's boxes removed by the filter.
- New card: weekday × hour heatmap of the congested share for the selected camera, with the test-set accuracy overall and on that camera.
- Two new modules, `src/demo/filtering.py` and `src/demo/congestion.py`, read only the artifacts above. Without them the page looks exactly as before.
- Existing files touched: `server.py` (two fields, one route), `index.html` (one card, one slot), `console.js`, `console.css`, `i18n.js` (English and Chinese UI strings).

### Committed results
`.gitignore` gets exceptions for `results/congestion/`: the human labels (`labeling/tasks.json`, `labeling/congestion_labels.json`), the frozen `ensemble_config.json`, the one-off test evaluation (`eval_test.json`, `eval_test_by_group.csv`) the per-frame levels `congestion_levels.csv` (1.4 MB) that the demo's heatmap reads, and the `report/` figures and summary table (1.4 MB). Voter scores and CLIP trials stay local.

### Run
See `src/camera_filter/README.md` and `src/congestion/README.md`. `clip_voter.py` needs `open_clip_torch`. After `python -m src.camera_filter.apply` and `python -m src.congestion.ensemble_eval apply`, start the demo with `python -m src.demo.web.server` and open the camera monitoring page.
