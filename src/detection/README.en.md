# detection — vehicle detection on LTA traffic-camera images

*English · [中文](README.md)*

Runs YOLO on Singapore LTA traffic-camera stills and writes three CSV files under `results/detection/`
(`detections.csv` / `frames.csv` / `cameras.csv`). The demo reads these files and **never imports this module**,
so the CSV column names and values are this module's public contract — do not change them.

## Default pipeline

```bash
python scripts/fetch_data.py lta                # LTA images -> datasets/lta/
python scripts/fetch_data.py weights            # COCO-pretrained weights -> yolo/
python -m src.detection.run_detect --limit 8    # smoke test on 8 images
python -m src.detection.run_detect              # full run
```

The default weights are the COCO-pretrained `yolo26m`; no configuration is needed.

## Using your own fine-tuned weights

Two steps. **The second one is easy to forget, and forgetting it produces an empty CSV without an error.**

1. Point the pipeline at the weights:

   ```bash
   python -m src.detection.run_detect --weights <your run>/weights/best.pt
   ```

2. Map the model's class names to the pipeline's four classes in `classes.yaml`. The pipeline counts only
   `car / motorcycle / bus / truck`. COCO weights already use these names (case-insensitive), so they need no mapping;
   weights fine-tuned on another dataset almost always do. List the model's classes with
   `python -c "from ultralytics import YOLO; print(YOLO('your.pt').names)"`, then write e.g.

   ```yaml
   Truck: truck
   Bus: bus
   Car: car
   Van: car        # vans count as cars, as in COCO
   ```

   Classes that are not vehicles can be omitted. **If no class maps at all, the program stops and lists every class name**
   instead of silently producing empty results. Weights fine-tuned in this module already use the four names and need no mapping.

3. Verify with `--limit 8`: the second line prints the vehicle classes found, and the detection count must not be 0.

**Changing weights or inference settings requires a clean output directory.** The three CSVs belong to one configuration;
`run_detect.py` refuses to append results from a different configuration (`run.json` records weights, class map and
inference parameters):

```bash
rm results/detection/detections.csv results/detection/frames.csv results/detection/run.json
```

## Fine-tuning on LTA pseudo-labels (exploratory)

```bash
python -m src.detection.sample      # stratified sample; downloads only the chosen images: week1 240 train, week2 80 val
python -m src.detection.prelabel    # yolo26x pre-labels at full resolution -> datasets/lta_yolo/ (YOLO format)
python -m src.detection.train --imgsz 960 --name m960_pseudo
python -m src.detection.evaluate --weights yolo/yolo26m.pt results/detection/finetune/m960_pseudo/weights/best.pt --imgsz 960
```

- **Split by week**: week1 for training, week2 for validation; no byte-identical image appears in both weeks. This measures
  *future periods of known cameras*; both weeks cover the same 8 cameras, so it says nothing about unseen cameras.
- **Day / night are stratified by Singapore time.** The dataset README says the `HHMM` in file names is Singapore time;
  it is actually **UTC**. `sample.py` converts with UTC+8.
- **Step 2 produces pseudo-labels** (model predictions). They are the starting point for human correction
  (`datasets/lta_yolo/` imports directly into CVAT as *Ultralytics YOLO Detection 1.0*, or into Label Studio / Roboflow)
  and allow fine-tuning before any human labels exist. **Metrics computed against pseudo-labels only measure agreement with
  the teacher**: real vehicles the teacher missed are counted as student false positives. They are neither an accuracy on
  Singapore cameras nor evidence of adaptation gains; `evaluate.py` prints this warning. Known issue: the teacher misses
  many distant motorcycles.
- `prelabel.py` refuses to run if `labels/` already contains labels (they may be human-corrected); `--overwrite` rebuilds.
- **Why not `model.val()`?** COCO weights have 80 classes whose indices differ from this 4-class dataset, and `val()` does not
  remap ground-truth indices. `evaluate.py` maps every model by class name and computes mAP itself (COCO 101-point
  interpolation, greedy matching, no crowd / ignore regions — values need not match Ultralytics or the official COCO
  evaluator to the last digit), plus precision / recall at confidence 0.25 and counting errors, split by day / night.
- **Memory**: yolo26m at 1280 with batch 4 needs about 19 GB. On an Apple M4 Pro (24 GB unified memory) this swaps
  (45 s per step); at 960 it uses about 11 GB (1 s per step). On an RTX 5070 (12 GB) use batch 2–4 at 1280, or 960.

## Fine-tuning and evaluation on MIO-TCD (human-annotated ground truth)

The LTA images have no human labels, so the experiment above cannot give an accuracy. Ground-truth numbers come from the
**MIO-TCD localization set** (8,000 traffic-surveillance cameras in Canada and the US, 110,000 labelled training images
of 720×480 or 342×228 pixels, CC BY-NC-SA 4.0; cite Luo et al., IEEE TIP 2018).

```bash
curl -L -o datasets/miotcd/MIO-TCD-Localization.tar https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar   # 3.7 GB
python -m src.detection.miotcd        # class mapping, scene grouping, train/dev/test split, verification -> datasets/miotcd_yolo_v4/, joint_yolo_v4/
python -m src.detection.audit_split   # audit residual same-camera overlap (contact sheets + a CSV to fill in)

# Formal comparison: both start from COCO weights; same image size, batch, augmentation; fixed 30 epochs (no early stopping)
python -m src.detection.train --data datasets/miotcd_yolo_v4/data.yaml --imgsz 736 --batch 8 --epochs 30 --name m736_mio_v4
python -m src.detection.train --data datasets/joint_yolo_v4/data.yaml  --imgsz 736 --batch 8 --epochs 30 --name m736_joint_v4

# Use dev while developing; once every decision is made, evaluate the three models on test and test_moto exactly once
python -m src.detection.evaluate --data datasets/miotcd_yolo_v4 --split dev  --imgsz 736 --weights yolo/yolo26m.pt \
    results/detection/finetune/m736_mio_v4/weights/best.pt results/detection/finetune/m736_joint_v4/weights/best.pt
python -m src.detection.evaluate --data datasets/miotcd_yolo_v4 --split test --imgsz 736 --weights ...   # same three
python -m src.detection.bootstrap --data datasets/miotcd_yolo_v4 --split test --imgsz 736 --weights ...  # confidence intervals
python -m src.detection.report_figures                                                                   # figures for the report
```

**Frozen data.** Before training, `train.py` checks the manifests that `data.yaml` actually references, and their label
contents, against the frozen record `dataset.json` (for joint training also the dev manifest and the LTA pseudo-labels).
`evaluate.py` checks the manifest and labels of the evaluated split. A versioned directory without a frozen record is
rejected. See `frozen.py`. The freeze covers manifests and labels only, **not image contents**: a replaced JPEG or a
re-pointed symlink would go unnoticed.

**Scene grouping (v4).** MIO-TCD has no camera IDs, and each camera contributes many consecutive frames (9,627 groups of
byte-identical files among the 110,000 images). Two images join the same group if any of four rules holds, and a whole
group goes to one split only: identical SHA-256; 256-bit dHash distance ≤ 16; background-thumbnail correlation ≥ 0.95;
correlation ≥ 0.93 between background thumbnails **after the labelled vehicles are masked out** (0.93 is an engineering
parameter, not a "93 % same-camera confidence"). Groups larger than 1,500 images, and groups with more than 30 motorcycle
images, go to the training pool. The two largest groups were inspected and each is a single camera (including correctly
merged day and night frames). Post-build checks: no shared file between splits; train~test minimum dHash distance 35,
maximum background correlation 0.94, maximum masked-background correlation 0.927.

**Residual same-camera overlap — must be disclosed.** Grouping is content-based, not true camera identity; large foreground
vehicles, hard shadows and lighting changes push frames of the same camera below the thresholds. `audit_split` sampled 40
test images with seed 42 and paired each with its **single most similar** training image (recorded in
`results/detection/finetune/audit/v4_test/audit.csv` with image IDs, timestamps and judgements). **Preliminary AI visual
review (Claude, pending human confirmation):** 9 pairs same camera, 2 probably, 4 undecidable (night test image vs. day
training image), 25 different cameras. Of the 9, two pairs are under 10 minutes apart (2 min 18 s and 8 min 27 s) and one
is about 10 minutes apart (10 min 03 s).

This is an observation from one audit, not the true overlap rate: only the single nearest training image was retrieved,
so other same-camera frames may be missed, and judgements may be wrong. The Wilson 95 % interval of 9/40 (22.5 %),
12 %–38 %, covers sampling error only, not retrieval misses or judgement errors. Conclusion: after duplicate and
near-duplicate filtering, **same-camera frames and near-in-time frames still exist**; test scores describe performance on
a *mix of seen and unseen cameras* and **cannot be claimed as generalisation to new cameras**.

| Split | Images | Motorcycle boxes | Purpose |
|---|---|---|---|
| train | 8,000 | 833 (15 groups, largest 15 %) | training; motorcycle images prioritised (enriched) |
| dev | 1,000 | 93 (20 groups, largest 12 %) | checkpoint selection and model comparison; **stratified**: every motorcycle image in the dev pool (≤ 10 per group) plus random fill |
| test | 2,000 | 33 (17 groups, largest 18 %) | **frozen**; evaluated once after all decisions; uniform sample of the test pool |
| test_moto | 161 | 173 (40 groups, largest 9 %) | supplementary diagnostic: remaining motorcycle images of the test pool (≤ 10 per group); disjoint from test but shares its scene groups |

Per-class group counts and largest-group shares are in `dataset.json` → `class_diversity`.

**Reporting rules**

- Results apply to a **filtered four-class subset**: images containing `motorized_vehicle` (official definition: motorised
  vehicles too small to classify) are removed entirely, about 18 %, which systematically drops scenes with hard small
  objects. **These are not official MIO-TCD benchmark results**, and mAP must not be written as "accuracy xx %".
- Class merging (work_van → car, three truck types → truck) is this project's definition and need not match COCO's boundaries.
- dev / test are **not the natural distribution**: very large groups and motorcycle-heavy groups go to training; other
  groups with motorcycles are split 30/20/50 (test/dev/train) by target, the rest 15/10/75, then sampled within pools.
  Actual shares are in `dataset.json` → `pool_rare_share_actual`. Describe it as *stratified by the rare class, isolated by
  scene group, sampled within pools*.
- Report all four class APs on test and note that motorcycles are few; test_moto is a supplementary diagnostic only — **do not
  substitute its AP for the test motorcycle AP when computing mAP**. The two share scene groups and are not independent
  evidence; box counts are not independent sample sizes.
- `best.pt` is selected by Ultralytics' dev mAP@0.5:0.95 (not by this module's mAP@0.5). Both runs train the full 30 epochs,
  but joint training sees about 6 % more samples per epoch (8,480 vs 8,000), so the comparison is between practical recipes
  and cannot fully separate the extra LTA data from the extra compute.

**Joint training.** `joint_yolo_v4/train.txt` = the MIO-TCD training images + the LTA training images × `--joint-lta-repeat`
(default 2); validation stays on the MIO-TCD dev ground truth. By default LTA is about 6 % of training images but about
21 % of boxes (LTA frames are crowded), so its influence cannot be judged by image share alone; more repetitions add no new
information and increase exposure to the teacher's false positives and misses.

### Formal results (v4)

Setup: both fine-tuned models start from COCO weights, 736 input, batch 8, 30 fixed epochs, trained on the same Apple
M4 Pro (MIO-TCD only 10.1 h, joint 9.8 h); `best.pt` is epoch 30 in both. One training run per configuration.

**Frozen test split** (MIO-TCD v4 test, 2,000 images, 31 scene groups, human ground truth, evaluated once):

| Model | mAP@0.5 (95 % CI) | mAP@0.5:0.95 | Car | Motorcycle* | Bus | Truck | Count MAE |
|---|---|---|---|---|---|---|---|
| COCO-pretrained yolo26m | 0.430 (0.337–0.516) | 0.320 | 0.656 | 0.136 | 0.674 | 0.253 | 1.21 |
| Fine-tuned, MIO-TCD only | 0.709 (0.626–0.817) | 0.561 | 0.883 | 0.570 | 0.920 | 0.463 | 0.65 |
| Fine-tuned, MIO-TCD + LTA | 0.720 (0.645–0.825) | 0.563 | 0.873 | 0.646 | 0.908 | 0.453 | 0.70 |

\* only 33 motorcycle boxes in test; this column is noisy. Intervals come from `bootstrap.py`: percentile intervals over
2,000 scene-group resamples, covering sampling variation only (not training randomness). Paired differences:

| Difference | mAP@0.5 | 95 % CI |
|---|---|---|
| MIO-TCD only − pretrained | +0.279 | 0.194–0.407 |
| Joint − pretrained | +0.291 | 0.213–0.411 |
| Joint − MIO-TCD only | +0.011 | −0.024–0.048 (includes 0) |

**Other splits**

| Model | dev mAP@0.5 | test_moto motorcycle AP@0.5 | LTA val agreement with teacher, mAP@0.5 (960) |
|---|---|---|---|
| COCO-pretrained | 0.370 | 0.305 | 0.555 |
| MIO-TCD only | 0.750 | 0.830 | 0.174 (misses 9.2 vehicles per image on average) |
| Joint | 0.703 | 0.838 | 0.634 |

**Conclusions**

1. Both fine-tuned models beat the pretrained model by about 0.28–0.29 mAP@0.5 on test, with intervals excluding 0;
   counting error roughly halves, and motorcycles improve the most.
2. The two fine-tuned models are not distinguishable on MIO-TCD: MIO-TCD-only leads by 0.047 on dev, joint leads by 0.011
   on test, and the paired interval includes 0.
3. The MIO-TCD-only model largely fails on Singapore frames; joint training keeps the Singapore scenes and agrees with the
   teacher better than the pretrained model does.
4. The demo and the detection pipeline therefore use **`m736_joint_v4`**. Trucks (three merged truck types) are the weakest
   class at about 0.45.
5. The exploratory claim that joint training "costs nothing on either side" is corrected by the formal runs: joint training
   does cost about 0.05 on dev, although no significant difference shows on test.

**Deployment on all LTA images (used by the demo).** `m736_joint_v4` was run at 1280 input (agreement with the teacher on
LTA val: 0.640 at 1280, 0.634 at 960, 0.599 at 736) over all 15,316 images, detecting 279,476 vehicles (car 187,928,
motorcycle 41,109, truck 35,328, bus 15,111) in 15 minutes on an Apple M4 Pro. Frames contain 18.2 vehicles on average
(27.2 by day, 8.9 at night); 1,478 frames, mostly at night, have no detection. Motorcycles are now detected on the Causeway,
but spot checks of peak-hour frames still show many missed riders, so motorcycle recall is incomplete. Outputs are the three
CSVs under `results/detection/`; `run.json` records the weights path.

**Exploratory and voided versions (not used for conclusions).** v1 split by contiguous ID blocks; 109 of 2,000 validation
images had byte-identical copies in training. v2 removed duplicate files but frames of the same camera seconds apart still
crossed train and test. v3 scaled mask boxes as if every image were 720×480, mis-masking the ~20 % of images that are
342×228. Exploratory runs also used inconsistent sizes and epochs (LTA pseudo-label fine-tune 960 / 30 epochs; MIO-TCD and
joint pilots 736 / 15 epochs). Their results remain in `eval/miotcd_yolo_v1/` and `eval/lta_yolo/` as directional
observations only. Formal conclusions use v4.

**`--base` and second-stage fine-tuning.** When `--base` points to fine-tuned weights, Ultralytics only loads the weights
and starts a new training stage (fresh optimiser state, learning-rate schedule and mosaic phase); it is not a resume.
Describe it as "second-stage fine-tuning initialised from …". The formal comparison starts from COCO weights.

Weights fine-tuned here already use the four class names, so no `classes.yaml` is needed:

```bash
python -m src.detection.run_detect --weights results/detection/finetune/<name>/weights/best.pt
```

## What is not here

Dataset choice is outside this module. The module defines only the **integration contract**:

```
your weights + classes.yaml  ->  detections.csv / frames.csv / cameras.csv  ->  demo
```

Any dataset or model works as long as the CSVs follow the column contract.
