# Vehicle Detection — report-ready text

Paste-ready paragraphs for the final report (vehicle-detection part). Numbers come from
`results/detection/finetune/eval/` and match `src/detection/README.en.md`. Figures are produced by
`python -m src.detection.report_figures` into `results/detection/finetune/figures/v4/`.

---

## 1. Problem and data

The detection module counts vehicles of four classes (car, motorcycle, bus, truck) in still images from Singapore
Land Transport Authority (LTA) traffic cameras. We collected 15,316 images from 8 cameras over two weeks in September 2026
at 10-minute intervals. The base images carry no annotations; a 63-image human-corrected validation subset is released separately in `datasets/lta_human_v1/`.

To obtain a ground-truth evaluation we used the MIO-TCD localization dataset (Luo et al., 2018): 110,000 labelled training
images from about 8,000 traffic-surveillance cameras in Canada and the United States, at 720×480 or 342×228 pixels.
The official test labels are not public, so all our splits are drawn from the official training set. We mapped its eleven
classes to our four (work vans to car; pickup, single-unit and articulated trucks to truck), removed pedestrian, bicycle and
non-motorised-vehicle boxes, and removed every image containing the class *motorized vehicle* (vehicles too small to be
classified), about 18 % of images. All MIO-TCD results therefore refer to this filtered four-class subset, not to the
official benchmark.

## 2. Evaluation protocol

**Leakage-aware splitting.** MIO-TCD contains many repeated frames: 9,627 groups of byte-identical files, and many
near-identical consecutive frames from the same camera. Camera identities are not provided. A first split by contiguous
image IDs left 109 of 2,000 validation images with exact copies in the training set. We therefore grouped images into
*scene groups* before splitting: two images are linked if their files are identical, if their 256-bit difference hashes
differ in at most 16 bits, if their low-resolution background thumbnails correlate at ≥ 0.95, or if the thumbnails
correlate at ≥ 0.93 after the annotated vehicles are masked out and in-painted. Connected components form the groups, and
each group is assigned to exactly one split. After building, we verified that no file is shared between splits and that no
cross-split pair meets any of the four rules.

**Splits.** Training uses 8,000 images (motorcycle images prioritised, as motorcycles are only 0.5 % of boxes). The
development set (1,000 images) is stratified so that motorcycles come from 20 different scene groups, with no group
contributing more than 12 % of them; it is used for checkpoint selection and model comparison. The test set (2,000 images,
31 scene groups) was frozen before training and evaluated exactly once, after all decisions were made. A supplementary
diagnostic set (161 images, 173 motorcycle boxes from 40 groups) measures motorcycle detection separately. Because groups
are allocated with rare-class stratification, the development and test sets are not the natural class distribution.
Manifests and labels are hashed, and training and evaluation refuse to run if they change.

**Residual overlap.** Content-based grouping is not a perfect substitute for camera identity. In an audit of 40 random test
images, each paired with its most similar training image, a preliminary visual review found 9 pairs from the same camera
(22.5 %; Wilson 95 % interval 12–38 %), two of them less than 10 minutes apart. Test results therefore describe performance
on a mix of seen and unseen cameras, not generalisation to new cameras.

**Metrics.** We report mAP at IoU 0.5 and averaged over IoU 0.5–0.95 (COCO 101-point interpolation), per-class AP at IoU 0.5,
precision and recall at confidence 0.25, and the mean absolute counting error per image. All models, including the
80-class COCO model, are evaluated by the same code after mapping class names to the four classes. Confidence intervals
come from 2,000 bootstrap resamples of scene groups (not individual images, which are correlated within a group); they
reflect sampling variation only, since each configuration was trained once.

## 3. Method

The detector is Ultralytics YOLO26m. We compare three models:

1. **Pretrained** — the official COCO weights, without fine-tuning (the pipeline's original detector).
2. **MIO-TCD only** — fine-tuned on the 8,000 MIO-TCD training images.
3. **Joint** — fine-tuned on the same MIO-TCD images plus the 240 LTA training images (each listed twice), whose labels are
   pseudo-labels produced by the larger YOLO26x model at full resolution. The LTA images give the model the Singapore
   viewpoint (high, distant, 1920×1080) that MIO-TCD lacks.

Both fine-tuned models start from the COCO weights and use identical settings: 736-pixel input, batch size 8, 30 epochs
without early stopping, and Ultralytics' default augmentation (mosaic, scaling, translation, colour jitter, horizontal flips;
vertical flips disabled). The checkpoint with the best development mAP@0.5:0.95 is kept; for both runs this was the final
epoch. Training took about 10 hours per model on an Apple M4 Pro.

## 4. Results

**Table 1. Frozen MIO-TCD test set (2,000 images, human ground truth).**

| Model | mAP@0.5 (95 % CI) | mAP@0.5:0.95 | Car | Motorcycle* | Bus | Truck | Count MAE |
|---|---|---|---|---|---|---|---|
| Pretrained | 0.430 (0.337–0.516) | 0.320 | 0.656 | 0.136 | 0.674 | 0.253 | 1.21 |
| MIO-TCD only | 0.709 (0.626–0.817) | 0.561 | 0.883 | 0.570 | 0.920 | 0.463 | 0.65 |
| Joint | 0.720 (0.645–0.825) | 0.563 | 0.873 | 0.646 | 0.908 | 0.453 | 0.70 |

\* 33 motorcycle boxes; on the dedicated motorcycle set, AP@0.5 is 0.305 (pretrained), 0.830 (MIO-TCD only) and 0.838 (joint).

Fine-tuning improves mAP@0.5 by 0.28–0.29 over the pretrained detector (paired 95 % intervals 0.19–0.41 and 0.21–0.41)
and roughly halves the counting error. The largest gains are for motorcycles and trucks, the classes the COCO model handles
worst in this viewpoint. The two fine-tuned models cannot be separated on MIO-TCD: the MIO-TCD-only model leads by 0.047 on
the development set, the joint model leads by 0.011 on the test set, and the paired interval of the test difference
(−0.024 to 0.048) includes zero.

**Table 2. Cross-domain behaviour (mAP@0.5).**

| Model | MIO-TCD test (human labels) | LTA validation (agreement with YOLO26x pseudo-labels) |
|---|---|---|
| Pretrained | 0.430 | 0.555 |
| MIO-TCD only | 0.709 | 0.174 |
| Joint | 0.720 | 0.634 |

The LTA column measures agreement with the teacher's predictions, not accuracy, but the contrast is large: the
MIO-TCD-only model misses on average 9.2 more vehicles per Singapore frame than the teacher, whereas the joint model
keeps, and slightly improves on, the pretrained model's agreement. We therefore use the joint model in the demo and the
detection pipeline.

**Deployment.** The joint model, run at 1280-pixel input, processed all 15,316 LTA images in 15 minutes on an Apple M4 Pro
and detected 279,476 vehicles (18.2 per frame on average; 27.2 by day and 8.9 at night). These detections feed the
monitoring demo. Motorcycles, which earlier models almost never found on the Causeway cameras, are now detected (41,109 in
total), although spot checks of peak-hour frames show that many riders are still missed.

## 5. Discussion

*Domain shift.* MIO-TCD frames are close-range, mostly 720×480 views of North American roads; LTA frames are high,
distant 1920×1080 views where far vehicles span only a few pixels. Fine-tuning on MIO-TCD alone specialised the model to the
former and degraded it on the latter. Adding a small set of pseudo-labelled target-domain images was enough to prevent this,
at no measurable cost on MIO-TCD test.

*Evaluation rigour changed the conclusion.* An early pilot with an ID-based split suggested both fine-tuned models were
equal everywhere; that split leaked exact duplicates into validation, and the pilots used inconsistent settings. Under the
leakage-aware, frozen protocol the joint model shows a small cost on the development set that is not significant on test.
We report the formal results and treat the pilot only as directional evidence.

*Trucks.* Trucks remain the weakest class (AP about 0.45). Our class definition merges pickups, single-unit trucks and
articulated trucks, which differ greatly in appearance and size, and pickups are easily confused with cars.

## 6. Limitations

- Results are on a filtered, stratified subset of MIO-TCD, not the official benchmark or the natural class distribution.
- Scene grouping approximates camera identity; about a fifth of audited test images had a same-camera training image.
- Each configuration was trained once; intervals exclude training randomness.
- The available human-corrected LTA set is a 63-image validation subset, not an independent test set; detection accuracy on Singapore cameras is measured on that subset with the limitations described in the report. LTA numbers are
  agreement with a teacher model whose own errors (notably missed distant motorcycles) are inherited.
- Motorcycle recall on LTA frames remains incomplete in peak-hour spot checks.
- The joint model sees about 6 % more images per epoch than the MIO-TCD-only model, so the comparison is between practical
  recipes rather than a perfectly controlled ablation.

## 7. Reproducibility

All steps are scripted in `src/detection/` (see `README.en.md`): `miotcd.py` builds the frozen splits and records manifest
and label hashes in `dataset.json`; `train.py` refuses to run on altered data and records weights and data hashes in
`provenance.json`; `evaluate.py` and `bootstrap.py` produce the tables; `audit_split.py` reproduces the overlap audit;
`report_figures.py` draws the figures.
