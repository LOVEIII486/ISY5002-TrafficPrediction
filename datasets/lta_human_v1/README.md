---
license: cc-by-4.0
task_categories:
  - object-detection
annotations_creators:
  - expert
language:
  - en
tags:
  - traffic
  - singapore
  - lta
  - vehicle-detection
  - human-annotated
size_categories:
  - n<1K
---

# LTA Human-Corrected Vehicle Boxes (Singapore, September 2026)

This directory contains the **human-corrected annotation subset** for the LTA
traffic-camera images in this dataset. It is intended to live at
`annotations/lta_human_v1/` inside
[`LOVEIII486/isy5002-lta-traffic-images`](https://huggingface.co/datasets/LOVEIII486/isy5002-lta-traffic-images).
It contains human-corrected vehicle boxes for a 63-image subset of the week-2
validation frames, together with the image manifest needed to join the labels to
the base images.

The images are **not duplicated in this folder**. Pair each `image_path` in
`images.csv` with the corresponding file at the dataset root. The base images
remain subject to the
[Singapore Open Data Licence 1.0](https://data.gov.sg/open-data-licence), and
the base dataset card warns that number plates may be legible in some
full-resolution frames.

## Contents

| Path | Description |
|---|---|
| `images.csv` | 63 rows: image filename, relative image path, camera, day/night, UTC time, batch, order, dimensions, SHA-256, box count |
| `split.csv` | The same rows in the evaluator's split format |
| `val.txt` | Image paths relative to the base LTA dataset |
| `labels/val/*.txt` | YOLO labels (`class cx cy w h`, normalized to the image) |
| `annotations/instances.json` | COCO detection annotations in absolute pixels |
| `annotations/annotations.jsonl` | One JSON object per image, convenient for `datasets` |
| `excluded.csv` | The 17 images the annotator removed, with the reason |
| `dataset.json` | Provenance, class names, counts and label-manifest hashes |
| `checksums.sha256` | SHA-256 for every file in this package |

Classes, in YOLO index order:

```text
0 car   1 motorcycle   2 bus   3 truck
```

Box counts: 611 cars, 459 motorcycles, 97 trucks and 30 buses (1,197 boxes).

## Provenance and label quality

The source images are the 80 week-2 validation frames used by the detection
experiment. An anonymous annotator reviewed all 80, removed 17 frames judged too dense or
too dark to label reliably, and corrected the remaining 63 in Label Studio.

The labels were **initialized from the YOLO26x teacher pre-labels and then
corrected by hand** (96% of the teacher boxes retained; 566 boxes added, of which
385 were drawn by hand). They are therefore *human-corrected*, not an
independent annotation from scratch. This also means the teacher family has an
advantage on this set. The report describes the set as a human-corrected
validation subset, not as a fully independent test set.

The 17 excluded frames are concentrated on camera 2702 and at night, so metrics
on this subset are likely optimistic for dense and nighttime scenes. A single anonymous
annotator produced all labels. See the report's limitations section for the
evaluation implications.

## Loading

The JSONL file can be loaded directly:

```python
from datasets import load_dataset

annotations = load_dataset(
    "json",
    data_files="annotations/annotations.jsonl",
    split="train",
)
print(annotations[0])
```

Each row contains `image`, `image_path`, camera/time metadata, `sha256`, and a
list of boxes with `category`, `category_id` and `bbox_xywh` in absolute pixels.

For YOLO training, use `images.csv` to resolve each image in the base dataset and
`labels/val/<image>.txt` for its labels. For COCO-style evaluation, use
`annotations/instances.json`.

## Reproducing this package

The package is generated from the Label Studio export with the repository's
collection command:

```bash
python -m src.detection.annotate collect <export.json> \
    --annotator "<annotator>" \
    --annotations-only \
    --excluded <deleted-images.csv> \
    --out datasets/lta_human_v1
```

`--annotations-only` deliberately skips copying images, so the repository and
the Hugging Face dataset contain labels rather than a second copy of the LTA
images.

## Publishing to Hugging Face

Upload this directory as a subfolder of the existing image dataset; do not
re-upload the images:

```bash
hf upload LOVEIII486/isy5002-lta-traffic-images datasets/lta_human_v1 \
    annotations/lta_human_v1 --repo-type dataset
```

The annotation files are released under CC BY 4.0. Attribution: "LTA
human-corrected vehicle boxes, ISY5002 Project Team (anonymous), CC BY 4.0."
The base images remain under the Singapore Open Data Licence 1.0. If the
annotation license changes, update both this card and `dataset.json`.
