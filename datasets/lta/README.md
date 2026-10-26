---
license: other
license_name: singapore-open-data-licence-1.0
license_link: https://data.gov.sg/open-data-licence
pretty_name: LTA Traffic Camera Images with Human-Corrected Vehicle Boxes (Singapore, September 2026)
tags:
  - traffic
  - singapore
  - lta
  - object-detection
  - images
  - time-series
  - human-corrected
  - annotations
size_categories:
  - 10K<n<100K
---

# LTA Traffic Camera Images with Human-Corrected Vehicle Boxes (Singapore, September 2026)

Two weeks of stills from Land Transport Authority (LTA) traffic cameras in Singapore, September 2026, together with a small human-corrected detection subset.

**15,316 images · 8 cameras · 2.6 GB · 10-minute sampling**

**63 human-corrected images · 1,197 vehicle boxes · 8 cameras**

## Attribution

> Contains information from the **Traffic Images** dataset accessed on
> 2026-09-13 to 2026-09-29 from **data.gov.sg** which is made available under the terms of the
> [Singapore Open Data Licence version 1.0](https://data.gov.sg/open-data-licence)

This dataset is not endorsed by, affiliated with, or an official product of the Singapore
Government or the Land Transport Authority. The human-corrected annotations were produced
by the project team and are not an LTA product.

## Contents

| Directory | Cameras | Images | Period (UTC) |
|---|---|---|---|
| `week1/` | 2701 2702 2704 4703 4712 4713 4798 4799 | 7,321 | 2026-09-13 → 2026-09-20 |
| `week2/` | the same 8 | 7,995 | 2026-09-22 → 2026-09-29 |
| `annotations/lta_human_v1/` | the same 8 | 63 annotated images | week 2 |

```
week1/2701/20260913T124023Z_2701_1240_97acd1b15491.jpg
└──┬─┘└┬─┘ └─────┬──────┘ └┬─┘ └┬─┘ └────┬─────┘
  week camera  captured_at  camera HHMM  sha256[:12]
                 (UTC)          (Singapore time)
```

## Human-corrected vehicle annotations

`annotations/lta_human_v1/` contains a 63-image human-corrected detection subset.
It is **not** a full annotation of all 15,316 images: it covers 63 of the 80
week-2 validation frames used by the project (78.8% of that validation set, or
0.41% of the full collection). The other 17 validation frames were removed
because the annotator judged them too dense or too dark to label reliably.

### What was annotated

One anonymous annotator reviewed the 80 week-2 validation frames in Label Studio. Each
frame started from YOLO26x teacher pre-labels generated at the original
1920-pixel resolution (confidence threshold 0.3). The annotator checked every
vehicle, corrected classes and box extents, added missed vehicles, and removed
boxes that were not vehicles. The labels are therefore **human-corrected
teacher pre-labels**, not an annotation from scratch. This package contains
detection boxes only; the project's separate 600-frame congestion-level labels
are not included here.

- 63 images retained, 17 removed
- 36 daytime and 27 nighttime images
- 1,197 boxes: 611 cars, 459 motorcycles, 30 buses and 97 trucks
- 96% of the teacher boxes retained
- 566 boxes added, including 385 drawn by hand
- One anonymous annotator; no inter-annotator agreement

| Camera | Images |
|---|---:|
| 2701 | 7 |
| 2702 | 3 |
| 2704 | 8 |
| 4703 | 10 |
| 4712 | 10 |
| 4713 | 8 |
| 4798 | 8 |
| 4799 | 9 |
| **Total** | **63** |

The 17 excluded frames are concentrated on camera 2702 and at night. Metrics on
this subset are therefore likely optimistic for dense and nighttime scenes, and
the subset is a **human-corrected validation subset, not an independent test
set**: the same images were used for teacher-reference comparisons and
resolution selection, and the teacher family has an advantage because the
annotations start from its boxes.

### Annotation files

| Path | Description |
|---|---|
| `images.csv` | 63 rows: image path, camera, day/night, UTC time, dimensions, SHA-256, box count |
| `split.csv` | The same rows in the evaluator's split format |
| `val.txt` | Image paths relative to the dataset root (`week2/...`) |
| `labels/val/*.txt` | YOLO labels (`class cx cy w h`, normalized) |
| `annotations/instances.json` | COCO detection annotations in absolute pixels |
| `annotations/annotations.jsonl` | One JSON object per image, convenient for `datasets` |
| `excluded.csv` | The 17 removed images and the reason |
| `dataset.json` | Provenance, class names, counts and label-manifest hashes |
| `checksums.sha256` | SHA-256 for every annotation-package file |

Images are not duplicated. Join each `image_path` in `images.csv` to the
corresponding file at the dataset root, and verify it against the recorded
SHA-256. The YOLO class order is:

```text
0 car   1 motorcycle   2 bus   3 truck
```

The JSONL file can be loaded directly:

```python
from datasets import load_dataset

annotations = load_dataset(
    "json",
    data_files="annotations/lta_human_v1/annotations/annotations.jsonl",
    split="train",
)
print(annotations[0])
```

## Source and collection

Images were fetched from the
[data.gov.sg real-time traffic images API](https://api.data.gov.sg/v1/transport/traffic-images)
by a small scheduled script run across roughly 30 collection windows. Every round requested
all 8 cameras; a frame was written only when it was fresh and its sha256 differed from the
previous one, so the archive contains no duplicate frames.

Camera feeds freeze overnight — the longest observed gap was 215 minutes — so the freshness
limit was raised from 15 to 240 minutes on 2026-09-16. The resulting bias is that
**00:00–07:00 on 2026-09-14 and 09-15 is markedly under-represented**.

The human-corrected annotations were added by the project team. The source images
are the week-2 validation frames; the annotation generation and filtering commands
are in `src/detection/annotate.py` in the project repository.

## Intended use

Vehicle detection and counting, traffic density estimation, diurnal pattern analysis,
per-camera time-series modelling, and evaluation on the human-corrected 63-image subset.

## Limitations

- **The base images have no annotations.** A 63-image human-corrected detection subset is
  provided under `annotations/lta_human_v1/`; it is not a full annotation of all images.
- **The annotation subset is teacher-assisted and small.** It was initialized from YOLO26x
  pre-labels, produced by one anonymous annotator, and covers only 63 images. It is a validation
  subset, not an independent test set; see `annotations/lta_human_v1/README.md`.
- **Night frames are dark**, and detection recall drops sharply.
- **No geographic metadata**, only camera IDs; lane geometry must be derived from the images
  if needed.
- **Number plates may be legible in some full-resolution frames.** The Singapore Open Data
  Licence does not extend to personal data; consider this before redistributing further.

## License

The images remain under the Singapore Open Data Licence 1.0. The human-corrected
annotation files are released under CC BY 4.0. Attribution: "LTA human-corrected
vehicle boxes, ISY5002 Project Team (anonymous), CC BY 4.0." See
`annotations/lta_human_v1/dataset.json` for the machine-readable record.
