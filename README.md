# ISY5002 Traffic Monitoring and Forecasting

A modular prototype for traffic monitoring and short-term forecasting. The
project combines vehicle recognition on Singapore LTA traffic-camera images with
short-term road-network forecasting on the public PEMSD8 benchmark, and presents
both outputs in a web workbench.

The recognition and forecasting components use different data domains. Their
outputs are joined only at the presentation layer, with the data source of each
view stated explicitly.

## Components

| Path | Purpose | Outputs |
|---|---|---|
| `src/detection/` | Run YOLO vehicle detection on LTA traffic-camera images and count vehicles | `results/detection/` |
| `src/camera_filter/` | Apply per-camera road regions and exclusion rules to remove recurring false positives | `results/camera_filter/` |
| `src/congestion/` | Estimate frame-level congestion from the detector, background differencing and a zero-shot vision-language model | `results/congestion/` |
| `src/forecast/` | Train and compare spatio-temporal forecasting models on PEMSD8, including the custom MultiHeadSTGCN | `results/forecast/` |
| `src/demo/` | Serve the Traffic Monitoring and Forecasting Workbench | Web interface |

The training and analysis modules communicate through files under `results/`.
The workbench reads those artifacts and does not import the training modules.

## Requirements

- Python 3.13
- A CUDA-capable PyTorch installation for training (CPU is sufficient for a
  small inference smoke test)
- Git submodules

`requirements.txt` does not include PyTorch. Install the PyTorch build that
matches your GPU and CUDA version first, then install the remaining packages.

```bash
conda create -n isy5002-project python=3.13 -y
conda activate isy5002-project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The forecasting code depends on the LibCity submodule:

```bash
git submodule update --init third_party/LibCity
```

## Setup

Clone the repository and fetch the datasets and pretrained detector weights:

```bash
git clone https://github.com/LOVEIII486/ISY5002-TrafficPrediction.git
cd ISY5002-TrafficPrediction

git submodule update --init third_party/LibCity
python scripts/fetch_data.py
```

`fetch_data.py` prepares the LTA images, the PEMSD8 data used by LibCity, the
optional METR-LA data and the YOLO weights. On Windows it also creates the
`third_party/LibCity/raw_data` directory junction used by LibCity.

## Quick checks

Run one short forecasting job and a small detection pass to verify the
environment:

```bash
python -m src.forecast.run_benchmark \
    --targets traffic_flow --models FNN --max-epoch 1

python -m src.detection.run_detect --limit 8
```

## Running the modules

### Vehicle detection

```bash
python -m src.detection.run_detect --limit 8
python -m src.detection.run_detect
python -m src.detection.run_detect \
    --weights results/detection/finetune/<run>/weights/best.pt
```

When using a fine-tuned detector, map its class names to `car`, `motorcycle`,
`bus` and `truck` in `src/detection/classes.yaml`. See
[`src/detection/README.md`](src/detection/README.md) for the input contract and
evaluation commands.

### Camera filter and congestion analysis

```bash
python -m src.camera_filter.apply
python -m src.congestion.report
```

See [`src/camera_filter/README.md`](src/camera_filter/README.md) and
[`src/congestion/README.md`](src/congestion/README.md) for calibration, labeling
and evaluation details.

### Forecasting

```bash
# A small baseline comparison
python -m src.forecast.run_benchmark \
    --dataset PEMSD8 --models FNN,STGCN,MTGNN

# The custom model
python -m src.forecast.run_benchmark \
    --dataset PEMSD8 --targets traffic_flow --models MultiHeadSTGCN

# Multiple seeds
python -m src.forecast.run_benchmark \
    --dataset PEMSD8 --models STGCN --seeds 0,1,2
```

Experiment records and resolved configurations are written under
`results/forecast/PEMSD8/<target>/<model>/<tag>/`; each run contains its
resolved configuration and metrics.

### Workbench

```bash
python -m src.demo.web.server
```

Open `http://127.0.0.1:8600`. The workbench provides Overview, Camera monitoring
and Forecast views. It reads the CSV/JSON artifacts produced by the modules
above and marks data sources explicitly.

## Human-corrected LTA annotations

A human-corrected detection subset is distributed with the LTA image dataset:

```text
annotations/lta_human_v1/
```

It contains 63 images and 1,197 vehicle boxes in YOLO, COCO and JSONL formats,
together with the image manifest, excluded-frame list and checksums. Images are
not duplicated. See
[`datasets/lta_human_v1/README.md`](datasets/lta_human_v1/README.md) and the
Hugging Face dataset:

<https://huggingface.co/datasets/LOVEIII486/isy5002-lta-traffic-images>

## Data sources and licenses

- LTA traffic-camera images: collected from the
  [data.gov.sg Traffic Images API](https://data.gov.sg/datasets/d_6cdb6b405b25aaaacbaf7689bcc6fae0/view)
  and used under the
  [Singapore Open Data Licence 1.0](https://data.gov.sg/open-data-licence).
- Human-corrected annotations: released under CC BY 4.0.
- MIO-TCD: human-labeled vehicle detection benchmark
  ([dataset page](https://tcd.miovision.com/)).
- PEMSD8: public road-network forecasting benchmark distributed by LibCity,
  with raw measurements from Caltrans PeMS.

## Documentation

| Document | Contents |
|---|---|
| [`src/detection/README.md`](src/detection/README.md) | Detection pipeline and evaluation contract |
| [`src/camera_filter/README.md`](src/camera_filter/README.md) | Camera-filter calibration, usage and evaluation |
| [`src/congestion/README.md`](src/congestion/README.md) | Congestion labeling, models and evaluation |
| [`datasets/lta_human_v1/README.md`](datasets/lta_human_v1/README.md) | Human-corrected annotation package and Hugging Face details |
