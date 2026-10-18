"""Validate the original PeMS08 archive and convert it to LibCity atomic files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/Davidham3/ASTGCN-2019-mxnet/tree/master/data/PEMS08"
EXPECTED_BLOBS = {
    "pems08.npz": "f5f3e84a00417573aa8487fe1e9e8cd13c6026c8",
    "distance.csv": "cf11bea3ff4229d2d900fdcf1085828d25429c88",
}
FIELDS = ["traffic_flow", "traffic_occupancy", "traffic_speed"]


def validate(directory: Path) -> tuple[np.ndarray, list[dict], dict]:
    checksums = {}
    for name, expected in EXPECTED_BLOBS.items():
        payload = (directory / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if blob != expected:
            raise ValueError(f"Source checksum mismatch: {name}: {blob}")
        checksums[name] = {"git_blob_sha1": blob, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    with np.load(directory / "pems08.npz", allow_pickle=False) as archive:
        data = archive["data"].astype(np.float32)
    if data.shape != (17856, 170, 3) or not np.isfinite(data).all():
        raise ValueError(f"Invalid PeMS08 tensor: {data.shape}")
    with (directory / "distance.csv").open(newline="") as stream:
        edges = list(csv.DictReader(stream))
    for edge in edges:
        if not all(0 <= int(edge[field]) < 170 for field in ("from", "to")):
            raise ValueError(f"Invalid node index: {edge}")
        if not np.isfinite(float(edge["cost"])) or float(edge["cost"]) < 0:
            raise ValueError(f"Invalid edge distance: {edge}")
    metadata = {
        "dataset": "PEMS08", "source": SOURCE, "shape": list(data.shape),
        "fields": FIELDS, "interval_minutes": 5, "edge_count": len(edges),
        "start_time": "2016-07-01T00:00:00", "end_time": "2016-08-31T23:55:00",
        "time_note": "Index-based timestamps follow LibCity's converter; timezone not verified.",
        "checksums": checksums,
    }
    return data, edges, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/external/PEMS08")
    parser.add_argument("--output", type=Path, default=ROOT / "third_party/LibCity/raw_data/PEMSD8")
    parser.add_argument("--target", choices=FIELDS, default="traffic_flow")
    args = parser.parse_args()
    data, edges, metadata = validate(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / "PEMSD8"
    with prefix.with_suffix(".geo").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["geo_id", "type", "coordinates"])
        writer.writerows((node, "Point", "[]") for node in range(data.shape[1]))
    with prefix.with_suffix(".rel").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rel_id", "type", "origin_id", "destination_id", "cost"])
        writer.writerows((i, "geo", edge["from"], edge["to"], edge["cost"]) for i, edge in enumerate(edges))
    times = np.datetime64("2016-07-01T00:00:00") + np.arange(data.shape[0]) * np.timedelta64(5, "m")
    stamps = np.datetime_as_string(times, unit="s")
    with prefix.with_suffix(".dyna").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dyna_id", "type", "time", "entity_id", *FIELDS])
        for node in range(data.shape[1]):
            writer.writerows((node * len(times) + t, "state", stamps[t] + "Z", node, *data[t, node].tolist()) for t in range(len(times)))
    config = {"info": {
        "data_col": [args.target], "output_dim": 1, "time_intervals": 300,
        "data_files": ["PEMSD8"], "geo_file": "PEMSD8", "rel_file": "PEMSD8",
        "weight_col": "cost", "init_weight_inf_or_zero": "zero",
        "set_weight_link_or_dist": "link", "calculate_weight_adj": False,
        "load_external": False, "add_time_in_day": False, "add_day_in_week": False,
    }}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.source / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"LibCity atomic files: {args.output}")


if __name__ == "__main__":
    main()
