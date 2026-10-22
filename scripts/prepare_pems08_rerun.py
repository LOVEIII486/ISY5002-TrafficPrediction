"""Prepare the verified sibling PeMS08 data for the pinned LibCity loader."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "ISY5002-Traffic-Recognition"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    raw = OLD / "datasets/external/PEMS08"
    metadata = json.loads((raw / "metadata.json").read_text())
    for name, record in metadata["checksums"].items():
        if sha256(raw / name) != record["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {name}")
    dest = ROOT / "datasets/libcity/PEMSD8"
    if dest.exists():
        raise FileExistsError(f"Preserving existing data directory: {dest}")
    dest.mkdir(parents=True)
    original = OLD / "third_party/LibCity/raw_data/PEMSD8"
    for name in ("PEMSD8.geo", "PEMSD8.dyna"):
        shutil.copyfile(original / name, dest / name)
    with (raw / "distance.csv").open(newline="") as stream:
        edges = list(csv.DictReader(stream))
    unique = {}
    for edge in edges:
        unique.setdefault((edge["from"], edge["to"]), edge["cost"])
    with (dest / "PEMSD8.rel").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rel_id", "type", "origin_id", "destination_id", "cost"])
        writer.writerows((i, "geo", a, b, cost) for i, ((a, b), cost) in enumerate(unique.items()))
    config = {
        "geo": {"including_types": ["Point"], "Point": {}},
        "rel": {"including_types": ["geo"], "geo": {"cost": "num"}},
        "dyna": {"including_types": ["state"], "state": {"entity_id": "geo_id", "traffic_flow": "num", "traffic_occupancy": "num", "traffic_speed": "num"}},
        "info": {"data_col": metadata["fields"], "output_dim": 3, "time_intervals": 300,
                 "data_files": ["PEMSD8"], "geo_file": "PEMSD8", "rel_file": "PEMSD8",
                 "weight_col": "cost", "init_weight_inf_or_zero": "zero", "set_weight_link_or_dist": "link",
                 "calculate_weight_adj": False, "weight_adj_epsilon": 0.1},
    }
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    link = ROOT / "third_party/LibCity/raw_data"
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Preserving existing LibCity data path: {link}")
    link.symlink_to(ROOT / "datasets/libcity", target_is_directory=True)
    audit = {"raw_source": metadata, "atomic_files_sha256": {p.name: sha256(p) for p in dest.iterdir()},
             "unique_directed_edges": len(unique),
             "note": "Data for this rerun; equality to the unavailable historical reference bundle remains unverified."}
    (dest / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {dest}; {len(unique)} unique directed edges")


if __name__ == "__main__":
    main()
