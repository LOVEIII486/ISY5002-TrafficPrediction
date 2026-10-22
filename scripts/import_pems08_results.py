"""Import recorded PeMS08 metrics; never import or run a training module.

Usage: python scripts/import_pems08_results.py [--source PATH]
The default source is the sibling ISY5002-Traffic-Recognition repository.
Raw records are copied unchanged. Protocol groups remain separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "flow_10epoch_batch8_seed0"
OUTPUT = ROOT / "results/imported/pems08_cpu10_seed0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "ISY5002-Traffic-Recognition")
    args = parser.parse_args()
    source = args.source.resolve()
    run = source / "results/pems08" / RUN_NAME
    config = json.loads((run / "run_config.json").read_text())
    protocol = config["protocol"]
    if protocol["dataset"] != "PEMS08" or protocol["target"] != "traffic_flow" or protocol["smoke"]:
        raise SystemExit("Expected the recorded full-test traffic_flow experiment, not a smoke run.")
    runner = source / "src/11_run_pems08_benchmarks.py"
    if digest(runner) != protocol["runner_sha256"]:
        raise SystemExit("Runner changed since this experiment; refusing to attach an incorrect snapshot.")
    verified = {}
    for name, expected in protocol["data_checksums"].items():
        actual = digest(source / "datasets/external/PEMS08" / name)
        if actual != expected["sha256"]:
            raise SystemExit(f"Dataset checksum mismatch: {name}")
        verified[name] = actual

    with (run / "comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["protocol_id"] != config["protocol_id"] for row in rows):
        raise SystemExit("Metric rows contain another protocol.")
    completed = set(config["completed_models"])
    recorded = {row["model"] for row in rows}
    naive = {"persistence", "moving_average", "seasonal_naive"}
    if recorded != completed | naive:
        raise SystemExit("Completed models and recorded scores disagree.")
    if OUTPUT.exists():
        raise SystemExit(f"Import already exists; preserving it: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    copies = []

    def copy(original: Path, relative: str) -> None:
        destination = OUTPUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        copies.append({"source": str(original.relative_to(source)), "destination": relative,
                       "bytes": destination.stat().st_size, "sha256": digest(destination)})

    for filename in ("run_config.json", "comparison.csv"):
        copy(run / filename, filename)
    copy(source / "datasets/external/PEMS08/metadata.json", "dataset_metadata.json")
    copy(runner, "source_snapshot/legacy_runner.py")
    copy(source / "src/10_prepare_pems08.py", "source_snapshot/legacy_prepare.py")
    histories = {}
    for model in config["models"]:
        folder = run / model.lower()
        if not folder.is_dir():
            histories[model] = {"status": "not_started", "epochs_recorded": 0}
            continue
        for name in ("architecture_config.json", "training_history.csv"):
            copy(folder / name, f"models/{model.lower()}/{name}")
        with (folder / "training_history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda row: float(row["validation_normalized_MAE"]))
        histories[model] = {
            "status": "completed" if model in completed else "interrupted_without_test_metrics",
            "epochs_recorded": len(history), "best_validation_epoch": int(best["epoch"]),
            "best_validation_normalized_MAE": float(best["validation_normalized_MAE"]),
            "last_validation_normalized_MAE": float(history[-1]["validation_normalized_MAE"]),
        }
    write_json(OUTPUT / "history_summary.json", histories)

    current_bundle = ROOT / "datasets/libcity/PEMSD8"
    audit = {
        "benchmark_lineage": "PEMS08 is the source benchmark converted to LibCity PEMSD8",
        "evidence_url": "https://github.com/LibCity/Bigscity-LibCity-Datasets/blob/master/pemsd8.py",
        "legacy_shape": [17856, 170, 3], "interval_minutes": 5,
        "fields": ["traffic_flow", "traffic_occupancy", "traffic_speed"],
        "legacy_files_sha256_verified": verified,
        "legacy_edge_rows": 295, "reference_unique_directed_edges": 277,
        "edge_note": "Official converter removes duplicate ordered pairs; binary adjacency ignores duplicate rows.",
        "reference_data_bundle_present_at_import": (current_bundle / "PEMSD8.dyna").is_file(),
        "reference_data_content_equality": "not_verified",
        "reason": "No original training data or dataset checksum was recorded in the new repository's run.json.",
        "same_test_samples": "not_verified; split construction and final-batch padding differ",
        "cross_protocol_ranking_allowed": False,
    }
    write_json(OUTPUT / "dataset_audit.json", audit)

    # Reference values retain their native metric definitions; no cross-group ranking/delta.
    comparison = []
    for row in rows:
        if row["horizon_minutes"] == "5":
            continue
        comparison.append({
            "protocol_group": "legacy_cpu10_60_20_20", "model": row["model"],
            "horizon_minutes": row["horizon_minutes"], "masked_MAE": row["masked_MAE"],
            "masked_RMSE": row["masked_RMSE"], "masked_MAPE_percent": row["masked_MAPE_percent"],
            "aggregation": "pooled elements" if row["horizon_minutes"] == "all" else "single horizon",
            "source_file": "results/imported/pems08_cpu10_seed0/comparison.csv",
            "cross_protocol_ranking_allowed": "false",
        })
    for model in ("FNN", "STGCN", "MTGNN", "MultiHeadSTGCN"):
        relative = f"results/forecast/PEMSD8/traffic_flow/{model}/seed0/run.json"
        record = json.loads((ROOT / relative).read_text())
        scores = record["per_target"]["traffic_flow"]
        for label, minutes in (("_avg", "all"), ("@15min", "15"), ("@30min", "30"), ("@60min", "60")):
            comparison.append({
                "protocol_group": "reference_cuda100_70_10_20", "model": model,
                "horizon_minutes": minutes, "masked_MAE": scores["masked_MAE"][label],
                "masked_RMSE": scores["masked_RMSE"][label],
                "masked_MAPE_percent": scores["masked_MAPE"][label] * 100,
                "aggregation": "mean of per-horizon metrics" if minutes == "all" else "single horizon",
                "source_file": relative, "cross_protocol_ranking_allowed": "false",
            })
    with (OUTPUT / "reference_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_json(OUTPUT / "import_manifest.json", {
        "source_repository": "https://github.com/LOVEIII486/ISY5002-Traffic-Recognition",
        "source_run": f"results/pems08/{RUN_NAME}",
        "source_protocol_id": config["protocol_id"], "source_run_status": config["status"],
        "source_state_note": "Experiment scripts/results were uncommitted; file hashes identify the evidence.",
        "completed_models": sorted(completed), "unfinished_models": config["unfinished_models"],
        "naive_baselines": sorted(naive), "original_files": copies,
        "note": "No training, model inference, or test evaluation was performed during import.",
    })
    print(f"Imported recorded metrics and provenance to {OUTPUT}")


if __name__ == "__main__":
    main()
