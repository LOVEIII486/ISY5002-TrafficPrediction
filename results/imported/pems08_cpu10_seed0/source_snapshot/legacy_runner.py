"""Compare LibCity models on original PeMS08 with one shared protocol.

Windows are loaded lazily from NPZ to avoid materializing millions of CSV rows
and all sliding windows. Model classes and architecture defaults come from the
local LibCity checkout; the PyTorch training/evaluation loop is shared here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]
LIBCITY = ROOT / "third_party/LibCity"
SUPPORTED = {
    "FNN": ("traffic_speed_prediction", "FNN", "FNN"),
    "STGCN": ("traffic_speed_prediction", "STGCN", "STGCN"),
    "MTGNN": ("traffic_speed_prediction", "MTGNN", "MTGNN"),
    "GRU": ("traffic_speed_prediction", "RNN", "RNN"),
    "LSTM": ("traffic_speed_prediction", "RNN", "RNN"),
    "GWNET": ("traffic_speed_prediction", "GWNET", "GWNET"),
    "AGCRN": ("traffic_flow_prediction", "AGCRN", "AGCRN"),
}
FIELDS = {"traffic_flow": 0, "traffic_occupancy": 1, "traffic_speed": 2}


class Windows(Dataset):
    def __init__(self, values, first_targets, input_window, output_window):
        self.values, self.first_targets = values, first_targets
        self.input_window, self.output_window = input_window, output_window

    def __len__(self):
        return len(self.first_targets)

    def __getitem__(self, index):
        start = self.first_targets[index]
        return (self.values[start - self.input_window:start].copy(),
                self.values[start:start + self.output_window].copy())


def make_splits(length, input_window, output_window, train_rate, val_rate):
    train_end, val_end = int(length * train_rate), int(length * (train_rate + val_rate))
    # Target windows cannot cross split boundaries. Earlier history is available
    # to the first validation/test prediction, as it would be in deployment.
    splits = {
        "train": np.arange(input_window, train_end - output_window + 1),
        "validation": np.arange(max(train_end, input_window), val_end - output_window + 1),
        "test": np.arange(max(val_end, input_window), length - output_window + 1),
    }
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("Empty split: reduce windows or change split rates")
    return splits, train_end, val_end


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def libcity_model(name, features, args):
    family, module_name, class_name = SUPPORTED[name]
    config = json.loads((LIBCITY / "libcity/config/model/traffic_state_pred" / f"{module_name}.json").read_text())
    config.update({"input_window": args.input_window, "output_window": args.output_window,
                   "output_dim": 1, "device": torch.device(args.device),
                   "teacher_forcing_ratio": 0, "use_curriculum_learning": False,
                   "stgcn_train_mode": "full", "max_epoch": args.max_epochs})
    if name in ("GRU", "LSTM"):
        config["rnn_type"] = name
    if name == "MTGNN":
        config["step_size1"] = 1  # Curriculum is disabled by the common trainer.
    # Load only the chosen model to avoid optional dependencies of other models.
    path = LIBCITY / "libcity/model" / family / f"{module_name}.py"
    spec = spec_from_file_location(f"pems08_{name.lower()}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if name == "GWNET":
        # This LibCity version still uses np.int, removed in NumPy 1.24.
        # Give this module a proxy rather than changing NumPy globally.
        class NumpyCompatibility:
            int = int

            def __getattr__(self, key):
                return getattr(np, key)

        module.np = NumpyCompatibility()
        # The old implementation constructs Conv1d with 2-D kernels and feeds
        # 4-D tensors. Use the intended spatial-temporal Conv2d in this module.
        class TorchLayersCompatibility:
            Conv1d = torch.nn.Conv2d

            def __getattr__(self, key):
                return getattr(torch.nn, key)

        module.nn = TorchLayersCompatibility()
        config["compatibility"] = ["np.int -> int", "2D-kernel Conv1d -> Conv2d"]
    model = getattr(module, class_name)(config, dict(features)).to(args.device)
    return model, {k: str(v) if isinstance(v, torch.device) else v for k, v in config.items()}


def prediction(model, x):
    # Future observations are not available at inference. With a single target
    # channel, RNN/STGCN need only the y shape; no future external data is used.
    dummy = torch.zeros((len(x), model.output_window, x.shape[2], 1), device=x.device)
    result = model.predict({"X": x, "y": dummy})
    if result.shape != dummy.shape or not torch.isfinite(result).all():
        raise ValueError(f"Invalid prediction shape/values: {result.shape} vs {dummy.shape}")
    return result


def masked_mae(predicted, target, scaler):
    mask = scaler.inverse_transform(target).abs() > 1e-6
    return (predicted - target).abs()[mask].mean() if mask.any() else predicted.sum() * 0


def evaluate_loss(model, loader, scaler, args):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            pred = prediction(model, x)
            valid = scaler.inverse_transform(y).abs() > 1e-6
            total += float((pred - y).abs()[valid].sum())
            count += int(valid.sum())
            if args.smoke and i >= 1:
                break
    if not count:
        raise ValueError("No nonzero validation targets")
    return total / count


def metric_rows(name, predicted, truth, protocol_id):
    rows = []
    horizons = [h for h in (1, 3, 6, 12) if h <= truth.shape[1]]
    for horizon in [*horizons, "all"]:
        p, y = (predicted, truth) if horizon == "all" else (predicted[:, horizon-1], truth[:, horizon-1])
        error = p.astype(np.float64) - y.astype(np.float64)
        valid = np.abs(y) > 1e-6
        rows.append({"model": name, "horizon_minutes": "all" if horizon == "all" else horizon * 5,
                     "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())),
                     "masked_MAE": float(np.abs(error[valid]).mean()) if valid.any() else None,
                     "masked_RMSE": float(np.sqrt(np.square(error[valid]).mean())) if valid.any() else None,
                     "masked_MAPE_percent": float(np.mean(np.abs(error[valid] / y[valid])) * 100) if valid.any() else None,
                     "nonzero_targets": int(valid.sum()), "total_targets": y.size,
                     "protocol_id": protocol_id})
    return rows


def save_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(name, loaders, features, args, output):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, config = libcity_model(name, features, args)
    output.mkdir()
    json_write(output / "architecture_config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = features["scaler"]
    best, stale, history = float("inf"), 0, []
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(loaders["train"].dataset, batch_size=args.batch_size,
                              shuffle=True, generator=generator)
    for epoch in range(1, args.max_epochs + 1):
        started = time.monotonic()
        model.train()
        total, samples = 0., 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            objective = masked_mae(prediction(model, x), y, scaler)
            if not torch.isfinite(objective):
                raise ValueError(f"Non-finite training loss for {name}")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(objective.detach()) * len(x)
            samples += len(x)
            if not args.smoke and (i + 1) % 200 == 0:
                print(f"{name} epoch {epoch}: batch {i+1}/{len(train_loader)}", flush=True)
            if args.smoke and i >= 1:
                break
        validation = evaluate_loss(model, loaders["validation"], scaler, args)
        history.append({"epoch": epoch, "train_normalized_MAE": total / samples,
                        "validation_normalized_MAE": validation,
                        "seconds": time.monotonic() - started})
        print(f"{name} epoch {epoch}/{args.max_epochs}: train={total/samples:.4f} val={validation:.4f}, {history[-1]['seconds']:.1f}s", flush=True)
        save_table(output / "training_history.csv", history)
        if validation < best:
            best, stale = validation, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "validation_normalized_MAE": best}, output / "best_model.pt")
        else:
            stale += 1
        if args.smoke or stale >= args.patience:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loaders["test"]):
            pred = prediction(model, x.to(args.device))
            predictions.append(scaler.inverse_transform(pred).cpu().numpy())
            truths.append(scaler.inverse_transform(y).numpy())
            if args.smoke and i >= 1:
                break
    return np.concatenate(predictions), np.concatenate(truths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/external/PEMS08")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED, default=list(SUPPORTED))
    parser.add_argument("--target", choices=FIELDS, default="traffic_flow")
    parser.add_argument("--input-window", type=int, default=12)
    parser.add_argument("--output-window", type=int, default=12)
    parser.add_argument("--train-rate", type=float, default=.6)
    parser.add_argument("--val-rate", type=float, default=.2)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Two batches per split; writes smoke outputs, never a benchmark")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, help="Earlier run directory; protocol must match before scores can be merged")
    args = parser.parse_args()
    if not 0 < args.train_rate < args.train_rate + args.val_rate < 1:
        parser.error("Invalid split rates")
    if min(args.input_window, args.output_window, args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Windows and training arguments must be positive")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(LIBCITY))
    from libcity.utils.normalization import StandardScaler
    prep_spec = importlib.util.spec_from_file_location("pems08_prepare", ROOT / "src/10_prepare_pems08.py")
    prep = importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    data, edges, metadata = prep.validate(args.source)
    raw = data[..., FIELDS[args.target]:FIELDS[args.target]+1]
    splits, train_end, val_end = make_splits(len(raw), args.input_window, args.output_window, args.train_rate, args.val_rate)
    scaler = StandardScaler(float(raw[:train_end].mean()), float(raw[:train_end].std()))
    if scaler.std <= 0:
        raise ValueError("Constant training series")
    values = scaler.transform(raw).astype(np.float32)
    loaders = {name: DataLoader(Windows(values, indices, args.input_window, args.output_window), batch_size=args.batch_size) for name, indices in splits.items()}
    adj = np.zeros((170, 170), dtype=np.float32)
    for edge in edges:
        origin, destination = int(edge["from"]), int(edge["to"])
        adj[origin, destination] = adj[destination, origin] = 1
    features = {"scaler": scaler, "num_nodes": 170, "feature_dim": 1, "output_dim": 1,
                "ext_dim": 0, "adj_mx": adj, "num_batches": len(loaders["train"])}
    sources = {name: sha256(LIBCITY / "libcity/model" / family / f"{module}.py") for name, (family, module, _) in SUPPORTED.items()}
    protocol = {"dataset": "PEMS08", "data_checksums": metadata["checksums"], "target": args.target,
                "inputs": [args.target], "interval_minutes": 5,
                "input_window": args.input_window, "output_window": args.output_window,
                "train_end_index": train_end, "validation_end_index": val_end,
                "split_policy": "chronological; target windows remain wholly inside each split",
                "split_samples": {k: len(v) for k, v in splits.items()},
                "scaler": {"type": "standard", "fit": "training timestamps only", "mean": scaler.mean, "std": scaler.std},
                "graph": "binary, bidirectional, source distance.csv edges",
                "optimizer": "Adam", "learning_rate": args.learning_rate, "loss": "MAE excluding zero targets",
                "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
                "gradient_clip_norm": 5, "seed": args.seed, "device": args.device, "threads": args.threads,
                "libcity_model_sha256": sources, "runner_sha256": sha256(Path(__file__)), "smoke": args.smoke}
    protocol["libcity_default_config_sha256"] = {
        name: sha256(LIBCITY / "libcity/config/model/traffic_state_pred" / f"{module}.json")
        for name, (_, module, _) in SUPPORTED.items()
    }
    protocol["software"] = {"torch": torch.__version__, "numpy": np.__version__}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    reference_rows = []
    if args.reference:
        prior = json.loads((args.reference / "run_config.json").read_text())
        if args.smoke or prior["protocol_id"] != protocol_id or prior.get("status") != "completed":
            raise ValueError("Reference protocol differs or run is incomplete; cannot combine benchmark scores")
        with (args.reference / "comparison.csv").open(newline="") as stream:
            reference_rows = list(csv.DictReader(stream))
    run = args.output or ROOT / "results/pems08" / (("smoke_" if args.smoke else "run_") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    config = {"protocol_id": protocol_id, "protocol": protocol, "models": args.models,
              "status": "running", "source": metadata["source"], "runner": "LibCity models with shared PyTorch loop",
              "reference": str(args.reference) if args.reference else None}
    json_write(run / "run_config.json", config)
    print(f"Run directory: {run}\nTarget: {args.target}, split samples: {protocol['split_samples']}", flush=True)
    rows = []
    test_indices = splits["test"][:args.batch_size * 2] if args.smoke else splits["test"]
    truth = np.stack([raw[start:start+args.output_window] for start in test_indices])
    np.savez_compressed(run / "test_targets.npz", truth=truth, first_target_indices=test_indices)
    for baseline in ("persistence", "moving_average", "seasonal_naive"):
        if baseline == "seasonal_naive":
            if args.output_window > 288 or np.min(test_indices) < 288:
                raise ValueError("Daily baseline requires enough history and a horizon no longer than one day")
            pred = np.stack([raw[start-288:start-288+args.output_window] for start in test_indices])
        else:
            history = np.stack([raw[start-args.input_window:start] for start in test_indices])
            anchor = history[:, -1:] if baseline == "persistence" else history.mean(axis=1, keepdims=True)
            pred = np.repeat(anchor, args.output_window, axis=1)
        rows.extend(metric_rows(baseline, pred, truth, protocol_id))
    table = run / ("smoke_metrics.csv" if args.smoke else "comparison.csv")
    save_table(table, rows)
    try:
        for name in args.models:
            pred, observed = train_one(name, loaders, features, args, run / name.lower())
            if not np.allclose(observed, truth, rtol=1e-5, atol=1e-4):
                raise ValueError("Test target alignment mismatch")
            np.savez_compressed(run / name.lower() / "predictions.npz", prediction=pred, first_target_indices=test_indices)
            model_rows = metric_rows(name, pred, truth, protocol_id)
            rows.extend(model_rows)
            save_table(table, rows)
            print(f"{name} complete: {model_rows[-1]}", flush=True)
        config["status"] = "smoke_completed" if args.smoke else "completed"
        json_write(run / "run_config.json", config)
        if reference_rows:
            current = {row["model"] for row in rows}
            save_table(run / "comparison_with_reference.csv", rows + [row for row in reference_rows if row["model"] not in current])
    except BaseException as exc:
        config.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        json_write(run / "run_config.json", config)
        raise
    print(f"Saved: {table}", flush=True)


if __name__ == "__main__":
    main()
