"""Run eight flow models sequentially with the repository's reference protocol.

Re-running the command skips completed runs and preserves their artifacts.
Each model runs in a separate process so dataset/GPU memory is released.
Status goes to results/reruns/<tag> for seed 0 and results/reruns/<tag>_seed<N> otherwise.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["FNN", "GRU", "LSTM", "MTGNN", "MultiHeadSTGCN", "GWNET", "AGCRN", "STGCN"]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="aligned_20261013_mps")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="passed to every run_benchmark call, e.g. --set add_day_in_week=true")
    args = parser.parse_args()
    # Seed 0 keeps the original folder so existing queues stay resumable.
    folder = ROOT / "results/reruns" / (args.tag if args.seed == 0 else f"{args.tag}_seed{args.seed}")
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "status.json"
    state = {"status": "running", "pid": os.getpid(), "target": "traffic_flow", "seed": args.seed,
             "models": {model: "pending" for model in args.models}, "tag": args.tag,
             "device": args.device, "overrides": args.set, "protocol": {"train_rate": .7, "eval_rate": .1, "input_window": 12,
             "output_window": 12, "batch_size": 64, "max_epoch": 100, "patience": 30}}
    records = []

    def summarize():
        save(state_path, state)
        rows = []
        for record in records:
            scores = record["per_target"]["traffic_flow"]
            for horizon in ("_avg", "@15min", "@30min", "@60min"):
                rows.append({"model": record["model"], "seed": record["seed"], "horizon": horizon,
                             "masked_MAE": scores["masked_MAE"][horizon],
                             "masked_RMSE": scores["masked_RMSE"][horizon],
                             "masked_MAPE_percent": scores["masked_MAPE"][horizon]*100,
                             "epochs_completed": record["epochs_completed"], "best_epoch": record["best_epoch"],
                             "input_channels": record["effective"]["input_channels"], "device": record["device"]})
        if rows:
            with (folder / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    summarize()
    child = None
    try:
        for model in args.models:
            run = ROOT / "results/forecast/PEMSD8/traffic_flow" / model / f"seed{args.seed}_{args.tag}"
            result_file = run / "run.json"
            if result_file.is_file() and json.loads((run / "status.json").read_text())["status"] == "completed":
                records.append(json.loads(result_file.read_text()))
                state["models"][model] = "completed"
                summarize()
                print(f"SKIP completed {model}", flush=True)
                continue
            if run.exists():
                state["models"][model] = "existing_incomplete"
                summarize()
                raise RuntimeError(f"Preserving incomplete run: {run}; inspect it and select a new tag.")
            state["models"][model] = "running"
            state["current_model"] = model
            command = [sys.executable, "-u", "-m", "src.forecast.run_benchmark", "--dataset", "PEMSD8",
                       "--targets", "traffic_flow", "--models", model, "--seeds", str(args.seed), "--device", args.device,
                       "--threads", "4", "--tag", args.tag]
            for item in args.set:
                command += ["--set", item]
            state["command"] = command
            summarize()
            print(f"START {model}, log: {folder / (model + '.log')}", flush=True)
            with (folder / f"{model}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                summarize()
                code = child.wait()
            if code:
                state["models"][model] = "failed"
                summarize()
                raise RuntimeError(f"{model} exited with {code}; see its log")
            record = json.loads(result_file.read_text())
            records.append(record)
            state["models"][model] = "completed"
            summarize()
            print(f"DONE {model}: MAE={record['per_target']['traffic_flow']['masked_MAE']['_avg']:.4f}", flush=True)
            child = None
        state["status"] = "completed"
        state.pop("current_model", None)
        summarize()
        print(f"COMPLETE: {folder / 'comparison.csv'}", flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            import signal
            child.send_signal(signal.SIGINT)
            child.wait()
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = repr(error)
        summarize()
        raise


if __name__ == "__main__":
    main()
