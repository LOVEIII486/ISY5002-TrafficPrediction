"""训练并评估一个模型。

产物写入 results/forecast/<数据集>/<目标>/<模型>/seed<N>/：run.json 是本次运行的完整
记录，libcity/ 下是 LibCity 的评估 CSV、预测 npz、日志与 TensorBoard 事件。
"""

from __future__ import annotations

import json
import time
import zlib
import csv
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import compat, datasets, paths
from . import metrics as metrics_module
from .models import registry
from .protocols import Protocol, default_protocol


def _run(
    dataset_name: str,
    targets: tuple[str, ...],
    model_name: str,
    protocol: Protocol | None = None,
    *,
    seed: int = 0,
    gpu: bool = True,
    max_epoch: int | None = None,
    batch_size: int | None = None,
    extra_args: dict | None = None,
    tag: str = "",
    requested_device: str | None = None,
) -> dict:
    """训练并评估一个模型，返回运行记录。"""
    dataset = datasets.resolve(dataset_name)
    spec = registry.resolve(model_name)
    protocol = protocol or default_protocol(*targets)

    variant = zlib.crc32(tag.encode()) % 1000 if tag else 0
    exp_id = paths.exp_id(
        dataset_name, spec.index, paths.target_mask(dataset_name, targets), seed, variant
    )
    compat_report = compat.activate(exp_id)

    use_cuda = requested_device == "cuda" if requested_device else gpu and torch.cuda.is_available()
    other_args = datasets.build_other_args(dataset, protocol, exp_id=exp_id, seed=seed, gpu=use_cuda)
    if max_epoch is not None:
        other_args["max_epoch"] = max_epoch
    if batch_size is not None:
        other_args["batch_size"] = batch_size
    merged_extra = dict(spec.default_args)
    if extra_args:
        merged_extra.update(extra_args)
    other_args.update(merged_extra)
    other_args["run_tag"] = tag

    with compat.libcity_cwd():
        from libcity.config import ConfigParser
        from libcity.data import get_dataset
        from libcity.utils import get_logger, set_random_seed

        # LibCity 会校验模型名是否在 task_config.json 的白名单内。自研模型借一个内置名字
        # 过校验（其 dataset_class/executor/evaluator 已在 default_args 中显式给出），
        # 构造完再改回真名，使缓存文件与日志使用真名。
        config_model = "FNN" if spec.source == "ours" else model_name
        config = ConfigParser("traffic_state_pred", config_model, dataset_name, other_args=other_args)
        if config.get("model") != model_name:
            config["model"] = model_name
        if requested_device:
            if requested_device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable in this process")
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this process")
            config["device"] = torch.device(requested_device)
        # 定种子并起日志。LibCity 的官方流程会做这两步，自己驱动时容易漏：漏了种子就
        # 不可复现，漏了 logger 则逐 epoch 进度与日志文件都不会产生。
        set_random_seed(config.get("seed", 0))
        get_logger(config).info("start %s / %s / seed %s", model_name, dataset_name, seed)

        libcity_dataset = get_dataset(config)
        train_data, valid_data, test_data = libcity_dataset.get_data()
        data_feature = libcity_dataset.get_data_feature()
        out_dir = paths.run_dir(dataset_name, targets, model_name, seed, tag)
        resolved = {key: str(value) if isinstance(value, torch.device) else value for key, value in config.config.items()}
        (out_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
        provenance = _provenance(libcity_dataset, train_data, valid_data, test_data, config)
        (out_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        problems = datasets.check_data_feature(dataset, data_feature, len(targets))
        if problems:
            raise RuntimeError("数据特征与注册元数据不一致：" + "；".join(problems))

        device = config.get("device")
        model = registry.build(model_name, config, data_feature).to(device)
        executor_class = compat.load_executor_class(config.get("executor", "TrafficStateExecutor"))
        executor = executor_class(config, model, data_feature)

        history = []
        best_epochs = []
        original_valid = executor._valid_epoch
        original_save = executor.save_model_with_epoch

        def record_validation(*args, **kwargs):
            value = original_valid(*args, **kwargs)
            history.append({"epoch": int(args[1]) + 1, "validation_loss": float(value)})
            with (out_dir / "training_history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            return value

        def record_save(epoch):
            result = original_save(epoch)
            best_epochs.append(int(epoch) + 1)
            return result

        executor._valid_epoch = record_validation
        executor.save_model_with_epoch = record_save

        started = time.time()
        executor.train(train_data, valid_data)
        train_seconds = time.time() - started
        torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "config": resolved, "best_epoch": best_epochs[-1] if best_epochs else None,
                    "data_provenance": provenance}, out_dir / "best_model.pt")
        executor.evaluate(test_data)
        evaluate_seconds = time.time() - started - train_seconds

        # LibCity 的评估器把全部输出列混在一起算，多目标时不可读，因此从它保存的预测
        # 数组按目标重算（数组已在原空间）。文件名以时间戳开头，排序即取最新一次。
        npz_files = sorted(compat.cache_dir(exp_id, "evaluate_cache").glob("*_predictions.npz"))
        if not npz_files:
            raise RuntimeError(f"未找到预测数组：{compat.cache_dir(exp_id, 'evaluate_cache')}")
        per_target = metrics_module.from_predictions(
            npz_files[-1], protocol, dataset.interval_seconds, targets
        )

        # 这些值大多由各模型自带的配置决定，记下来便于判断与论文是否同口径。
        effective = {
            "scaler": config.get("scaler"),
            "input_channels": data_feature.get("feature_dim"),
            "time_in_day": bool(config.get("add_time_in_day")),
            "day_in_week": bool(config.get("add_day_in_week")),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "max_epoch": config.get("max_epoch"),
            "use_early_stop": config.get("use_early_stop"),
            "patience": config.get("patience"),
            "clip_grad_norm": config.get("clip_grad_norm"),
            "pad_with_last_sample": config.get("pad_with_last_sample"),
        }
        for key in ("train_rate", "eval_rate", "input_window", "output_window", "learner", "weight_decay",
                    "lr_decay", "lr_scheduler", "steps", "step_size", "lr_decay_ratio", "max_grad_norm",
                    "head_mode", "spatial_mode", "bidir_adj_mx", "direct_multi_step", "scale_aware_loss",
                    "use_curriculum_learning", "step_size1", "rnn_type"):
            effective[key] = config.get(key, getattr(model, key, None))

    record = {
        "dataset": dataset_name,
        "targets": list(targets),
        "model": model_name,
        "source": spec.source,
        "seed": seed,
        "exp_id": exp_id,
        "interval_minutes": dataset.interval_minutes,
        "device": str(device),
        "train_seconds": round(train_seconds, 1),
        "evaluate_seconds": round(evaluate_seconds, 1),
        "compat": compat_report,
        "effective": effective,
        "overrides": extra_args or None,
        "per_target": per_target,
        "epochs_completed": len(history),
        "best_epoch": best_epochs[-1] if best_epochs else None,
        "final_task_level": getattr(model, "task_level", None),
        "data_provenance": provenance,
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "compatibility_notes": ["GWNET module Conv1d with 2-D kernels uses Conv2d"] if model_name == "GWNET" else [],
    }

    out_dir = paths.run_dir(dataset_name, targets, model_name, seed, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["artifacts"] = len(compat.collect_artifacts(exp_id, out_dir / "libcity"))
    return record


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(dataset, train_data, valid_data, test_data, config):
    folder = compat.LIBCITY_ROOT / "raw_data" / config["dataset"]
    raw_hashes = {path.name: _sha256(path) for path in sorted(folder.iterdir()) if path.suffix in (".dyna", ".geo", ".rel", ".json")}
    length = len(dataset.timesolts)  # Name used by the pinned LibCity loader.
    starts = np.arange(config["input_window"], length - config["output_window"] + 1)
    n_train = round(len(starts) * config["train_rate"])
    n_test = round(len(starts) * (1 - config["train_rate"] - config["eval_rate"]))
    test_starts = starts[-n_test:]
    out_dir = paths.run_dir(config["dataset"], tuple(config["data_col"]), config["model"], config["seed"], config.get("run_tag", ""))
    # Store original and padded order, matching this LibCity loader and evaluator exactly.
    padded = test_starts
    if config.get("pad_with_last_sample"):
        extra = (-len(test_starts)) % config["batch_size"]
        padded = np.concatenate([test_starts, np.repeat(test_starts[-1], extra)])
    np.savez_compressed(out_dir / "test_indices.npz", first_target_indices=padded, unpadded_first_target_indices=test_starts)
    scaler = dataset.scaler
    return {"raw_files_sha256": raw_hashes, "adjacency_sha256": hashlib.sha256(dataset.adj_mx.tobytes()).hexdigest(),
            "node_order": [str(node) for node in dataset.geo_ids],
            "timeline_length": length, "split_samples_unpadded": {"train": n_train, "validation": len(starts)-n_train-n_test, "test": n_test},
            "split_samples_padded": {"train": len(train_data.dataset), "validation": len(valid_data.dataset), "test": len(test_data.dataset)},
            "scaler": {"type": config.get("scaler"), "mean": float(getattr(scaler, "mean", 0)), "std": float(getattr(scaler, "std", 1)), "fit": "training input windows, LibCity default"},
            "libcity_commit": subprocess.check_output(["git", "-C", str(compat.LIBCITY_ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "split_policy": "LibCity windows then chronological 7:1:2; retain final-batch padding for reference compatibility"}


def run(dataset_name, targets, model_name, protocol=None, **kwargs):
    """Run once in a fresh directory; preserve failure status and existing results."""
    out_dir = paths.run_dir(dataset_name, targets, model_name, kwargs.get("seed", 0), kwargs.get("tag", ""))
    if out_dir.exists():
        raise FileExistsError(f"Run directory already exists; use a fresh --tag: {out_dir}")
    out_dir.mkdir(parents=True)
    status_file = out_dir / "status.json"
    status_file.write_text(json.dumps({"status": "running", "model": model_name}))
    try:
        result = _run(dataset_name, targets, model_name, protocol, **kwargs)
    except BaseException as error:
        status_file.write_text(json.dumps({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)}))
        raise
    status_file.write_text(json.dumps({"status": "completed", "epochs_completed": result["epochs_completed"]}))
    return result
