"""训练并评估一个模型。

产物写入 results/forecast/<数据集>/<目标>/<模型>/seed<N>/：run.json 是本次运行的完整
记录，libcity/ 下是 LibCity 的评估 CSV、预测 npz、日志与 TensorBoard 事件。
"""

from __future__ import annotations

import json
import time
import zlib

from . import compat, datasets, paths
from . import metrics as metrics_module
from .models import registry
from .protocols import Protocol, default_protocol


def run(
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

    other_args = datasets.build_other_args(dataset, protocol, exp_id=exp_id, seed=seed, gpu=gpu)
    if max_epoch is not None:
        other_args["max_epoch"] = max_epoch
    if batch_size is not None:
        other_args["batch_size"] = batch_size
    merged_extra = dict(spec.default_args)
    if extra_args:
        merged_extra.update(extra_args)
    other_args.update(merged_extra)

    with compat.libcity_cwd():
        from libcity.config import ConfigParser
        from libcity.data import get_dataset
        from libcity.utils import get_logger, set_random_seed

        # LibCity 会校验模型名是否在 task_config.json 的白名单内。自研模型借一个内置名字
        # 过校验（其 dataset_class/executor/evaluator 已在 default_args 中显式给出），
        # 构造完再改回真名，使缓存文件与日志使用真名。
        config_model = "FNN" if spec.source == "ours" else model_name
        config = ConfigParser("traffic_state_pred", config_model, dataset_name, other_args=other_args)
        if config_model != model_name:
            config["model"] = model_name
        # 定种子并起日志。LibCity 的官方流程会做这两步，自己驱动时容易漏：漏了种子就
        # 不可复现，漏了 logger 则逐 epoch 进度与日志文件都不会产生。
        set_random_seed(config.get("seed", 0))
        get_logger(config).info("start %s / %s / seed %s", model_name, dataset_name, seed)

        libcity_dataset = get_dataset(config)
        train_data, valid_data, test_data = libcity_dataset.get_data()
        data_feature = libcity_dataset.get_data_feature()

        problems = datasets.check_data_feature(dataset, data_feature, len(targets))
        if problems:
            raise RuntimeError("数据特征与注册元数据不一致：" + "；".join(problems))

        device = config.get("device")
        model = registry.build(model_name, config, data_feature).to(device)
        executor_class = compat.load_executor_class(config.get("executor", "TrafficStateExecutor"))
        executor = executor_class(config, model, data_feature)

        started = time.time()
        executor.train(train_data, valid_data)
        train_seconds = time.time() - started
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
    }

    out_dir = paths.run_dir(dataset_name, targets, model_name, seed, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record["artifacts"] = len(compat.collect_artifacts(exp_id, out_dir / "libcity"))
    return record
