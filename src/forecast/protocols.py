"""实验协议：一次基准实验的口径。

除学习率与种子数外全部对齐 LibCity 基准论文（arXiv:2308.12899）的实验设置：7:1:2 划分、
12 步预测 12 步、batch 64、验证集连续 30 轮不降则早停、梯度裁剪到 5。论文要求重复 5 次
取平均，种子数由命令行 --seeds 给出（当前只跑 seed 0）。学习率论文未报告，由各模型按各自
原论文的设置（论文原话：model parameters are set according to the settings in the
original paper），因此本协议不覆盖它。
"""

from __future__ import annotations

from dataclasses import dataclass

# 5 分钟采样下，15 / 30 / 60 分钟对应第 3 / 6 / 12 步。
HORIZON_STEPS = (3, 6, 12)
HORIZON_MINUTES = (15, 30, 60)


@dataclass(frozen=True)
class Protocol:
    """基准实验口径。"""

    targets: tuple[str, ...]
    train_rate: float = 0.7
    eval_rate: float = 0.1
    input_window: int = 12
    output_window: int = 12
    batch_size: int = 64
    # 运行预算，不是模型超参，各模型自带值不同（TGCN 是 5000）。
    max_epoch: int = 100
    use_early_stop: bool = True
    patience: int = 30
    clip_grad_norm: bool = True
    max_grad_norm: float = 5.0
    horizon_steps: tuple[int, ...] = HORIZON_STEPS
    # LibCity 的 masked_* 指标以 null_val=0 屏蔽零值，即文献口径。
    null_val: float = 0.0

    @property
    def label(self) -> str:
        """目标组合的显示名。"""
        return "+".join(self.targets)

    def horizon_minutes(self, interval_seconds: int) -> tuple[int, ...]:
        """把预测步数换算成分钟。"""
        return tuple(step * interval_seconds // 60 for step in self.horizon_steps)

    def as_other_args(self) -> dict:
        """转成传给 ConfigParser 的 other_args；这些键优先于数据包自带的 config.json。

        给的是论文的公共训练协议（划分、窗口、批次、早停、裁剪、预算）与目标选择。不覆盖
        学习率，论文要求各模型按各自原论文的设置，而各模型自带配置正是如此。同理不动
        scaler、load_external、add_time_in_day、add_day_in_week：各模型取值不同（如 STGCN
        不带时间特征、MTGNN 带、TGCN 用 normal 而非 standard），且时间特征只在
        load_external 为真时才会加进输入。
        """
        return {
            "train_rate": self.train_rate,
            "eval_rate": self.eval_rate,
            "input_window": self.input_window,
            "output_window": self.output_window,
            "batch_size": self.batch_size,
            "max_epoch": self.max_epoch,
            "use_early_stop": self.use_early_stop,
            "patience": self.patience,
            "clip_grad_norm": self.clip_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "data_col": list(self.targets),
            "output_dim": len(self.targets),
            # 缓存按 exp_id 落到 LibCity 的 cache 目录，撞号会静默读到别的数据集。
            "cache_dataset": False,
        }


def default_protocol(*targets: str) -> Protocol:
    """按目标列给出默认协议；传多个目标即多目标联合预测。"""
    return Protocol(targets=targets)
