"""自研模型：STGCN 的时空块 + 每目标一个输出头，直接预测全部时间步。

改动 1 `direct_multi_step`：一次前向出全部时间步，替掉 STGCN 的单步 + 自回归滚动。
改动 2 各目标一个输出头：由 output_dim 决定，单目标时退化为普通单头模型。
改动 3 `scale_aware_loss`：各目标的损失先按自身尺度归一化再平均。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from .. import compat

# 直接复用 STGCN 的时空块，主干与基线逐行一致，增益只能来自本文件明说的那几处改动。
_stgcn = compat.load_model_module("STGCN")


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行归一化；度数为 0 的行保持全 0。"""
    degree = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(degree > 0, degree, 1.0)


class DirectedSpatialConv(nn.Module):
    """有向双路传播：一路顺道路方向（上游→下游），一路逆向，各自归一化与权重。

    邻接沿用 LibCity 的约定 A[i, j]=1 表示 i → j（i 在上游）。
    STGCN 把邻接对称化、MTGNN 直接学一张新图，都把数据集原生的有向路网丢掉了。
    """

    def __init__(self, c_in: int, c_out: int, adj: np.ndarray, device):
        super().__init__()
        self.register_buffer("along", torch.FloatTensor(_row_normalize(adj.T)).to(device))
        self.register_buffer("against", torch.FloatTensor(_row_normalize(adj)).to(device))
        self.w_along = nn.Linear(c_in, c_out)
        self.w_against = nn.Linear(c_in, c_out)
        self.bias = nn.Parameter(torch.zeros(1, c_out, 1, 1))
        self.align = _stgcn.Align(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C_in, T, N)
        along = torch.einsum("nm,bctm->bctn", self.along, x)
        against = torch.einsum("nm,bctm->bctn", self.against, x)
        h = self.w_along(along.permute(0, 2, 3, 1)) + self.w_against(against.permute(0, 2, 3, 1))
        h = h.permute(0, 3, 1, 2) + self.bias
        return torch.relu(h + self.align(x))


class DirectedSTConvBlock(nn.Module):
    """与 STGCN 的 STConvBlock 同构，只把空间算子换成有向双路传播。"""

    def __init__(self, kt: int, n: int, c: list, p: float, adj: np.ndarray, device):
        super().__init__()
        self.tconv1 = _stgcn.TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = DirectedSpatialConv(c[1], c[1], adj, device)
        self.tconv2 = _stgcn.TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t1 = self.tconv1(x)
        x_t2 = self.tconv2(self.sconv(x_t1))
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class MultiHeadSTGCN(AbstractTrafficStateModel):
    """STGCN 主干 + 每目标一个输出头。"""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.output_dim = data_feature.get("output_dim", 1)
        self._scaler = data_feature.get("scaler")
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get("output_window", 12)
        self.device = config.get("device", torch.device("cpu"))

        self.Ks = config.get("Ks", 3)
        self.Kt = config.get("Kt", 3)
        self.blocks = [list(block) for block in config.get("blocks", [[1, 32, 64], [64, 32, 128]])]
        self.drop_prob = config.get("dropout", 0)

        self.direct_multi_step = config.get("direct_multi_step", True)
        self.scale_aware_loss = config.get("scale_aware_loss", True)

        self.blocks[0][0] = self.feature_dim
        # 每个时空块含两层时间卷积，各缩短 Kt-1 步，与 STGCN 的约束相同。
        self.trunk_steps = self.input_window - len(self.blocks) * 2 * (self.Kt - 1)
        if self.trunk_steps <= 0:
            raise ValueError(
                f"input_window={self.input_window} 太小，时空块要求大于 "
                f"{len(self.blocks) * 2 * (self.Kt - 1)}"
            )

        # spatial_mode：
        #   "cheb"     沿用 STGCN 的 Chebyshev 谱卷积。它要求拉普拉斯对称，因此要配
        #              bidir_adj_mx=true；若拿到有向矩阵，特征值含复部、lambda_max 失去意义；
        #   "directed" 改用有向双路传播，把数据集原生的 277 条有向边真正用起来
        #              （需配 bidir_adj_mx=false）。
        self.spatial_mode = config.get("spatial_mode", "cheb")
        if self.spatial_mode == "cheb":
            laplacian = _stgcn.calculate_scaled_laplacian(data_feature["adj_mx"])
            self.Lk = torch.FloatTensor(_stgcn.calculate_cheb_poly(laplacian, self.Ks)).to(self.device)
            self.st_conv = nn.ModuleList(
                [
                    _stgcn.STConvBlock(
                        self.Ks, self.Kt, self.num_nodes, block, self.drop_prob, self.Lk, self.device
                    )
                    for block in self.blocks
                ]
            )
        else:
            adjacency = np.asarray(data_feature["adj_mx"], dtype=np.float64) + np.eye(self.num_nodes)
            self.st_conv = nn.ModuleList(
                [
                    DirectedSTConvBlock(
                        self.Kt, self.num_nodes, block, self.drop_prob, adjacency, self.device
                    )
                    for block in self.blocks
                ]
            )

        channels = self.blocks[-1][2]
        steps = self.output_window if self.direct_multi_step else 1
        # head_mode：
        #   "collapse" 沿用 STGCN 的输出层，先把时间轴压成一步再预测。那是为单步预测设计的，
        #              用在直接多步上等于先丢掉时间结构；
        #   "flatten"  保留时间轴直接投影到全部预测步，与 MTGNN / Graph WaveNet 一致。
        self.head_mode = config.get("head_mode", "flatten")
        if self.head_mode == "collapse":
            self.time_collapse = _stgcn.TemporalConvLayer(self.trunk_steps, channels, channels, "GLU")
            self.ln = nn.LayerNorm([self.num_nodes, channels])
            flat_dim = channels
        else:
            self.ln = nn.LayerNorm([self.trunk_steps, channels])
            flat_dim = channels * self.trunk_steps
        self.heads = nn.ModuleList([nn.Linear(flat_dim, steps) for _ in range(self.output_dim)])

    def forward(self, batch) -> torch.Tensor:
        x = batch["X"].permute(0, 3, 1, 2)  # (B, T_in, N, F) -> (B, F, T_in, N)
        for block in self.st_conv:
            x = block(x)
        if self.head_mode == "collapse":
            x = self.time_collapse(x)  # (B, C, 1, N)
            x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x.squeeze(2).permute(0, 2, 1)  # (B, N, C)
        else:
            x = self.ln(x.permute(0, 3, 2, 1))  # (B, N, T, C)
            x = x.flatten(2)  # (B, N, T*C)
        stacked = torch.stack([head(x) for head in self.heads], dim=-1)  # (B, N, steps, out_dim)
        return stacked.permute(0, 2, 1, 3)  # (B, steps, N, out_dim)

    def predict(self, batch) -> torch.Tensor:
        if self.direct_multi_step:
            return self.forward(batch)
        # 自回归滚动，与 STGCN 的 predict() 同构
        x, y = batch["X"].clone(), batch["y"]
        preds = []
        for i in range(self.output_window):
            step = self.forward({"X": x})
            preds.append(step.clone())
            if step.shape[-1] < x.shape[-1]:
                step = torch.cat([step, y[:, i : i + 1, :, self.output_dim :]], dim=3)
            x = torch.cat([x[:, 1:, :, :], step], dim=1)
        return torch.cat(preds, dim=1)

    def calculate_loss(self, batch) -> torch.Tensor:
        if self.direct_multi_step:
            y_true, y_pred = batch["y"], self.forward(batch)
        else:
            # 与 STGCN 一致：训练只看第一个预测步，其余靠评估时滚动
            y_true, y_pred = batch["y"][:, 0:1, :, :], self.forward(batch)
        y_true = self._scaler.inverse_transform(y_true[..., : self.output_dim])
        y_pred = self._scaler.inverse_transform(y_pred[..., : self.output_dim])
        per_target = []
        for i in range(self.output_dim):
            value = loss.masked_mae_torch(y_pred[..., i], y_true[..., i])
            if self.scale_aware_loss:
                # 按该目标自身的平均量级归一，使三个目标对梯度的贡献相当。
                value = value / y_true[..., i].abs().mean().clamp_min(1e-6)
            per_target.append(value)
        return sum(per_target) / len(per_target)
