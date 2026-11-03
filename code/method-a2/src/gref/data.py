"""GReF 的只读数组加载、训练侧标准化与批次构建。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_arrays(directory: Path) -> dict:
    """以内存映射读取既有数据，不复制完整签名库或改写原数组。"""
    directory = Path(directory)
    names = ("X_obs", "signature_bank", "mask", "candidate_mask", "edge_index",
             "edge_attr", "edge_mask", "group_id", "y_loc", "y_resist")
    data = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
    data["meta"] = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    x, bank = data["X_obs"], data["signature_bank"]
    if x.ndim != 4 or bank.shape != (x.shape[0], x.shape[1] + 1, *x.shape[1:]):
        raise ValueError("数据必须采用 N 个故障候选加末尾正常候选的响应轴")
    if data["candidate_mask"].shape != bank.shape[:2] or data["mask"].shape != x.shape[:2]:
        raise ValueError("数据候选或节点观测掩码形状不匹配")
    groups, labels = data["group_id"], data["y_loc"]
    if groups.shape != (len(x),) or labels.shape != groups.shape:
        raise ValueError("位置和分组必须为逐样本向量")
    if np.any((labels < -1) | (labels >= x.shape[1])):
        raise ValueError("故障标签越界")
    if np.any((labels >= 0) != (groups >= 0)) or np.any(groups < -1):
        raise ValueError("故障样本必须有非负分组，正常样本必须使用 -1 分组")
    for name in ("candidate_mask", "mask"):
        if not np.isin(data[name], [0, 1]).all():
            raise ValueError(f"{name} 必须为零一掩码")
    if (data["mask"].sum(axis=1) == 0).any():
        raise ValueError("数据含无有效观测的样本")
    return data


def fit_normalization(data: dict, train_indices: np.ndarray, progress=None) -> dict[str, np.ndarray]:
    """仅在最终训练索引上按节点、通道累计有效候选与时间的统计量。"""
    shape = (data["X_obs"].shape[1], data["X_obs"].shape[-1])
    count, total, squares = (np.zeros(shape, dtype=np.float64) for _ in range(3))
    for start in range(0, len(train_indices), 8):
        indices = train_indices[start:start + 8]
        bank = np.asarray(data["signature_bank"][indices], dtype=np.float64)
        weight = data["candidate_mask"][indices, :, None, None, None] * data["mask"][indices, None, :, None, None]
        weight = np.broadcast_to(weight, bank.shape)
        if not np.isfinite(bank[weight.astype(bool)]).all():
            raise ValueError("有效训练响应包含非有限值")
        values = np.where(weight.astype(bool), bank, 0.0)
        count += weight.sum(axis=(0, 1, 3))
        total += values.sum(axis=(0, 1, 3))
        squares += (values * values).sum(axis=(0, 1, 3))
        if progress is not None:
            progress.update("normalization", "rows", min(start + 8, len(train_indices)), len(train_indices))
    if (count == 0).any():
        raise ValueError("训练侧存在完全不可观测的节点或通道，首版不支持其外推")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "count": count,
            "train_indices": np.asarray(train_indices, dtype=np.int64)}


def make_batch(data: dict, indices: np.ndarray, normalization: dict, device: torch.device) -> dict:
    """在原存储坐标中应用训练侧仿射变换，然后构建设备张量。"""
    mean, std = normalization["mean"][:, None, :], normalization["std"][:, None, :]
    observation = (data["X_obs"][indices] - mean) / std
    target = (data["signature_bank"][indices] - mean) / std
    arrays = {
        "observation": observation, "target": target,
        "node_mask": data["mask"][indices], "candidate_mask": data["candidate_mask"][indices],
        "edge_mask": data["edge_mask"][indices], "location": data["y_loc"][indices],
        "edge_index": np.array(data["edge_index"]), "edge_attr": np.array(data["edge_attr"]),
    }
    batch = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
    batch["observation"] = batch["observation"].float()
    batch["target"] = batch["target"].float()
    return batch
