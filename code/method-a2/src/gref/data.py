"""GReF 的只读数组加载、训练侧标准化与批次构建。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data_generation.topology import validate_edge_masks


def load_arrays(directory: Path) -> dict:
    """以内存映射读取既有数据，不复制完整签名库或改写原数组。"""
    directory = Path(directory)
    names = ("X_obs", "signature_bank", "mask", "candidate_mask", "edge_index",
             "edge_attr", "edge_mask", "group_id", "y_loc", "y_resist")
    data = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
    optional_names = (
        "edge_index_universe", "edge_attr_universe", "edge_pair_id",
        "edge_direction", "edge_type", "edge_attr_source",
        "edge_mask_true", "edge_mask_obs",
    )
    for name in optional_names:
        path = directory / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    if "edge_index_universe" not in data:
        data["edge_index_universe"] = data["edge_index"]
        data["edge_attr_universe"] = data["edge_attr"]
        data["edge_pair_id"] = np.repeat(
            np.arange(len(data["edge_index"]) // 2, dtype=np.int64), 2
        )
        data["edge_direction"] = np.tile(
            np.asarray([0, 1], dtype=np.int64), len(data["edge_index"]) // 2
        )
        data["edge_type"] = np.zeros(len(data["edge_index"]), dtype=np.int64)
        data["edge_attr_source"] = np.zeros(len(data["edge_index"]), dtype=np.int64)
        data["edge_mask_true"] = data["edge_mask"]
        data["edge_mask_obs"] = data["edge_mask"]
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
    if data["edge_index_universe"].ndim != 2 or data["edge_index_universe"].shape[1] != 2:
        raise ValueError("候选全集边索引必须为 [E,2]")
    if data["edge_attr_universe"].shape[0] != data["edge_index_universe"].shape[0]:
        raise ValueError("候选全集边属性与边索引行数不匹配")
    if data["edge_pair_id"].shape != (data["edge_index_universe"].shape[0],):
        raise ValueError("edge_pair_id 形状不匹配")
    if data["edge_mask_true"].shape != data["edge_mask_obs"].shape:
        raise ValueError("真实与观测边掩码形状不一致")
    if data["edge_mask_true"].shape[0] != len(x) or data["edge_mask_true"].shape[1] != data["edge_index_universe"].shape[0]:
        raise ValueError("边掩码与样本或候选全集形状不匹配")
    validate_edge_masks(
        data["edge_pair_id"], data["edge_mask_true"], data["edge_mask_obs"]
    )
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


def make_batch(
    data: dict, indices: np.ndarray, normalization: dict, device: torch.device,
    graph_view: str = "true",
) -> dict:
    """在原存储坐标中应用训练侧仿射变换，然后构建设备张量。

    `true` 保持旧训练路径；`universe` 使用候选全集和逐样本观测边掩码。
    """
    if graph_view not in ("true", "universe"):
        raise ValueError("graph_view 必须为 true 或 universe")
    if graph_view == "true":
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        edge_mask = data["edge_mask"][indices]
    else:
        edge_index = data["edge_index_universe"]
        edge_attr = data["edge_attr_universe"]
        edge_mask = data["edge_mask_obs"][indices]
    mean, std = normalization["mean"][:, None, :], normalization["std"][:, None, :]
    observation = (data["X_obs"][indices] - mean) / std
    target = (data["signature_bank"][indices] - mean) / std
    arrays = {
        "observation": observation, "target": target,
        "node_mask": data["mask"][indices], "candidate_mask": data["candidate_mask"][indices],
        "edge_mask": edge_mask, "location": data["y_loc"][indices],
        "edge_index": np.array(edge_index), "edge_attr": np.array(edge_attr),
    }
    batch = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
    batch["observation"] = batch["observation"].float()
    batch["target"] = batch["target"].float()
    return batch


def cache_cuda_batches(data, indices, normalization, device, batch_size):
    """分块缓存已标准化的训练和验证张量，显存不足时由调用方沿用逐批加载。"""
    fields = ("observation", "target", "node_mask", "candidate_mask", "edge_mask", "location")
    n_rows = len(indices)
    bytes_per_row = 4 * (data["X_obs"][0].size + data["signature_bank"][0].size)
    bytes_per_row += sum(data[key][0].nbytes for key in ("mask", "candidate_mask", "edge_mask", "y_loc"))
    free_bytes, _ = torch.cuda.mem_get_info(device.index)
    if n_rows * bytes_per_row > min(free_bytes // 3, free_bytes - 2 * 1024**3):
        return None

    row_lookup = np.full(len(data["y_loc"]), -1, dtype=np.int64)
    row_lookup[indices] = np.arange(n_rows)
    cache = {}
    for start in range(0, n_rows, batch_size):
        batch = make_batch(data, indices[start:start + batch_size], normalization, device)
        if not cache:
            cache = {key: batch[key].new_empty((n_rows, *batch[key].shape[1:])) for key in fields}
            graph = {key: batch[key] for key in ("edge_index", "edge_attr")}
        for key in fields:
            cache[key][start:start + len(batch["location"])].copy_(batch[key])

    def load_batch(rows):
        """按原始数据索引从显存读取独立批次。"""
        slots = row_lookup[rows]
        if (slots < 0).any():
            raise ValueError("批次包含显存缓存之外的样本")
        index = torch.as_tensor(slots, device=device)
        return {**graph, **{key: tensor.index_select(0, index) for key, tensor in cache.items()}}

    return load_batch


def sample_node_masks(
    locations: np.ndarray, n_nodes: int, rng: np.random.Generator, *,
    full_probability: float = 0.25, min_observed: int = 15, max_observed: int = 37,
    force_observed_probability: float = 0.5,
) -> np.ndarray:
    """为训练批次生成节点观测掩码，返回 `[B,n_nodes]`。

    每个事件先以 `full_probability` 保留完整观测；其余事件随机抽取
    `[min_observed, max_observed]` 个节点，并对故障事件以
    `force_observed_probability` 强制包含或强制排除真实故障节点。这样一批数据
    同时覆盖完整观测、真实节点被观测、真实节点未被观测与随机部分观测。
    """
    locations = np.asarray(locations, dtype=np.int64)
    if not 0.0 <= full_probability <= 1.0:
        raise ValueError("full_probability 必须位于 [0,1]")
    if not 0.0 <= force_observed_probability <= 1.0:
        raise ValueError("force_observed_probability 必须位于 [0,1]")
    if n_nodes < 1 or not 1 <= min_observed <= max_observed <= n_nodes:
        raise ValueError("观测节点数区间必须满足 1<=min<=max<=n_nodes")
    masks = np.zeros((locations.size, n_nodes), dtype=np.float32)
    all_nodes = np.arange(n_nodes)
    for row, location in enumerate(locations):
        if rng.random() < full_probability:
            masks[row] = 1.0
            continue
        count = int(rng.integers(min_observed, max_observed + 1))
        if count >= n_nodes:
            masks[row] = 1.0
            continue
        if location >= 0 and rng.random() < force_observed_probability:
            others = all_nodes[all_nodes != int(location)]
            selected = np.concatenate([[int(location)], rng.choice(others, size=count - 1, replace=False)])
        else:
            pool = all_nodes[all_nodes != int(location)] if location >= 0 else all_nodes
            selected = rng.choice(pool, size=count, replace=False)
        masks[row, selected] = 1.0
    return masks


def build_undirected_edge_groups(edge_index: np.ndarray) -> list[np.ndarray]:
    """把双向边按无向边分组，返回每个无向边对应的全部行索引。

    当前数据把每条无向边展开为两个有向边。训练期边删除必须成对处理，
    否则会得到只有单方向消息的非物理图。本函数按端点排序后的二元组分组，
    并保持组顺序确定，便于复现。
    """
    grouped: dict[tuple[int, int], list[int]] = {}
    for row, (source, target) in enumerate(np.asarray(edge_index, dtype=np.int64)):
        key = (int(min(source, target)), int(max(source, target)))
        grouped.setdefault(key, []).append(int(row))
    return [np.asarray(grouped[key], dtype=np.int64) for key in sorted(grouped)]


def apply_edge_drop(
    batch: dict,
    edge_groups: list[np.ndarray],
    drop_prob: float,
    max_drop: int,
    rng: np.random.Generator,
) -> dict:
    """对批次中的每个样本随机删除若干无向边，并原地修改 edge_mask。

    `drop_prob` 为单个样本启用删除的概率；启用后从 1 到 `max_drop` 中均匀抽取
    删除条数，成对清除该无向边对应的全部有向边。该函数只修改模型输入的掩码，
    不改变观测、目标或标签，因此物理响应仍来自真实拓扑。
    """
    if not 0.0 <= float(drop_prob) <= 1.0:
        raise ValueError("drop_prob 必须位于 [0, 1]")
    if int(max_drop) < 1:
        raise ValueError("max_drop 必须为正整数")
    mask = batch["edge_mask"]
    if mask.ndim != 2:
        raise ValueError("edge_mask 必须为 [B,E]")
    if not edge_groups:
        return batch
    # 在 CPU 上批量修改小掩码，避免逐条边触发 GPU 同步。
    mask_np = mask.detach().cpu().numpy().copy()
    for row in range(mask_np.shape[0]):
        if float(rng.random()) >= float(drop_prob):
            continue
        available = [group for group in edge_groups if mask_np[row, group[0]] > 0]
        if not available:
            continue
        count = min(int(rng.integers(1, int(max_drop) + 1)), len(available))
        chosen = rng.choice(len(available), size=count, replace=False)
        for index in np.atleast_1d(chosen):
            mask_np[row, available[int(index)]] = 0.0
    batch["edge_mask"] = torch.as_tensor(mask_np, device=mask.device, dtype=mask.dtype)
    return batch
