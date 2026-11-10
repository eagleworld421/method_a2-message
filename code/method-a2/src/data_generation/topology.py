"""拓扑工具：候选边、拓扑掩码和节点观测掩码。"""

from typing import Optional, Tuple

import numpy as np


def build_edge_universe(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    extra_edges: Optional[np.ndarray] = None,
    extra_edge_attr: Optional[np.ndarray] = None,
    extra_edge_type: Optional[np.ndarray] = None,
    extra_attr_source: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """构造包含真实边和候选虚增边的双向边全集。

    `extra_edges` 按无向边给出；每条候选边会展开为两个方向，并分配独立的
    `edge_pair_id`。因此即使端点与真实边相同，也能表示一条额外的平行边。
    真实边的数组顺序保持不变，旧模型可以继续使用其前缀。
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edge_index 必须为 [E,2]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[0]:
        raise ValueError("edge_attr 必须与 edge_index 行数一致")
    if len(edge_index) % 2:
        raise ValueError("双向边数组必须包含偶数行")
    n_true_pairs = len(edge_index) // 2
    if n_true_pairs and not np.array_equal(edge_index[1::2], edge_index[0::2, ::-1]):
        raise ValueError("真实边数组必须按每条无向边的两个方向交错排列")

    if extra_edges is None:
        extra_edges = np.empty((0, 2), dtype=np.int64)
    extra_edges = np.asarray(extra_edges, dtype=np.int64)
    if extra_edges.ndim != 2 or extra_edges.shape[1] != 2:
        raise ValueError("extra_edges 必须为 [K,2]")
    if np.any(extra_edges[:, 0] == extra_edges[:, 1]):
        raise ValueError("候选边不能是自环")
    n_extra = len(extra_edges)
    if extra_edge_attr is None:
        extra_edge_attr = np.ones((n_extra, edge_attr.shape[1]), dtype=np.float32)
    extra_edge_attr = np.asarray(extra_edge_attr, dtype=np.float32)
    if extra_edge_attr.shape != (n_extra, edge_attr.shape[1]):
        raise ValueError("extra_edge_attr 形状必须为 [K,F]")
    if extra_edge_type is None:
        extra_edge_type = np.full(n_extra, 3, dtype=np.int64)
    if extra_attr_source is None:
        extra_attr_source = np.full(n_extra, 2, dtype=np.int64)
    extra_edge_type = np.asarray(extra_edge_type, dtype=np.int64)
    extra_attr_source = np.asarray(extra_attr_source, dtype=np.int64)
    if extra_edge_type.shape != (n_extra,) or extra_attr_source.shape != (n_extra,):
        raise ValueError("候选边类型和属性来源必须为 [K]")

    directed_extra = np.stack([extra_edges, extra_edges[:, ::-1]], axis=1).reshape(-1, 2)
    attrs_extra = np.repeat(extra_edge_attr, 2, axis=0)
    index_universe = np.concatenate([edge_index, directed_extra], axis=0)
    attr_universe = np.concatenate([edge_attr, attrs_extra], axis=0)
    pair_id = np.concatenate([
        np.repeat(np.arange(n_true_pairs, dtype=np.int64), 2),
        np.repeat(np.arange(n_true_pairs, n_true_pairs + n_extra, dtype=np.int64), 2),
    ])
    direction = np.tile(np.asarray([0, 1], dtype=np.int64), n_true_pairs + n_extra)
    edge_type = np.concatenate([
        np.zeros(len(edge_index), dtype=np.int64),
        np.repeat(extra_edge_type, 2),
    ])
    attr_source = np.concatenate([
        np.zeros(len(edge_index), dtype=np.int64),
        np.repeat(extra_attr_source, 2),
    ])
    mask_true = np.concatenate([
        np.ones(len(edge_index), dtype=np.float32),
        np.zeros(len(directed_extra), dtype=np.float32),
    ])
    return {
        "edge_index_universe": index_universe,
        "edge_attr_universe": attr_universe,
        "edge_pair_id": pair_id,
        "edge_direction": direction,
        "edge_type": edge_type,
        "edge_attr_source": attr_source,
        "edge_mask_true": mask_true,
        "edge_mask_obs": mask_true.copy(),
    }


def expand_undirected_mask(
    pair_mask: np.ndarray, edge_pair_id: np.ndarray,
) -> np.ndarray:
    """按 `edge_pair_id` 将无向边掩码展开为有向边掩码。"""
    pair_mask = np.asarray(pair_mask)
    edge_pair_id = np.asarray(edge_pair_id, dtype=np.int64)
    if pair_mask.ndim not in (1, 2) or edge_pair_id.ndim != 1:
        raise ValueError("掩码必须为 [P] 或 [B,P]，edge_pair_id 必须为 [E]")
    n_pairs = int(edge_pair_id.max()) + 1 if edge_pair_id.size else 0
    if pair_mask.shape[-1] != n_pairs:
        raise ValueError("无向边掩码列数与 edge_pair_id 不匹配")
    return pair_mask[..., edge_pair_id]


def validate_edge_masks(
    edge_pair_id: np.ndarray,
    edge_mask_true: np.ndarray,
    edge_mask_obs: np.ndarray,
) -> None:
    """校验真实与观测边掩码的形状、取值和双向一致性。"""
    pair_id = np.asarray(edge_pair_id, dtype=np.int64)
    true = np.asarray(edge_mask_true)
    obs = np.asarray(edge_mask_obs)
    if pair_id.ndim != 1 or true.ndim != 2 or obs.shape != true.shape:
        raise ValueError("边掩码必须为 [B,E] 且形状一致")
    if true.shape[1] != len(pair_id):
        raise ValueError("边掩码列数与 edge_pair_id 不匹配")
    if not np.isin(true, [0, 1]).all() or not np.isin(obs, [0, 1]).all():
        raise ValueError("边掩码必须为二值")
    for pair in np.unique(pair_id):
        columns = np.flatnonzero(pair_id == pair)
        if not np.all(true[:, columns] == true[:, columns[0:1]]):
            raise ValueError("edge_mask_true 的双向边状态不一致")
        if not np.all(obs[:, columns] == obs[:, columns[0:1]]):
            raise ValueError("edge_mask_obs 的双向边状态不一致")


def build_candidate_edges(adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从无向邻接矩阵构造双向候选边及其基础属性。

    每条无向边 `(i, j)` 展开为 `(i, j)` 和 `(j, i)`，供消息传递分别聚合。
    """
    n = int(adj_matrix.shape[0])
    rows, cols = np.where(np.triu(adj_matrix, k=1) > 0)
    undirected = np.stack([rows, cols], axis=1).astype(np.int64)
    edge_index = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    length = np.ones(len(edge_index), dtype=np.float32)
    edge_attr = np.stack([
        np.ones(len(edge_index), dtype=np.float32),
        np.ones(len(edge_index), dtype=np.float32),
        length,
        np.ones(len(edge_index), dtype=np.float32),
        np.ones(len(edge_index), dtype=np.float32),
    ], axis=1)
    if n == 0 or len(undirected) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 5), dtype=np.float32)
    return edge_index, edge_attr


def inject_topology_error(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    n_nodes: int,
    error_rate: float,
    error_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成观测拓扑边掩码和真实拓扑边标签。"""
    del edge_attr, n_nodes
    n_edges = int(len(edge_index))
    y_edge = np.ones(n_edges, dtype=np.float32)
    edge_mask = y_edge.copy()
    if error_rate > 0 and n_edges > 0:
        n_err = min(n_edges, max(1, int(round(error_rate * n_edges))))
        chosen = rng.choice(n_edges, size=n_err, replace=False)
        if error_type == "missing":
            edge_mask[chosen] = 0.0
        elif error_type == "flip":
            edge_mask[chosen] = 1.0 - y_edge[chosen]
        else:
            raise ValueError(f"未知拓扑错误类型：{error_type}")
    elif error_type not in ("missing", "flip"):
        raise ValueError(f"未知拓扑错误类型：{error_type}")
    return edge_mask, y_edge


def build_observation_mask(
    n_nodes: int,
    scheme: str,
    rate: float,
    rng: np.random.Generator,
    key_nodes: Optional[list] = None,
) -> np.ndarray:
    """按 full、key 或 random 方案构造节点观测掩码。"""
    if scheme == "full":
        return np.ones(n_nodes, dtype=bool)
    if scheme == "key":
        selected = key_nodes or [0, n_nodes - 1, n_nodes // 2]
        mask = np.zeros(n_nodes, dtype=bool)
        mask[np.asarray(selected, dtype=np.int64)] = True
        return mask
    if scheme == "random":
        n_obs = max(1, min(n_nodes, int(round(rate * n_nodes))))
        idx = rng.choice(n_nodes, size=n_obs, replace=False)
        mask = np.zeros(n_nodes, dtype=bool)
        mask[idx] = True
        return mask
    raise ValueError(f"未知观测方案：{scheme}")
