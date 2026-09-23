"""拓扑工具：候选边、拓扑掩码和节点观测掩码。"""

from typing import Optional, Tuple

import numpy as np


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
