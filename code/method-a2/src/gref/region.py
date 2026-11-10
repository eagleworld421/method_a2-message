"""RaReF 区域概率损失所需的拓扑距离、区域核、区域目标与区域指标工具。

本模块只提供可复用的确定性计算，不涉及模型结构：候选图上的无向跳数距离、
固定扩散核、由单节点概率到区域概率的变换、区域集合、区域目标分布以及区域
覆盖率、区域大小、拓扑半径和中心距离等指标。模型前向仍只输出候选响应。
"""

from __future__ import annotations

import numpy as np


def graph_distance_hops(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """按无向边计算候选拓扑的跳数距离矩阵，返回 `[n_nodes,n_nodes]` 整数矩阵。

    候选轴必须与拓扑节点身份一致：函数要求边端点严格位于 `[0,n_nodes)`，且
    图连通；任何越界或不可达都会直接报错，避免按数组位置臆测邻接关系。
    """
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_index 必须为 [E,2]")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index 必须为整数数组")
    if n_nodes < 1:
        raise ValueError("n_nodes 必须为正整数")
    if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("边端点越界，候选轴与拓扑节点身份不一致")
    adjacency: list[set[int]] = [set() for _ in range(n_nodes)]
    for source, target in edges:
        source, target = int(source), int(target)
        adjacency[source].add(target)
        adjacency[target].add(source)
    distance = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for start in range(n_nodes):
        distance[start, start] = 0
        frontier = [start]
        while frontier:
            following = []
            for node in frontier:
                for neighbour in adjacency[node]:
                    if distance[start, neighbour] < 0:
                        distance[start, neighbour] = distance[start, node] + 1
                        following.append(neighbour)
            frontier = following
    if (distance < 0).any():
        raise ValueError("拓扑不连通，无法构造区域核")
    if not np.array_equal(distance, distance.T):
        raise ValueError("跳数距离矩阵必须对称")
    return distance


def region_kernel(distance: np.ndarray, sigma: float) -> np.ndarray:
    """由跳数距离构造列随机区域核 `K[r,k]`，返回 `[n_nodes,n_nodes]`。

    每一列对应真实故障节点 `k`，在全部候选区域 `r` 上按
    `exp(-d(r,k)/sigma)` 归一化，因此 `sum_r K[r,k]=1`，用于把单节点概率扩散
    为区域概率，也用于构造软区域监督目标。
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma 必须为正有限数")
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance 必须为方阵")
    if (distance < 0).any() or (distance != distance.T).any():
        raise ValueError("distance 必须为非负对称矩阵")
    with np.errstate(divide="ignore"):
        logits = -distance / float(sigma)
    logits -= logits.max(axis=0, keepdims=True)
    kernel = np.exp(logits)
    kernel /= kernel.sum(axis=0, keepdims=True)
    if not np.isfinite(kernel).all():
        raise ValueError("区域核包含非有限值")
    if not np.allclose(kernel.sum(axis=0), 1.0, atol=1e-6):
        raise ValueError("区域核未按候选归一")
    return kernel.astype(np.float32)


def region_probabilities(probabilities: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """由单节点概率计算区域概率 `q_r = sum_k p_k K[r,k]`，返回与输入同形状。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if probabilities.shape[-1] != kernel.shape[0]:
        raise ValueError("单节点概率的候选轴与区域核不一致")
    if (probabilities < -1e-12).any() or (kernel < 0).any():
        raise ValueError("概率与区域核必须非负")
    # 核按 K[r,k] 存储，区域概率为 q_r = sum_k p_k K[r,k] = p @ K^T
    region = probabilities @ kernel.T
    return np.clip(region, 1e-12, None)


def minimal_regions(region: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """返回最小区域集合 `[E,N]` 布尔矩阵与逐事件区域大小。

    按区域概率降序累加，取首次达到 `1-alpha` 的前若干候选；概率并列时按候选
    索引稳定排序。区域概率必须先归一。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0,1)")
    region = np.asarray(region, dtype=np.float64)
    if region.ndim != 2:
        raise ValueError("region 必须为 [E,N]")
    if not np.allclose(region.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("区域概率未归一")
    order = np.argsort(-region, axis=1, kind="stable")
    ordered = np.take_along_axis(region, order, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    # 浮点累计可能略小于阈值，扣除容差避免把恰好达到 1-alpha 的区域多算一个节点
    size = (cumulative < (1.0 - alpha) - 1e-12).sum(axis=1) + 1
    size = np.minimum(size, region.shape[1])
    membership = np.zeros(region.shape, dtype=bool)
    rows = np.repeat(np.arange(region.shape[0]), size)
    columns = np.concatenate([order[row, :size[row]] for row in range(region.shape[0])])
    membership[rows, columns] = True
    return membership, size.astype(np.int64)


def region_target(
    kernel: np.ndarray, location: int, observed: bool,
    beta_obs: float, beta_miss: float,
) -> np.ndarray:
    """构造单个故障事件的区域监督目标 `y=(1-beta)*onehot+beta*K(:,k*)`。"""
    kernel = np.asarray(kernel, dtype=np.float64)
    if not 0 <= int(location) < kernel.shape[0]:
        raise ValueError("真实故障节点越界")
    for name, value in (("beta_obs", beta_obs), ("beta_miss", beta_miss)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须位于 [0,1]")
    beta = float(beta_obs) if observed else float(beta_miss)
    target = beta * kernel[:, int(location)].astype(np.float64)
    target[int(location)] += 1.0 - beta
    if not np.isclose(target.sum(), 1.0, atol=1e-9):
        raise ValueError("区域监督目标未归一")
    return target


def probability_statistics(probabilities: np.ndarray) -> dict:
    """计算单节点概率的熵（nats）、最大概率与首位概率间隔。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities 必须为 [E,N]")
    if (probabilities < -1e-12).any():
        raise ValueError("概率必须非负")
    clipped = np.clip(probabilities, 1e-12, None)
    entropy = -(np.where(probabilities > 0, probabilities * np.log(clipped), 0.0)).sum(axis=1)
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    maximum = ordered[:, 0]
    gap = ordered[:, 0] - ordered[:, 1] if ordered.shape[1] > 1 else np.zeros_like(maximum)
    return {"entropy": entropy, "max_probability": maximum, "top_gap": gap}


def region_event_metrics(
    region: np.ndarray, locations: np.ndarray, distance: np.ndarray, alpha: float,
) -> dict:
    """逐事件计算区域覆盖率、区域大小、拓扑半径与中心距离。

    区域中心取区域概率最大的候选；拓扑半径为区域内距离中心的最远跳数；
    中心距离为真实故障节点到区域中心的跳数。区域概率未覆盖真实节点时仍按
    预测区域计算大小与半径，覆盖率单独统计。
    """
    membership, size = minimal_regions(region, alpha)
    locations = np.asarray(locations, dtype=np.int64)
    if membership.shape[0] != locations.size:
        raise ValueError("区域集合与真实位置事件数不一致")
    if (locations < 0).any() or (locations >= membership.shape[1]).any():
        raise ValueError("真实位置必须为有效故障候选")
    distance = np.asarray(distance, dtype=np.int64)
    center = np.argmax(region, axis=1)
    center_distance = distance[center, locations]
    to_center = distance[center]
    radius = np.where(membership, to_center, -1).max(axis=1)
    coverage = membership[np.arange(locations.size), locations]
    mass = np.where(membership, region, 0.0).sum(axis=1)
    return {
        "coverage": float(coverage.mean()),
        "size": size,
        "radius": radius.astype(np.int64),
        "center_distance": center_distance.astype(np.int64),
        "center": center.astype(np.int64),
        "mass": mass,
    }


def calibrate_temperature(
    energies: np.ndarray, fault_mask: np.ndarray, locations: np.ndarray,
    grid: int = 401, low: float = 1e-6, high: float = 1e2,
) -> dict:
    """在给定事件上按真实候选负对数似然最小化标定单节点概率温度。

    只使用传入的事件（应来自验证集），在 `logspace(low, high, grid)` 上搜索，
    返回温度、事件数与平均负对数似然；温度按现有排序温度同尺度使用。
    """
    energies = np.asarray(energies, dtype=np.float64)
    fault = np.asarray(fault_mask, dtype=bool)
    locations = np.asarray(locations, dtype=np.int64)
    rows = (locations >= 0) & fault.any(axis=1)
    rows &= fault[np.arange(locations.size), np.clip(locations, 0, fault.shape[1] - 1)]
    if not rows.any():
        raise ValueError("没有可用于标定温度的故障事件")
    selected = energies[rows]
    selected_mask = fault[rows]
    labels = locations[rows]
    logits = np.where(selected_mask, -selected, -np.inf)
    candidates = np.logspace(np.log10(low), np.log10(high), grid)
    best = None
    for temperature in candidates:
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        log_probability = scaled - log_normalizer
        nll = float(-log_probability[np.arange(labels.size), labels].mean())
        if best is None or nll < best[0]:
            best = (nll, float(temperature))
    return {
        "temperature": best[1], "negative_log_likelihood": best[0],
        "n_events": int(labels.size), "grid": int(grid),
        "range": [float(low), float(high)],
        "source": "验证集完整观测，单节点概率真实候选负对数似然最小",
    }
