"""Node Marker MVP 的分组划分、评价指标与表示探针工具。

本模块把训练脚本中的评价逻辑集中在一处，供各阶段共用：

- `split_by_group`：按反事实族编号划分训练、验证与测试，保证同一分组不跨集合；
- `rank_metrics`：在故障候选上计算 Top-K、真实候选名次、signed margin、
  hardest-negative 命中率与候选间隔；
- `oracle_rank_metrics`：用真实反事实响应计算同一组指标，作为可达性上界；
- 岭回归探针：在冻结表示上估计故障位置可恢复性与事件参数可恢复性。
"""

from __future__ import annotations

import numpy as np


def split_by_group(
    group_id: np.ndarray,
    location: np.ndarray,
    seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """按分组编号划分训练、验证与测试索引，验证与测试只含故障样本。"""
    fault_groups = np.unique(group_id[group_id >= 0])
    if fault_groups.size < 3:
        raise ValueError("故障分组少于 3 个，无法同时给出训练、验证与测试集合")
    rng = np.random.default_rng(seed)
    shuffled = fault_groups.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(shuffled.size * val_ratio)))
    n_test = max(1, int(round(shuffled.size * test_ratio)))
    n_train = shuffled.size - n_val - n_test
    if n_train < 1:
        raise ValueError("训练分组不足，请减少验证或测试比例")
    train_groups = set(shuffled[:n_train].tolist())
    val_groups = set(shuffled[n_train:n_train + n_val].tolist())
    test_groups = set(shuffled[n_train + n_val:].tolist())
    rows = np.arange(group_id.size)
    train_idx = rows[np.isin(group_id, sorted(train_groups)) | (group_id == -1)]
    val_idx = rows[np.isin(group_id, sorted(val_groups)) & (location >= 0)]
    test_idx = rows[np.isin(group_id, sorted(test_groups)) & (location >= 0)]
    return {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "train_groups": np.asarray(sorted(train_groups), dtype=np.int64),
        "val_groups": np.asarray(sorted(val_groups), dtype=np.int64),
        "test_groups": np.asarray(sorted(test_groups), dtype=np.int64),
    }


def _response_distance(
    observation: np.ndarray,
    response: np.ndarray,
    variance: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """计算事实观测与候选响应之间的距离，返回 `[M,C]`。

    observation：`[M,N,T,F]`；response：`[M,C,N,T,F]`；
    variance：与 `response` 同形状的预测方差，给出时作为不确定性校正；
    scale：逐通道尺度 `[F]`，给出时按通道归一化。
    """
    difference = response - observation[:, None, :, :, :]
    if scale is not None:
        difference = difference / scale.reshape(1, 1, 1, 1, -1)
    distance = difference.astype(np.float64).__pow__(2).mean(axis=(2, 3, 4))
    if variance is not None:
        distance = distance / (variance.astype(np.float64).mean(axis=(2, 3, 4)) + epsilon)
    return distance


def rank_metrics(
    scores: np.ndarray,
    locations: np.ndarray,
    margins: np.ndarray | None = None,
) -> dict:
    """由候选择优评分计算定位指标。

    scores：`[M,C]`，评分越小越可能是真实故障位置；
    locations：`[M]` 真实故障位置，取值必须落在候选轴内；
    margins：`[M]` 真实物理候选间隔，给出时计算预测误差与间隔的比值。
    """
    if scores.ndim != 2 or scores.shape[0] != locations.size:
        raise ValueError("scores 必须为 [M,C] 且与 locations 同长")
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = (order == locations[:, None]).nonzero()[1] + 1
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    true_scores = scores[np.arange(scores.shape[0]), locations]
    hardest = np.where(order[:, 0] == locations, sorted_scores[:, 1], sorted_scores[:, 0])
    signed_margin = hardest - true_scores
    result = {
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "signed_margin_median": float(np.median(signed_margin)),
        "signed_margin_negative_ratio": float((signed_margin < 0).mean()),
        "n_events": int(scores.shape[0]),
        "n_candidates": int(scores.shape[1]),
    }
    if margins is not None:
        margins = np.asarray(margins, dtype=np.float64)
        error = np.abs(signed_margin)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(margins > 0, error / np.maximum(margins, 1e-30), np.inf)
        result["prediction_error_to_margin_ratio_median"] = float(np.median(ratio))
        result["margin_below_error_budget_ratio"] = float((margins < np.median(error)).mean())
    return result


def oracle_rank_metrics(
    observation: np.ndarray,
    target: np.ndarray,
    locations: np.ndarray,
    n_fault_candidates: int,
    scale: np.ndarray | None = None,
    error_budget: float | None = None,
) -> dict:
    """用真实反事实响应计算 Oracle 定位指标与候选间隔。"""
    scores = _response_distance(
        observation, target[:, :n_fault_candidates], scale=scale
    )
    valid = locations >= 0
    scores = scores[valid]
    subset_locations = locations[valid]
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = (order == subset_locations[:, None]).nonzero()[1] + 1
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    true_scores = scores[np.arange(scores.shape[0]), subset_locations]
    hardest = np.where(
        order[:, 0] == subset_locations, sorted_scores[:, 1], sorted_scores[:, 0]
    )
    margins = hardest - true_scores
    result = {
        "oracle_top1": float((ranks == 1).mean()),
        "oracle_top3": float((ranks <= 3).mean()),
        "oracle_top5": float((ranks <= 5).mean()),
        "oracle_mean_rank": float(ranks.mean()),
        "margin_median": float(np.median(margins)),
        "margin_p05": float(np.percentile(margins, 5)),
        "margin_min": float(margins.min()),
        "margin_exact_zero_count": int((margins == 0).sum()),
        "n_events": int(scores.shape[0]),
        "margins": margins,
    }
    if error_budget is not None:
        result["margin_below_budget_ratio"] = float((margins < float(error_budget)).mean())
        result["error_budget"] = float(error_budget)
    return result


def ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    task: str = "regression",
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> dict:
    """在冻结特征上拟合岭回归或岭惩罚多类逻辑回归并返回测试指标。

    features：`[M,D]` 特征；targets：`[M]` 或多维目标；
    train_mask、test_mask：布尔索引，必须按分组划分以避免泄漏；
    task 为 regression 时报告 R²，为 classification 时报告准确率。
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    x_train = features[train_mask].astype(np.float64)
    x_test = features[test_mask].astype(np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    # 划分内部再切出调参用的验证子集，避免用测试集选超参数
    n_train = x_train.shape[0]
    n_inner = max(1, int(round(n_train * 0.2)))
    inner_train = slice(0, n_train - n_inner)
    inner_val = slice(n_train - n_inner, n_train)

    if task == "regression":
        y_train = targets[train_mask].astype(np.float64)
        y_test = targets[test_mask].astype(np.float64)
        if y_train.ndim == 1:
            y_train = y_train[:, None]
            y_test = y_test[:, None]
        best = None
        for alpha in alphas:
            weights, intercept = _ridge_fit(x_train[inner_train], y_train[inner_train], alpha)
            prediction = x_train[inner_val] @ weights + intercept
            score = _r2(y_train[inner_val], prediction)
            if best is None or score > best[0]:
                best = (score, alpha)
        weights, intercept = _ridge_fit(x_train, y_train, best[1])
        prediction = x_test @ weights + intercept
        return {
            "task": task,
            "alpha": float(best[1]),
            "test_r2": _r2(y_test, prediction),
            "validation_r2": float(best[0]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
        }

    y_train = targets[train_mask].astype(np.int64)
    y_test = targets[test_mask].astype(np.int64)
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    best = None
    for alpha in alphas:
        weights, intercept = _logistic_fit(
            x_train[inner_train], y_train[inner_train], n_classes, alpha
        )
        score = float(
            (x_train[inner_val] @ weights + intercept).argmax(axis=1)
            .__eq__(y_train[inner_val]).mean()
        )
        if best is None or score > best[0]:
            best = (score, alpha)
    weights, intercept = _logistic_fit(x_train, y_train, n_classes, best[1])
    accuracy = float(
        (x_test @ weights + intercept).argmax(axis=1).__eq__(y_test).mean()
    )
    return {
        "task": task,
        "alpha": float(best[1]),
        "test_accuracy": accuracy,
        "validation_accuracy": float(best[0]),
        "chance_accuracy": float(1.0 / n_classes),
        "n_classes": int(n_classes),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """带截距的岭回归闭式解，返回系数 `[D,K]` 与截距 `[K]`。"""
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    regularizer = np.eye(design.shape[1]) * float(alpha)
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return solution[:-1], solution[-1]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算多输出平均 R²。"""
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    residual = ((y_true - y_pred) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(total > 0, 1.0 - residual / np.maximum(total, 1e-30), 0.0)
    return float(np.mean(values))


def _logistic_fit(
    x: np.ndarray, y: np.ndarray, n_classes: int, alpha: float, steps: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """用梯度下降拟合岭惩罚多类逻辑回归，返回权重 `[D,K]` 与偏置 `[K]`。"""
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    weights = np.zeros((design.shape[1], n_classes), dtype=np.float64)
    one_hot = np.zeros((y.size, n_classes), dtype=np.float64)
    one_hot[np.arange(y.size), y] = 1.0
    learning_rate = 0.5
    for _ in range(int(steps)):
        logits = design @ weights
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        gradient = design.T @ (probabilities - one_hot) / design.shape[0]
        gradient[:-1] += float(alpha) * weights[:-1]
        weights -= learning_rate * gradient
    return weights[:-1], weights[-1]
