"""阶段零：计算真实反事实响应的 Oracle 可达性指标。

本脚本只读取既有数据数组，不训练模型、不修改数据。它在两种距离口径下分别
计算故障候选集合上的 Oracle Top-1、Top-3、真实候选名次与候选间隔，用于判定
当前数据与新事件参数空间是否具有达到 Top-1 0.60、Top-3 0.85 的真实物理可分辨性。

口径说明：
- 原始口径 `d_raw`：对节点、时间、通道求未加权均方差，与当前 MVP 实现一致；
- 归一化口径 `d_S`：按训练集逐通道尺度 `s_f` 归一化后再求均方，对应核心公式第 4 节
  的归一化距离。当前数据只保存逐通道尺度 `feature_scaler.npz`，不保存逐节点尺度，
  因此本脚本实现的是按通道归一化的可计算形式，并在产物中显式声明该差异。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.dataset_builder import load_dataset  # noqa: E402


def _candidate_scores(
    observation: np.ndarray,
    responses: np.ndarray,
    scale: np.ndarray | None,
) -> np.ndarray:
    """按事件计算故障候选的响应距离，返回 `[M,C]`。

    observation：事实观测 `[M,N,T,F]`；
    responses：候选真实响应 `[M,C,N,T,F]`；
    scale：逐通道尺度 `[F]`；为 None 时使用未加权均方口径。
    """
    difference = responses - observation[:, None, :, :, :]
    if scale is not None:
        # 尺度向量只作用于最后一个通道轴，其余轴按观测维度广播
        shape = (1,) * (difference.ndim - 1) + (-1,)
        difference = difference / scale.reshape(shape)
    return difference.astype(np.float64).__pow__(2).mean(axis=(2, 3, 4))


def _rank_metrics(scores: np.ndarray, locations: np.ndarray) -> dict:
    """在故障候选上计算 Top-K 命中率与真实候选名次。"""
    if scores.shape[1] < 2:
        raise ValueError("候选数少于 2 时名次不可定义，请改用量级比较")
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = (order == locations[:, None]).nonzero()[1] + 1
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    true_scores = scores[np.arange(scores.shape[0]), locations]
    # 最难错误候选为排除真实候选后的最小评分
    hardest = np.where(order[:, 0] == locations, sorted_scores[:, 1], sorted_scores[:, 0])
    return {
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "ranks": ranks.tolist(),
        "true_scores": true_scores.tolist(),
        "hardest_scores": hardest.tolist(),
        "margins": (hardest - true_scores).tolist(),
        "n_events": int(scores.shape[0]),
        "n_candidates": int(scores.shape[1]),
    }


def _hop_distance(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """由边表计算节点之间的最短跳数距离矩阵。"""
    adjacency = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(adjacency, 0.0)
    for source, target in edge_index:
        adjacency[int(source), int(target)] = 1.0
    for middle in range(n_nodes):
        adjacency = np.minimum(
            adjacency, adjacency[:, middle, None] + adjacency[None, middle, :]
        )
    return adjacency


def _channel_strata(
    observation: np.ndarray,
    responses: np.ndarray,
    locations: np.ndarray,
    scale: np.ndarray | None,
) -> list[dict]:
    """按通道分组计算单通道口径下的 Oracle 指标。"""
    rows = []
    for channel in range(observation.shape[-1]):
        channel_scores = _candidate_scores(
            observation[..., channel:channel + 1],
            responses[..., channel:channel + 1],
            None if scale is None else scale[channel:channel + 1],
        )
        metrics = _rank_metrics(channel_scores, locations)
        rows.append({
            "channel": int(channel),
            "top1": metrics["top1"],
            "top3": metrics["top3"],
            "mean_rank": metrics["mean_rank"],
            "margin_median": float(np.median(metrics["margins"])),
        })
    return rows


def _node_strata(
    observation: np.ndarray,
    responses: np.ndarray,
    locations: np.ndarray,
    scale: np.ndarray | None,
) -> list[dict]:
    """按节点分组计算单节点口径下的 Oracle 指标。

    单节点口径在节点轴上只有一个位置，不能直接复用带候选广播的通用评分函数，
    因此此处按 `[M,C,T,F]` 的显式形状单独计算。
    """
    rows = []
    for node in range(observation.shape[1]):
        difference = responses[:, :, node, :, :] - observation[:, node, :, :][:, None, :, :]
        if scale is not None:
            difference = difference / scale.reshape(1, 1, 1, -1)
        node_scores = difference.astype(np.float64).__pow__(2).mean(axis=(2, 3))
        metrics = _rank_metrics(node_scores, locations)
        rows.append({
            "node": int(node),
            "top1": metrics["top1"],
            "top3": metrics["top3"],
            "mean_rank": metrics["mean_rank"],
            "margin_median": float(np.median(metrics["margins"])),
        })
    return rows


def _distance_strata(
    observation: np.ndarray,
    responses: np.ndarray,
    locations: np.ndarray,
    edge_index: np.ndarray,
    scale: np.ndarray | None,
) -> list[dict]:
    """按真实候选与各候选之间的拓扑跳数分层计算 Oracle 指标。"""
    hops = _hop_distance(edge_index, observation.shape[1])
    scores = _candidate_scores(observation, responses, scale)
    n_candidates = scores.shape[1]
    # 逐事件取真实候选到各候选的拓扑跳数，形状为 [M,C]
    event_hops = hops[locations][:, :n_candidates]
    rows = []
    for level in range(1, int(event_hops[np.isfinite(event_hops)].max()) + 1):
        # 逐事件只在跳数不小于该层的候选上比较；条件需广播到候选轴
        masked = np.where((event_hops >= level)[:, None, :], scores, np.inf)
        complete = np.isfinite(masked).all(axis=2)
        subset = masked[complete]
        if subset.shape[0] == 0:
            break
        subset_locations = locations[complete]
        order = np.argsort(subset, axis=1, kind="stable")
        ranks = (order == subset_locations[:, None]).nonzero()[1] + 1
        rows.append({
            "hop_level": int(level),
            "n_events_with_full_candidate_set": int(subset.shape[0]),
            "top1": float((ranks == 1).mean()),
            "top3": float((ranks <= 3).mean()),
            "mean_rank": float(ranks.mean()),
        })
    return rows


def run(args: argparse.Namespace) -> dict:
    """执行阶段零计算并写出 oracle_metrics.json。"""
    data = load_dataset(args.data_dir)
    observation = data["X_obs"]
    bank = data["signature_bank"]
    locations = data["y_loc"]
    test_idx = data["test_idx"]
    train_idx = data["train_idx"]
    meta = data["meta"]
    n_nodes = int(meta["n_nodes"])

    scaler = np.load(Path(args.data_dir) / meta.get("feature_scaler", "feature_scaler.npz"))
    scale = scaler["std"].astype(np.float64)
    scale = np.where(scale < 1e-8, 1.0, scale)

    # 只在故障候选（索引 0 至 N-1）与真实故障样本上计算
    fault_columns = slice(0, n_nodes)
    fault_rows = locations >= 0
    if args.split == "test":
        split_rows = fault_rows & np.isin(np.arange(locations.size), test_idx)
    elif args.split == "val":
        if data["val_idx"].size == 0:
            raise ValueError("数据目录不含 val_idx，无法按验证集切分")
        split_rows = fault_rows & np.isin(np.arange(locations.size), data["val_idx"])
    elif args.split == "train":
        split_rows = fault_rows & np.isin(np.arange(locations.size), train_idx)
    else:
        split_rows = fault_rows

    subset_observation = observation[split_rows]
    subset_responses = bank[split_rows][:, fault_columns]
    subset_locations = locations[split_rows]

    raw_metrics = _rank_metrics(
        _candidate_scores(subset_observation, subset_responses, None), subset_locations
    )
    normalized_metrics = _rank_metrics(
        _candidate_scores(subset_observation, subset_responses, scale), subset_locations
    )

    margins = np.asarray(normalized_metrics["margins"], dtype=np.float64)
    raw_margins = np.asarray(raw_metrics["margins"], dtype=np.float64)

    # 预测误差预算：可用候选响应重构误差。未显式给出时读取参考运行的测试族损失。
    budget = args.error_budget
    budget_note = "命令行显式给出" if budget is not None else ""
    if budget is None:
        metrics_path = Path(args.reference_metrics) if args.reference_metrics else None
        if metrics_path is not None and metrics_path.exists():
            reference = json.loads(metrics_path.read_text(encoding="utf-8"))
            budget = float(reference.get("test_family_loss", 0.0))
            budget_note = f"取自 {metrics_path} 的 test_family_loss"
        else:
            budget = None
            budget_note = "未提供预测误差预算，间隔比例不可计算"

    result = {
        "run_id": args.run_id,
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "n_events": int(subset_locations.size),
        "n_fault_candidates": int(n_nodes),
        "normal_candidate_included": False,
        "scale_definition": "训练集逐通道标准差（feature_scaler.npz 的 std）",
        "scale_note": "核心公式的 s_{n,f} 为逐节点逐通道尺度，当前数据只保存逐通道尺度，本计算为按通道归一化的可计算形式",
        "raw": {key: value for key, value in raw_metrics.items() if key not in {"ranks", "true_scores", "hardest_scores", "margins"}},
        "normalized": {key: value for key, value in normalized_metrics.items() if key not in {"ranks", "true_scores", "hardest_scores", "margins"}},
        "raw_margin_median": float(np.median(raw_margins)),
        "normalized_margin_median": float(np.median(margins)),
        "normalized_margin_min": float(margins.min()),
        "normalized_margin_p05": float(np.percentile(margins, 5)),
        "error_budget": budget,
        "error_budget_source": args.error_budget_source,
        "normalized_margin_below_budget_ratio": float((margins < budget).mean()),
        "raw_margin_below_budget_ratio": float((raw_margins < budget).mean()),
        "channel_strata": _channel_strata(
            subset_observation, subset_responses, subset_locations, scale
        ),
        "node_strata": _node_strata(
            subset_observation, subset_responses, subset_locations, scale
        ),
        "topology_strata": _distance_strata(
            subset_observation, subset_responses, subset_locations,
            data["edge_index"], scale,
        ),
        "gate": {
            "oracle_top3_target": 0.85,
            "oracle_top1_target": 0.60,
            "oracle_top3_pass": bool(normalized_metrics["top3"] >= 0.85),
            "oracle_top1_pass": bool(normalized_metrics["top1"] >= 0.60),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "oracle_per_event.npz",
        event_index=np.flatnonzero(split_rows),
        location=subset_locations,
        raw_margin=np.asarray(raw_metrics["margins"]),
        normalized_margin=margins,
        raw_rank=np.asarray(raw_metrics["ranks"]),
        normalized_rank=np.asarray(normalized_metrics["ranks"]),
    )
    return result


def parse_args() -> argparse.Namespace:
    """解析阶段零参数。"""
    parser = argparse.ArgumentParser(description="计算 Method-A2 阶段零 Oracle 可达性指标")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--run-id", default="oracle-run1")
    parser.add_argument("--error-budget", type=float, default=None,
                        help="预测误差预算；缺省时读取 --reference-metrics 中的 test_family_loss")
    parser.add_argument("--error-budget-source", default="train-s0-marker-run1 的 test_family_loss")
    parser.add_argument("--reference-metrics", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outcome = run(args)
    print(json.dumps({
        "run_id": outcome["run_id"],
        "split": outcome["split"],
        "n_events": outcome["n_events"],
        "raw_top1": outcome["raw"]["top1"],
        "raw_top3": outcome["raw"]["top3"],
        "normalized_top1": outcome["normalized"]["top1"],
        "normalized_top3": outcome["normalized"]["top3"],
        "normalized_margin_median": outcome["normalized_margin_median"],
        "error_budget": outcome["error_budget"],
        "margin_below_budget_ratio": outcome["normalized_margin_below_budget_ratio"],
        "gate": outcome["gate"],
    }, ensure_ascii=False, indent=2))
