"""计算达到 Top-1 目标所需的响应预测精度，用于判定误差预算是否可行。

对验证集与测试集分别计算：真实候选与最难错误候选之间的候选间隔（Oracle 口径），
以及当前模型的部署评分误差。两者之比即模型需要提升的精度倍数。
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """解析参数。"""
    parser = argparse.ArgumentParser(description="计算所需预测精度倍数")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    return parser.parse_args()


def main() -> int:
    """打印候选间隔、预测误差与精度倍数。"""
    args = parse_args()
    payload = np.load(args.run_dir / "predictions.npz")
    response = payload["response"]
    observation = payload["observation"]
    target = payload["target"]
    locations = payload["location"]
    n_nodes = observation.shape[1]

    fault_response = response[:, :n_nodes]
    fault_target = target[:, :n_nodes]
    difference = fault_response - observation[:, None]
    scores = (difference.astype(np.float64) ** 2).mean(axis=(2, 3, 4))
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = (order == locations[:, None]).nonzero()[1] + 1
    true_scores = scores[np.arange(scores.shape[0]), locations]

    # Oracle 口径：观测与真实候选响应的距离，逐通道未加权
    oracle_difference = fault_target - observation[:, None]
    oracle_scores = (oracle_difference.astype(np.float64) ** 2).mean(axis=(2, 3, 4))

    margins = []
    for index in range(scores.shape[0]):
        hardest = [k for k in np.argsort(oracle_scores[index]) if k != locations[index]][0]
        margins.append(float(oracle_scores[index, hardest] - oracle_scores[index, locations[index]]))
    margins = np.asarray(margins, dtype=np.float64)

    print(json.dumps({
        "split": args.split,
        "n_events": int(scores.shape[0]),
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "oracle_top1_unweighted": float(
            (np.argsort(oracle_scores, axis=1, kind="stable")[:, 0] == locations).mean()
        ),
        "margin_median": float(np.median(margins)),
        "margin_p05": float(np.percentile(margins, 5)),
        "true_candidate_score_median": float(np.median(true_scores)),
        "prediction_error_median": float(np.median(np.abs(true_scores))),
        "required_precision_factor_median": float(
            np.median(np.abs(true_scores) / np.maximum(margins, 1e-30))
        ),
        "events_with_margin_below_model_error": float(
            (margins < np.median(np.abs(true_scores))).mean()
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
