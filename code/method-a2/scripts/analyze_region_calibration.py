"""在已保存的验证集区域概率上计算置信度—覆盖率校准指标。

只读取运行目录中的 `validation_region_predictions.npz`（含单节点概率温度、区域概率、
条件掩码与真实位置），对 α∈{0.05,0.1,0.2} 重新构造最小区域，按完整观测、真实节点
被观测与未被观测分别报告经验覆盖率、区域大小与最差掩码。不重新前向、不训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.region import minimal_regions


def parse_args() -> argparse.Namespace:
    """解析校准分析参数。"""
    parser = argparse.ArgumentParser(description="计算区域概率的覆盖率—区域大小校准")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _stats(values: list[float]) -> dict:
    """给出均值、最小值、最大值与标准差。"""
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "min": float(array.min()),
            "max": float(array.max()), "std": float(array.std())}


def _subset_metrics(
    probability: np.ndarray, locations: np.ndarray,
    select: np.ndarray, alphas: list[float],
) -> dict:
    """在给定事件子集上计算覆盖率、区域大小与半径。"""
    result = {}
    for alpha in alphas:
        coverage, sizes = [], []
        for index in range(probability.shape[0]):
            membership, size = minimal_regions(probability[index][select], alpha)
            coverage.append(float(membership[np.arange(select.sum()), locations[select]].mean()))
            sizes.append(float(size.mean()))
        result[str(alpha)] = {"coverage": _stats(coverage),
                              "size": _stats(sizes),
                              "nominal": 1.0 - alpha,
                              "coverage_gap": _stats([value - (1.0 - alpha) for value in coverage])}
    return result


def main() -> int:
    """执行校准计算并写出 JSON。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    payload = np.load(run_dir / "validation_region_predictions.npz")
    probability = payload["region_probability"]
    masks = payload["condition_masks"]
    locations = payload["location"]
    rows = np.arange(locations.size)
    # 条件 0 为完整观测，其余为部分观测
    conditions = {"full": 0, "partial": list(range(1, probability.shape[0]))}
    observed = {
        "full": np.ones(locations.size, dtype=bool),
        "observed": masks[conditions["partial"][0]][rows, locations] > 0,
        "missing": masks[conditions["partial"][0]][rows, locations] == 0,
    }
    # 完整观测条件
    full_metrics = _subset_metrics(probability, locations,
                                   np.ones(locations.size, dtype=bool), args.alphas)
    partial_metrics = {}
    worst = {}
    for name in ("observed", "missing"):
        per_mask = {str(alpha): {"coverage": [], "size": []} for alpha in args.alphas}
        for index in conditions["partial"]:
            select = masks[index][rows, locations] > 0
            if name == "missing":
                select = ~select
            if not select.any():
                continue
            for alpha in args.alphas:
                membership, size = minimal_regions(probability[index][select], alpha)
                hit = membership[np.arange(select.sum()), locations[select]].mean()
                per_mask[str(alpha)]["coverage"].append(float(hit))
                per_mask[str(alpha)]["size"].append(float(size.mean()))
        metrics = {}
        for alpha in args.alphas:
            values = per_mask[str(alpha)]["coverage"]
            metrics[str(alpha)] = {
                "coverage": _stats(values), "size": _stats(per_mask[str(alpha)]["size"]),
                "nominal": 1.0 - alpha,
                "coverage_gap": _stats([v - (1.0 - alpha) for v in values]),
            }
        partial_metrics[name] = metrics
        worst[name] = sorted(
            [{"mask": index, "coverage": per_mask[str(args.alphas[0])]["coverage"][position]}
             for position, index in enumerate(conditions["partial"])
             if position < len(per_mask[str(args.alphas[0])]["coverage"])],
            key=lambda item: item["coverage"])[:5]
    result = {
        "run_dir": str(run_dir),
        "calibrated_temperature": float(payload["calibrated_temperature"][0]),
        "alphas": args.alphas,
        "conditions": {"full": full_metrics, "partial": partial_metrics},
        "worst_masks_missing_alpha_first": worst["missing"],
        "worst_masks_observed_alpha_first": worst["observed"],
        "note": "区域概率由验证集标定的单节点概率扩散得到；覆盖率是经验命中率。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
