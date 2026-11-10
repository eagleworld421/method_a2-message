"""汇总RaReF 区域概率损失多种子配对复核的验证集定位与区域指标。

读取三种配置在多个种子上的验证集指标与验证集区域指标，按同种子配对计算配置间
差值，并给出种子间均值、标准差与逐级均值。只使用验证集，不读取测试集产物。
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

LEVELS = ("35", "30", "25", "20", "15")
CONFIG_TEMPLATES = {
    "tau1x": "three-term-ieee37-a-tau1x-seed{seed}",
    "phase1": "region-term-ieee37-a-seed{seed}",
    "phase2": "region-sweep-e1-m30-seed{seed}",
}
PAIRS = (("phase2", "phase1"), ("phase2", "tau1x"), ("phase1", "tau1x"))


def parse_args() -> argparse.Namespace:
    """解析汇总参数。"""
    parser = argparse.ArgumentParser(description="汇总RaReF 区域概率损失多种子配对复核")
    parser.add_argument("--root", type=Path, default=Path("code/method-a2/output/gref"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[342, 343, 344])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """返回算术平均。"""
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    """返回样本标准差（ddof=0）。"""
    return float(np.std(values))


def _extract(run_dir: Path) -> dict:
    """读取单个运行的验证定位与区域指标。"""
    validation = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    region = json.loads((run_dir / "validation_region_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    partial = region["partial"]
    missing = [partial[level]["conditional"]["missing"] for level in LEVELS]
    observed = [partial[level]["conditional"]["observed"] for level in LEVELS]
    return {
        "run_id": run_dir.name,
        "best_epoch": config.get("best_epoch"),
        "full_top1": region["full"]["rank"]["top1"],
        "full_top3": region["full"]["rank"]["top3"],
        "full_mean_rank": region["full"]["rank"]["mean_rank"],
        "partial_top1_mean": _mean([partial[level]["rank"]["top1"]["mean"] for level in LEVELS]),
        "observed_top1_mean": _mean([item["top1"]["mean"] for item in observed]),
        "observed_top1_by_level": [item["top1"]["mean"] for item in observed],
        "missing_top1_mean": _mean([item["top1"]["mean"] for item in missing]),
        "missing_top1_by_level": [item["top1"]["mean"] for item in missing],
        "coverage01_mean": _mean([item["region_0.1"]["coverage"]["mean"] for item in missing]),
        "coverage01_by_level": [item["region_0.1"]["coverage"]["mean"] for item in missing],
        "coverage02_mean": _mean([item["region_0.2"]["coverage"]["mean"] for item in missing]),
        "size_mean": _mean([partial[level]["region"]["0.1"]["size_mean"]["mean"] for level in LEVELS]),
        "size_p90_mean": _mean([partial[level]["region"]["0.1"]["size_p90"]["mean"] for level in LEVELS]),
        "radius_mean": _mean([partial[level]["region"]["0.1"]["radius_mean"]["mean"] for level in LEVELS]),
        "center_distance_mean": _mean(
            [partial[level]["region"]["0.1"]["center_distance_mean"]["mean"] for level in LEVELS]),
        "observed_coverage01_mean": _mean(
            [item["region_0.1"]["coverage"]["mean"] for item in observed]),
        "validation_losses": validation["losses"],
    }


def _aggregate(rows: list[dict], key: str) -> dict:
    """给出某个指标在种子间的均值、标准差与范围。"""
    values = [row[key] for row in rows]
    return {"mean": _mean(values), "std": _std(values),
            "min": float(np.min(values)), "max": float(np.max(values))}


def main() -> int:
    """执行汇总并写出 JSON。"""
    args = parse_args()
    root = args.root.resolve()
    runs: dict[dict[str, str]] = {}
    table: dict[str, dict[int, dict]] = {name: {} for name in CONFIG_TEMPLATES}
    for name, template in CONFIG_TEMPLATES.items():
        for seed in args.seeds:
            run_dir = root / template.format(seed=seed)
            if not (run_dir / "validation_region_metrics.json").exists():
                raise FileNotFoundError(f"缺少 {run_dir} 的验证区域指标")
            table[name][seed] = _extract(run_dir)
    key_metrics = ("full_top1", "full_top3", "partial_top1_mean", "observed_top1_mean",
                   "missing_top1_mean", "coverage01_mean", "coverage02_mean", "size_mean",
                   "size_p90_mean", "radius_mean", "center_distance_mean",
                   "observed_coverage01_mean")
    aggregates = {
        name: {key: _aggregate(list(table[name].values()), key) for key in key_metrics}
        for name in table
    }
    paired = {}
    for left, right in PAIRS:
        diffs = {}
        for key in key_metrics:
            values = [table[left][seed][key] - table[right][seed][key] for seed in args.seeds]
            diffs[key] = {"mean": _mean(values), "std": _std(values),
                          "by_seed": {str(seed): value for seed, value in zip(args.seeds, values)}}
        paired[f"{left}-{right}"] = diffs
    by_level = {
        name: {
            "missing_top1": {level: _mean([table[name][seed]["missing_top1_by_level"][index]
                                           for seed in args.seeds])
                             for index, level in enumerate(LEVELS)},
            "coverage01": {level: _mean([table[name][seed]["coverage01_by_level"][index]
                                        for seed in args.seeds])
                           for index, level in enumerate(LEVELS)},
            "observed_top1": {level: _mean([table[name][seed]["observed_top1_by_level"][index]
                                            for seed in args.seeds])
                              for index, level in enumerate(LEVELS)},
        } for name in table
    }
    result = {
        "root": str(root), "seeds": args.seeds,
        "config_templates": CONFIG_TEMPLATES,
        "note": "多种子复核只使用验证集；测试集仍保持每种配置仅在种子 342 上评价一次。",
        "runs": {name: {str(seed): row for seed, row in rows.items()} for name, rows in table.items()},
        "aggregates": aggregates,
        "paired_differences": paired,
        "by_level": by_level,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({
        "seeds": args.seeds,
        "aggregates": {name: {key: round(value["mean"], 4) for key, value in metrics.items()}
                       for name, metrics in aggregates.items()},
        "paired": {pair: {key: round(value["mean"], 4) for key, value in diffs.items()}
                   for pair, diffs in paired.items()},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
