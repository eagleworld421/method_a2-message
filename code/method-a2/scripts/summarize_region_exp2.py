"""汇总RaReF 区域概率损失未观测条件定向实验（beta_miss × sigma）并执行选模。

读取 6 组候选（beta_obs=0.05，beta_miss∈{0.2,0.3,0.4}，sigma∈{0.5,1.0}）与第二阶段
对照（beta_obs=0.1，beta_miss=0.4，sigma=1.0）的验证集区域指标，按补充实验计划
第 4.3 节的约束筛选，并在满足约束的候选中按未观测子集单节点 Top-1 选择。
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


def parse_args() -> argparse.Namespace:
    """解析汇总参数。"""
    parser = argparse.ArgumentParser(description="汇总RaReF 区域概率损失未观测定向实验")
    parser.add_argument("--root", type=Path, default=Path("code/method-a2/output/gref"))
    parser.add_argument("--control", default="region-sweep-e1-m30-seed342")
    parser.add_argument("--pattern", default="region-exp2-*-seed342")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-floor", type=float, default=0.95)
    parser.add_argument("--size-tolerance", type=float, default=0.5)
    parser.add_argument("--observed-tolerance", type=float, default=0.02)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """返回算术平均。"""
    return float(np.mean(values))


def _extract(run_dir: Path) -> dict:
    """读取单个运行的验证区域指标与配置。"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    region = json.loads((run_dir / "validation_region_metrics.json").read_text(encoding="utf-8"))
    partial = region["partial"]
    missing = [partial[level]["conditional"]["missing"] for level in LEVELS]
    observed = [partial[level]["conditional"]["observed"] for level in LEVELS]
    return {
        "run_id": run_dir.name,
        "beta_obs": config["region_kernel"]["beta_obs"],
        "beta_miss": config["region_kernel"]["beta_miss"],
        "sigma": config["region_kernel"]["sigma"],
        "best_epoch": config.get("best_epoch"),
        "full_top1": region["full"]["rank"]["top1"],
        "full_top3": region["full"]["rank"]["top3"],
        "observed_top1_mean": _mean([item["top1"]["mean"] for item in observed]),
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
        "validation_losses": validation["losses"],
    }


def main() -> int:
    """执行汇总、约束判定与选择，并写出 JSON。"""
    args = parse_args()
    root = args.root.resolve()
    control = _extract(root / args.control)
    runs = sorted(path for path in root.glob(args.pattern) if path.is_dir())
    if not runs:
        raise FileNotFoundError("未找到定向实验运行目录")
    rows = []
    for run_dir in runs:
        row = _extract(run_dir)
        row["coverage_ok"] = bool(row["coverage01_mean"] >= args.coverage_floor)
        row["size_ok"] = bool(row["size_mean"] <= control["size_mean"] + args.size_tolerance)
        row["observed_ok"] = bool(
            row["observed_top1_mean"] >= control["observed_top1_mean"] - args.observed_tolerance)
        row["constraints_passed"] = bool(
            row["coverage_ok"] and row["size_ok"] and row["observed_ok"])
        rows.append(row)
    eligible = [row for row in rows if row["constraints_passed"]]
    pool = eligible if eligible else rows
    selected = sorted(pool, key=lambda row: (-row["missing_top1_mean"], row["size_mean"]))[0]
    result = {
        "root": str(root), "pattern": args.pattern,
        "control": control,
        "constraints": {
            "coverage_floor": args.coverage_floor,
            "size_tolerance_nodes": args.size_tolerance,
            "observed_top1_tolerance": args.observed_tolerance,
            "definition": "未观测覆盖率 α=0.1 ≥ 0.95；平均区域不超过对照 0.5 个节点；"
                          "观测子集 Top-1 不低于对照 2 个百分点",
        },
        "n_runs": len(rows), "n_eligible": len(eligible),
        "selected_run_id": selected["run_id"],
        "selection_fallback": not bool(eligible),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({
        "control": {key: control[key] for key in
                    ("run_id", "coverage01_mean", "size_mean", "observed_top1_mean",
                     "missing_top1_mean", "full_top1")},
        "selected": selected["run_id"], "fallback": result["selection_fallback"],
        "table": [{key: row[key] for key in
                   ("run_id", "beta_obs", "beta_miss", "sigma", "full_top1",
                    "observed_top1_mean", "missing_top1_mean", "coverage01_mean",
                    "size_mean", "radius_mean", "constraints_passed")} for row in rows],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
