"""汇总RaReF 区域概率损失第二阶段权重扫描的验证集指标、约束判定与选模结果。

只读取各扫描运行目录的配置、验证集指标与验证集区域指标，不训练、不访问测试集。
按补遗登记的协议选择配置：先满足完整观测与观测子集 Top-1 约束，再按未观测子集
区域覆盖率、区域大小、观测子集 Top-1 与完整观测 Top-1 依次比较。
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
    parser = argparse.ArgumentParser(description="汇总RaReF 区域概率损失权重扫描")
    parser.add_argument("--root", type=Path, default=Path("code/method-a2/output/gref"))
    parser.add_argument("--pattern", default="region-sweep-*-seed342")
    parser.add_argument("--phase1-dir", type=Path, default=None)
    parser.add_argument("--gradient-root", type=Path, default=None,
                        help="可选的逐运行梯度分析目录，含 <run-id>/region_gradient_analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """返回算术平均。"""
    return float(np.mean(values))


def _subset_metrics(region_payload: dict) -> dict:
    """从验证区域指标中抽取选模需要的分层指标。"""
    missing_coverage = [region_payload["partial"][level]["conditional"]["missing"]
                        ["region_0.1"]["coverage"]["mean"] for level in LEVELS]
    missing_coverage_02 = [region_payload["partial"][level]["conditional"]["missing"]
                           ["region_0.2"]["coverage"]["mean"] for level in LEVELS]
    missing_size = [region_payload["partial"][level]["conditional"]["missing"]
                    ["region_0.1"]["size_mean"]["mean"] for level in LEVELS]
    missing_radius = [region_payload["partial"][level]["conditional"]["missing"]
                      ["region_0.1"]["radius_mean"]["mean"] for level in LEVELS]
    observed_top1 = [region_payload["partial"][level]["conditional"]["observed"]
                     ["top1"]["mean"] for level in LEVELS]
    overall_top1 = [region_payload["partial"][level]["rank"]["top1"]["mean"]
                    for level in LEVELS]
    return {
        "missing_coverage_alpha01_mean": _mean(missing_coverage),
        "missing_coverage_alpha01_by_level": missing_coverage,
        "missing_coverage_alpha02_mean": _mean(missing_coverage_02),
        "missing_size_alpha01_mean": _mean(missing_size),
        "missing_radius_alpha01_mean": _mean(missing_radius),
        "observed_top1_mean": _mean(observed_top1),
        "observed_top1_by_level": observed_top1,
        "partial_top1_mean": _mean(overall_top1),
        "full_top1": region_payload["full"]["rank"]["top1"],
        "full_region_coverage_alpha01": region_payload["full"]["region"]["0.1"]["coverage"],
    }


def _load_run(run_dir: Path, gradient_root: Path | None) -> dict:
    """读取单个运行的配置、验证指标与验证区域指标。"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    region = json.loads((run_dir / "validation_region_metrics.json").read_text(encoding="utf-8"))
    row = {
        "run_id": run_dir.name,
        "lambda_energy": config["loss_weights"]["energy"],
        "lambda_rank": config["loss_weights"]["rank"],
        "lambda_region": config["loss_weights"]["region"],
        "best_epoch": config.get("best_epoch"),
        "validation_losses": validation["losses"],
        "validation_score": {key: validation["score"][key]
                             for key in ("top1", "top3", "top5", "mean_rank")},
    }
    row.update(_subset_metrics(region))
    if gradient_root is not None:
        gradient_path = gradient_root / run_dir.name / "region_gradient_analysis.json"
        if gradient_path.exists():
            gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
            row["gradient_projection_masked"] = gradient["derived"]["masked"]["current"]["projection"]
            row["gradient_share_masked"] = gradient["derived"]["masked"]["current"]["norm_share"]
    return row


def main() -> int:
    """执行汇总、约束判定与选模，并写出 JSON。"""
    args = parse_args()
    root = args.root.resolve()
    runs = sorted(path for path in root.glob(args.pattern) if path.is_dir())
    if not runs:
        raise FileNotFoundError("未找到扫描运行目录")
    phase1_dir = args.phase1_dir
    if phase1_dir is None:
        phase1_dir = root / "region-term-ieee37-a-seed342"
    phase1 = _load_run(phase1_dir, None)
    # 约束以第一阶段验证指标为锚点：完整观测 Top-1 不超过 1 个百分点退化，
    # 观测子集 Top-1 均值不超过 2 个百分点退化
    full_floor = phase1["full_top1"] - 0.01
    observed_floor = phase1["observed_top1_mean"] - 0.02
    rows = []
    for run_dir in runs:
        row = _load_run(run_dir, args.gradient_root)
        row["full_top1_floor"] = full_floor
        row["observed_top1_floor"] = observed_floor
        row["constraints_passed"] = bool(
            row["full_top1"] >= full_floor and row["observed_top1_mean"] >= observed_floor)
        rows.append(row)
    eligible = [row for row in rows if row["constraints_passed"]]
    pool = eligible if eligible else rows
    selected = sorted(pool, key=lambda row: (
        -row["missing_coverage_alpha01_mean"],
        row["missing_size_alpha01_mean"],
        -row["observed_top1_mean"],
        -row["full_top1"],
    ))[0]
    result = {
        "root": str(root), "pattern": args.pattern,
        "phase1_reference": {key: phase1[key] for key in
                             ("run_id", "full_top1", "observed_top1_mean",
                              "missing_coverage_alpha01_mean", "missing_size_alpha01_mean")},
        "constraints": {
            "full_top1_floor": full_floor,
            "observed_top1_floor": observed_floor,
            "definition": "完整观测验证 Top-1 不低于第一阶段减 1 个百分点；"
                          "观测子集验证 Top-1 均值不低于第一阶段减 2 个百分点",
        },
        "priority": ["未观测子集区域覆盖率 α=0.1 均值最大",
                     "平均区域节点数 α=0.1 最小",
                     "观测子集 Top-1 均值最大",
                     "完整观测 Top-1 最大"],
        "n_runs": len(rows), "n_eligible": len(eligible),
        "selected_run_id": selected["run_id"],
        "selection_fallback": not bool(eligible),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({
        "phase1": result["phase1_reference"], "constraints": result["constraints"],
        "selected": selected["run_id"], "fallback": result["selection_fallback"],
        "eligible": [row["run_id"] for row in eligible],
        "table": [{key: row[key] for key in
                   ("run_id", "lambda_energy", "lambda_rank", "lambda_region",
                    "full_top1", "observed_top1_mean", "missing_coverage_alpha01_mean",
                    "missing_size_alpha01_mean", "missing_coverage_alpha02_mean",
                    "partial_top1_mean", "constraints_passed")} for row in rows],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
