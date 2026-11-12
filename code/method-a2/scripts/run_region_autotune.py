"""无人值守执行 RaReF 参数搜索、验证集选模与最终测试评价。"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


METHOD_DIR = Path(__file__).resolve().parents[1]
TRAIN = METHOD_DIR / "scripts" / "train_gref.py"
EVAL = METHOD_DIR / "scripts" / "eval_region_term.py"
TEST = METHOD_DIR / "scripts" / "evaluate_gref.py"


def parse_args() -> argparse.Namespace:
    """解析无人值守搜索参数。"""
    parser = argparse.ArgumentParser(description="RaReF 区域损失自动调参与最终评价")
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--max-runs", type=int, default=26)
    parser.add_argument("--resume", action="store_true", help="跳过已有验证产物的候选")
    parser.add_argument("--dry-run", action="store_true", help="只写候选清单，不启动训练")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(value: object) -> float:
    """读取带 mean 字段或直接为数值的指标。"""
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def summarize_validation(payload: dict) -> dict:
    """提取候选的完整、条件分解和区域覆盖率摘要。"""
    levels = []
    for level, item in payload["partial"].items():
        missing = item["conditional"]["missing"]
        observed = item["conditional"]["observed"]
        levels.append({
            "observed_nodes": int(level),
            "top1": _metric(item["rank"]["top1"]),
            "observed_top1": _metric(observed["top1"]),
            "missing_top1": _metric(missing["top1"]),
            "missing_coverage": _metric(missing["region_0.1"]["coverage"]),
            "missing_size": _metric(missing["region_0.1"]["size_mean"]),
        })
    levels.sort(key=lambda row: row["observed_nodes"], reverse=True)
    full_top1 = _metric(payload["full"]["rank"]["top1"])
    return {
        "full_top1": full_top1,
        "partial_top1_mean": sum(row["top1"] for row in levels) / len(levels),
        "observed_top1_mean": sum(row["observed_top1"] for row in levels) / len(levels),
        "missing_top1_mean": sum(row["missing_top1"] for row in levels) / len(levels),
        "missing_coverage_min": min(row["missing_coverage"] for row in levels),
        "missing_coverage_mean": sum(row["missing_coverage"] for row in levels) / len(levels),
        "missing_size_mean": sum(row["missing_size"] for row in levels) / len(levels),
        "levels": levels,
    }


def build_candidates(max_runs: int) -> list[dict]:
    """构造围绕第二阶段配置的有限搜索空间。"""
    candidates = []
    mixes = [(0.141, 0.194), (0.194, 0.141), (0.234, 0.139)]
    for sigma, beta_obs, beta_miss, (lambda_rank, lambda_region) in itertools.product(
        (0.75, 1.0), (0.05, 0.1), (0.35, 0.4), mixes
    ):
        candidates.append({
            "lambda_energy": 1.0,
            "lambda_rank": lambda_rank,
            "lambda_region": lambda_region,
            "region_sigma": sigma,
            "beta_obs": beta_obs,
            "beta_miss": beta_miss,
        })
    # 补充一个能量权重对照，检验是否存在小幅能量增强的安全收益。
    candidates.extend([
        {"lambda_energy": 3.0, "lambda_rank": 0.141, "lambda_region": 0.194,
         "region_sigma": 1.0, "beta_obs": 0.1, "beta_miss": 0.4},
        {"lambda_energy": 3.0, "lambda_rank": 0.194, "lambda_region": 0.141,
         "region_sigma": 1.0, "beta_obs": 0.1, "beta_miss": 0.4},
    ])
    unique = []
    seen = set()
    for item in candidates:
        key = tuple(sorted(item.items()))
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique[:max_runs]


def _run(command: list[str], log_path: Path) -> None:
    """执行子进程并把标准输出和错误追加到统一日志。"""
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        result = subprocess.run(command, cwd=METHOD_DIR, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"子命令失败，退出码={result.returncode}，详见 {log_path}")


def _clear_stale_candidate(out_dir: Path) -> None:
    """清理上次失败尝试留下的候选目录与进度日志。

    `train_gref.py` 为保护既有结果而拒绝复用输出目录，也拒绝复用同名进度日志，
    因此重试候选前必须先清空这两处残留。
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    progress_dir = METHOD_DIR / "logs" / out_dir.name
    if progress_dir.exists():
        shutil.rmtree(progress_dir)


def _candidate_name(index: int, params: dict) -> str:
    """生成稳定的候选目录名。"""
    return (
        f"cand-{index:03d}-e{params['lambda_energy']:g}-r{params['lambda_rank']:g}"
        f"-q{params['lambda_region']:g}-s{params['region_sigma']:g}"
        f"-bo{params['beta_obs']:g}-bm{params['beta_miss']:g}"
    ).replace(".", "p")


def _train_command(args: argparse.Namespace, config: dict, params: dict, out_dir: Path) -> list[str]:
    """生成单个候选的训练命令。"""
    data_dir = Path(args.data_dir or config["data_dir"]).resolve()
    return [
        sys.executable, str(TRAIN), "--data-dir", str(data_dir), "--output-dir", str(out_dir),
        "--energy-tolerance", str(config["energy_tolerance"]),
        "--tolerance-source", str(config["tolerance_source"]),
        "--temperature", str(config["temperature"]), "--loss-mode", "region",
        "--lambda-energy", str(params["lambda_energy"]), "--lambda-rank", str(params["lambda_rank"]),
        "--lambda-region", str(params["lambda_region"]), "--region-sigma", str(params["region_sigma"]),
        "--beta-obs", str(params["beta_obs"]), "--beta-miss", str(params["beta_miss"]),
        "--rank-temperature", str(config.get("rank_temperature", 0.0040640985)),
        "--node-mask-full-probability", str(config.get("node_mask_full_probability", 0.25)),
        "--node-mask-min-observed", str(config.get("node_mask_min_observed", 15)),
        "--node-mask-max-observed", str(config.get("node_mask_max_observed", 37)),
        "--node-mask-force-observed-probability", str(config.get("node_mask_force_observed_probability", 0.5)),
        "--epochs", str(args.epochs), "--early-stopping-patience", str(args.early_stopping_patience),
        "--batch-size", str(args.batch_size), "--seed", str(args.seed),
        "--device", args.device, "--skip-test-evaluation", "--no-terminal-progress",
    ]


def main() -> int:
    """训练全部候选、按验证约束选模并完成一次测试评价。"""
    args = parse_args()
    baseline = args.baseline_run_dir.resolve()
    config = _load_json(baseline / "config.json")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "autotune.log"
    candidates = build_candidates(args.max_runs)
    (root / "candidate_grid.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"candidate_count": len(candidates), "output_root": str(root)}, ensure_ascii=False))
        return 0

    data_dir = Path(args.data_dir or config["data_dir"]).resolve()
    if not (data_dir / "X_obs.npy").is_file():
        raise FileNotFoundError(f"数据集目录无效，缺少 X_obs.npy：{data_dir}")
    args.data_dir = data_dir
    baseline_validation = _load_json(baseline / "validation_region_metrics.json")
    baseline_summary = summarize_validation(baseline_validation)
    rows = []
    for index, params in enumerate(candidates):
        name = _candidate_name(index, params)
        out_dir = root / name
        validation_file = out_dir / "validation_region_metrics.json"
        try:
            if not (args.resume and validation_file.exists()):
                _clear_stale_candidate(out_dir)
                _run(_train_command(args, config, params, out_dir), log_path)
                _run([sys.executable, str(EVAL), "--run-dir", str(out_dir),
                      "--data-dir", str(Path(args.data_dir or config["data_dir"]).resolve()),
                      "--output-dir", str(out_dir), "--split", "validation", "--device", args.device,
                      "--mask-seed-offset", "1000", "--no-merge-predictions"], log_path)
            summary = summarize_validation(_load_json(validation_file))
            row = {"index": index, "name": name, "params": params, "summary": summary,
                   "status": "ok"}
        except Exception as error:  # 候选失败不应阻断无人值守搜索。
            row = {"index": index, "name": name, "params": params, "status": "failed",
                   "error": str(error)}
        rows.append(row)
        (root / "progress.json").write_text(
            json.dumps({"baseline": baseline_summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    valid = [row for row in rows if row["status"] == "ok"]
    def accepted(row: dict) -> bool:
        s = row["summary"]
        return (s["missing_coverage_mean"] >= 0.95
                and s["missing_coverage_min"] >= 0.90
                and s["missing_size_mean"] <= baseline_summary["missing_size_mean"] + 0.5
                and s["observed_top1_mean"] >= baseline_summary["observed_top1_mean"] - 0.02
                and s["full_top1"] >= baseline_summary["full_top1"] - 0.01)

    accepted_rows = [row for row in valid if accepted(row)]
    pool = accepted_rows or valid
    if not pool:
        raise RuntimeError(
            f"所有候选均失败，未生成最终配置；详见 {root / 'progress.json'} 与 {log_path}")
    selected = max(pool, key=lambda row: (
        row["summary"]["missing_top1_mean"], row["summary"]["missing_coverage_min"],
        row["summary"]["observed_top1_mean"], row["summary"]["full_top1"],
        -row["summary"]["missing_size_mean"]))
    selected_dir = root / selected["name"]
    _run([sys.executable, str(TEST), "--run-dir", str(selected_dir),
          "--data-dir", str(Path(args.data_dir or config["data_dir"]).resolve()),
          "--device", args.device, "--batch-size", str(args.batch_size)], log_path)
    _run([sys.executable, str(EVAL), "--run-dir", str(selected_dir),
          "--data-dir", str(Path(args.data_dir or config["data_dir"]).resolve()),
          "--output-dir", str(selected_dir), "--split", "test", "--device", args.device], log_path)
    result = {"baseline": baseline_summary, "selected": selected,
              "accepted_count": len(accepted_rows), "candidate_count": len(candidates),
              "rows": rows}
    (root / "autotune_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected["name"], "accepted_count": len(accepted_rows),
                      "output_root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
