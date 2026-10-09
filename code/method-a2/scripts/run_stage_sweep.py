"""按阶段批量运行训练配置，汇总验证集结果供阶段判定使用。

本脚本只做调度与汇总，不改动模型或数据。每个配置独立运行一次训练脚本，并把
验证集指标汇总为 JSON 与 CSV，便于按《Method-A2 排除既有方法后的准确率提升验证计划》
的阶段门做选择与报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

METHOD_DIR = Path(__file__).absolute().parents[1]
REPO_DIR = METHOD_DIR.parents[1]
TRAIN_SCRIPT = METHOD_DIR / "scripts" / "train_marker_mvp.py"


def parse_args() -> argparse.Namespace:
    """解析批量运行参数。"""
    parser = argparse.ArgumentParser(description="批量运行 Node Marker 训练配置")
    parser.add_argument("--stage", required=True, help="阶段名，用于命名输出目录前缀")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[342])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-pair", type=float, default=0.1)
    parser.add_argument("--lambda-list", type=float, default=0.1)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--pooling", nargs="+", default=["mean"])
    parser.add_argument("--temporal-mode", nargs="+", default=["single"])
    parser.add_argument("--derived-channels", action="store_true")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--arm-name", default=None,
                        help="配置名称，用于输出目录；给出时忽略 pooling 与 temporal-mode 的多值展开")
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--encoder-layers", type=int, default=2)
    return parser.parse_args()


def build_configurations(args: argparse.Namespace) -> list[dict]:
    """由命令行参数展开待运行配置列表。"""
    configurations = []
    if args.arm_name:
        configurations.append({
            "arm": args.arm_name,
            "pooling": args.pooling[0],
            "temporal_mode": args.temporal_mode[0],
        })
        return configurations
    for pooling in args.pooling:
        for temporal_mode in args.temporal_mode:
            configurations.append({
                "arm": f"{pooling}__{temporal_mode}",
                "pooling": pooling,
                "temporal_mode": temporal_mode,
            })
    return configurations


def run_configuration(
    args: argparse.Namespace, configuration: dict, seed: int
) -> dict:
    """运行单个配置并读取其验证集与测试集指标。"""
    run_id = f"{args.stage}-{configuration['arm']}-seed{seed}"
    output_dir = Path(args.output_root) / run_id
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--lambda-pair", str(args.lambda_pair),
        "--lambda-list", str(args.lambda_list),
        "--rank-temperature", str(args.rank_temperature),
        "--pooling", configuration["pooling"],
        "--temporal-mode", configuration["temporal_mode"],
        "--temporal-layers", str(args.temporal_layers),
        "--encoder-layers", str(args.encoder_layers),
        "--seed", str(seed),
        "--run-id", run_id,
        "--no-terminal-progress",
    ]
    if args.derived_channels:
        command.append("--derived-channels")
    if args.uncertainty:
        command.append("--uncertainty")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(REPO_DIR), capture_output=True, text=True)
    duration = time.perf_counter() - started
    record = {
        "run_id": run_id,
        "arm": configuration["arm"],
        "pooling": configuration["pooling"],
        "temporal_mode": configuration["temporal_mode"],
        "derived_channels": bool(args.derived_channels),
        "uncertainty": bool(args.uncertainty),
        "seed": seed,
        "seconds": round(duration, 1),
        "returncode": int(completed.returncode),
    }
    if completed.returncode != 0:
        record["stderr_tail"] = completed.stderr.strip().splitlines()[-3:]
        return record
    validation = json.loads((output_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    probes = json.loads((output_dir / "probes.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    record.update({
        "val_top1": validation["plain"]["top1"],
        "val_top3": validation["plain"]["top3"],
        "val_mean_rank": validation["plain"]["mean_rank"],
        "val_family_loss": validation["losses"]["family"],
        "val_signed_margin_median": validation["plain"]["signed_margin_median"],
        "test_top1": test["plain"]["top1"],
        "test_top3": test["plain"]["top3"],
        "test_mean_rank": test["plain"]["mean_rank"],
        "test_family_loss": test["losses"]["family"],
        "oracle_val_top1": validation["oracle"]["oracle_top1"],
        "oracle_val_top3": validation["oracle"]["oracle_top3"],
        "oracle_test_top1": test["oracle"]["oracle_top1"],
        "oracle_test_top3": test["oracle"]["oracle_top3"],
        "probe_z_shared_fault_location": probes["z_shared_fault_location"]["test_accuracy"],
        "probe_node_repr_fault_location": probes["node_repr_fault_location"]["test_accuracy"],
        "probe_chance": probes["z_shared_fault_location"]["chance_accuracy"],
        "trainable_parameters": config["trainable_parameters"],
        "best_epoch": config["best_epoch"],
        "n_train": config["split"]["n_train"],
        "n_val": config["split"]["n_val"],
        "n_test": config["split"]["n_test"],
    })
    if "uncertainty" in test:
        record["val_uncertainty_top1"] = validation["uncertainty"]["top1"]
        record["val_uncertainty_top3"] = validation["uncertainty"]["top3"]
        record["test_uncertainty_top1"] = test["uncertainty"]["top1"]
        record["test_uncertainty_top3"] = test["uncertainty"]["top3"]
    if "variance_error_correlation" in test:
        record["variance_error_correlation"] = test["variance_error_correlation"]
    return record


def main() -> int:
    """依次运行全部配置并写出汇总结果。"""
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for configuration in build_configurations(args):
        for seed in args.seeds:
            record = run_configuration(args, configuration, seed)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
    summary_path = output_root / f"{args.stage}-summary.json"
    summary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = sorted({key for record in records for key in record})
    with (output_root / f"{args.stage}-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"汇总写入 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
