"""生成 Method-A2 S0 数据并持久化可读取的运行进度。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.dataset_builder import build_dataset
from src.progress import ProgressWriter


def parse_args() -> argparse.Namespace:
    """解析数据生成参数。"""
    parser = argparse.ArgumentParser(description="生成 Method-A2 S0 数据")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name", default="ieee13")
    parser.add_argument("--samples-per-bus", type=int, default=1)
    parser.add_argument("--normal-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="给出时按训练/验证/测试三段按分组划分，验证集只含故障样本")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="三段划分时测试集分组占比")
    parser.add_argument("--res-min", type=float, default=0.1,
                        help="故障阻抗下界，单位为欧姆")
    parser.add_argument("--res-max", type=float, default=100.0,
                        help="故障阻抗上界，单位为欧姆")
    parser.add_argument("--impedance-sampling", choices=("uniform", "log_uniform"),
                        default="uniform", help="故障阻抗采样方式")
    parser.add_argument("--fault-classes", type=int, nargs="+", default=[0],
                        help="事件可用的故障类型；默认只用单相接地故障以保证全母线可承载")
    parser.add_argument("--hard-pair-file", type=Path, default=None,
                        help="困难候选对 JSON（hard_negative_pairs.json）；给出时启用主动困难候选采样")
    parser.add_argument("--hard-pair-max-attempts", type=int, default=12,
                        help="每个事件族为命中困难候选对的最大重采样次数")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--no-terminal-progress", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None, progress: ProgressWriter | None = None) -> int:
    """运行数据生成并在异常时写入 error 记录。

    参数全部由命令行解析得到；`args` 与 `progress` 供调用方（如 `main.py` 运行
    入口）直接注入，为 None 时维持原有命令行行为。
    """
    args = parse_args() if args is None else args
    run_id = args.run_id or f"dataset-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-seed{args.seed}"
    progress_path = args.progress or (METHOD_DIR / "logs" / run_id / "progress.jsonl")
    if progress is None:
        progress = ProgressWriter(
            progress_path, run_id, terminal=not args.no_terminal_progress
        )
    try:
        hard_pairs = None
        if args.hard_pair_file is not None:
            payload = json.loads(Path(args.hard_pair_file).read_text(encoding="utf-8"))
            hard_pairs = {
                tuple(sorted((int(row["true_location"]), int(row["oracle_hardest_location"]))))
                for row in payload["pairs"]
            }
        build_dataset(
            args.output_dir,
            case_name=args.case_name,
            samples_per_bus=args.samples_per_bus,
            normal_ratio=args.normal_ratio,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            res_min=args.res_min,
            res_max=args.res_max,
            impedance_sampling=args.impedance_sampling,
            fault_classes=tuple(args.fault_classes),
            hard_pairs=hard_pairs,
            hard_pair_max_attempts=args.hard_pair_max_attempts,
            progress=progress,
        )
        progress.finish("data_generation")
    except Exception as exc:
        progress.error("data_generation", f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
