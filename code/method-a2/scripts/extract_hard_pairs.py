"""从已完成运行中提取困难候选对，供主动困难候选仿真使用。

只读取指定运行目录下的 `hard_negative_pairs.json`，该文件由训练脚本在验证集上
生成，因此不会把测试集信息带入新数据。输出用于 `generate_dataset.py` 的
`--hard-pair-file`。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析参数。"""
    parser = argparse.ArgumentParser(description="提取困难候选对")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=1,
                        help="候选对在验证集事件中出现的最少次数")
    return parser.parse_args()


def main() -> int:
    """统计并写出困难候选对。"""
    args = parse_args()
    payload = json.loads((args.run_dir / "hard_negative_pairs.json").read_text(encoding="utf-8"))
    counter: Counter = Counter(
        tuple(sorted((int(row["true_location"]), int(row["oracle_hardest_location"]))))
        for row in payload["pairs"]
    )
    selected = {pair: count for pair, count in counter.items() if count >= args.min_count}
    ordered = sorted(selected.items(), key=lambda item: -item[1])
    output = {
        "source_run": str(args.run_dir),
        "source_split": payload.get("source_split", "validation"),
        "n_validation_events": payload.get("n_pairs", 0),
        "n_distinct_pairs": len(counter),
        "min_count": int(args.min_count),
        "pairs": [
            {"locations": list(pair), "count": count} for pair, count in ordered
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_distinct_pairs": len(counter),
        "n_selected": len(selected),
        "top": ordered[:8],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
