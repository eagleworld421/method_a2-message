"""读取 Method-A2 运行进度快照，供用户和 Agent 查询。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    """逐行读取进度记录，跳过进程中断留下的残缺 JSON 行。"""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _age_s(utc_text: str) -> float:
    """计算进度记录距离当前时间的秒数。"""
    stamped = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    return max(0.0, datetime.now(timezone.utc).timestamp() - stamped.timestamp())


def summarize(records: list[dict], stale_after_s: float = 60.0) -> dict:
    """汇总最新阶段、完成度、ETA、新鲜度和停滞状态。"""
    if not records:
        return {"status": "no_records"}
    latest = records[-1]
    age = _age_s(latest["utc"])
    kind = latest.get("kind")
    if kind == "end" or (kind == "stage_done" and latest.get("total") == latest.get("done")):
        status = "completed"
    elif kind == "error":
        status = "error"
    elif age > float(stale_after_s):
        status = "stalled_suspected"
    else:
        status = "running"
    rates: dict[str, list[float]] = {}
    for record in records:
        unit = record.get("unit")
        rate = record.get("rate_s")
        if unit and rate is not None:
            rates.setdefault(unit, []).append(float(rate))
    remaining = None
    if latest.get("total") is not None:
        remaining = max(int(latest["total"]) - int(latest.get("done") or 0), 0)
    values = rates.get(latest.get("unit"), [])
    rate = sum(values) / len(values) if values else None
    eta = None if rate is None or remaining is None or remaining == 0 or status != "running" else rate * remaining
    if remaining == 0:
        basis = "本阶段单元已全部完成"
    elif eta is not None:
        basis = "按同类单元历史平均耗时外推"
    else:
        basis = "尚无同类单元完成，无法估计"
    return {
        "status": status,
        "run_id": latest.get("run_id"),
        "stage": latest.get("stage"),
        "unit": latest.get("unit"),
        "done": latest.get("done"),
        "total": latest.get("total"),
        "percent": latest.get("percent"),
        "elapsed_s": latest.get("elapsed_s"),
        "age_s": round(age, 2),
        "eta_s": None if eta is None else round(eta, 2),
        "eta_range_s": None if eta is None else [round(eta * 0.8, 2), round(eta * 1.3, 2)],
        "basis": basis,
        "note": latest.get("note"),
        "record_count": len(records),
        "pid": latest.get("pid"),
    }


def resolve_progress(progress: Path | None, run_id: str | None, log_root: Path) -> Path | None:
    """根据显式路径、运行标识或日志根目录定位进度文件。"""
    if progress is not None:
        return progress
    if run_id:
        return log_root / run_id / "progress.jsonl"
    candidates = sorted(log_root.glob("*/progress.jsonl"), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def main() -> int:
    """输出 JSON 汇总，便于人和 Agent 共同读取。"""
    parser = argparse.ArgumentParser(description="读取 Method-A2 运行进度")
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--log-root", type=Path, default=Path(__file__).absolute().parents[1] / "logs")
    parser.add_argument("--stale-after-s", type=float, default=60.0)
    args = parser.parse_args()
    path = resolve_progress(args.progress, args.run_id, args.log_root)
    if path is None:
        print(json.dumps({"status": "no_progress_file"}, ensure_ascii=False))
        return 1
    summary = summarize(load_records(path), stale_after_s=args.stale_after_s)
    summary["progress_file"] = str(path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] != "no_records" else 1


if __name__ == "__main__":
    sys.exit(main())
