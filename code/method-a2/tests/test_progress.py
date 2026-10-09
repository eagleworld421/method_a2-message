"""进度落盘、ETA 和 Agent 读取入口测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.progress import ProgressWriter

REPORT_PATH = METHOD_DIR / "scripts" / "report_progress.py"
SPEC = importlib.util.spec_from_file_location("report_progress", REPORT_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_progress_writer_persists_latest_and_eta():
    """每次推进都能被快照读取，并按同类单元计算 ETA。"""
    path = METHOD_DIR / "logs" / "test-progress" / "progress.jsonl"
    writer = ProgressWriter(path, "test-run", min_interval_s=0.0, terminal=False)
    writer.stage_start("training", 4)
    writer.update("training", "batch", 1, 4, note="第一批", force=True, unit_seconds=2.0)
    writer.update("training", "batch", 2, 4, note="第二批", force=True, unit_seconds=2.0)
    records = REPORT.load_records(path)
    latest = json.loads((path.with_name("progress.latest.json")).read_text(encoding="utf-8"))
    assert latest["percent"] == pytest.approx(50.0)
    assert latest["eta_s"] == pytest.approx(4.0)
    assert len(records) == 3
    for candidate in (path, path.with_name("progress.latest.json")):
        if candidate.exists():
            candidate.unlink()
    if path.parent.exists():
        path.parent.rmdir()


def test_progress_reader_skips_truncated_line_and_reports_completed():
    """读取入口跳过残缺行，完成状态不提供虚假的 ETA。"""
    records = [
        {"kind": "unit", "utc": "2999-01-01T00:00:00+00:00", "run_id": "r",
         "stage": "training", "unit": "batch", "done": 2, "total": 2,
         "percent": 100.0, "elapsed_s": 4.0, "rate_s": 2.0, "eta_s": 0.0, "note": "完成"},
    ]
    path = Path(__file__).parent / "_progress_test.jsonl"
    try:
        path.write_text("\n".join(json.dumps(item) for item in records) + "\n残缺", encoding="utf-8")
        loaded = REPORT.load_records(path)
        summary = REPORT.summarize(loaded)
        assert len(loaded) == 1
        assert summary["status"] == "running"
        assert summary["eta_s"] is None
        assert summary["basis"] == "本阶段单元已全部完成"
    finally:
        if path.exists():
            path.unlink()
