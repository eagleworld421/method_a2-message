"""运行进度落盘工具：同时服务终端进度条和 Agent 读取。"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class ProgressWriter:
    """按阶段记录完成度、实测耗时和 ETA，并立即写入磁盘。"""

    def __init__(
        self,
        path: Path,
        run_id: str,
        min_interval_s: float = 1.0,
        terminal: bool = True,
    ) -> None:
        self.path = Path(path)
        self.latest_path = self.path.with_name("progress.latest.json")
        self.run_id = str(run_id)
        self.min_interval_s = float(min_interval_s)
        self.terminal = bool(terminal)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._last_write = 0.0
        self._unit_totals: dict[str, list[float | int]] = {}
        self._last_record: dict | None = None

    def _elapsed(self) -> float:
        """返回本次运行已经过的秒数。"""
        return time.perf_counter() - self.started

    @staticmethod
    def _utc() -> str:
        """返回带 UTC 时区的 ISO 时间。"""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _write(self, record: dict) -> None:
        """追加 JSONL 记录并更新快照文件。"""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.latest_path.open("w", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._last_record = record

    def _terminal(self, record: dict) -> None:
        """在终端显示单行进度，不影响磁盘记录。"""
        if not self.terminal:
            return
        percent = record.get("percent")
        percent_text = "--" if percent is None else f"{percent:6.2f}%"
        filled = min(20, max(0, int((percent or 0) / 5)))
        bar = "#" * filled + "-" * (20 - filled)
        eta = record.get("eta_s")
        if eta is None:
            eta_text = "--"
        elif eta == 0:
            eta_text = "0.0s"
        else:
            eta_text = f"{_format_seconds(eta * 0.8)}–{_format_seconds(eta * 1.3)}"
        elapsed_text = _format_seconds(float(record.get("elapsed_s") or 0.0))
        total = record.get("total")
        total_text = "?" if total is None else str(total)
        text = (
            f"[{record.get('stage', '')}] [{bar}] {percent_text} "
            f"{record.get('done', 0)}/{total_text} "
            f"已用 {elapsed_text} ETA {eta_text} {record.get('note', '')}"
        )
        if sys.stdout.isatty():
            end = "\n" if record.get("kind") in {"stage_done", "end", "error"} else ""
            print("\r" + text[:160].ljust(160), end=end, flush=True)
        else:
            print(text.rstrip(), flush=True)

    def stage_start(self, stage: str, total: int | None) -> None:
        """写入阶段开始记录并清空该阶段的分类速率。"""
        self._unit_totals = {}
        self.update(stage, None, 0, total, kind="start", force=True)

    def update(
        self,
        stage: str,
        unit: str | None,
        done: int,
        total: int | None,
        note: str = "",
        kind: str = "unit",
        force: bool = False,
        unit_seconds: float | None = None,
    ) -> None:
        """写入一次阶段进度，并按同类单元估计 ETA。"""
        now = time.perf_counter()
        if unit is not None and unit_seconds is not None:
            history = self._unit_totals.setdefault(unit, [0.0, 0])
            history[0] = float(history[0]) + float(unit_seconds)
            history[1] = int(history[1]) + 1
        if not force and now - self._last_write < self.min_interval_s:
            return
        rate = None
        if unit in self._unit_totals and self._unit_totals[unit][1]:
            rate = float(self._unit_totals[unit][0]) / int(self._unit_totals[unit][1])
        remaining = None if total is None else max(int(total) - int(done), 0)
        eta = 0.0 if remaining == 0 else (None if rate is None or remaining is None else rate * remaining)
        record = {
            "kind": kind,
            "utc": self._utc(),
            "run_id": self.run_id,
            "stage": stage,
            "unit": unit,
            "done": int(done),
            "total": total,
            "percent": None if total is None else round(100.0 * done / max(total, 1), 2),
            "elapsed_s": round(self._elapsed(), 3),
            "rate_s": None if rate is None else round(rate, 4),
            "eta_s": None if eta is None else round(eta, 3),
            "note": note,
            "pid": os.getpid(),
        }
        self._last_write = now
        self._write(record)
        self._terminal(record)

    def stage_done(self, stage: str, total: int | None, note: str = "阶段完成") -> None:
        """写入阶段完成记录。"""
        self.update(stage, None, total or 0, total, note=note, kind="stage_done", force=True)

    def finish(self, stage: str = "writeout", note: str = "运行完成") -> None:
        """写入整次运行结束记录。"""
        last = self._last_record or {}
        self.update(
            stage,
            last.get("unit"),
            last.get("done") or 0,
            last.get("total"),
            note=note,
            kind="end",
            force=True,
        )

    def error(self, stage: str, note: str) -> None:
        """记录异常并保留原始异常退出语义。"""
        self.update(stage, None, 0, None, note=note, kind="error", force=True)


def _format_seconds(seconds: float) -> str:
    """把秒数格式化为适合终端阅读的时长。"""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{remainder:02d}s"
