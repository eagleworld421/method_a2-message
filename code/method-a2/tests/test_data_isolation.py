"""数据生成子进程隔离与完成判定的测试。

以 OpenDSS COM 求解数据的进程无法正常退出，因此数据生成在子进程中执行，父进程
只依据进度文件判定结果。本文件覆盖该判定逻辑，测试中用假进程替代真实子进程。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

METHOD_DIR = Path(__file__).absolute().parents[1]
for _extra_path in (str(METHOD_DIR), str(METHOD_DIR / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

MAIN_PATH = METHOD_DIR / "main.py"
SPEC = importlib.util.spec_from_file_location("method_a2_main", MAIN_PATH)
MAIN = importlib.util.module_from_spec(SPEC)
sys.modules["method_a2_main"] = MAIN
SPEC.loader.exec_module(MAIN)

RUN_ID = "unittest-isolated"


def write_progress(records: list[dict], run_id: str = RUN_ID) -> None:
    """把给定的进度记录写入该运行的进度文件与快照文件。"""
    path = MAIN.progress_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    MAIN.latest_progress_path(run_id).write_text(lines[-1] + "\n", encoding="utf-8")


def clear_progress(run_id: str = RUN_ID) -> None:
    """删除该运行的进度文件、快照文件、归档文件与子进程日志。"""
    directory = MAIN.progress_path(run_id).parent
    for path in (
        MAIN.progress_path(run_id),
        MAIN.latest_progress_path(run_id),
        MAIN.child_log_path(run_id),
    ):
        if path.is_file():
            path.unlink()
    if directory.is_dir():
        for path in directory.glob("progress*.attempt*.*"):
            path.unlink()
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


@pytest.fixture(autouse=True)
def clean_run_logs():
    """每个测试前后都清理该测试使用的运行目录。"""
    clear_progress()
    yield
    clear_progress()


def data_namespace(**overrides) -> argparse.Namespace:
    """构造数据生成参数，默认值与测试无关的字段取配置文件默认值。"""
    values = {
        "case_name": "ieee13",
        "seed": 342,
        "run_id": RUN_ID,
        "output_dir": METHOD_DIR / "data" / "unittest-output",
        "samples_per_bus": 1,
        "normal_ratio": 1.0,
        "val_ratio": None,
        "test_ratio": 0.15,
        "res_min": 100.0,
        "res_max": 1000.0,
        "impedance_sampling": "log_uniform",
        "fault_classes": [0],
        "hard_pair_file": None,
        "hard_pair_max_attempts": 12,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def end_record(run_id: str = RUN_ID) -> dict:
    """构造一条数据生成结束记录。"""
    return {"kind": "end", "utc": "t2", "run_id": run_id, "stage": "data_generation",
            "done": 2, "total": 2, "percent": 100.0, "elapsed_s": 1.0}


def error_record(run_id: str = RUN_ID, note: str = "RuntimeError: 测试") -> dict:
    """构造一条数据生成失败记录。"""
    return {"kind": "error", "utc": "t1", "run_id": run_id, "stage": "data_generation",
            "done": 0, "total": None, "percent": None, "elapsed_s": 0.5, "note": note}


class FakeProcess:
    """替代 subprocess.Popen 的假子进程：按轮询次数返回预设状态。"""

    def __init__(self, states: list[int | None]) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.kill_calls = 0
        self._states = list(states)

    def poll(self) -> int | None:
        """返回当前退出码；None 表示仍在运行。"""
        if self._states:
            self.returncode = self._states.pop(0)
        return self.returncode

    def kill(self) -> None:
        """记录强制终止。"""
        self.kill_calls += 1
        self.returncode = -9

    def wait(self) -> int | None:
        """返回退出码。"""
        return self.returncode


def test_data_arguments_command_carries_all_parameters():
    """子进程命令行必须完整携带数据生成参数，且路径为绝对路径。"""
    args = data_namespace(val_ratio=0.2, hard_pair_file=Path("data/pairs.json"))
    command = MAIN.data_arguments_command(args)
    assert "--case-name" in command and "ieee13" in command
    assert command[command.index("--seed") + 1] == "342"
    assert command[command.index("--samples-per-bus") + 1] == "1"
    assert command[command.index("--val-ratio") + 1] == "0.2"
    assert command[command.index("--fault-classes") + 1] == "0"
    assert Path(command[command.index("--output-dir") + 1]).is_absolute()
    assert Path(command[command.index("--hard-pair-file") + 1]).is_absolute()
    assert "--no-terminal-progress" in command


def test_data_arguments_command_omits_optional_none_values():
    """未设置的可选参数不得出现在子进程命令行中。"""
    command = MAIN.data_arguments_command(data_namespace())
    assert "--val-ratio" not in command
    assert "--hard-pair-file" not in command


def test_relative_output_dir_resolves_against_cwd_not_method_dir(tmp_path, monkeypatch):
    """命令行给出的相对路径按当前工作目录解析，不叠加方法目录前缀。"""
    monkeypatch.chdir(tmp_path)
    command = MAIN.data_arguments_command(data_namespace(output_dir=Path("code/method-a2/data/x")))
    resolved = Path(command[command.index("--output-dir") + 1])
    assert resolved == (tmp_path / "code" / "method-a2" / "data" / "x")
    assert "method-a2" not in str(resolved.parent.parent.parent.name)


def test_run_completion_is_detected_from_progress_file():
    """结束时序的判定以进度文件最后一条记录为准。"""
    write_progress([
        {"kind": "start", "utc": "t0", "run_id": RUN_ID, "stage": "data_generation",
         "done": 0, "total": 2, "percent": 0.0, "elapsed_s": 0.0},
    ])
    assert MAIN.data_run_completed(RUN_ID) is False
    assert MAIN.data_run_failed(RUN_ID) is False
    write_progress([
        {"kind": "start", "utc": "t0", "run_id": RUN_ID, "stage": "data_generation",
         "done": 0, "total": 2, "percent": 0.0, "elapsed_s": 0.0},
        {"kind": "end", "utc": "t1", "run_id": RUN_ID, "stage": "data_generation",
         "done": 2, "total": 2, "percent": 100.0, "elapsed_s": 1.0},
    ])
    assert MAIN.data_run_completed(RUN_ID) is True


def test_run_completion_ignores_truncated_last_line():
    """最后一行写入不完整时不得判定为完成，并继续向前解析。"""
    path = MAIN.progress_path(RUN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kind": "unit", "utc": "t0", "run_id": RUN_ID,
                    "stage": "data_generation", "done": 1, "total": 2}) + "\n"
        + '{"kind": "end", "utc": "t1", "run_id": "' + RUN_ID,
        encoding="utf-8",
    )
    assert MAIN.data_run_completed(RUN_ID) is False


def test_run_failure_is_detected_from_progress_file():
    """失败记录必须被识别，供父进程区分失败与未完成。"""
    write_progress([
        {"kind": "error", "utc": "t0", "run_id": RUN_ID, "stage": "data_generation",
         "done": 0, "total": None, "percent": None, "elapsed_s": 0.5,
         "note": "RuntimeError: 测试"},
    ])
    assert MAIN.data_run_failed(RUN_ID) is True
    assert MAIN.data_run_completed(RUN_ID) is False


def test_latest_progress_snapshot_tolerates_missing_and_broken_file():
    """进度快照缺失或写入不完整时返回 None，不得抛异常。"""
    assert MAIN.read_latest_progress(RUN_ID) is None
    path = MAIN.latest_progress_path(RUN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert MAIN.read_latest_progress(RUN_ID) is None
    path.write_text(json.dumps({"kind": "unit", "utc": "t0", "done": 3}), encoding="utf-8")
    assert MAIN.read_latest_progress(RUN_ID)["done"] == 3


def test_isolated_run_force_kills_hung_child_after_payload(monkeypatch):
    """子进程写完结束记录后仍不退出时必须强制终止，父进程按成功返回。"""
    write_progress([
        {"kind": "start", "utc": "t0", "run_id": RUN_ID, "stage": "data_generation",
         "done": 0, "total": 2, "percent": 0.0, "elapsed_s": 0.0},
        end_record(),
    ])
    process = FakeProcess([None, None, None, None])
    monkeypatch.setattr(MAIN.subprocess, "Popen", lambda *a, **k: process)
    code = MAIN.run_data_generation_isolated(data_namespace(), timeout_s=5.0)
    assert process.kill_calls == 1
    assert MAIN.data_run_completed(RUN_ID) is True
    assert code == -9


def test_isolated_run_reports_failure_when_child_writes_error_record(monkeypatch):
    """子进程写入失败记录时父进程必须报错，而不是当成完成。"""
    write_progress([error_record()])
    process = FakeProcess([None, None, None, None])
    monkeypatch.setattr(MAIN.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="数据生成失败"):
        MAIN.run_data_generation_isolated(
            data_namespace(), timeout_s=5.0, max_attempts=1
        )
    assert process.kill_calls == 1


def test_isolated_run_reports_timeout(monkeypatch):
    """子进程长期不写结束记录时必须在超时后终止并报错。"""
    process = FakeProcess([None] * 8)
    monkeypatch.setattr(MAIN.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(TimeoutError, match="仍未写入结束记录"):
        MAIN.run_data_generation_isolated(data_namespace(), timeout_s=0.0)
    assert process.kill_calls == 1


def test_isolated_run_reports_incomplete_when_child_dies_without_end_record(monkeypatch):
    """子进程退出但没有结束记录时必须报未完成，不得判为成功。"""
    process = FakeProcess([1])
    monkeypatch.setattr(MAIN.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="数据生成未完成"):
        MAIN.run_data_generation_isolated(
            data_namespace(), timeout_s=5.0, max_attempts=1
        )
    assert process.kill_calls == 0


def test_isolated_run_retries_after_failure_record(monkeypatch):
    """首次尝试写入失败记录后必须重新启动子进程，第二次成功即判为完成。"""
    write_progress([error_record()])
    processes: list[FakeProcess] = []
    popen_kwargs: list[dict] = []

    def fake_popen(*args, **kwargs):
        """第一次返回失败进程，第二次在启动时写入结束记录模拟成功。"""
        popen_kwargs.append(kwargs)
        if len(processes) == 1:
            write_progress([end_record()])
        processes.append(FakeProcess([None, None]))
        return processes[-1]

    monkeypatch.setattr(MAIN.subprocess, "Popen", fake_popen)
    code = MAIN.run_data_generation_isolated(data_namespace(), timeout_s=5.0)

    assert len(processes) == 2
    assert processes[0].kill_calls == 1
    assert code == -9
    archived = MAIN.progress_path(RUN_ID).with_name("progress.attempt1.jsonl")
    assert archived.is_file()
    assert popen_kwargs[0]["stdout"] is not None
    assert popen_kwargs[0]["stderr"] == MAIN.subprocess.STDOUT


def test_failure_message_includes_child_log_tail(monkeypatch):
    """失败信息必须带上子进程日志末尾，使原生引擎的报错可见。"""
    write_progress([error_record(note="RuntimeError: OpenDSS 基态求解失败")])
    log_path = MAIN.child_log_path(RUN_ID)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Runtime error 216 at 00000070E27E79\n", encoding="utf-8")
    process = FakeProcess([None, None, None, None])
    monkeypatch.setattr(MAIN.subprocess, "Popen", lambda *a, **k: process)

    with pytest.raises(RuntimeError) as excinfo:
        MAIN.run_data_generation_isolated(
            data_namespace(), timeout_s=5.0, max_attempts=1
        )

    message = str(excinfo.value)
    assert "数据生成失败" in message
    assert "已尝试 1 次" in message
    assert "OpenDSS 基态求解失败" in message
    assert "Runtime error 216" in message
