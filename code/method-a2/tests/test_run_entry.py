"""运行入口 main.py 的参数映射、路径解析、进度显示与参数漂移测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

METHOD_DIR = Path(__file__).absolute().parents[1]
for _extra_path in (str(METHOD_DIR), str(METHOD_DIR / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

MAIN_PATH = METHOD_DIR / "main.py"
# 入口模块含 dataclass，加载前必须先注册到 sys.modules 供 dataclasses 解析注解
SPEC = importlib.util.spec_from_file_location("method_a2_main", MAIN_PATH)
MAIN = importlib.util.module_from_spec(SPEC)
sys.modules["method_a2_main"] = MAIN
SPEC.loader.exec_module(MAIN)

from generate_dataset import parse_args as parse_data_args  # noqa: E402
from train_marker_mvp import parse_args as parse_train_args  # noqa: E402


def script_actions(parse_args) -> dict:
    """读取阶段脚本自身的参数定义，用于与本入口的参数定义比对。

    脚本的 `parse_args()` 只读 `sys.argv`，因此临时替换
    `ArgumentParser.parse_args`，在解析前截获其参数定义。
    """
    import argparse

    captured: list = []
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):  # noqa: ANN001 - 测试替身
        captured.extend(self._actions)
        return argparse.Namespace()

    argparse.ArgumentParser.parse_args = capture
    try:
        parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original
    return {action.dest: action for action in captured}


def entry_actions(parser) -> dict:
    """取出本入口解析器的参数定义。"""
    return {action.dest: action for action in parser._actions}


def parse_data(command: list[str]):
    """按数据生成阶段解析参数。"""
    return MAIN.parse_stage_arguments(
        MAIN.build_data_parser(), command, MAIN.DATA_CONFIG_FIELDS
    )


def parse_train(command: list[str]):
    """按训练阶段解析参数。"""
    return MAIN.parse_stage_arguments(
        MAIN.build_train_parser(), command, MAIN.TRAIN_CONFIG_FIELDS
    )


def test_entry_parsers_cover_scripts_parameters():
    """入口解析器必须逐项覆盖脚本的参数名、取值集合与多值形式。"""
    for builder, parse_args in (
        (MAIN.build_data_parser, parse_data_args),
        (MAIN.build_train_parser, parse_train_args),
    ):
        script = script_actions(parse_args)
        entry = entry_actions(builder())
        missing = sorted(set(script) - set(entry))
        assert missing == [], f"入口解析器缺少脚本参数：{missing}"
        for name, action in script.items():
            if action.choices is not None:
                assert tuple(entry[name].choices) == tuple(action.choices), name
            if action.nargs == "+":
                assert entry[name].nargs == "+", name
            if action.option_strings and action.dest not in (
                MAIN.COMMON_FIELDS + MAIN.DATA_FIELDS + MAIN.TRAIN_FIELDS
            ):
                assert entry[name].option_strings == action.option_strings, name


def test_entry_parsers_match_script_defaults():
    """入口解析器的默认值必须来自配置文件，并与配置文件取值一致。"""
    data = entry_actions(MAIN.build_data_parser())
    train = entry_actions(MAIN.build_train_parser())
    assert data["case_name"].default == MAIN.COMMON.case_name
    assert data["seed"].default == MAIN.COMMON.seed
    assert data["samples_per_bus"].default == MAIN.DATA.samples_per_bus
    assert data["hard_pair_max_attempts"].default == MAIN.DATA.hard_pair_max_attempts
    assert train["seed"].default == MAIN.COMMON.seed
    assert train["data_dir"].default == MAIN.resolve_path(MAIN.TRAIN.data_dir)
    assert train["learning_rate"].default == MAIN.TRAIN.learning_rate
    assert train["pooling"].default == MAIN.TRAIN.pooling
    assert train["temporal_mode"].default == MAIN.TRAIN.temporal_mode


def test_case_name_is_not_treated_as_path():
    """算例名不是路径，不得被解析为方法目录下的路径。"""
    assert parse_data([]).case_name == "ieee13"


def test_relative_paths_resolve_against_method_dir():
    """相对路径一律相对方法目录解析，与运行时工作目录无关。"""
    assert MAIN.resolve_path("data/s0-lg-run3") == METHOD_DIR / "data" / "s0-lg-run3"
    absolute = METHOD_DIR / "data" / "s0-lg-run3"
    assert MAIN.resolve_path(absolute) == absolute
    assert MAIN.resolve_path(None) is None


def test_config_values_reach_the_stage_namespace(monkeypatch):
    """配置文件取值必须逐项进入传给阶段函数的参数对象。"""
    monkeypatch.setattr(MAIN.TRAIN, "epochs", 3)
    monkeypatch.setattr(MAIN.TRAIN, "pooling", "trimmed_mean")
    monkeypatch.setattr(MAIN.TRAIN, "derived_channels", True)
    monkeypatch.setattr(MAIN.TRAIN, "max_batches", 1)
    monkeypatch.setattr(MAIN.COMMON, "run_id", "unittest-run")
    train_args = parse_train([])
    assert train_args.epochs == 3
    assert train_args.pooling == "trimmed_mean"
    assert train_args.derived_channels is True
    assert train_args.max_batches == 1
    assert train_args.run_id == "unittest-run"
    assert train_args.data_dir == METHOD_DIR / "data" / "s0-lg-run3"
    assert train_args.output_dir is None

    monkeypatch.setattr(MAIN.DATA, "samples_per_bus", 2)
    monkeypatch.setattr(MAIN.DATA, "val_ratio", 0.2)
    data_args = parse_data([])
    assert data_args.samples_per_bus == 2
    assert data_args.val_ratio == pytest.approx(0.2)
    assert data_args.fault_classes == [0]
    assert data_args.output_dir == METHOD_DIR / "data" / "s4-high-impedance-100-1000"


def test_command_line_overrides_config_file():
    """命令行覆盖配置文件时以命令行为准，便于从 PyCharm 临时改参。"""
    train_args = parse_train(["--epochs", "7", "--pooling", "mean"])
    assert train_args.epochs == 7
    assert train_args.pooling == "mean"
    data_args = parse_data(["--samples-per-bus", "4", "--fault-classes", "0"])
    assert data_args.samples_per_bus == 4
    assert data_args.fault_classes == [0]
    assert data_args.case_name == MAIN.COMMON.case_name


def test_unknown_stage_option_fails_loudly():
    """当前阶段不存在的参数必须立即报错，而不是被静默忽略。"""
    with pytest.raises(SystemExit):
        parse_train(["--samples-per-bus", "4"])
    with pytest.raises(SystemExit):
        parse_data(["--epochs", "4"])


def test_unused_config_field_is_reported(monkeypatch, capsys):
    """与当前阶段无关且被改动的配置字段只警告，不报错。"""
    monkeypatch.setattr(MAIN.TRAIN, "resplit", True)
    MAIN.warn_unused_config(MAIN.TRAIN, MAIN.TRAIN_FIELDS, "数据生成")
    assert "[警告] 以下字段在数据生成阶段不生效" in capsys.readouterr().out


def test_misplaced_config_field_fails_loudly():
    """配置字段未在解析器中声明时必须报错，避免配置被静默丢弃。"""
    parser = MAIN.build_data_parser()
    parser._actions = [
        action for action in parser._actions if action.dest != "samples_per_bus"
    ]
    with pytest.raises(ValueError, match="配置字段未在参数解析器中声明"):
        MAIN.parse_stage_arguments(
            parser, [], MAIN.COMMON_FIELDS + MAIN.DATA_FIELDS
        )


def test_progress_display_formats_stage_and_eta():
    """进度显示包含阶段、完成度、已用时长与 ETA，并能接受缺失的 ETA。"""
    path = Path("logs") / "unittest" / "progress.jsonl"
    line = MAIN.format_progress(
        {
            "kind": "unit",
            "stage": "training",
            "done": 3,
            "total": 12,
            "percent": 25.0,
            "elapsed_s": 1.5,
            "eta_s": 4.5,
            "note": "epoch=1/2",
        },
        "unittest-run",
        path,
    )
    assert "阶段 training" in line
    assert "3/12" in line
    assert "25.00%" in line
    assert "已用 1.5s" in line
    assert "ETA 4.5s" in line
    assert "进度文件" in line
    empty_eta = MAIN.format_progress(
        {"kind": "start", "stage": "training", "done": 0, "total": None},
        "unittest-run",
        path,
    )
    assert "ETA --" in empty_eta
    assert "0/?" in empty_eta
    assert MAIN.format_seconds(3725.0) == "1h02m"
    assert MAIN.format_seconds(125.0) == "2m05s"
    assert MAIN.format_seconds(None) == "--"


def test_progress_writer_writes_record_and_displays_line(tmp_path, capsys):
    """进度写入器保持落盘行为，同时把进度打到控制台。"""
    writer = MAIN.make_progress_writer("unittest-run")
    assert writer.path == MAIN.METHOD_DIR / "logs" / "unittest-run" / "progress.jsonl"
    writer.update("training", "batch", 1, 4, force=True)
    output = capsys.readouterr().out
    assert "[进度]" in output
    assert "阶段 training" in output
    assert "1/4" in output
    for candidate in (
        writer.path,
        writer.path.with_name("progress.latest.json"),
    ):
        assert candidate.is_file()
        candidate.unlink()


def test_data_arguments_keep_two_way_split_by_default():
    """默认不给出 val_ratio 时保持两段划分，并拒绝非法故障类型。"""
    with pytest.raises(SystemExit):
        parse_data(["--fault-classes", "9"])
    namespace = parse_data([])
    assert namespace.val_ratio is None
    assert namespace.test_ratio == pytest.approx(0.15)
    assert namespace.output_dir == METHOD_DIR / "data" / "s4-high-impedance-100-1000"


def test_s4_data_configuration_uses_log_uniform_high_impedance_defaults():
    """入口默认配置应对应独立 S4 高阻数据生成，而不是训练阶段。"""
    namespace = parse_data([])
    assert namespace.samples_per_bus == 20
    assert namespace.normal_ratio == pytest.approx(0.8)
    assert namespace.res_min == pytest.approx(100.0)
    assert namespace.res_max == pytest.approx(1000.0)
    assert namespace.impedance_sampling == "log_uniform"
    assert namespace.output_dir == METHOD_DIR / "data" / "s4-high-impedance-100-1000"


def test_run_id_defaults_to_stage_time_and_seed():
    """未显式给出运行标识时，按阶段、时间与种子生成。"""
    assert MAIN.build_run_id("train", 342, "explicit") == "explicit"
    generated = MAIN.build_run_id("train", 342, None)
    assert generated.startswith("train-")
    assert generated.endswith("-seed342")
    assert MAIN.progress_path(generated).name == "progress.jsonl"
