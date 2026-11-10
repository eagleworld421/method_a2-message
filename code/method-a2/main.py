"""Method-A2 统一运行入口：在文件顶部设置参数后直接运行数据生成或训练。

本程序的定位是替代手工拼接命令行：把 `scripts/generate_dataset.py` 与
`scripts/train_marker_mvp.py` 的参数集中到本文件顶部的配置块中，运行时先打印本次
实际使用的参数，再用这些参数直接调用两个脚本原有的执行函数。

使用方式（PyCharm 直接对该文件点运行，或在项目根目录执行
`python code/method-a2/main.py`）：

1. 在顶部的 `CommonConfig` 与对应阶段配置对象里改动参数；
2. 用 `MODE` 选择 `"data"`、`"train"` 或 `"all"`；
3. 运行后在控制台查看实时进度，进度记录同时落盘到 `logs/<run-id>/progress.jsonl`。

约定与边界：

- 参数含义、取值范围与校验规则沿用两个阶段脚本的 argparse 定义，本文件只为同一
  组参数提供"由配置文件提供默认值"的入口；配置字段名与 CLI 参数名逐字一致。
- `tests/test_run_entry.py` 会把本文件的参数定义与两个脚本的命令行程式逐项比对，
  若脚本侧的参数被改名、改变默认值或改变取值集合，测试会失败而不是静默使用旧值。
- `CommonConfig.case_name` 是 OpenDSS 算例名（如 `ieee13`），不是路径；配置中的
  其余路径若为相对路径，一律相对本方法目录解析，因此与运行时的工作目录无关。
- 控制台输出同时写入 `logs/main-<mode>.log`，便于运行结束后复查。
- 本程序不新增产物：数据、模型与报告仍由既有脚本写入 `data/`、`output/` 与 `logs/`。
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TextIO

METHOD_DIR = Path(__file__).absolute().parents[0]
for _extra_path in (METHOD_DIR, METHOD_DIR / "scripts"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from generate_dataset import main as run_data_generation
from train_marker_mvp import run as run_training

from src.data_generation.opendss_sim import FAULT_CLASSES
from src.progress import ProgressWriter

# ============================================================================
# 运行模式：`"data"` 只生成数据；`"train"` 只训练；`"all"` 先生成数据再训练
# ============================================================================
MODE = "data"

POOLING_CHOICES = ("mean", "trimmed_mean", "mean_median_concat")
TEMPORAL_MODE_CHOICES = ("single", "dual")


@dataclass
class CommonConfig:
    """两个阶段共用的参数。"""

    case_name: str = "ieee13"
    """OpenDSS 算例名，不是路径。"""

    seed: int = 342
    """数据生成与训练共用的随机种子。"""

    run_id: str | None = None
    """运行标识；为 None 时按阶段、时间与种子自动生成，日志目录用它命名。"""


@dataclass
class DataConfig:
    """数据生成参数，字段名与 generate_dataset.py 的命令行参数一一对应。"""

    output_dir: Path = Path("data/s4-high-impedance-100-1000")
    """数据输出目录。"""

    samples_per_bus: int = 20
    """每个母线的故障样本数，同时等于反事实族（group）的数量。"""

    normal_ratio: float = 0.8
    """无故障样本数与故障样本数之比。"""

    val_ratio: float | None = None
    """给出时按训练/验证/测试三段划分且验证集只含故障样本；为 None 时两段划分。"""

    test_ratio: float = 0.15
    """三段划分时测试集分组占比。"""

    fault_classes: tuple[int, ...] = (0,)
    """事件可用的故障类型；默认只用单相接地，以保证全部母线都能承载。"""

    hard_pair_file: Path | None = None
    """困难候选对 JSON 路径；给出时启用主动困难候选采样。"""

    hard_pair_max_attempts: int = 12
    """每个事件族为命中困难候选对的最大重采样次数。"""

    res_min: float = 100.0
    """S4 故障阻抗下界，单位为欧姆。"""

    res_max: float = 1000.0
    """S4 故障阻抗上界，单位为欧姆。"""

    impedance_sampling: str = "log_uniform"
    """故障阻抗采样方式；S4 使用对数均匀采样。"""


@dataclass
class TrainConfig:
    """训练参数，字段名与 train_marker_mvp.py 的命令行参数一一对应。"""

    data_dir: Path = Path("data/s0-lg-run3")
    """输入数据目录，必须包含 meta.json 与 feature_scaler.npz。"""

    output_dir: Path | None = None
    """产物目录；为 None 时写入 output/node-marker-mvp/<run-id>/。"""

    epochs: int = 200
    batch_size: int = 16
    max_batches: int = 0
    """每个轮次最多使用的批次数；0 表示不限制。"""

    learning_rate: float = 1e-3
    lambda_rank: float = 0.1
    lambda_response: float = 0.1
    rank_margin: float = 0.1
    rank_temperature: float = 1.0
    uncertainty_weight: float = 1.0
    logvar_clamp: float = 8.0

    temporal_layers: int = 3
    """时间编码器残差块层数。"""

    encoder_layers: int = 2
    """共享图编码器消息传递层数。"""

    pooling: str = "mean"
    """候选池化方式；可选 mean、trimmed_mean、mean_median_concat。"""

    temporal_mode: str = "single"
    """时间编码尺度；可选 single、dual。"""

    derived_channels: bool = False
    """是否加入派生观测通道。"""

    uncertainty: bool = False
    """是否启用候选响应不确定性分支。"""

    val_ratio: float = 0.15
    """重新划分时的验证集占比，仅在 resplit 为真时生效。"""

    test_ratio: float = 0.15
    """重新划分时的测试集占比，仅在 resplit 为真时生效。"""

    resplit: bool = False
    """为真时忽略数据目录内置划分，按分组重新划分。"""

    device: str = "cpu"
    """训练设备，例如 cpu 或 cuda。"""


COMMON = CommonConfig()
DATA = DataConfig()
TRAIN = TrainConfig()

COMMON_FIELDS = tuple(field.name for field in fields(CommonConfig))
DATA_FIELDS = tuple(field.name for field in fields(DataConfig))
TRAIN_FIELDS = tuple(field.name for field in fields(TrainConfig))

# 本入口为各阶段提供默认值的配置字段：训练阶段不使用 OpenDSS 算例名
DATA_CONFIG_FIELDS = COMMON_FIELDS + DATA_FIELDS
TRAIN_CONFIG_FIELDS = ("seed", "run_id") + TRAIN_FIELDS


class LogTee:
    """把控制台输出同时写入日志文件的转发流。"""

    def __init__(self, stream: TextIO, handle: TextIO) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        """同时写入日志文件与控制台，并立即刷新以保证长任务期间可见。"""
        self._handle.write(text)
        self._handle.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        """刷新两个目标流。"""
        self._handle.flush()
        self._stream.flush()


def make_utf8_stream(stream: TextIO) -> TextIO | None:
    """把文本流重建为 UTF-8 且编码失败时替换字符的新流，不支持时返回 None。"""
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        return None
    try:
        return io.TextIOWrapper(
            buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except (ValueError, OSError):
        return None


def ensure_console_encodable() -> None:
    """确保控制台能输出本程序的中文文本。

    Windows 控制台默认编码可能是 GBK，无法表示部分符号；本函数优先把标准输出与
    标准错误改为 UTF-8，若当前环境不支持重建文本流，则退回可替换错误的显示方式，
    保证长任务的进度输出不会因编码问题中断。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except (ValueError, OSError, LookupError):
                pass
        replaced = make_utf8_stream(stream)
        if replaced is not None:
            setattr(sys, name, replaced)


def log(message: str = "") -> None:
    """打印一行控制台消息，与既有脚本的 print 输出保持一致。"""
    print(message, flush=True)


def add_progress_arguments(parser: argparse.ArgumentParser) -> None:
    """添加与阶段脚本同名的进度参数。

    本入口始终自行构造进度写入器（`terminal=False`）并负责控制台显示，因此
    `--no-terminal-progress` 默认取真，避免脚本自身再输出一条覆盖式进度行。
    """
    parser.add_argument("--progress", type=Path, default=None,
                        help="进度文件路径；默认为 logs/<run-id>/progress.jsonl")
    parser.add_argument("--no-terminal-progress", action="store_true", default=True,
                        help="不把进度写往终端；本入口已在控制台显示进度")


def resolve_path(path: Path | str | None) -> Path | None:
    """把配置中的相对路径解析为相对本方法目录的绝对路径。"""
    if path is None:
        return None
    resolved = Path(path)
    return resolved if resolved.is_absolute() else (METHOD_DIR / resolved).resolve()


def absolute_from_cwd(path: Path | str | None) -> Path | None:
    """把命令行给出的相对路径解析为相对当前工作目录的绝对路径。"""
    if path is None:
        return None
    return Path(path).absolute()


def build_data_parser() -> argparse.ArgumentParser:
    """构造数据生成参数解析器，默认值来自配置文件与数据生成脚本。"""
    parser = argparse.ArgumentParser(
        description="生成 Method-A2 S0 数据",
        argument_default=argparse.SUPPRESS,
    )
    # 共用参数：默认值来自 CommonConfig
    parser.add_argument("--case-name", default=COMMON.case_name)
    parser.add_argument("--seed", type=int, default=COMMON.seed)
    parser.add_argument("--run-id", default=COMMON.run_id)
    # 数据生成参数：默认值来自 DataConfig
    parser.add_argument("--output-dir", type=Path, default=resolve_path(DATA.output_dir))
    parser.add_argument("--samples-per-bus", type=int, default=DATA.samples_per_bus)
    parser.add_argument("--normal-ratio", type=float, default=DATA.normal_ratio)
    parser.add_argument("--val-ratio", type=float, default=DATA.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=DATA.test_ratio)
    parser.add_argument("--res-min", type=float, default=DATA.res_min)
    parser.add_argument("--res-max", type=float, default=DATA.res_max)
    parser.add_argument("--impedance-sampling", choices=("uniform", "log_uniform"),
                        default=DATA.impedance_sampling)
    parser.add_argument(
        "--fault-classes", type=int, nargs="+", default=list(DATA.fault_classes),
        choices=sorted(FAULT_CLASSES),
        help="事件可用的故障类型；默认只用单相接地故障以保证全母线可承载",
    )
    parser.add_argument(
        "--hard-pair-file", type=Path, default=resolve_path(DATA.hard_pair_file),
        help="困难候选对 JSON（hard_negative_pairs.json）；给出时启用主动困难候选采样",
    )
    parser.add_argument(
        "--hard-pair-max-attempts", type=int, default=DATA.hard_pair_max_attempts,
        help="每个事件族为命中困难候选对的最大重采样次数",
    )
    add_progress_arguments(parser)
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    """构造训练参数解析器，默认值来自配置文件与训练脚本。"""
    parser = argparse.ArgumentParser(
        description="训练 Method-A2 Node Marker MVP",
        argument_default=argparse.SUPPRESS,
    )
    # 共用参数：默认值来自 CommonConfig
    parser.add_argument("--seed", type=int, default=COMMON.seed)
    parser.add_argument("--run-id", default=COMMON.run_id)
    # 训练参数：默认值来自 TrainConfig
    parser.add_argument("--data-dir", type=Path, default=resolve_path(TRAIN.data_dir))
    parser.add_argument(
        "--output-dir", type=Path,
        default=None if TRAIN.output_dir is None else resolve_path(TRAIN.output_dir),
    )
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--max-batches", type=int, default=TRAIN.max_batches)
    parser.add_argument("--learning-rate", type=float, default=TRAIN.learning_rate)
    parser.add_argument("--lambda-rank", type=float, default=TRAIN.lambda_rank)
    parser.add_argument("--lambda-response", type=float, default=TRAIN.lambda_response)
    parser.add_argument("--rank-margin", type=float, default=TRAIN.rank_margin)
    parser.add_argument("--rank-temperature", type=float, default=TRAIN.rank_temperature)
    parser.add_argument("--uncertainty-weight", type=float, default=TRAIN.uncertainty_weight)
    parser.add_argument("--logvar-clamp", type=float, default=TRAIN.logvar_clamp)
    parser.add_argument("--temporal-layers", type=int, default=TRAIN.temporal_layers,
                        help="时间编码器残差块层数")
    parser.add_argument("--encoder-layers", type=int, default=TRAIN.encoder_layers,
                        help="共享图编码器消息传递层数")
    parser.add_argument("--pooling", choices=POOLING_CHOICES, default=TRAIN.pooling)
    parser.add_argument("--temporal-mode", choices=TEMPORAL_MODE_CHOICES,
                        default=TRAIN.temporal_mode)
    parser.add_argument("--derived-channels", action="store_true",
                        default=TRAIN.derived_channels)
    parser.add_argument("--uncertainty", action="store_true", default=TRAIN.uncertainty)
    parser.add_argument("--val-ratio", type=float, default=TRAIN.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=TRAIN.test_ratio)
    parser.add_argument("--resplit", action="store_true", default=TRAIN.resplit,
                        help="忽略数据目录内置划分，按分组重新划分")
    parser.add_argument("--device", default=TRAIN.device)
    add_progress_arguments(parser)
    return parser


def parse_stage_arguments(
    parser: argparse.ArgumentParser,
    command: list[str],
    config_names: tuple[str, ...],
) -> argparse.Namespace:
    """解析命令行覆盖项，并校验本阶段的配置字段都在解析器中声明。"""
    namespace = parser.parse_args(command)
    declared = {action.dest for action in parser._actions}
    misplaced = [name for name in config_names if name not in declared]
    if misplaced:
        raise ValueError(f"配置字段未在参数解析器中声明：{misplaced}")
    return namespace


def warn_unused_config(source: object, names: tuple[str, ...], stage: str) -> None:
    """警告与当前阶段无关且被改动的配置字段，避免设置被静默忽略。"""
    defaults = type(source)()
    changed = [
        name for name in names if getattr(source, name) != getattr(defaults, name)
    ]
    if changed:
        log(f"[警告] 以下字段在{stage}阶段不生效，改动已被忽略：{changed}")


def build_run_id(stage: str, seed: int, run_id: str | None) -> str:
    """返回本次运行的标识，未显式给出时按阶段、时间与种子生成。"""
    if run_id:
        return run_id
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stage}-{stamp}-seed{seed}"


def stage_run_id(mode: str, category: str, base_run_id: str) -> str:
    """返回某阶段的运行标识；`all` 模式下两个阶段必须使用不同标识。"""
    if mode == "all":
        return f"{base_run_id}-{category}"
    return base_run_id


def stage_output_dir(mode: str, category: str, output_dir: Path | str | None):
    """返回某阶段的产物目录；`all` 模式下两个阶段必须写入不同目录。"""
    if output_dir is None or mode != "all":
        return output_dir
    resolved = Path(output_dir)
    return resolved.with_name(f"{resolved.name}-{category}")


def progress_path(run_id: str) -> Path:
    """返回运行进度文件路径。"""
    return METHOD_DIR / "logs" / run_id / "progress.jsonl"


def latest_progress_path(run_id: str) -> Path:
    """返回进度快照文件路径，供父进程读取子进程进度。"""
    return progress_path(run_id).with_name("progress.latest.json")


def read_latest_progress(run_id: str) -> dict | None:
    """读取进度快照；文件不存在或写入不完整时返回 None。"""
    path = latest_progress_path(run_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def data_run_completed(run_id: str) -> bool:
    """判断数据生成是否已写入结束记录，即产物落盘完成的标志。"""
    path = progress_path(run_id)
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        return record.get("kind") == "end"
    return False


def data_run_failed(run_id: str) -> bool:
    """判断数据生成是否已写入失败记录。"""
    path = progress_path(run_id)
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        return record.get("kind") == "error"
    return False


def data_arguments_command(args: argparse.Namespace) -> list[str]:
    """把数据生成参数还原为子进程命令行参数。

    路径以绝对路径传递：父进程与子进程的工作目录不保证相同，绝对路径可避免把
    数据写到意外位置。命令行本身给出的相对路径按标准语义相对当前工作目录解析。
    """
    command = [
        "--case-name", str(args.case_name),
        "--seed", str(args.seed),
        "--run-id", str(args.run_id),
        "--output-dir", str(absolute_from_cwd(args.output_dir)),
        "--samples-per-bus", str(args.samples_per_bus),
        "--normal-ratio", str(args.normal_ratio),
        "--test-ratio", str(args.test_ratio),
        "--res-min", str(args.res_min),
        "--res-max", str(args.res_max),
        "--impedance-sampling", str(args.impedance_sampling),
        "--fault-classes", *[str(value) for value in args.fault_classes],
        "--hard-pair-max-attempts", str(args.hard_pair_max_attempts),
        "--no-terminal-progress",
    ]
    if args.val_ratio is not None:
        command += ["--val-ratio", str(args.val_ratio)]
    if args.hard_pair_file is not None:
        command += ["--hard-pair-file", str(absolute_from_cwd(args.hard_pair_file))]
    return command


def child_log_path(run_id: str) -> Path:
    """返回子进程控制台日志路径，用于承载原生引擎写到标准错误的输出。"""
    return progress_path(run_id).with_name("child.log")


def read_child_log_tail(run_id: str, limit: int = 12) -> str:
    """读取子进程控制台日志末尾的非空行，供失败时报告真实原因。"""
    path = child_log_path(run_id)
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = [line.strip() for line in lines if line.strip()][-limit:]
    return "\n".join(tail)


def archive_previous_progress(run_id: str, attempt: int) -> None:
    """重试前把上一次尝试的进度文件改名保留。

    进度文件以追加方式写入，若不清走，上一次尝试的失败记录会被父进程当成本次尝试
    的结果，从而在下一次尝试刚启动时就被判为失败。改名而不是删除，是为了保留每次
    尝试的原始记录。
    """
    for path in (progress_path(run_id), latest_progress_path(run_id)):
        if not path.is_file():
            continue
        target = path.with_name(f"{path.stem}.attempt{attempt}{path.suffix}")
        try:
            path.replace(target)
        except OSError:
            path.unlink(missing_ok=True)


def describe_attempt(run_id: str, status: str, code: int | None) -> str:
    """把一次子进程尝试的结果描述为可读文本。"""
    reason = {
        "failed": "子进程写入失败记录",
        "incomplete": "子进程在写入结束记录前退出",
    }.get(status, status)
    record = read_latest_progress(run_id) or {}
    parts = [reason, f"退出码 {code}"]
    if record.get("note"):
        parts.append(f"记录原因：{record['note']}")
    return "；".join(parts)


def run_child_attempt(
    data_args: argparse.Namespace, run_id: str, timeout_s: float, attempt: int
) -> tuple[str, int | None, bool]:
    """执行一次子进程数据生成并轮询进度，返回 (结果状态, 退出码, 是否强制终止)。"""
    command = [sys.executable, str(Path(__file__).resolve()), "--child-data"]
    command += data_arguments_command(data_args)
    log(f"数据生成在子进程中执行：{command}")
    log_path = child_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_seen: tuple | None = None
    killed = False
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            f"===== 第 {attempt} 次尝试 "
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n"
        )
        handle.flush()
        # 子进程的标准输出与标准错误写入日志文件而不是控制台：OpenDSS 引擎在
        # 进程收尾时会把原生的 "Runtime error 216" 一类消息写到标准错误，直接
        # 继承到控制台会被误读成本次运行的失败原因。
        process = subprocess.Popen(
            command,
            cwd=str(METHOD_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while True:
            record = read_latest_progress(run_id)
            if record is not None:
                marker = (record.get("utc"), record.get("done"))
                if marker != last_seen:
                    last_seen = marker
                    log(format_progress(record, run_id, progress_path(run_id)))
            if data_run_completed(run_id):
                break
            if process.poll() is not None:
                break
            if data_run_failed(run_id):
                process.kill()
                process.wait()
                return "failed", process.returncode, False
            if time.perf_counter() - started > timeout_s:
                process.kill()
                process.wait()
                raise TimeoutError(
                    f"数据生成超过 {timeout_s:.0f} 秒仍未写入结束记录，已终止子进程"
                )
            time.sleep(0.5)

        if process.poll() is None:
            process.kill()
            process.wait()
            killed = True
        code = process.returncode

    if data_run_completed(run_id):
        return "completed", code, killed
    return "incomplete", code, killed


def run_data_generation_isolated(
    data_args: argparse.Namespace,
    timeout_s: float = 7200.0,
    max_attempts: int = 2,
) -> int:
    """在子进程中运行数据生成，并由父进程判定结果。

    以 OpenDSS COM 求解数据的进程无法正常退出：引擎被 `Dispatch` 加载后，解释器
    收尾会以访问冲突（0xC0000005）或无限挂起结束，`CoUninitialize`、`os._exit` 与
    保留引用均无法避免。因此数据生成放在子进程执行，父进程只轮询进度文件：子进程
    写完结束记录即视为成功，父进程随即强制终止子进程并自行正常退出。

    引擎本身是原生 Delphi 组件，偶发在首次求解时就抛出异常（控制台表现为
    `Runtime error 216 at ...`，Windows 应用日志记为 OpenDSSEngine.dll 访问冲突）。
    该失败与数据内容无关且不可预测，因此一次尝试失败后会重新启动子进程重试，最多
    `max_attempts` 次；超时不重试，避免把一次卡死放大成两倍等待。重试会从头重新
    求解，已堆积在子进程内存中的样本随之丢弃。

    返回子进程的原始退出码，仅用于如实记录，不参与成功判定。
    """
    run_id = str(data_args.run_id)
    if max_attempts < 1:
        raise ValueError("max_attempts 必须为正整数")
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            archive_previous_progress(run_id, attempt - 1)
        started = time.perf_counter()
        status, code, killed = run_child_attempt(
            data_args, run_id, timeout_s, attempt
        )
        if status == "completed":
            log(
                f"数据生成完成，已用 {format_seconds(time.perf_counter() - started)}；"
                f"子进程退出码 {code}"
                + ("（子进程挂起，已强制终止）" if killed else "")
            )
            return code
        detail = describe_attempt(run_id, status, code)
        if attempt < max_attempts:
            log(f"第 {attempt} 次尝试未完成：{detail}；重新启动子进程重试")
            continue
        label = "失败" if status == "failed" else "未完成"
        tail = read_child_log_tail(run_id)
        raise RuntimeError(
            f"数据生成{label}，已尝试 {max_attempts} 次；{detail}；"
            f"请检查 {console_log_path('data')}、{progress_path(run_id)} 与 "
            f"{child_log_path(run_id)}"
            + (f"\n子进程日志末尾：\n{tail}" if tail else "")
        )


def console_log_path(mode: str) -> Path:
    """返回指定模式的控制台日志路径。"""
    return METHOD_DIR / "logs" / f"main-{mode}.log"


def format_seconds(seconds: float | None) -> str:
    """把秒数格式化为便于阅读的时长；给不出估计时返回 `--`。"""
    if seconds is None:
        return "--"
    value = max(float(seconds), 0.0)
    if value < 60:
        return f"{value:.1f}s"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{remainder:02d}s"


def format_progress(record: dict, run_id: str, path: Path) -> str:
    """把一条进度记录格式化为控制台显示行。"""
    kinds = {
        "start": "开始",
        "unit": "进行",
        "stage_done": "阶段完成",
        "end": "运行结束",
        "error": "失败",
    }
    kind = str(record.get("kind"))
    total = record.get("total")
    percent = record.get("percent")
    parts = [
        "[进度]",
        f"运行标识 {run_id}",
        f"阶段 {record.get('stage')} {kinds.get(kind, kind)}",
        f"{record.get('done', 0)}/{'?' if total is None else total}",
        "--" if percent is None else f"{float(percent):.2f}%",
        f"已用 {format_seconds(record.get('elapsed_s'))}",
        f"ETA {format_seconds(record.get('eta_s'))}",
    ]
    if record.get("note"):
        parts.append(str(record["note"]))
    return " ".join(parts) + f" | 进度文件 {path}"


def make_progress_writer(run_id: str) -> ProgressWriter:
    """构造进度写入器：落盘行为不变，显示方式改为控制台逐行输出。"""
    path = progress_path(run_id)
    writer = ProgressWriter(path, run_id, terminal=False)
    writer._terminal = lambda record: log(format_progress(record, run_id, path))  # type: ignore[method-assign]
    return writer


def check_data_dir(data_dir: Path | str) -> None:
    """在训练前检查数据目录的必要文件，避免启动后才失败。"""
    directory = Path(data_dir)
    missing = [
        name
        for name in ("meta.json", "feature_scaler.npz")
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"数据目录 {directory} 缺少必要文件：{missing}")


def check_hard_pair_file(path: Path | str | None) -> None:
    """在数据生成前检查困难候选对文件是否存在。"""
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(f"困难候选对文件不存在：{path}")


def print_plan(mode: str, data_args, train_args, data_run_id, train_run_id) -> None:
    """打印本次运行实际使用的参数。"""
    log("-" * 72)
    log(f"Method-A2 运行入口 | 模式 {mode} | 方法目录 {METHOD_DIR}")
    if data_args is not None:
        log(f"数据生成 | 运行标识 {data_run_id}")
        log(f"  输出目录：{data_args.output_dir}")
        log(f"  算例与种子：{data_args.case_name} / seed={data_args.seed}")
        log(
            f"  样本设置：samples_per_bus={data_args.samples_per_bus} "
            f"normal_ratio={data_args.normal_ratio} "
            f"fault_classes={data_args.fault_classes}"
        )
        log(
            f"  划分设置：val_ratio={data_args.val_ratio} "
            f"test_ratio={data_args.test_ratio}"
        )
        log(f"  困难候选：{data_args.hard_pair_file}")
        log(f"  进度文件：{progress_path(str(data_run_id))}")
    if train_args is not None:
        log(f"训练 | 运行标识 {train_run_id}")
        log(f"  数据目录：{train_args.data_dir}")
        log(f"  产物目录：{train_args.output_dir}")
        log(
            f"  优化设置：epochs={train_args.epochs} batch_size={train_args.batch_size} "
            f"lr={train_args.learning_rate} max_batches={train_args.max_batches} "
            f"device={train_args.device}"
        )
        log(
            f"  损失权重：rank={train_args.lambda_rank} "
            f"response={train_args.lambda_response} margin={train_args.rank_margin} "
            f"temperature={train_args.rank_temperature} "
            f"uncertainty={train_args.uncertainty_weight} "
            f"logvar_clamp={train_args.logvar_clamp}"
        )
        log(
            f"  模型结构：pooling={train_args.pooling} "
            f"temporal_mode={train_args.temporal_mode} "
            f"temporal_layers={train_args.temporal_layers} "
            f"encoder_layers={train_args.encoder_layers} "
            f"derived_channels={train_args.derived_channels} "
            f"uncertainty={train_args.uncertainty}"
        )
        log(
            f"  数据划分：resplit={train_args.resplit} "
            f"val_ratio={train_args.val_ratio} test_ratio={train_args.test_ratio}"
        )
        log(f"  进度文件：{progress_path(str(train_run_id))}")
    log("-" * 72)


def filter_stage_command(
    parser: argparse.ArgumentParser, command: list[str]
) -> list[str]:
    """只保留当前阶段认识的命令行参数，用于 `all` 模式下按阶段分发。

    透传参数中属于另一阶段的选项会被丢弃，由 `warn_unused_config` 与调用顺序保证
    不会静默改变语义；`--` 分隔符之后的内容一律保留。
    """
    known = {option for action in parser._actions for option in action.option_strings}
    kept: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == "--":
            kept.extend(command[index:])
            break
        if token.startswith("--") and token.split("=", 1)[0] not in known:
            index += 2 if "=" not in token else 1
            continue
        kept.append(token)
        index += 1
    return kept


def run_child_data(command: list[str]) -> int:
    """子进程侧：按父进程传入的参数执行数据生成。

    本函数运行在会被 OpenDSS 引擎卡住退出的子进程里，因此只负责完成数据与进度
    落盘，退出码由父进程按进度记录判定。
    """
    data_args = parse_stage_arguments(
        build_data_parser(), command, DATA_CONFIG_FIELDS
    )
    data_args.run_id = build_run_id("dataset", data_args.seed, data_args.run_id)
    check_hard_pair_file(data_args.hard_pair_file)
    run_data_generation(
        data_args, progress=make_progress_writer(data_args.run_id)
    )
    sys.stdout.flush()
    return 0


def execute(mode: str, data_args, train_args) -> int:
    """按运行模式执行各阶段，并把控制台输出同时写入日志文件。"""
    log_dir = METHOD_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    console_path = console_log_path(mode)
    with console_path.open("w", encoding="utf-8") as handle:
        original = sys.stdout
        sys.stdout = LogTee(original, handle)  # type: ignore[assignment]
        try:
            log(f"控制台完整日志：{console_path}")
            if data_args is not None:
                check_hard_pair_file(data_args.hard_pair_file)
                run_data_generation_isolated(data_args)
            if train_args is not None:
                check_data_dir(train_args.data_dir)
                run_training(
                    train_args, progress=make_progress_writer(train_args.run_id)
                )
        finally:
            sys.stdout = original
    return 0


def parse_entry_args() -> tuple[argparse.Namespace, list[str]]:
    """解析入口参数，其余参数透传给阶段参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Method-A2 数据生成与训练统一入口",
        epilog="其余参数按 --name value 形式透传给当前阶段，例如 --epochs 5。",
    )
    parser.add_argument("--mode", choices=["data", "train", "all"], default=None,
                        help="覆盖文件顶部的 MODE")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印本次实际参数，不执行数据生成或训练")
    parser.add_argument("--child-data", action="store_true",
                        help="内部参数：由父进程调用，在当前进程内执行数据生成")
    return parser.parse_known_args()


def main() -> int:
    """读取顶部配置，打印实际参数并执行所选阶段。"""
    ensure_console_encodable()
    overrides, passthrough = parse_entry_args()
    if overrides.child_data:
        return run_child_data(passthrough)
    mode = overrides.mode or MODE
    if mode not in {"data", "train", "all"}:
        raise SystemExit(f"MODE 取值非法：{mode!r}，只能为 data、train 或 all")

    data_args = None
    train_args = None
    data_run_id = None
    train_run_id = None
    if mode in {"data", "all"}:
        warn_unused_config(TRAIN, TRAIN_FIELDS, "数据生成")
        data_parser = build_data_parser()
        data_args = parse_stage_arguments(
            data_parser,
            filter_stage_command(data_parser, passthrough),
            DATA_CONFIG_FIELDS,
        )
        data_run_id = stage_run_id(
            mode, "data", build_run_id("dataset", data_args.seed, data_args.run_id)
        )
        data_args.run_id = data_run_id
        data_args.output_dir = stage_output_dir(mode, "data", data_args.output_dir)
    if mode in {"train", "all"}:
        warn_unused_config(DATA, DATA_FIELDS, "训练")
        train_parser = build_train_parser()
        train_args = parse_stage_arguments(
            train_parser,
            filter_stage_command(train_parser, passthrough),
            TRAIN_CONFIG_FIELDS,
        )
        train_run_id = stage_run_id(
            mode, "train", build_run_id("train", train_args.seed, train_args.run_id)
        )
        train_args.run_id = train_run_id
        train_args.output_dir = stage_output_dir(mode, "train", train_args.output_dir)
        if train_args.output_dir is None:
            train_args.output_dir = (
                METHOD_DIR / "output" / "node-marker-mvp" / train_run_id
            )

    print_plan(mode, data_args, train_args, data_run_id, train_run_id)
    if overrides.dry_run:
        log("dry-run：仅打印参数，未执行数据生成或训练。")
        return 0
    return execute(mode, data_args, train_args)


if __name__ == "__main__":
    raise SystemExit(main())
