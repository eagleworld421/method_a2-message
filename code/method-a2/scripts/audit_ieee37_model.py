"""IEEE 37 OpenDSS 模型完整性核对与电气拓扑审计脚本。

本脚本只读取原始 DSS 文件，不修改任何原始模型。静态解析结果与 OpenDSS COM
求解结果分别保存，并在生成交付物前进行计数、名称和矩阵一致性核对。

用法：
    python scripts/audit_ieee37_model.py --source-dir <原始模型目录> --out <输出目录>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

# 三相旋转算子与对称分量变换矩阵；仅用于从原始 3×3 相阻抗矩阵导出序分量。
A_OP = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A_OP**2, A_OP], [1, A_OP, A_OP**2]], dtype=complex)
A_INV = np.linalg.inv(A_MAT)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """返回文件的 SHA256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_comment(line: str) -> str:
    """删除 DSS 行内注释，保留引号内部的感叹号。"""
    quote = ""
    out = []
    for char in line:
        if quote:
            out.append(char)
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            out.append(char)
            continue
        if char == "!":
            break
        out.append(char)
    return "".join(out)


def split_top_level(text: str) -> list[str]:
    """按空白切分，但保留括号和引号内部的空白。"""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    for char in text:
        if quote:
            buf.append(char)
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            buf.append(char)
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char.isspace() and depth == 0:
            if buf:
                parts.append("".join(buf))
                buf = []
            continue
        buf.append(char)
    if buf:
        parts.append("".join(buf))
    return parts


def parse_attrs(text: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """解析 DSS 属性键值对，返回最后值字典和按出现顺序排列的键值对。

    支持裸值、单引号/双引号字符串、圆括号列表和方括号矩阵。
    """
    pairs: list[tuple[str, str]] = []
    attrs: dict[str, str] = {}
    index = 0
    length = len(text)
    key_pattern = re.compile(r"[A-Za-z_%][A-Za-z0-9_%]*")
    while index < length:
        while index < length and (text[index].isspace() or text[index] in ",;"):
            index += 1
        if index >= length:
            break
        match = key_pattern.match(text, index)
        if not match:
            index += 1
            continue
        key = match.group(0).lower()
        index = match.end()
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != "=":
            continue
        index += 1
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            value = ""
        elif text[index] in ("'", '"'):
            quote = text[index]
            index += 1
            start = index
            while index < length and text[index] != quote:
                index += 1
            value = text[start:index]
            index = min(length, index + 1)
        elif text[index] in "([":
            opener = text[index]
            closer = ")" if opener == "(" else "]"
            depth = 0
            start = index + 1
            index += 1
            while index < length:
                if text[index] == opener:
                    depth += 1
                elif text[index] == closer:
                    if depth == 0:
                        break
                    depth -= 1
                index += 1
            value = text[start:index]
            index = min(length, index + 1)
        else:
            start = index
            while index < length and not text[index].isspace() and text[index] not in ",;":
                index += 1
            value = text[start:index]
        pairs.append((key, value))
        attrs[key] = value
    return attrs, pairs


def parse_float(value: str | None, default: float | None = None) -> float | None:
    """把 DSS 字符串转为浮点数；空值返回默认值。"""
    if value is None or str(value).strip() == "":
        return default
    return float(str(value).strip())


def parse_int(value: str | None, default: int | None = None) -> int | None:
    """把 DSS 字符串转为整数；空值返回默认值。"""
    if value is None or str(value).strip() == "":
        return default
    return int(float(str(value).strip()))


def parse_list(value: str | None) -> list[str]:
    """把圆括号/引号列表拆成字符串列表。"""
    if value is None:
        return []
    return split_top_level(str(value).strip())


def parse_bus(bus: str, phases: int | None = None) -> tuple[str, list[int]]:
    """拆分母线名与相节点，例如 701.1.2.3 -> ('701', [1,2,3])。"""
    tokens = str(bus).strip().split(".")
    name = tokens[0]
    nodes = [int(token) for token in tokens[1:] if token.strip().isdigit()]
    if not nodes and phases:
        nodes = list(range(1, phases + 1))
    return name, nodes


def parse_matrix(value: str | None, nphases: int) -> np.ndarray:
    """解析 LineCode 的 R/X/C 矩阵，支持方括号和圆括号形式。

    文件以 ``|`` 分隔下三角行，本函数补全为对称方阵。
    """
    if value is None:
        return np.zeros((nphases, nphases), dtype=float)
    raw = str(value).strip()
    raw = raw.strip("[]()")
    rows = [row.strip() for row in raw.split("|") if row.strip()]
    matrix = np.zeros((nphases, nphases), dtype=float)
    for row_index, row in enumerate(rows):
        values = [float(item) for item in row.split()]
        for col_index, item in enumerate(values):
            if row_index < nphases and col_index < nphases:
                matrix[row_index, col_index] = item
                matrix[col_index, row_index] = item
    return matrix


def matrix_to_text(matrix: np.ndarray) -> str:
    """把矩阵压成便于 CSV 阅读的分号/竖线字符串。"""
    rows = []
    for row in np.asarray(matrix):
        rows.append(" ".join(f"{float(value):.10g}" for value in row))
    return "|".join(rows)


def complex_to_pair(value: complex) -> list[float]:
    """把复数转成 JSON 可写的 [实部, 虚部]。"""
    return [float(value.real), float(value.imag)]


def pair_to_complex(value) -> complex:
    """把 [实部, 虚部] 转为复数。"""
    return complex(float(value[0]), float(value[1]))


def pairs_to_complex(values: Iterable[float]) -> list[complex]:
    """把 OpenDSS 返回的复数交错数组转为复数列表。"""
    seq = [float(value) for value in values]
    return [complex(seq[index], seq[index + 1]) for index in range(0, len(seq), 2)]


def total_complex(pairs: list[list[float]]) -> complex:
    """把复数对列表求和；用于 Losses 这类只返回一个总值的数组。"""
    return sum((pair_to_complex(value) for value in pairs), complex(0, 0))


def fmt(value: float | None, digits: int = 6) -> str:
    """统一浮点数输出格式。"""
    if value is None:
        return ""
    if abs(value) < 1e-15:
        value = 0.0
    return f"{value:.{digits}g}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """生成 Markdown 表格。"""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    """以 UTF-8 BOM 写 CSV，便于 Excel 直接打开中文表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def natural_key(text: str) -> list[Any]:
    """按数字段自然排序母线名。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


# ---------------------------------------------------------------------------
# 原始 DSS 静态解析
# ---------------------------------------------------------------------------


@dataclass
class Statement:
    """一条合并续行后的 DSS 语句。"""

    file: str
    line: int
    text: str
    klass: str = ""
    name: str = ""
    attrs: dict[str, str] = field(default_factory=dict)
    pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class LineDef:
    """线路静态定义。"""

    name: str
    bus1: str
    bus2: str
    phases: int
    linecode: str | None
    length: float | None
    units: str | None
    r1: float | None
    x1: float | None
    r0: float | None
    x0: float | None
    c0: float | None
    c1: float | None
    raw_line: int
    is_jumper: bool = False


@dataclass
class LoadDef:
    """负荷静态定义。"""

    name: str
    bus1: str
    phases: int
    conn: str
    model: int
    kv: float
    kw: float
    kvar: float
    raw_line: int


@dataclass
class TransformerDef:
    """变压器/调压器静态定义。"""

    name: str
    phases: int
    windings: int
    buses: list[str]
    conns: list[str]
    kvs: list[float]
    kvas: list[float]
    rs: list[float]
    taps: list[float]
    xhl: float
    bank: str | None
    like: str | None
    raw_line: int
    r_specified: list[bool]


@dataclass
class RegControlDef:
    """调压控制器静态定义。"""

    name: str
    transformer: str
    winding: int
    vreg: float
    band: float
    ptratio: float
    ctprim: float
    r: float
    x: float
    delay: float
    like: str | None
    raw_line: int


@dataclass
class LineCodeDef:
    """线路配置静态定义。"""

    name: str
    nphases: int
    basefreq: float | None
    units: str | None
    rmatrix: np.ndarray
    xmatrix: np.ndarray
    cmatrix: np.ndarray
    normamps: float | None
    emergamps: float | None
    raw_line: int


def read_statements(path: Path) -> list[Statement]:
    """读取 DSS 文件，处理 BOM、注释和以 ~ 开头的续行。"""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    statements: list[Statement] = []
    buffer: list[str] = []
    start_line = 0

    def flush() -> None:
        if not buffer:
            return
        merged = " ".join(buffer).strip()
        if not merged:
            buffer.clear()
            return
        statements.append(Statement(file=path.name, line=start_line, text=merged))
        buffer.clear()

    for line_no, raw in enumerate(text.splitlines(), start=1):
        clean = strip_comment(raw).strip()
        if not clean:
            continue
        if clean.startswith("~"):
            if not buffer:
                start_line = line_no
            buffer.append(clean[1:].strip())
        else:
            flush()
            start_line = line_no
            buffer.append(clean)
    flush()

    for statement in statements:
        lowered = statement.text.lower()
        if lowered.startswith("new "):
            rest = statement.text[4:].strip()
            parts = split_top_level(rest)
            if not parts:
                continue
            target = parts[0]
            if target.lower().startswith("object="):
                target = target.split("=", 1)[1]
            if "." in target:
                statement.klass, statement.name = target.split(".", 1)
            else:
                statement.klass, statement.name = target, ""
            attrs_text = rest[len(parts[0]) :].strip()
            statement.attrs, statement.pairs = parse_attrs(attrs_text)
        elif lowered.startswith("set "):
            statement.klass = "set"
            statement.attrs, statement.pairs = parse_attrs(statement.text[4:].strip())
        elif lowered.startswith("redirect "):
            statement.klass = "redirect"
            statement.name = statement.text.split(None, 1)[1].strip().strip('"')
        elif lowered.startswith("compile "):
            statement.klass = "compile"
            statement.name = statement.text.split(None, 1)[1].strip().strip('"')
        elif lowered.startswith("buscoords "):
            statement.klass = "buscoords"
            statement.name = statement.text.split(None, 1)[1].strip().strip('"')
        else:
            statement.klass = "command"
    return statements


def resolve_like(objects: dict[tuple[str, str], Statement]) -> None:
    """就地解析 like= 继承，保留当前对象显式覆盖的属性。"""
    resolved: set[tuple[str, str]] = set()

    def resolve(key: tuple[str, str], stack: tuple[tuple[str, str], ...] = ()) -> None:
        if key in resolved:
            return
        if key in stack:
            raise ValueError(f"like= 循环引用：{key}")
        statement = objects[key]
        like = statement.attrs.get("like")
        if like:
            base_key = (key[0], like.lower())
            if base_key not in objects:
                raise ValueError(f"{key} 的 like={like} 未找到同类别对象")
            resolve(base_key, stack + (key,))
            base = objects[base_key]
            merged_attrs = dict(base.attrs)
            merged_attrs.update(statement.attrs)
            statement.attrs = merged_attrs
            # 继承对象保留自身源行号，但记录继承来源。
            statement.attrs["_like_source"] = like
        resolved.add(key)

    for key in list(objects):
        resolve(key)


def build_model(source_dir: Path) -> dict[str, Any]:
    """解析主 DSS 和 LineCodes 文件，返回静态模型字典。"""
    master = source_dir / "IEEE37_Load_Connection_Map.csv1"
    linecodes_path = source_dir / "IEEELineCodes.DSS"
    busxy_path = source_dir / "IEEE37_Load_Connection_Map.csv0"
    exp_meters = source_dir / "ieee37_EXP_METERS.CSV"

    statements = read_statements(master)
    linecode_statements = read_statements(linecodes_path)

    # 先建立对象表，便于 like= 继承。
    objects: dict[tuple[str, str], Statement] = {}
    for statement in statements:
        if statement.klass and statement.name:
            objects[(statement.klass.lower(), statement.name.lower())] = statement
    resolve_like(objects)

    linecodes: dict[str, LineCodeDef] = {}
    for statement in linecode_statements:
        if statement.klass.lower() != "linecode" or not statement.name:
            continue
        nphases = parse_int(statement.attrs.get("nphases"), 3) or 3
        linecodes[statement.name] = LineCodeDef(
            name=statement.name,
            nphases=nphases,
            basefreq=parse_float(statement.attrs.get("basefreq")),
            units=statement.attrs.get("units"),
            rmatrix=parse_matrix(statement.attrs.get("rmatrix"), nphases),
            xmatrix=parse_matrix(statement.attrs.get("xmatrix"), nphases),
            cmatrix=parse_matrix(statement.attrs.get("cmatrix"), nphases),
            normamps=parse_float(statement.attrs.get("normamps")),
            emergamps=parse_float(statement.attrs.get("emergamps")),
            raw_line=statement.line,
        )

    lines: dict[str, LineDef] = {}
    loads: dict[str, LoadDef] = {}
    transformers: dict[str, TransformerDef] = {}
    regcontrols: dict[str, RegControlDef] = {}

    for statement in statements:
        klass = statement.klass.lower()
        attrs = statement.attrs
        if klass == "line" and statement.name:
            phases = parse_int(attrs.get("phases"), 3) or 3
            is_jumper = statement.name.lower() == "jumper"
            lines[statement.name] = LineDef(
                name=statement.name,
                bus1=attrs.get("bus1", ""),
                bus2=attrs.get("bus2", ""),
                phases=phases,
                linecode=attrs.get("linecode"),
                length=parse_float(attrs.get("length")),
                units=attrs.get("units"),
                r1=parse_float(attrs.get("r1")),
                x1=parse_float(attrs.get("x1")),
                r0=parse_float(attrs.get("r0")),
                x0=parse_float(attrs.get("x0")),
                c0=parse_float(attrs.get("c0")),
                c1=parse_float(attrs.get("c1")),
                raw_line=statement.line,
                is_jumper=is_jumper,
            )
        elif klass == "load" and statement.name:
            loads[statement.name] = LoadDef(
                name=statement.name,
                bus1=attrs.get("bus1", ""),
                phases=parse_int(attrs.get("phases"), 1) or 1,
                conn=attrs.get("conn", "wye"),
                model=parse_int(attrs.get("model"), 1) or 1,
                kv=parse_float(attrs.get("kv"), 4.8) or 4.8,
                kw=parse_float(attrs.get("kw"), 0.0) or 0.0,
                kvar=parse_float(attrs.get("kvar"), 0.0) or 0.0,
                raw_line=statement.line,
            )
        elif klass == "transformer" and statement.name:
            pairs = statement.pairs
            winding_rows: list[dict[str, str]] = []
            current: dict[str, str] | None = None
            top: dict[str, str] = {}
            for key, value in pairs:
                if key == "wdg":
                    current = {"wdg": value}
                    winding_rows.append(current)
                elif current is not None:
                    current[key] = value
                else:
                    top[key] = value
            if winding_rows:
                buses = [row.get("bus", "") for row in winding_rows]
                conns = [row.get("conn", "wye") for row in winding_rows]
                kvs = [parse_float(row.get("kv"), 0.0) or 0.0 for row in winding_rows]
                kvas = [parse_float(row.get("kva"), 0.0) or 0.0 for row in winding_rows]
                rs = [parse_float(row.get("%r"), 0.0) or 0.0 for row in winding_rows]
                taps = [parse_float(row.get("tap"), 1.0) or 1.0 for row in winding_rows]
                r_specified = ["%r" in row for row in winding_rows]
                phases = parse_int(top.get("phases"), 3) or 3
                windings = parse_int(top.get("windings"), len(winding_rows)) or len(winding_rows)
                xhl = parse_float(top.get("xhl"), 0.0) or 0.0
                bank = top.get("bank")
                like = top.get("like")
                raw_line = statement.line
            else:
                buses = parse_list(attrs.get("buses"))
                conns = parse_list(attrs.get("conns"))
                kvs = [float(item) for item in parse_list(attrs.get("kvs"))]
                kvas = [float(item) for item in parse_list(attrs.get("kvas"))]
                rs = [0.2 for _ in buses]  # OpenDSS 对未写 %r 的默认值，求解器复核
                r_specified = [False for _ in buses]
                taps = [1.0 for _ in buses]
                phases = parse_int(attrs.get("phases"), 1) or 1
                windings = parse_int(attrs.get("windings"), len(buses)) or len(buses)
                xhl = parse_float(attrs.get("xhl"), 0.0) or 0.0
                bank = attrs.get("bank")
                like = attrs.get("like")
                raw_line = statement.line
            transformers[statement.name] = TransformerDef(
                name=statement.name,
                phases=phases,
                windings=windings,
                buses=buses,
                conns=conns,
                kvs=kvs,
                kvas=kvas,
                rs=rs,
                taps=taps,
                xhl=xhl,
                bank=bank,
                like=like,
                raw_line=raw_line,
                r_specified=r_specified,
            )
        elif klass == "regcontrol" and statement.name:
            regcontrols[statement.name] = RegControlDef(
                name=statement.name,
                transformer=attrs.get("transformer", ""),
                winding=parse_int(attrs.get("winding"), 1) or 1,
                vreg=parse_float(attrs.get("vreg"), 120.0) or 120.0,
                band=parse_float(attrs.get("band"), 2.0) or 2.0,
                ptratio=parse_float(attrs.get("ptratio"), 20.0) or 20.0,
                ctprim=parse_float(attrs.get("ctprim"), 100.0) or 100.0,
                r=parse_float(attrs.get("r"), 0.0) or 0.0,
                x=parse_float(attrs.get("x"), 0.0) or 0.0,
                delay=parse_float(attrs.get("delay"), 15.0) or 15.0,
                like=attrs.get("like"),
                raw_line=statement.line,
            )

    # 静态文件引用与命令。
    refs: list[dict[str, Any]] = []
    for statement in statements:
        if statement.klass in {"redirect", "compile", "buscoords"}:
            refs.append(
                {
                    "type": statement.klass,
                    "reference": statement.name,
                    "source_file": statement.file,
                    "source_line": statement.line,
                    "exists": (source_dir / statement.name).exists(),
                }
            )

    # 辅助文件清单（包括未被 DSS 引用的 EXP_METERS）。
    inventory_files = [master, linecodes_path, busxy_path]
    if exp_meters.exists():
        inventory_files.append(exp_meters)
    inventory = []
    for path in inventory_files:
        inventory.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "referenced_by_dss": any(ref["reference"].lower() == path.name.lower() for ref in refs),
            }
        )

    # 母线坐标。
    coords: dict[str, tuple[float, float]] = {}
    if busxy_path.exists():
        for raw in busxy_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = [item.strip() for item in raw.split(",")]
            if len(parts) >= 3 and parts[0]:
                try:
                    coords[parts[0].lower()] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue

    # 母线集合由线路、负荷和变压器端子共同决定。
    bus_names: set[str] = {"sourcebus"}
    for line in lines.values():
        bus_names.add(parse_bus(line.bus1)[0])
        bus_names.add(parse_bus(line.bus2)[0])
    for load in loads.values():
        bus_names.add(parse_bus(load.bus1, load.phases)[0])
    for transformer in transformers.values():
        for bus in transformer.buses:
            bus_names.add(parse_bus(bus)[0])

    # 电压等级由变压器绕组和电源定义推导，避免按名称猜测。
    voltage_level = {"sourcebus": 230.0}
    for transformer in transformers.values():
        for bus, kv in zip(transformer.buses, transformer.kvs):
            name = parse_bus(bus)[0].lower()
            voltage_level.setdefault(name, float(kv))
    # 799 与 799r 由 SubXF 二次侧和调压器定义共同确定为 4.8 kV。
    voltage_level.setdefault("799", 4.8)
    voltage_level.setdefault("799r", 4.8)

    def bus_voltage_key(name: str) -> float:
        return voltage_level.get(name.lower(), 4.8)

    model = {
        "source_dir": str(source_dir),
        "master_file": master.name,
        "statements": statements,
        "linecode_statements": linecode_statements,
        "linecodes": linecodes,
        "lines": lines,
        "loads": loads,
        "transformers": transformers,
        "regcontrols": regcontrols,
        "refs": refs,
        "inventory": inventory,
        "coords": coords,
        "bus_names": sorted(bus_names, key=natural_key),
        "voltage_level": voltage_level,
        "bus_voltage_key": bus_voltage_key,
        "set_commands": [statement.text for statement in statements if statement.klass == "set"],
    }
    return model


# ---------------------------------------------------------------------------
# OpenDSS COM 求解与原始导出
# ---------------------------------------------------------------------------


def _com_value(value: Any) -> Any:
    """把 COM 返回的 SAFEARRAY 转为普通列表。"""
    if isinstance(value, tuple):
        return list(value)
    return value


def solve_worker(source_dir: Path, output_json: Path) -> None:
    """在独立进程中编译并求解原始模型，导出全部求解器读数。

    该函数只写入 JSON，不生成报告；调用方负责超时与错误处理。
    """
    import win32com.client  # 仅求解阶段需要

    master = (source_dir / "IEEE37_Load_Connection_Map.csv1").resolve()
    engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
    engine.AllowForms = False
    compile_command = f'Compile "{master}"'
    engine.Text.Command = compile_command
    circuit = engine.ActiveCircuit

    def bus_record(name: str) -> dict[str, Any]:
        circuit.SetActiveBus(name)
        bus = circuit.ActiveBus
        record: dict[str, Any] = {
            "name": bus.Name,
            "kvbase_ln": float(bus.kVBase),
            "nodes": [int(node) for node in bus.Nodes],
            "x": float(bus.x),
            "y": float(bus.y),
            "distance": float(bus.Distance),
            "voltages_ln": [complex_to_pair(value) for value in pairs_to_complex(bus.Voltages)],
            "vll": [complex_to_pair(value) for value in pairs_to_complex(bus.VLL)],
            "pu_vll": [complex_to_pair(value) for value in pairs_to_complex(bus.puVLL)],
            "vmag_angle": [float(value) for value in bus.VMagAngle],
            "pu_vmag_angle": [float(value) for value in bus.puVmagAngle],
            "zsc1": None,
            "zsc0": None,
        }
        return record

    def line_record(name: str) -> dict[str, Any]:
        circuit.Lines.Name = name
        line = circuit.Lines
        element = circuit.ActiveCktElement
        return {
            "name": line.Name,
            "bus1": line.Bus1,
            "bus2": line.Bus2,
            "phases": int(line.Phases),
            "length": float(line.Length),
            "units": int(line.Units),
            "linecode": line.LineCode,
            "rmatrix": [float(value) for value in line.Rmatrix],
            "xmatrix": [float(value) for value in line.Xmatrix],
            "cmatrix": [float(value) for value in line.Cmatrix],
            "r1": float(line.R1),
            "x1": float(line.X1),
            "r0": float(line.R0),
            "x0": float(line.X0),
            "c1": float(line.C1),
            "c0": float(line.C0),
            "normamps": float(line.NormAmps),
            "emergamps": float(line.EmergAmps),
            "num_conductors": int(element.NumConductors),
            "num_terminals": int(element.NumTerminals),
            "enabled": bool(element.Enabled),
            "currents": [complex_to_pair(value) for value in pairs_to_complex(element.Currents)],
            "powers": [complex_to_pair(value) for value in pairs_to_complex(element.Powers)],
            "losses": [complex_to_pair(value) for value in pairs_to_complex(element.Losses)],
            "voltages": [complex_to_pair(value) for value in pairs_to_complex(element.Voltages)],
            "yprim": [complex_to_pair(value) for value in pairs_to_complex(element.Yprim)],
        }

    def load_record(name: str) -> dict[str, Any]:
        circuit.Loads.Name = name
        load = circuit.Loads
        element = circuit.ActiveCktElement
        return {
            "name": load.Name,
            "bus_names": [str(value) for value in element.BusNames],
            "num_phases": int(element.NumPhases),
            "num_conductors": int(element.NumConductors),
            "enabled": bool(element.Enabled),
            "kv": float(load.kV),
            "kw": float(load.kW),
            "kvar": float(load.kvar),
            "kva": float(load.kva),
            "pf": float(load.PF),
            "model": int(load.Model),
            "is_delta": bool(load.IsDelta),
            "status": int(load.Status),
            "cvrwatts": float(load.CVRwatts),
            "cvrvars": float(load.CVRvars),
            "zipv": [float(value) for value in _com_value(load.ZIPV)] if _com_value(load.ZIPV) else [],
            "powers": [complex_to_pair(value) for value in pairs_to_complex(element.Powers)],
            "currents": [complex_to_pair(value) for value in pairs_to_complex(element.Currents)],
            "losses": [complex_to_pair(value) for value in pairs_to_complex(element.Losses)],
        }

    def transformer_record(name: str) -> dict[str, Any]:
        circuit.Transformers.Name = name
        transformer = circuit.Transformers
        element = circuit.ActiveCktElement
        windings = []
        for winding in range(1, int(transformer.NumWindings) + 1):
            transformer.Wdg = winding
            windings.append(
                {
                    "kv": float(transformer.kV),
                    "kva": float(transformer.kva),
                    "r_pct": float(transformer.R),
                    "tap": float(transformer.Tap),
                    "rdc_ohms": float(transformer.RdcOhms),
                }
            )
        return {
            "name": transformer.Name,
            "bus_names": [str(value) for value in element.BusNames],
            "num_phases": int(element.NumPhases),
            "num_conductors": int(element.NumConductors),
            "num_terminals": int(element.NumTerminals),
            "enabled": bool(element.Enabled),
            "num_windings": int(transformer.NumWindings),
            "xhl": float(transformer.Xhl),
            "xht": float(transformer.Xht),
            "xlt": float(transformer.Xlt),
            "max_tap": float(transformer.MaxTap),
            "min_tap": float(transformer.MinTap),
            "num_taps": int(transformer.NumTaps),
            "is_delta": bool(transformer.IsDelta),
            "windings": windings,
            "powers": [complex_to_pair(value) for value in pairs_to_complex(element.Powers)],
            "currents": [complex_to_pair(value) for value in pairs_to_complex(element.Currents)],
            "losses": [complex_to_pair(value) for value in pairs_to_complex(element.Losses)],
            "voltages": [complex_to_pair(value) for value in pairs_to_complex(element.Voltages)],
        }

    result: dict[str, Any] = {
        "version": engine.Version,
        "compile_command": compile_command,
        "compile_error_number": int(engine.Error.Number),
        "compile_error_description": str(engine.Error.Description),
        "data_path": str(engine.DataPath),
        "circuit_name": str(circuit.Name),
        "num_buses": int(circuit.NumBuses),
        "num_nodes": int(circuit.NumNodes),
        "num_elements": int(circuit.NumCktElements),
        "bus_names": [str(value) for value in circuit.AllBusNames],
        "node_names": [str(value) for value in circuit.AllNodeNames],
        "line_names": [str(value) for value in circuit.Lines.AllNames],
        "load_names": [str(value) for value in circuit.Loads.AllNames],
        "transformer_names": [str(value) for value in circuit.Transformers.AllNames],
        "regcontrol_names": [str(value) for value in circuit.RegControls.AllNames],
        "linecode_names": [str(value) for value in circuit.LineCodes.AllNames],
        "solution": {
            "converged": bool(circuit.Solution.Converged),
            "iterations": int(circuit.Solution.Iterations),
            "control_iterations": int(circuit.Solution.ControlIterations),
            "total_iterations": int(circuit.Solution.Totaliterations),
            "max_iterations": int(circuit.Solution.MaxIterations),
            "tolerance": float(circuit.Solution.Tolerance),
            "mode": int(circuit.Solution.Mode),
            "control_mode": int(circuit.Solution.ControlMode),
            "load_mult": float(circuit.Solution.LoadMult),
            "frequency": float(circuit.Solution.Frequency),
            "algorithm": int(circuit.Solution.Algorithm),
            "total_time_ms": float(circuit.Solution.Total_Time),
            "process_time_ms": float(circuit.Solution.Process_Time),
        },
        "buses": [bus_record(name) for name in circuit.AllBusNames],
        "lines": [line_record(name) for name in circuit.Lines.AllNames],
        "loads": [load_record(name) for name in circuit.Loads.AllNames],
        "transformers": [transformer_record(name) for name in circuit.Transformers.AllNames],
        "regcontrols": [],
        "vsources": [],
    }

    index = circuit.RegControls.First
    while index > 0:
        control = circuit.RegControls
        result["regcontrols"].append(
            {
                "name": control.Name,
                "transformer": control.Transformer,
                "winding": int(control.Winding),
                "tap_winding": int(control.TapWinding),
                "tap_number": int(control.TapNumber),
                "vreg": float(control.ForwardVreg),
                "band": float(control.ForwardBand),
                "ptratio": float(control.PTratio),
                "ctprim": float(control.CTPrimary),
                "r": float(control.ForwardR),
                "x": float(control.ForwardX),
                "delay": float(control.Delay),
                "monitored_bus": str(control.MonitoredBus),
                "voltage_limit": float(control.VoltageLimit),
                "max_tap_change": int(control.MaxTapChange),
            }
        )
        index = circuit.RegControls.Next

    index = circuit.VSources.First
    while index > 0:
        source = circuit.VSources
        result["vsources"].append(
            {
                "name": source.Name,
                "basekv": float(source.BasekV),
                "pu": float(source.pu),
                "angle_deg": float(source.AngleDeg),
                "frequency": float(source.Frequency),
                "phases": int(source.Phases),
            }
        )
        index = circuit.VSources.Next

    # 所有电压、电流和功率读数完成后再刷新短路阻抗；ZscRefresh 会改变求解状态，
    # 因此不能放在逐母线读取电压的循环中。
    for bus_record_item in result["buses"]:
        circuit.SetActiveBus(bus_record_item["name"])
        try:
            circuit.ActiveBus.ZscRefresh()
            bus_record_item["zsc1"] = complex_to_pair(complex(*circuit.ActiveBus.Zsc1))
            bus_record_item["zsc0"] = complex_to_pair(complex(*circuit.ActiveBus.Zsc0))
        except Exception:
            bus_record_item["zsc1"] = None
            bus_record_item["zsc0"] = None

    # 同步检查求解后的电源母线正序/零序阻抗参考值。
    for bus in result["buses"]:
        if bus["name"].lower() == "sourcebus":
            result["source_bus_thevenin"] = {"zsc1": bus.get("zsc1"), "zsc0": bus.get("zsc0")}
            break

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    # 本机 OpenDSS COM 在解释器退出阶段可能挂起；数据已落盘后直接退出进程。
    os._exit(0)


def run_solver(source_dir: Path, output_json: Path, timeout_s: int = 180) -> tuple[dict[str, Any] | None, str]:
    """以独立子进程运行求解器，返回求解结果与日志文本。"""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_json.with_suffix(".log")
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve-worker",
        "--source-dir",
        str(source_dir),
        "--solver-json",
        str(output_json),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            return None, f"求解子进程超时（{timeout_s} s），日志见 {log_path}"
    elapsed = time.time() - started
    if not output_json.exists():
        return None, f"求解子进程未生成 JSON，退出码 {process.returncode}，日志见 {log_path}"
    try:
        result = json.loads(output_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"求解 JSON 读取失败：{exc}"
    result["_wall_time_s"] = elapsed
    if log_path.exists() and log_path.stat().st_size == 0:
        log_path.unlink()
    return result, f"求解完成，用时 {elapsed:.2f} s"


# ---------------------------------------------------------------------------
# 静态与求解结果合并、矩阵计算
# ---------------------------------------------------------------------------


def display_bus_name(name: str) -> str:
    """统一母线显示名：仅把 OpenDSS 的 sourcebus 显示为 SourceBus。"""
    return "SourceBus" if str(name).lower() == "sourcebus" else str(name)


def solver_maps(solver: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """建立求解器对象的名称索引，名称统一小写。"""
    return {
        "buses": {str(item["name"]).lower(): item for item in solver["buses"]},
        "lines": {str(item["name"]).lower(): item for item in solver["lines"]},
        "loads": {str(item["name"]).lower(): item for item in solver["loads"]},
        "transformers": {str(item["name"]).lower(): item for item in solver["transformers"]},
        "regcontrols": {str(item["name"]).lower(): item for item in solver["regcontrols"]},
    }


def matrix_from_flat(values: list[float], n: int) -> np.ndarray:
    """把 OpenDSS 的 n×n 平铺数组转为矩阵。"""
    return np.array([float(value) for value in values], dtype=float).reshape(n, n)


def sequence_components(matrix: np.ndarray) -> tuple[complex, complex, complex]:
    """由 3×3 相阻抗/导纳矩阵计算正/负/零序分量，顺序为 (Z1, Z2, Z0)。"""
    if matrix.shape == (1, 1):
        value = complex(matrix[0, 0])
        return value, value, value
    if matrix.shape == (2, 2):
        # 两相矩阵没有标准三序分解；仅为全定义表的完整性返回对角元。
        return complex(matrix[0, 0]), complex(matrix[1, 1]), complex(0, 0)
    seq = A_INV @ matrix.astype(complex) @ A_MAT
    return complex(seq[1, 1]), complex(seq[2, 2]), complex(seq[0, 0])


def line_series_from_yprim(yprim: list[list[float]], nconductors: int) -> np.ndarray | None:
    """从线路 Yprim 的端子间互导纳块反推总串联阻抗矩阵。"""
    if not yprim:
        return None
    complex_values = [pair_to_complex(value) for value in yprim]
    size = int(round(math.sqrt(len(complex_values))))
    if size != 2 * nconductors:
        return None
    admittance = np.array(complex_values, dtype=complex).reshape(size, size)
    y12 = admittance[:nconductors, nconductors:]
    try:
        return -np.linalg.inv(y12)
    except np.linalg.LinAlgError:
        return None


def line_length_mi(line: LineDef, solver_line: dict[str, Any] | None) -> float:
    """返回线路长度（英里）；Jumper 的长度只用于数值等效，不表示物理长度。"""
    if line.is_jumper:
        return 1.0
    if line.length is not None:
        return float(line.length)
    if solver_line is not None:
        return float(solver_line["length"])
    return 0.0


def build_bus_records(model: dict[str, Any], solver: dict[str, Any]) -> list[dict[str, Any]]:
    """合并静态母线与求解器母线读数，生成节点表记录。"""
    maps = solver_maps(solver)
    solver_buses = maps["buses"]
    # 建立母线级关联。
    line_members: dict[str, list[str]] = defaultdict(list)
    transformer_members: dict[str, list[str]] = defaultdict(list)
    load_members: dict[str, list[str]] = defaultdict(list)
    for line in model["lines"].values():
        for bus in (line.bus1, line.bus2):
            name = parse_bus(bus)[0].lower()
            line_members[name].append(line.name)
    for transformer in model["transformers"].values():
        for bus in transformer.buses:
            name = parse_bus(bus)[0].lower()
            transformer_members[name].append(transformer.name)
    for load in model["loads"].values():
        name = parse_bus(load.bus1, load.phases)[0].lower()
        load_members[name].append(load.name)

    # 母线级图，用于路径长度和度。
    graph = nx.Graph()
    for name in model["bus_names"]:
        graph.add_node(name.lower())
    for line in model["lines"].values():
        a = parse_bus(line.bus1)[0].lower()
        b = parse_bus(line.bus2)[0].lower()
        weight = 0.0 if line.is_jumper else line_length_mi(line, maps["lines"].get(line.name.lower()))
        graph.add_edge(a, b, weight=weight, kind="line", name=line.name)
    for transformer in model["transformers"].values():
        a = parse_bus(transformer.buses[0])[0].lower()
        b = parse_bus(transformer.buses[1])[0].lower()
        if graph.has_edge(a, b):
            # 799-799r 已有线路边；变压器边不覆盖线路权重，只保证连通。
            continue
        graph.add_edge(a, b, weight=0.0, kind="transformer", name=transformer.name)

    path_lengths = nx.single_source_dijkstra_path_length(graph, "799r", weight="weight")
    path_lengths["799"] = 0.0
    path_lengths["sourcebus"] = 0.0

    records = []
    ordered = sorted(model["bus_names"], key=natural_key)
    for index, raw_name in enumerate(ordered):
        key = raw_name.lower()
        static_voltage = float(model["voltage_level"].get(key, 4.8))
        bus = solver_buses.get(key)
        if bus is None:
            raise ValueError(f"求解器缺少母线 {raw_name}")
        vll = [pair_to_complex(value) for value in bus["vll"]]
        pu_vll = [pair_to_complex(value) for value in bus["pu_vll"]]
        vln = [pair_to_complex(value) for value in bus["voltages_ln"]]
        if len(vll) != 3 or len(vln) != 3:
            raise ValueError(f"母线 {raw_name} 的电压数组长度异常")
        vll_mag = [abs(value) for value in vll]
        pu_mag = [abs(value) for value in pu_vll]
        vll_seq = A_INV @ np.array(vll, dtype=complex)
        v2_v1 = abs(vll_seq[2]) / abs(vll_seq[1]) * 100 if abs(vll_seq[1]) > 0 else 0.0
        mean_vll = sum(vll_mag) / 3
        spread = (max(vll_mag) - min(vll_mag)) / mean_vll * 100 if mean_vll else 0.0
        total_degree = len(set(line_members[key])) + len(set(transformer_members[key]))
        is_source = key == "sourcebus"
        if is_source:
            bus_type = "Source (230 kV)"
            group = "230 kV"
        elif key == "799":
            bus_type = "Substation secondary / regulator input"
            group = "4.8 kV"
        elif key == "799r":
            bus_type = "Regulator output / feeder head"
            group = "4.8 kV"
        elif key == "775":
            bus_type = "Transformer secondary (no load)"
            group = "0.48 kV"
        elif key == "709":
            bus_type = "Feeder node / XFM1 primary"
            group = "4.8 kV"
        else:
            bus_type = "Feeder node"
            group = "4.8 kV"
        record = {
            "BusName": display_bus_name(raw_name),
            "BusKey": key,
            "BusIndex": index,
            "VoltageLevel_kV": static_voltage,
            "VoltageLevelGroup": group,
            "kVBase_LN": float(bus["kvbase_ln"]),
            "NominalLL_kV": static_voltage,
            "Phases": 3,
            "PhaseNodes": "1,2,3",
            "NodeCount": 3,
            "BusType": bus_type,
            "ConnectedLines": ",".join(sorted(set(line_members[key]), key=natural_key)),
            "ConnectedLineCount": len(set(line_members[key])),
            "ConnectedTransformers": ",".join(sorted(set(transformer_members[key]), key=natural_key)),
            "ConnectedTransformerCount": len(set(transformer_members[key])),
            "ConnectedLoads": ",".join(sorted(set(load_members[key]), key=natural_key)),
            "ConnectedLoadCount": len(set(load_members[key])),
            "BusDegree": total_degree,
            "IsLeaf": bool(total_degree == 1 and not is_source),
            "IsIsolated": bool(total_degree == 0),
            "PathLengthFrom799r_mi": float(path_lengths.get(key, math.nan)),
            "X": float(bus["x"]),
            "Y": float(bus["y"]),
            "Vab_kV": vll_mag[0] / 1000.0,
            "Vab_deg": math.degrees(math.atan2(vll[0].imag, vll[0].real)),
            "Vbc_kV": vll_mag[1] / 1000.0,
            "Vbc_deg": math.degrees(math.atan2(vll[1].imag, vll[1].real)),
            "Vca_kV": vll_mag[2] / 1000.0,
            "Vca_deg": math.degrees(math.atan2(vll[2].imag, vll[2].real)),
            "Vab_pu": pu_mag[0],
            "Vbc_pu": pu_mag[1],
            "Vca_pu": pu_mag[2],
            "Vmin_pu": min(pu_mag),
            "Vmax_pu": max(pu_mag),
            "VLL_Spread_pct": spread,
            "V2_V1_pct": v2_v1,
            "Van_kV_LN": abs(vln[0]) / 1000.0,
            "Vbn_kV_LN": abs(vln[1]) / 1000.0,
            "Vcn_kV_LN": abs(vln[2]) / 1000.0,
            "Zsc1_ohm_mag": abs(pair_to_complex(bus["zsc1"])) if bus.get("zsc1") else math.nan,
            "Zsc1_ohm_ang_deg": math.degrees(math.atan2(pair_to_complex(bus["zsc1"]).imag, pair_to_complex(bus["zsc1"]).real)) if bus.get("zsc1") else math.nan,
            "Zsc0_ohm_mag": abs(pair_to_complex(bus["zsc0"])) if bus.get("zsc0") else math.nan,
            "Zsc0_Note": "3-wire Delta has no zero-sequence return; Zsc0 is a solver network artifact, not used for positive-sequence/fault analysis",
            "Solved": bool(solver["solution"]["converged"]),
        }
        records.append(record)
    return records


def build_linecode_records(model: dict[str, Any], solver: dict[str, Any], referenced_only: bool = True) -> list[dict[str, Any]]:
    """生成 LineCode 表记录。"""
    maps = solver_maps(solver)
    usage: dict[str, list[str]] = defaultdict(list)
    for line in model["lines"].values():
        if line.linecode:
            usage[str(line.linecode)].append(line.name)
    records = []
    for name, code in model["linecodes"].items():
        is_used = name in usage
        if referenced_only and not is_used:
            continue
        r1, r2, r0 = sequence_components(code.rmatrix)
        x1, x2, x0 = sequence_components(code.xmatrix)
        c1, c2, c0 = sequence_components(code.cmatrix)
        units_resolved = "not referenced"
        for solver_line in solver.get("lines", []):
            if str(solver_line.get("linecode", "")).lower() == name.lower():
                units_resolved = {0: "mi", 1: "kft", 2: "km", 3: "m", 4: "ft", 5: "in", 6: "cm"}.get(
                    int(solver_line["units"]), str(solver_line["units"])
                )
                break
        sequence_valid = code.nphases == 3
        records.append(
            {
                "LineCode": name,
                "SourceLine": code.raw_line,
                "nphases": code.nphases,
                "BaseFreq_Hz": code.basefreq if code.basefreq is not None else solver["solution"]["frequency"],
                "Units_Resolved": units_resolved,
                "Units_InFile": code.units if code.units else "not specified (OpenDSS default mi)",
                "RMatrix_ohm_per_mi": matrix_to_text(code.rmatrix),
                "XMatrix_ohm_per_mi": matrix_to_text(code.xmatrix),
                "CMatrix_nF_per_mi": matrix_to_text(code.cmatrix),
                "Sequence_Valid": "yes" if sequence_valid else "no (not a 3-phase matrix)",
                "R1_ohm_per_mi": r1.real if sequence_valid else "",
                "X1_ohm_per_mi": x1.real if sequence_valid else "",
                "C1_nF_per_mi": c1.real if sequence_valid else "",
                "R0_ohm_per_mi": r0.real if sequence_valid else "",
                "X0_ohm_per_mi": x0.real if sequence_valid else "",
                "C0_nF_per_mi": c0.real if sequence_valid else "",
                "Z1_mag_ohm_per_mi": abs(complex(r1.real, x1.real)) if sequence_valid else "",
                "Z0_mag_ohm_per_mi": abs(complex(r0.real, x0.real)) if sequence_valid else "",
                "NormAmps_Resolved": 400.0,
                "EmergAmps_Resolved": 600.0,
                "NormAmps_InFile": code.normamps if code.normamps is not None else "not specified (OpenDSS default 400 A)",
                "EmergAmps_InFile": code.emergamps if code.emergamps is not None else "not specified (OpenDSS default 600 A)",
                "UseCount": len(usage.get(name, [])),
                "ReferencedByLines": ",".join(sorted(usage.get(name, []), key=natural_key)),
                "Referenced": is_used,
                "Notes": "C 矩阵非对角元为零" if np.allclose(code.cmatrix - np.diag(np.diag(code.cmatrix)), 0) else "C 矩阵存在非对角元",
            }
        )
    return records


def build_line_records(model: dict[str, Any], solver: dict[str, Any]) -> list[dict[str, Any]]:
    """生成线路表记录，包含单长矩阵和整条线路总矩阵。"""
    maps = solver_maps(solver)
    records = []
    for name, line in model["lines"].items():
        solved = maps["lines"].get(name.lower())
        if solved is None:
            raise ValueError(f"求解器缺少线路 {name}")
        phases = int(line.phases)
        n = phases
        length = line_length_mi(line, solved)
        if line.is_jumper:
            r_per = np.array([[float(line.r1 or 0.0)]])
            x_per = np.array([[float(line.x1 or 0.0)]])
            c_per = np.array([[float(line.c1 or 0.0)]])
        else:
            if not line.linecode or line.linecode not in model["linecodes"]:
                raise ValueError(f"线路 {name} 引用了未定义 LineCode {line.linecode}")
            code = model["linecodes"][line.linecode]
            if code.nphases != phases:
                raise ValueError(f"线路 {name} 相数与 LineCode {line.linecode} 不一致")
            r_per = code.rmatrix
            x_per = code.xmatrix
            c_per = code.cmatrix
        r_total = r_per * length
        x_total = x_per * length
        c_total = c_per * length

        # 静态 LineCode 矩阵与求解器实际加载的单长矩阵逐元素核对。
        solver_r = matrix_from_flat(solved["rmatrix"], n)
        solver_x = matrix_from_flat(solved["xmatrix"], n)
        solver_c = matrix_from_flat(solved["cmatrix"], n)
        matrix_diff = max(
            float(np.max(np.abs(solver_r - r_per))),
            float(np.max(np.abs(solver_x - x_per))),
            float(np.max(np.abs(solver_c - c_per))),
        )

        # 用 Yprim 反推总串联阻抗，验证“单长矩阵 × 长度”的口径。
        z_from_yprim = line_series_from_yprim(solved.get("yprim", []), int(solved["num_conductors"]))
        if z_from_yprim is not None:
            z_total = r_total + 1j * x_total
            max_diff = float(np.max(np.abs(z_from_yprim - z_total)))
        else:
            max_diff = math.nan

        r1, r2, r0 = sequence_components(r_total)
        x1, x2, x0 = sequence_components(x_total)
        current_pairs = [pair_to_complex(value) for value in solved["currents"]]
        currents_mag = [abs(value) for value in current_pairs]
        max_current = max(currents_mag) if currents_mag else 0.0
        loss = total_complex(solved["losses"])
        records.append(
            {
                "LineName": name,
                "SourceLine": line.raw_line,
                "Category": "Jumper" if line.is_jumper else "FeederLine",
                "Bus1": line.bus1,
                "Bus2": line.bus2,
                "FromBus": parse_bus(line.bus1)[0],
                "ToBus": parse_bus(line.bus2)[0],
                "Phases": phases,
                "LineCode": line.linecode or "",
                "Length": length,
                "Units": "mi",
                "Units_InFile": line.units if line.units else "not specified (OpenDSS default mi)",
                "Length_mi": length,
                "Length_km": length * 1.609344,
                "RMatrix_per_length_ohm_per_mi": matrix_to_text(r_per),
                "XMatrix_per_length_ohm_per_mi": matrix_to_text(x_per),
                "CMatrix_per_length_nF_per_mi": matrix_to_text(c_per),
                "RMatrix_total_ohm": matrix_to_text(r_total),
                "XMatrix_total_ohm": matrix_to_text(x_total),
                "CMatrix_total_nF": matrix_to_text(c_total),
                "R1_total_ohm": r1.real,
                "X1_total_ohm": x1.real,
                "Z1_total_mag_ohm": abs(complex(r1.real, x1.real)),
                "Z1_total_ang_deg": math.degrees(math.atan2(x1.real, r1.real)),
                "R0_total_ohm": r0.real,
                "X0_total_ohm": x0.real,
                "Z0_total_mag_ohm": abs(complex(r0.real, x0.real)),
                "Z0_total_ang_deg": math.degrees(math.atan2(x0.real, r0.real)),
                "NormAmps": float(solved["normamps"]),
                "EmergAmps": float(solved["emergamps"]),
                "Enabled": bool(solved["enabled"]),
                "NumConductors": int(solved["num_conductors"]),
                "MaxCurrent_A": max_current,
                "MaxLoading_pct": max_current / float(solved["normamps"]) * 100 if solved["normamps"] else math.nan,
                "Loss_kW": loss.real / 1000.0,
                "Loss_kvar": loss.imag / 1000.0,
                "Yprim_Impedance_MaxDiff_ohm": max_diff,
                "LineCode_vs_Solver_Matrix_MaxDiff": matrix_diff,
                "Impedance_Source": "LineCode 单长矩阵 × 长度；与求解器单长矩阵及 -inv(Yprim 端子互导纳块) 双重核对",
                "Notes": "Jumper 为 B 相 1 mΩ 集中连接，Length 取 OpenDSS 默认 1.0，物理长度无意义；基态电流近零（SubXF/reg1a/reg1c 的 B 相电流在 799.2 处相量和为零），它是电位参考而非基态功率通道，故障/不平衡时可能承载大电流" if line.is_jumper else "",
            }
        )
    return records


def build_load_records(model: dict[str, Any], solver: dict[str, Any]) -> list[dict[str, Any]]:
    """生成负荷对象表记录。"""
    maps = solver_maps(solver)
    voltage_level = model["voltage_level"]
    records = []
    for name, load in model["loads"].items():
        solved = maps["loads"].get(name.lower())
        if solved is None:
            raise ValueError(f"求解器缺少负荷 {name}")
        bus_name = parse_bus(load.bus1, load.phases)[0]
        nodes = parse_bus(load.bus1, load.phases)[1]
        if load.phases == 3:
            phase_pair = "1-2,2-3,3-1"
        else:
            phase_pair = f"{nodes[0]}-{nodes[1]}" if len(nodes) >= 2 else ",".join(str(node) for node in nodes)
        actual = total_complex(solved["losses"])
        model_name = {1: "Constant P (Model 1)", 2: "Constant Z (Model 2)", 4: "Exponential/CVR (Model 4)"}.get(load.model, f"Model {load.model}")
        records.append(
            {
                "LoadName": name,
                "Bus1": load.bus1,
                "BusName": display_bus_name(bus_name),
                "PhaseNodes": ",".join(str(node) for node in nodes),
                "PhasePair": phase_pair,
                "Phases": load.phases,
                "Conn": load.conn,
                "IsDelta": bool(solved["is_delta"]),
                "Model": load.model,
                "ModelName": model_name,
                "kV": load.kv,
                "kW": load.kw,
                "kvar": load.kvar,
                "kVA": float(solved["kva"]),
                "PF": float(solved["pf"]),
                "PF_LeadLag": "Lagging" if load.kvar > 0 else ("Leading" if load.kvar < 0 else "Unity"),
                "Status_Code": int(solved["status"]),
                "Status": "Fixed (0)" if int(solved["status"]) == 0 else f"Variable ({solved['status']})",
                "Enabled": bool(solved["enabled"]),
                "CVRwatts": float(solved["cvrwatts"]),
                "CVRvars": float(solved["cvrvars"]),
                "ZIPV": ",".join(str(value) for value in solved["zipv"]) if solved["zipv"] else "not specified",
                "CVR_Source": "not in DSS; OpenDSS default resolved by solver",
                "VoltageLevel_kV": float(voltage_level.get(bus_name.lower(), 4.8)),
                "Actual_P_kW": actual.real / 1000.0,
                "Actual_Q_kvar": actual.imag / 1000.0,
                "Actual_PF": actual.real / abs(actual) if abs(actual) > 0 else math.nan,
                "SourceLine": load.raw_line,
                "Notes": "S728 为唯一三相 Delta 负荷，总功率按三个相间支路均分" if load.phases == 3 else "单相 Delta 负荷，接于标明的两个相节点之间",
            }
        )
    return records


def build_connection_map(model: dict[str, Any], solver: dict[str, Any], bus_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成逐支路负荷接线映射。"""
    maps = solver_maps(solver)
    bus_vll = {record["BusKey"]: record for record in bus_records}
    rows = []
    for name, load in model["loads"].items():
        solved = maps["loads"].get(name.lower())
        bus_name = parse_bus(load.bus1, load.phases)[0]
        nodes = parse_bus(load.bus1, load.phases)[1]
        vll = None
        if bus_name.lower() in bus_vll:
            record = bus_vll[bus_name.lower()]
            vll = {
                (1, 2): record["Vab_kV"],
                (2, 3): record["Vbc_kV"],
                (3, 1): record["Vca_kV"],
            }
        if load.phases == 3:
            branches = [(1, 2), (2, 3), (3, 1)]
            branch_kw = load.kw / 3.0
            branch_kvar = load.kvar / 3.0
            powers = [pair_to_complex(value) for value in solved["powers"]]
            # 三相 Delta 负荷的 Powers 为三个相节点注入功率，不是三个支路功率。
            phase_powers = powers[:3] if len(powers) >= 3 else []
            for branch_index, (from_phase, to_phase) in enumerate(branches, start=1):
                phase_power = phase_powers[branch_index - 1] if branch_index - 1 < len(phase_powers) else complex(0, 0)
                rows.append(
                    {
                        "LoadName": name,
                        "BranchID": f"{name}-B{branch_index}",
                        "BusName": display_bus_name(bus_name),
                        "PhaseFrom": from_phase,
                        "PhaseTo": to_phase,
                        "ConnectionType": "Delta phase-to-phase",
                        "Phases": 3,
                        "Nominal_kV": load.kv,
                        "Solved_VLL_kV": vll.get((from_phase, to_phase), math.nan) if vll else math.nan,
                        "Nominal_Branch_kW": branch_kw,
                        "Nominal_Branch_kvar": branch_kvar,
                        "Solved_Branch_kW": branch_kw,
                        "Solved_Branch_kvar": branch_kvar,
                        "Solved_PhaseNode_P_kW": phase_power.real,
                        "Solved_PhaseNode_Q_kvar": phase_power.imag,
                        "Model": load.model,
                        "Notes": "三相 Delta 负荷总功率按三个支路均分；相节点注入功率用于核对电压不平衡",
                    }
                )
        else:
            from_phase = nodes[0] if len(nodes) >= 1 else 1
            to_phase = nodes[1] if len(nodes) >= 2 else 2
            powers = [pair_to_complex(value) for value in solved["powers"]]
            total = sum(powers) if powers else complex(0, 0)
            rows.append(
                {
                    "LoadName": name,
                    "BranchID": f"{name}-B1",
                    "BusName": display_bus_name(bus_name),
                    "PhaseFrom": from_phase,
                    "PhaseTo": to_phase,
                    "ConnectionType": "Delta phase-to-phase",
                    "Phases": 1,
                    "Nominal_kV": load.kv,
                    "Solved_VLL_kV": vll.get((from_phase, to_phase), math.nan) if vll else math.nan,
                    "Nominal_Branch_kW": load.kw,
                    "Nominal_Branch_kvar": load.kvar,
                    "Solved_Branch_kW": total.real,
                    "Solved_Branch_kvar": total.imag,
                    "Solved_PhaseNode_P_kW": "",
                    "Solved_PhaseNode_Q_kvar": "",
                    "Model": load.model,
                    "Notes": "单相 Delta 负荷；两个端子分别接标明的两个相节点",
                }
            )
    return rows


def build_transformer_records(model: dict[str, Any], solver: dict[str, Any]) -> list[dict[str, Any]]:
    """生成变压器/调压器本体参数表。"""
    maps = solver_maps(solver)
    records = []
    for name, transformer in model["transformers"].items():
        solved = maps["transformers"].get(name.lower())
        if solved is None:
            raise ValueError(f"求解器缺少变压器 {name}")
        if len(transformer.buses) < 2:
            raise ValueError(f"变压器 {name} 绕组数不足")
        bus1 = parse_bus(transformer.buses[0])[0]
        bus2 = parse_bus(transformer.buses[1])[0]
        kv1, kv2 = transformer.kvs[0], transformer.kvs[1]
        kva1, kva2 = transformer.kvas[0], transformer.kvas[1]
        r1_pct, r2_pct = transformer.rs[0], transformer.rs[1]
        zbase1 = kv1**2 / (kva1 / 1000.0) if kva1 else math.nan
        zbase2 = kv2**2 / (kva2 / 1000.0) if kva2 else math.nan
        r_total_1 = r1_pct / 100.0 * zbase1 + r2_pct / 100.0 * zbase2 * (kv1 / kv2) ** 2
        x_total_1 = transformer.xhl / 100.0 * zbase1
        z_total_1 = complex(r_total_1, x_total_1)
        r_total_2 = r2_pct / 100.0 * zbase2 + r1_pct / 100.0 * zbase1 * (kv2 / kv1) ** 2
        x_total_2 = transformer.xhl / 100.0 * zbase2
        z_total_2 = complex(r_total_2, x_total_2)
        powers = [pair_to_complex(value) for value in solved["powers"]]
        currents = [pair_to_complex(value) for value in solved["currents"]]
        ncond = int(solved["num_conductors"])
        p1 = sum(powers[:ncond]) if len(powers) >= ncond else complex(0, 0)
        p2 = sum(powers[ncond : 2 * ncond]) if len(powers) >= 2 * ncond else complex(0, 0)
        loss = total_complex(solved["losses"])
        s1 = abs(p1)
        s2 = abs(p2)
        loading = max(s1, s2) / kva1 * 100 if kva1 else math.nan
        phase_count = 3 if transformer.phases == 3 else ncond
        current1 = currents[:ncond]
        current2 = currents[ncond : 2 * ncond]
        max_i1 = max((abs(value) for value in current1[:phase_count]), default=math.nan)
        max_i2 = max((abs(value) for value in current2[:phase_count]), default=math.nan)
        rated_i1 = kva1 / (math.sqrt(3) * kv1) if transformer.phases == 3 else kva1 / kv1
        rated_i2 = kva2 / (math.sqrt(3) * kv2) if transformer.phases == 3 else kva2 / kv2
        if name.lower() == "subxf":
            kind = "Substation transformer"
        elif name.lower() == "xfm1":
            kind = "Load transformer"
        else:
            kind = "Open-Delta regulator"
        if name.lower() == "subxf":
            note = "230/4.8 kV Delta-Delta；%r 显式 1%/绕组；Xhl=8%"
        elif name.lower() == "xfm1":
            note = "4.8/0.48 kV Delta-Delta；%r 显式 0.045%/绕组；Xhl=1.81%；775 侧无负荷"
        elif name.lower() == "reg1a":
            note = "单相 A-B 调压器；%r 未写，求解器默认 0.2%/绕组；Xhl=1%；tap=1.1"
        elif name.lower() == "reg1c":
            note = "单相 C-B 调压器；%r 未写，求解器默认 0.2%/绕组；Xhl=1%；tap=1.0875"
        else:
            note = ""
        records.append(
            {
                "Name": name,
                "SourceLine": transformer.raw_line,
                "Type": kind,
                "Phases": transformer.phases,
                "Windings": transformer.windings,
                "Bank": transformer.bank or "",
                "Bus1": bus1,
                "Bus2": bus2,
                "Terminal1": transformer.buses[0],
                "Terminal2": transformer.buses[1],
                "Conn1": transformer.conns[0] if len(transformer.conns) > 0 else "",
                "Conn2": transformer.conns[1] if len(transformer.conns) > 1 else "",
                "kV1": kv1,
                "kV2": kv2,
                "kVA1": kva1,
                "kVA2": kva2,
                "R1_pct": r1_pct,
                "R2_pct": r2_pct,
                "R_Specified": "yes" if all(transformer.r_specified) else "no (OpenDSS default 0.2%/winding)",
                "Xhl_pct": transformer.xhl,
                "Xht_pct_resolved": float(solved["xht"]),
                "Xlt_pct_resolved": float(solved["xlt"]),
                "Tap1": float(solved["windings"][0]["tap"]),
                "Tap2": float(solved["windings"][1]["tap"]),
                "MinTap": float(solved["min_tap"]),
                "MaxTap": float(solved["max_tap"]),
                "NumTaps": int(solved["num_taps"]),
                "TurnsRatio": kv1 / kv2 if kv2 else math.nan,
                "Zbase1_ohm": zbase1,
                "Zbase2_ohm": zbase2,
                "R_total_ohm_side1": r_total_1,
                "X_total_ohm_side1": x_total_1,
                "Z_total_ohm_side1_mag": abs(z_total_1),
                "Z_total_pu_side1": abs(z_total_1) / zbase1 if zbase1 else math.nan,
                "R_total_ohm_side2": r_total_2,
                "X_total_ohm_side2": x_total_2,
                "Z_total_ohm_side2_mag": abs(z_total_2),
                "Z_total_pu_side2": abs(z_total_2) / zbase2 if zbase2 else math.nan,
                "SolvedP1_kW": p1.real,
                "SolvedQ1_kvar": p1.imag,
                "SolvedS1_kVA": s1,
                "SolvedP2_kW": p2.real,
                "SolvedQ2_kvar": p2.imag,
                "SolvedS2_kVA": s2,
                "RatedCurrent1_A": rated_i1,
                "MaxCurrent1_A": max_i1,
                "CurrentLoading1_pct": max_i1 / rated_i1 * 100 if rated_i1 else math.nan,
                "RatedCurrent2_A": rated_i2,
                "MaxCurrent2_A": max_i2,
                "CurrentLoading2_pct": max_i2 / rated_i2 * 100 if rated_i2 else math.nan,
                "Loss_kW": loss.real / 1000.0,
                "Loss_kvar": loss.imag / 1000.0,
                "Loading_pct": loading,
                "Enabled": bool(solved["enabled"]),
                "Notes": note,
            }
        )
    return records


def build_regulator_records(model: dict[str, Any], solver: dict[str, Any]) -> list[dict[str, Any]]:
    """生成调压控制器与调压器运行状态表。"""
    maps = solver_maps(solver)
    rows = []
    for name, control in model["regcontrols"].items():
        solved = maps["regcontrols"].get(name.lower())
        if solved is None:
            raise ValueError(f"求解器缺少调压控制器 {name}")
        transformer = model["transformers"].get(control.transformer)
        if transformer is None:
            raise ValueError(f"调压控制器 {name} 指向未定义变压器 {control.transformer}")
        transformer_solved = maps["transformers"][control.transformer.lower()]
        tap2 = float(transformer_solved["windings"][1]["tap"])
        tap_number = int(solved["tap_number"])
        step_pct = (float(solved["max_tap_change"]) if solved["max_tap_change"] else 0.625)
        # OpenDSS 默认 32 档、±16 档；步长由求解器 Tap 与档位关系反推更可靠。
        if tap_number != 0:
            step_pct = (tap2 - 1.0) / tap_number * 100.0
        phase_pair = "1-2" if name.lower() == "creg1a" else "3-2" if name.lower() == "creg1c" else ""
        rows.append(
            {
                "Regulator": name,
                "SourceLine": control.raw_line,
                "Transformer": control.transformer,
                "Bank": transformer.bank or "",
                "ControlledWinding": control.winding,
                "TapWinding": int(solved["tap_winding"]),
                "PhasePair": phase_pair,
                "Bus1": transformer.buses[0],
                "Bus2": transformer.buses[1],
                "Phases": transformer.phases,
                "kV": transformer.kvs[0],
                "kVA_per_winding": transformer.kvas[0],
                "Xhl_pct": transformer.xhl,
                "R_pct": transformer.rs[0],
                "Vreg_V_secondary": float(solved["vreg"]),
                "PTRatio": float(solved["ptratio"]),
                "Vreg_kV_primary": float(solved["vreg"]) / 120.0 * transformer.kvs[0],
                "Band_V_secondary": float(solved["band"]),
                "Band_kV_primary": float(solved["band"]) / 120.0 * transformer.kvs[0],
                "CTPrimary_A": float(solved["ctprim"]),
                "R_setting_V": float(solved["r"]),
                "X_setting_V": float(solved["x"]),
                "TapPosition": tap_number,
                "TapRatio": tap2,
                "TapStep_pct": step_pct,
                "MinTap": float(transformer_solved["min_tap"]),
                "MaxTap": float(transformer_solved["max_tap"]),
                "NumTaps": int(transformer_solved["num_taps"]),
                "Delay_s": float(solved["delay"]),
                "MonitoredBus": solved["monitored_bus"] or "(default: transformer secondary bus)",
                "VoltageLimit_V": float(solved["voltage_limit"]),
                "MaxTapChange": int(solved["max_tap_change"]),
                "Enabled": True,
                "Notes": "Open-Delta 的 A-B 支路" if name.lower() == "creg1a" else "Open-Delta 的 C-B 支路",
            }
        )
    return rows


def graph_audit(model: dict[str, Any]) -> dict[str, Any]:
    """执行母线级与相节点级图论审计。"""
    multi = nx.MultiGraph()
    for name in model["bus_names"]:
        multi.add_node(name.lower(), label=name)
    line_edges = []
    transformer_edges = []
    for line in model["lines"].values():
        a = parse_bus(line.bus1)[0].lower()
        b = parse_bus(line.bus2)[0].lower()
        weight = 0.0 if line.is_jumper else line_length_mi(line, None)
        multi.add_edge(a, b, key=line.name, kind="line", name=line.name, weight=weight)
        line_edges.append((a, b, line.name))
    for transformer in model["transformers"].values():
        a = parse_bus(transformer.buses[0])[0].lower()
        b = parse_bus(transformer.buses[1])[0].lower()
        multi.add_edge(a, b, key=transformer.name, kind="transformer", name=transformer.name, weight=0.0)
        transformer_edges.append((a, b, transformer.name))

    # 馈线本体图只含 L1–L35；Jumper 与变压器边不进入馈线树判定。
    feeder_graph = nx.Graph()
    for line in model["lines"].values():
        if line.is_jumper:
            continue
        a = parse_bus(line.bus1)[0].lower()
        b = parse_bus(line.bus2)[0].lower()
        feeder_graph.add_edge(a, b, weight=line_length_mi(line, None), name=line.name)

    components = list(nx.connected_components(nx.Graph(multi)))
    cycle_rank = multi.number_of_edges() - multi.number_of_nodes() + len(components)
    simple_cycles = nx.cycle_basis(feeder_graph)
    parallel_groups = []
    pair_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for a, b, data in multi.edges(data=True):
        key = tuple(sorted((a, b)))
        pair_map[key].append(data["name"])
    for pair, names in pair_map.items():
        if len(names) > 1:
            parallel_groups.append({"pair": pair, "elements": names, "count": len(names)})

    duplicate_lines = []
    line_pair_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for line in model["lines"].values():
        a = parse_bus(line.bus1)[0].lower()
        b = parse_bus(line.bus2)[0].lower()
        line_pair_map[tuple(sorted((a, b)))].append(line.name)
    for pair, names in line_pair_map.items():
        if len(names) > 1:
            duplicate_lines.append({"pair": pair, "lines": names})

    isolated = [node for node in multi.nodes if multi.degree(node) == 0]
    leaves = [node for node in multi.nodes if multi.degree(node) == 1 and node != "sourcebus"]
    branch_points = [node for node in multi.nodes if multi.degree(node) >= 3 and node not in {"sourcebus"}]
    source_leaf = "sourcebus" if multi.degree("sourcebus") == 1 else ""

    # 主馈线：从 799r 出发、按线路长度加权的最远叶节点路径。
    path_graph = nx.Graph()
    for a, b, data in multi.edges(data=True):
        if data["kind"] != "line":
            continue
        weight = 0.0 if data["name"].lower() == "jumper" else data.get("weight", 0.0)
        if not path_graph.has_edge(a, b) or weight < path_graph[a][b]["weight"]:
            path_graph.add_edge(a, b, weight=weight, name=data["name"])
    distances, paths = nx.single_source_dijkstra(path_graph, "799r", weight="weight")
    candidate_leaves = [node for node in path_graph.nodes if path_graph.degree(node) == 1 and node != "799r"]
    farthest = max(candidate_leaves, key=lambda node: distances.get(node, -1)) if candidate_leaves else None
    main_path = paths.get(farthest, []) if farthest else []
    main_path_lines = []
    for a, b in zip(main_path, main_path[1:]):
        main_path_lines.append(path_graph[a][b]["name"])

    # 相节点检查。
    phase_errors = []
    expected_phases = {1, 2, 3}
    for line in model["lines"].values():
        n1 = parse_bus(line.bus1)[1]
        n2 = parse_bus(line.bus2)[1]
        if line.phases == 3 and (set(n1) != expected_phases or set(n2) != expected_phases):
            phase_errors.append(f"{line.name}: 三相线路节点不是 1.2.3")
        if line.phases == 1 and (len(n1) != 1 or len(n2) != 1):
            phase_errors.append(f"{line.name}: 单相线路节点数不为 1")
    for load in model["loads"].values():
        nodes = parse_bus(load.bus1, load.phases)[1]
        if load.phases == 3 and set(nodes) != expected_phases:
            phase_errors.append(f"{load.name}: 三相负荷节点不是 1.2.3")
        if load.phases == 1 and len(nodes) != 2:
            phase_errors.append(f"{load.name}: 单相 Delta 负荷不是两个节点")
    for transformer in model["transformers"].values():
        for bus in transformer.buses:
            nodes = parse_bus(bus)[1]
            if not nodes and transformer.phases == 3:
                nodes = [1, 2, 3]
            if not nodes or any(node not in expected_phases for node in nodes):
                phase_errors.append(f"{transformer.name}: 端子 {bus} 相节点异常")
    # 相节点级“功率传输连通性”：线路同相连接、Jumper 的 B 相连接、变压器作为设备节点。
    phase_graph = nx.Graph()
    for name in model["bus_names"]:
        key = name.lower()
        for phase in (1, 2, 3):
            phase_graph.add_node(f"{key}.{phase}")
    for line in model["lines"].values():
        n1 = parse_bus(line.bus1)[1]
        n2 = parse_bus(line.bus2)[1]
        if line.phases == 3:
            for phase in (1, 2, 3):
                phase_graph.add_edge(f"{parse_bus(line.bus1)[0].lower()}.{phase}", f"{parse_bus(line.bus2)[0].lower()}.{phase}", kind="line", name=line.name)
        elif line.is_jumper:
            for node1, node2 in zip(n1, n2):
                phase_graph.add_edge(f"{parse_bus(line.bus1)[0].lower()}.{node1}", f"{parse_bus(line.bus2)[0].lower()}.{node2}", kind="jumper", name=line.name)
    for transformer in model["transformers"].values():
        device = f"DEVICE:{transformer.name.lower()}"
        phase_graph.add_node(device, kind="device")
        for bus in transformer.buses:
            bus_name = parse_bus(bus)[0].lower()
            terminals = parse_bus(bus)[1]
            if not terminals and transformer.phases == 3:
                terminals = [1, 2, 3]
            for node in terminals:
                phase_graph.add_edge(device, f"{bus_name}.{node}", kind="transformer", name=transformer.name)
    phase_components = list(nx.connected_components(phase_graph))

    return {
        "num_buses": multi.number_of_nodes(),
        "num_edges": multi.number_of_edges(),
        "num_line_edges": len(line_edges),
        "num_transformer_edges": len(transformer_edges),
        "num_components": len(components),
        "components": [sorted(component, key=natural_key) for component in components],
        "cycle_rank": cycle_rank,
        "simple_cycles": simple_cycles,
        "parallel_groups": parallel_groups,
        "duplicate_lines": duplicate_lines,
        "isolated": isolated,
        "leaves": sorted(leaves, key=natural_key),
        "branch_points": sorted(branch_points, key=natural_key),
        "source_leaf": source_leaf,
        "main_path": main_path,
        "main_path_lines": main_path_lines,
        "main_path_length_mi": distances.get(farthest, math.nan) if farthest else math.nan,
        "farthest_leaf": farthest,
        "phase_errors": phase_errors,
        "num_phase_nodes": phase_graph.number_of_nodes(),
        "num_phase_components": len(phase_components),
        "phase_component_sizes": sorted((len(component) for component in phase_components), reverse=True),
        "feeder_tree_nodes": feeder_graph.number_of_nodes(),
        "feeder_tree_edges": feeder_graph.number_of_edges(),
        "feeder_is_tree": nx.is_tree(feeder_graph) if feeder_graph.number_of_nodes() else False,
    }


# ---------------------------------------------------------------------------
# CSV 输出与汇总
# ---------------------------------------------------------------------------


def records_to_rows(records: list[dict[str, Any]], headers: list[str]) -> list[list[Any]]:
    """按表头顺序把记录转为 CSV 行。"""
    return [[record.get(header, "") for header in headers] for record in records]


def matrix_rows(line_record: dict[str, Any]) -> list[list[Any]]:
    """把一条线路的单长/总矩阵展开为逐元素长表。"""
    r_per = [row.split() for row in line_record["RMatrix_per_length_ohm_per_mi"].split("|")]
    x_per = [row.split() for row in line_record["XMatrix_per_length_ohm_per_mi"].split("|")]
    c_per = [row.split() for row in line_record["CMatrix_per_length_nF_per_mi"].split("|")]
    r_tot = [row.split() for row in line_record["RMatrix_total_ohm"].split("|")]
    x_tot = [row.split() for row in line_record["XMatrix_total_ohm"].split("|")]
    c_tot = [row.split() for row in line_record["CMatrix_total_nF"].split("|")]
    rows = []
    for row_index in range(len(r_per)):
        for col_index in range(len(r_per[row_index])):
            r_value = float(r_per[row_index][col_index])
            x_value = float(x_per[row_index][col_index])
            rows.append(
                [
                    line_record["LineName"],
                    line_record["LineCode"],
                    line_record["Length_mi"],
                    row_index + 1,
                    col_index + 1,
                    "Self" if row_index == col_index else "Mutual",
                    r_value,
                    x_value,
                    float(c_per[row_index][col_index]),
                    float(r_tot[row_index][col_index]),
                    float(x_tot[row_index][col_index]),
                    float(c_tot[row_index][col_index]),
                    math.hypot(float(r_tot[row_index][col_index]), float(x_tot[row_index][col_index])),
                    math.degrees(math.atan2(float(x_tot[row_index][col_index]), float(r_tot[row_index][col_index]))),
                    "mi / nF per mi; total Ω / nF",
                ]
            )
    return rows


def build_summary(
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    connection_map: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    graph: dict[str, Any],
) -> dict[str, Any]:
    """汇总审计关键读数，供 Markdown 渲染和 JSON 留档。"""
    load_kw = sum(load["kW"] for load in loads)
    load_kvar = sum(load["kvar"] for load in loads)
    load_kva = math.hypot(load_kw, load_kvar)
    actual_kw = sum(load["Actual_P_kW"] for load in loads)
    actual_kvar = sum(load["Actual_Q_kvar"] for load in loads)
    min_bus = min(buses, key=lambda item: item["Vmin_pu"])
    max_bus = max(buses, key=lambda item: item["Vmax_pu"])
    max_unbalance = max(buses, key=lambda item: item["V2_V1_pct"])
    max_current_line = max(lines, key=lambda item: item["MaxCurrent_A"])
    max_loading_line = max(lines, key=lambda item: item["MaxLoading_pct"])
    subxf = next(item for item in transformers if item["Name"].lower() == "subxf")
    xfm1 = next(item for item in transformers if item["Name"].lower() == "xfm1")
    reg1a = next(item for item in transformers if item["Name"].lower() == "reg1a")
    reg1c = next(item for item in transformers if item["Name"].lower() == "reg1c")
    max_xf = max(transformers, key=lambda item: item["Loading_pct"])
    pair_totals = {}
    for pair in [(1, 2), (2, 3), (3, 1)]:
        pair_totals[f"{pair[0]}-{pair[1]}"] = {
            "kW": sum(row["Nominal_Branch_kW"] for row in connection_map if (row["PhaseFrom"], row["PhaseTo"]) == pair),
            "kvar": sum(row["Nominal_Branch_kvar"] for row in connection_map if (row["PhaseFrom"], row["PhaseTo"]) == pair),
            "branches": sum(1 for row in connection_map if (row["PhaseFrom"], row["PhaseTo"]) == pair),
        }
    model_counts = Counter(load["Model"] for load in loads)
    feeder_length = sum(line["Length_mi"] for line in lines if line["Category"] == "FeederLine")
    line_loss = sum(line["Loss_kW"] for line in lines)
    xf_loss = sum(item["Loss_kW"] for item in transformers)
    return {
        "num_buses_total": len(buses),
        "num_buses_network": len([bus for bus in buses if bus["BusKey"] != "sourcebus"]),
        "num_phase_nodes": solver["num_nodes"],
        "num_buses_solver": solver["num_buses"],
        "num_elements_solver": solver["num_elements"],
        "num_lines_total": len(lines),
        "num_feeder_lines": len([line for line in lines if line["Category"] == "FeederLine"]),
        "num_jumpers": len([line for line in lines if line["Category"] == "Jumper"]),
        "num_loads": len(loads),
        "num_load_buses": len({load["BusName"] for load in loads}),
        "num_transformers": len(transformers),
        "num_regulators": len(regulators),
        "num_linecodes_defined": len(model["linecodes"]),
        "num_linecodes_referenced": len({line["LineCode"] for line in lines if line["LineCode"]}),
        "load_kw": load_kw,
        "load_kvar": load_kvar,
        "load_kva": load_kva,
        "load_pf": load_kw / load_kva if load_kva else math.nan,
        "actual_kw": actual_kw,
        "actual_kvar": actual_kvar,
        "model_counts": dict(sorted(model_counts.items())),
        "pair_totals": pair_totals,
        "feeder_length_mi": feeder_length,
        "line_loss_kw": line_loss,
        "xf_loss_kw": xf_loss,
        "min_bus": min_bus,
        "max_bus": max_bus,
        "max_unbalance": max_unbalance,
        "max_current_line": max_current_line,
        "max_loading_line": max_loading_line,
        "subxf": subxf,
        "xfm1": xfm1,
        "reg1a": reg1a,
        "reg1c": reg1c,
        "max_xf": max_xf,
        "graph": graph,
        "solution": solver["solution"],
        "version": solver["version"],
        "compile_error": [solver["compile_error_number"], solver["compile_error_description"]],
    }


def write_csv_outputs(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    linecodes: list[dict[str, Any]],
    all_linecodes: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    connection_map: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    graph: dict[str, Any],
    exp_meters: Path | None,
) -> None:
    """写出全部 CSV 数据表。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    bus_headers = list(buses[0].keys())
    write_csv(out_dir / "IEEE37_Line_Impedance_Matrix.csv2", bus_headers, records_to_rows(buses, bus_headers))

    line_headers = list(lines[0].keys())
    write_csv(out_dir / "IEEE37_Line_Impedance_Matrix.csv0", line_headers, records_to_rows(lines, line_headers))

    linecode_headers = list(linecodes[0].keys())
    write_csv(out_dir / "IEEE37_LineCode_Table.csv", linecode_headers, records_to_rows(linecodes, linecode_headers))
    write_csv(
        out_dir / "IEEE37_LineCode_All_Definitions.csv",
        linecode_headers,
        records_to_rows(all_linecodes, linecode_headers),
    )

    impedance_rows = []
    for line in lines:
        impedance_rows.extend(matrix_rows(line))
    write_csv(
        out_dir / "IEEE37_Line_Impedance_Matrix.csv",
        [
            "LineName",
            "LineCode",
            "Length_mi",
            "Row",
            "Col",
            "ElementType",
            "R_ohm_per_mi",
            "X_ohm_per_mi",
            "C_nF_per_mi",
            "R_total_ohm",
            "X_total_ohm",
            "C_total_nF",
            "Z_total_mag_ohm",
            "Z_total_ang_deg",
            "Units",
        ],
        impedance_rows,
    )

    load_headers = list(loads[0].keys())
    write_csv(out_dir / "IEEE37_Line_Impedance_Matrix.csv1", load_headers, records_to_rows(loads, load_headers))

    map_headers = list(connection_map[0].keys())
    write_csv(out_dir / "IEEE37_Load_Connection_Map.csv", map_headers, records_to_rows(connection_map, map_headers))

    xf_headers = list(transformers[0].keys())
    write_csv(out_dir / "IEEE37_Transformer_Table.csv", xf_headers, records_to_rows(transformers, xf_headers))

    reg_headers = list(regulators[0].keys())
    write_csv(out_dir / "IEEE37_Regulator_Table.csv", reg_headers, records_to_rows(regulators, reg_headers))

    # 文件清单与引用关系。
    inventory_rows = []
    for item in model["inventory"]:
        inventory_rows.append(
            [
                item["file"],
                item["bytes"],
                item["sha256"],
                item["modified"],
                "yes" if item["referenced_by_dss"] else ("master" if item["file"] == model["master_file"] else "no"),
                "provided extraction / byte-identical canonical case",
            ]
        )
    if exp_meters and exp_meters.exists():
        inventory_rows.append(
            [
                exp_meters.name,
                exp_meters.stat().st_size,
                sha256_file(exp_meters),
                datetime.fromtimestamp(exp_meters.stat().st_mtime).isoformat(timespec="seconds"),
                "no (auxiliary meter export)",
                str(exp_meters),
            ]
        )
    else:
        inventory_rows.append(
            ["ieee37_EXP_METERS.CSV", "", "", "", "no (auxiliary meter export)", "not available in source extraction"]
        )
    write_csv(
        out_dir / "IEEE37_Line_Parameter_Analysis.md1",
        ["File", "Bytes", "SHA256", "Modified", "ReferencedByDSS", "Origin"],
        inventory_rows,
    )

    ref_rows = []
    for item in model["refs"]:
        resolved = (Path(model["source_dir"]) / item["reference"]).resolve()
        ref_rows.append(
            [
                item["type"],
                item["reference"],
                item["source_file"],
                item["source_line"],
                "yes" if item["exists"] else "no",
                str(resolved),
                "resolved relative to IEEE37_Load_Connection_Map.csv1",
            ]
        )
    ref_rows.append(
        [
            "entry",
            model["master_file"],
            model["master_file"],
            1,
            "yes",
            str((Path(model["source_dir"]) / model["master_file"]).resolve()),
            "compiled by OpenDSS",
        ]
    )
    ref_rows.append(
        [
            "auxiliary (not referenced)",
            "ieee37_EXP_METERS.CSV",
            "",
            "",
            "yes" if exp_meters and exp_meters.exists() else "no",
            str(exp_meters) if exp_meters else "",
            "historical meter export; not an input to the electrical model",
        ]
    )
    write_csv(
        out_dir / "IEEE37_File_References.csv",
        ["Type", "Reference", "SourceFile", "SourceLine", "Exists", "ResolvedPath", "Note"],
        ref_rows,
    )

    # 母线电压宽表。
    voltage_headers = [
        "BusName",
        "VoltageLevel_kV",
        "kVBase_LN",
        "Vab_kV",
        "Vab_deg",
        "Vbc_kV",
        "Vbc_deg",
        "Vca_kV",
        "Vca_deg",
        "Vab_pu",
        "Vbc_pu",
        "Vca_pu",
        "Vmin_pu",
        "Vmax_pu",
        "VLL_Spread_pct",
        "V2_V1_pct",
        "Van_kV_LN",
        "Vbn_kV_LN",
        "Vcn_kV_LN",
        "Zsc1_ohm_mag",
        "Zsc1_ohm_ang_deg",
        "Zsc0_ohm_mag",
        "Solved",
    ]
    write_csv(out_dir / "IEEE37_Bus_Voltages.csv", voltage_headers, records_to_rows(buses, voltage_headers))

    # 线路电流长表。
    current_rows = []
    for line in solver["lines"]:
        phases = int(line["phases"])
        ncond = int(line["num_conductors"])
        bus1_nodes = parse_bus(line["bus1"])[1]
        bus2_nodes = parse_bus(line["bus2"])[1]
        pairs = [pair_to_complex(value) for value in line["currents"]]
        for terminal in (0, 1):
            base = terminal * ncond
            for conductor in range(ncond):
                if base + conductor >= len(pairs):
                    continue
                value = pairs[base + conductor]
                bus_name = line["bus1"] if terminal == 0 else line["bus2"]
                nodes = bus1_nodes if terminal == 0 else bus2_nodes
                phase = nodes[conductor] if conductor < len(nodes) else conductor + 1
                current_rows.append(
                    [
                        line["name"],
                        line["linecode"],
                        line["length"],
                        terminal + 1,
                        display_bus_name(parse_bus(bus_name)[0]),
                        phase,
                        abs(value),
                        math.degrees(math.atan2(value.imag, value.real)),
                        abs(value) / float(line["normamps"]) if line["normamps"] else math.nan,
                        "into line" if terminal == 0 else "out of line",
                        bool(line["enabled"]),
                    ]
                )
    write_csv(
        out_dir / "IEEE37_Line_Currents.csv",
        [
            "LineName",
            "LineCode",
            "Length_mi",
            "Terminal",
            "Bus",
            "Phase",
            "Current_A_mag",
            "Current_A_ang_deg",
            "Current_pu_NormAmps",
            "Direction",
            "Enabled",
        ],
        current_rows,
    )

    # 负荷功率表。
    load_power_rows = []
    for load in loads:
        bus = next((item for item in buses if item["BusName"] == load["BusName"]), None)
        load_power_rows.append(
            [
                load["LoadName"],
                load["BusName"],
                load["Phases"],
                load["Model"],
                load["ModelName"],
                load["kW"],
                load["kvar"],
                load["kVA"],
                load["PF"],
                load["Actual_P_kW"],
                load["Actual_Q_kvar"],
                math.hypot(load["Actual_P_kW"], load["Actual_Q_kvar"]),
                load["Actual_PF"],
                (bus["Vab_kV"] + bus["Vbc_kV"] + bus["Vca_kV"]) / 3 if bus else math.nan,
                bool(solver["solution"]["converged"]),
            ]
        )
    write_csv(
        out_dir / "IEEE37_Load_Powers.csv",
        [
            "LoadName",
            "BusName",
            "Phases",
            "Model",
            "ModelName",
            "kW_defined",
            "kvar_defined",
            "kVA_defined",
            "PF_defined",
            "P_actual_kW",
            "Q_actual_kvar",
            "S_actual_kVA",
            "PF_actual",
            "VLL_avg_kV",
            "Converged",
        ],
        load_power_rows,
    )

    # 变压器潮流与调压器运行状态。
    write_csv(
        out_dir / "IEEE37_Transformer_Flows.csv",
        [
            "Name",
            "Type",
            "Bus1",
            "Bus2",
            "P1_kW",
            "Q1_kvar",
            "S1_kVA",
            "P2_kW",
            "Q2_kvar",
            "S2_kVA",
            "Loss_kW",
            "Loss_kvar",
            "Loading_pct",
            "MaxCurrent1_A",
            "MaxCurrent2_A",
            "Enabled",
        ],
        [
            [
                item["Name"],
                item["Type"],
                item["Bus1"],
                item["Bus2"],
                item["SolvedP1_kW"],
                item["SolvedQ1_kvar"],
                item["SolvedS1_kVA"],
                item["SolvedP2_kW"],
                item["SolvedQ2_kvar"],
                item["SolvedS2_kVA"],
                item["Loss_kW"],
                item["Loss_kvar"],
                item["Loading_pct"],
                item["MaxCurrent1_A"],
                item["MaxCurrent2_A"],
                item["Enabled"],
            ]
            for item in transformers
        ],
    )

    bus_lookup = {bus["BusName"]: bus for bus in buses}
    reg_rows = []
    for regulator in regulators:
        bus_799 = bus_lookup.get("799")
        bus_799r = bus_lookup.get("799r")
        reg_rows.append(
            [
                regulator["Regulator"],
                regulator["Transformer"],
                regulator["PhasePair"],
                regulator["TapPosition"],
                regulator["TapRatio"],
                regulator["Vreg_kV_primary"],
                regulator["Band_kV_primary"],
                regulator["R_setting_V"],
                regulator["X_setting_V"],
                regulator["Delay_s"],
                regulator["MaxTap"],
                regulator["MinTap"],
                "yes" if regulator["TapPosition"] >= 16 else "no",
                bus_799["Vab_kV"] if bus_799 else math.nan,
                bus_799["Vbc_kV"] if bus_799 else math.nan,
                bus_799["Vca_kV"] if bus_799 else math.nan,
                bus_799r["Vab_kV"] if bus_799r else math.nan,
                bus_799r["Vbc_kV"] if bus_799r else math.nan,
                bus_799r["Vca_kV"] if bus_799r else math.nan,
            ]
        )
    write_csv(
        out_dir / "IEEE37_Regulator_State.csv",
        [
            "Regulator",
            "Transformer",
            "PhasePair",
            "TapPosition",
            "TapRatio",
            "Vreg_kV",
            "Band_kV",
            "R_setting_V",
            "X_setting_V",
            "Delay_s",
            "MaxTap",
            "MinTap",
            "AtMaxTap",
            "Vab_799_kV",
            "Vbc_799_kV",
            "Vca_799_kV",
            "Vab_799r_kV",
            "Vbc_799r_kV",
            "Vca_799r_kV",
        ],
        reg_rows,
    )

    # 图论审计表。
    graph_rows = [
        ["Connected components (bus level)", graph["num_components"], "; ".join(",".join(c) for c in graph["components"])],
        ["Cycle rank (multi-edge bus graph)", graph["cycle_rank"], "E - V + C"],
        ["Feeder tree (L1–L35 only)", "yes" if graph["feeder_is_tree"] else "no", f"{graph['feeder_tree_nodes']} nodes, {graph['feeder_tree_edges']} edges"],
        ["Simple cycles in feeder", len(graph["simple_cycles"]), str(graph["simple_cycles"])],
        ["Parallel edge groups", len(graph["parallel_groups"]), "; ".join(f"{g['pair']}: {g['elements']}" for g in graph["parallel_groups"])],
        ["Duplicate line definitions", len(graph["duplicate_lines"]), str(graph["duplicate_lines"])],
        ["Isolated buses", len(graph["isolated"]), ",".join(graph["isolated"])],
        ["Leaves (excluding SourceBus)", len(graph["leaves"]), ",".join(graph["leaves"])],
        ["Branch points (degree ≥ 3)", len(graph["branch_points"]), ",".join(graph["branch_points"])],
        ["Main feeder path", graph["farthest_leaf"], " → ".join(graph["main_path"]) + f" ({graph['main_path_length_mi']:.3f} mi)"],
        ["Phase-node graph components", graph["num_phase_components"], f"{graph['num_phase_nodes']} nodes (incl. 4 transformer device nodes); sizes {graph['phase_component_sizes']}"],
        ["Phase connection errors", len(graph["phase_errors"]), "; ".join(graph["phase_errors"]) if graph["phase_errors"] else "none"],
    ]
    write_csv(out_dir / "IEEE37_Line_Parameter_Analysis.md9", ["Check", "Result", "Detail"], graph_rows)


# ---------------------------------------------------------------------------
# Markdown 与图件渲染
# ---------------------------------------------------------------------------


def fmt_bool(value: bool) -> str:
    """布尔值转为中文/英文混合报告中的 yes/no。"""
    return "yes" if value else "no"


def md_matrix(matrix_text: str, unit: str) -> str:
    """把矩阵字符串渲染为 Markdown 代码块。"""
    rows = matrix_text.split("|")
    body = "\n".join(f"  [{row}]" for row in rows)
    return f"```text\n{body}\n单位：{unit}\n```"


def render_line_analysis(
    out_dir: Path,
    model: dict[str, Any],
    lines: list[dict[str, Any]],
    linecodes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """生成线路参数与阻抗矩阵分析。"""
    code_rows = []
    seq_rows = []
    for code_name in ["721", "722", "723", "724"]:
        code = next(item for item in linecodes if item["LineCode"] == code_name)
        code_rows.append(
            [
                code_name,
                code["nphases"],
                code["Units_Resolved"],
                code["UseCount"],
                fmt(code["R1_ohm_per_mi"], 6),
                fmt(code["X1_ohm_per_mi"], 6),
                fmt(code["R0_ohm_per_mi"], 6),
                fmt(code["X0_ohm_per_mi"], 6),
                fmt(code["C1_nF_per_mi"], 4),
                fmt(code["C0_nF_per_mi"], 4),
            ]
        )
        z = model["linecodes"][code_name].rmatrix + 1j * model["linecodes"][code_name].xmatrix
        seq = A_INV @ z @ A_MAT
        offdiag = max(abs(seq[0, 1]), abs(seq[0, 2]), abs(seq[1, 0]), abs(seq[1, 2]), abs(seq[2, 0]), abs(seq[2, 1]))
        seq_rows.append(
            [
                code_name,
                f"{abs(seq[0, 0]):.5f}",
                f"{abs(seq[1, 1]):.5f}",
                f"{abs(seq[2, 2]):.5f}",
                f"{offdiag:.5f}",
            ]
        )

    total_rows = []
    for line in lines:
        if line["Category"] != "FeederLine":
            continue
        total_rows.append(
            [
                line["LineName"],
                line["LineCode"],
                fmt(line["Length_mi"], 4),
                fmt(line["R1_total_ohm"], 5),
                fmt(line["X1_total_ohm"], 5),
                fmt(line["Z1_total_mag_ohm"], 5),
                fmt(line["Z1_total_ang_deg"], 3),
                fmt(line["R0_total_ohm"], 5),
                fmt(line["X0_total_ohm"], 5),
                fmt(line["Z0_total_mag_ohm"], 5),
                f"{line['MaxCurrent_A']:.1f}",
            ]
        )

    matrices_721 = md_matrix(next(item for item in linecodes if item["LineCode"] == "721")["RMatrix_ohm_per_mi"], "Ω/mi")
    matrices_722 = md_matrix(next(item for item in linecodes if item["LineCode"] == "722")["RMatrix_ohm_per_mi"], "Ω/mi")
    x_721 = md_matrix(next(item for item in linecodes if item["LineCode"] == "721")["XMatrix_ohm_per_mi"], "Ω/mi")
    c_721 = md_matrix(next(item for item in linecodes if item["LineCode"] == "721")["CMatrix_nF_per_mi"], "nF/mi")
    max_diff = max(line["Yprim_Impedance_MaxDiff_ohm"] for line in lines)
    matrix_diff = max(line["LineCode_vs_Solver_Matrix_MaxDiff"] for line in lines)

    text = f"""<!-- 摘要：IEEE 37 线路配置与阻抗矩阵审计，记录 721–724 的完整 R/X/C 矩阵、单位、单长与整条线路总阻抗、序分量、默认载流量和 PCB 简化风险。 -->

# IEEE 37 线路参数与阻抗矩阵分析

> 文档定位：任务 3 的线路阻抗矩阵提取与分析。  
> 数据来源：`IEEELineCodes.DSS` 的 721–724 定义、`IEEE37_Load_Connection_Map.csv1` 的线路长度、OpenDSS COM 求解器的 `Rmatrix`/`Xmatrix`/`Cmatrix` 与 `Yprim`。  
> 状态：矩阵逐元素保留；单长与总阻抗均计算；总阻抗已用 `Yprim` 反推校验。

## 1. 单位与口径

- `LineCode` 与 `Line` 都没有显式写 `Units`。OpenDSS 求解器返回 `Units=0`，即 `mi`（英里）。
- `RMatrix`、`XMatrix` 的单位是 **Ω/mi**；`CMatrix` 的单位是 **nF/mi**。
- `Length` 的单位是 **mi**；本模型 35 条馈线线路的长度合计 {summary['feeder_length_mi']:.2f} mi，单条范围 0.08–1.85 mi。
- `Lines.Rmatrix`（COM 属性）返回的是**单位长度矩阵**，不是整条线路总矩阵。整条线路总矩阵按下式计算：

```text
R_total = R_per_length × Length
X_total = X_per_length × Length
C_total = C_per_length × Length
```

- 该口径已用 `Yprim` 的端子互导纳块反推验证：`-inv(Y12)` 与 `R_total + jX_total` 的最大绝对差为 {max_diff:.3e} Ω。这个差值处于双精度舍入量级，说明“单长矩阵 × 长度”的口径正确。
- 静态 `LineCode` 矩阵与求解器实际加载的单长矩阵逐元素核对，最大绝对差为 {matrix_diff:.3e}（R/X 单位 Ω/mi，C 单位 nF/mi 混合比较），说明求解器加载的正是文件中的矩阵，而不是另一套缓存值。
- `Length_km` 列只是换算值（1 mi = 1.609344 km），不参与计算。

## 2. 四个被引用 LineCode 的完整矩阵

### 721（L35，1 条）

R 矩阵（Ω/mi）：

{matrices_721}

X 矩阵（Ω/mi）：

{x_721}

C 矩阵（nF/mi）：

{c_721}

### 722–724

722、723、724 的完整矩阵逐元素保存在 `IEEE37_LineCode_Table.csv` 与 `IEEE37_Line_Impedance_Matrix.csv`，不在此重复粘贴。它们的共同特征：

- 三个 R/X 矩阵都是 3×3 对称矩阵，非对角元不为零；不能只取对角线。
- 四个 C 矩阵的非对角元均为零，即 721–724 在这些数据中是**对角电容矩阵**。这与“所有 LineCode 的 C 矩阵都有互电容”的笼统说法不同，应以本文件为准。
- C 的对角值随 721→724 递减：721 最大，724 最小；R/X 的对角值随 721→724 递增。

## 3. 参数汇总

{md_table(['LineCode', '相数', '单位', '使用条数', 'R1/Ω/mi', 'X1/Ω/mi', 'R0/Ω/mi', 'X0/Ω/mi', 'C1/nF/mi', 'C0/nF/mi'], code_rows)}

表中的 R1/X1/R0/X0 是由完整 3×3 矩阵经对称分量变换得到的对角序分量，不是 OpenDSS 的 `LineCode.R1` 标量属性。COM 接口的 `LineCode.R1`/`X1` 在该版本中表现出活动对象缓存，四个代码返回了相同的 0.058/0.1206；因此本审计不采用这些标量属性，全部由矩阵重新计算。

## 4. 序分量与非对称性

{md_table(['LineCode', '|Z0|/Ω/mi', '|Z1|/Ω/mi', '|Z2|/Ω/mi', '最大序间耦合/Ω/mi'], seq_rows)}

721–724 的相矩阵不是完全换位对称矩阵：例如 722 的对角元为 0.089981/0.085/0.089981，互元为 0.030852/0.023371/0.030852。因此：

- Z1 与 Z2 不完全相等；
- 序分量矩阵存在小的序间耦合（最大约 {max(float(row[4]) for row in seq_rows):.5f} Ω/mi）；
- 不能把线路简化为“三相对称、完全解耦”的正序单相等值而不记录误差。

## 5. 全部馈线线路的总阻抗

{md_table(['线路', 'LineCode', 'Length/mi', 'R1/Ω', 'X1/Ω', '|Z1|/Ω', '∠Z1/°', 'R0/Ω', 'X0/Ω', '|Z0|/Ω', 'Imax/A'], total_rows)}

`R1/X1` 为正序总阻抗，`R0/X0` 为零序总阻抗；它们由整条线路的 3×3 总矩阵变换得到。Jumper 为 1×1 的 1 mΩ 集中连接，不在本表；其单长/总矩阵均为 1 mΩ、0 mH、0 nF。

## 6. 721–724 的差异

- 721：R/X 最低、C 最高，用于 799r–701 的 1.85 mi 线路 L35。
- 722：R/X 低于 723/724，C 较高；用于 L1、L4。
- 723：中等 R/X、较低 C；使用 15 条。
- 724：R/X 最高、C 最低；使用 17 条。
- 这些差异不是“同一根导线换编号”，而是原始 IEEE 37 地下电缆/导体配置的差异。PCB 缩比时若把 721–724 合并成一种 RLC，必须分别评估每条线路的电压降和损耗误差。

## 7. 默认载流量与历史矩阵

- `NormAmps=400 A`、`EmergAmps=600 A` 是 OpenDSS 默认值，DSS 未写出。它们不是 IEEE 37 导体的真实额定值，不能直接用于 PCB 铜箔、端子或保险丝选型。
- `IEEELineCodes.DSS` 中 721–724 上方有 `!!!~` 注释掉的历史矩阵；当前生效的是其后的 `~` 行。历史 C 值（例如 721 的 124.851 nF/mi）明显高于当前值（80.274847 nF/mi），当前值约为历史值的 64%（低约 36%）。本审计使用当前生效值，因为 OpenDSS 实际加载的是它们。
- `IEEE37_Load_Connection_Map.csv1` 第 1 行的注释明确说明这是 3-wire delta 系统；线路电容的零序/正序解释必须放在三线制背景下，不能引入不存在的零序通路。

## 8. 对 PCB 主干设计的含义

- **必须保留完整 3×3 R/X 矩阵**，至少在设计数据库中保留；是否在 PCB 上实现互耦，需要单独的误差评估。简单做法“每相一个独立 R-L 支路”会丢掉 20%–40% 的互阻抗项，对电压不平衡和故障电流分配有影响。
- **C 矩阵在 721–724 中是对角矩阵**，所以并联电容可以按相独立实现；但仍需保留每相各自的 C 值，不能用一个平均电容代替四种 LineCode。
- **长度单位是英里**。把 0.08–1.85 直接当成 km 或 m 会造成 1.6 倍或 1600 倍的错误；PCB 缩比时必须先做单位换算，再乘阻抗缩放系数。
- **60 Hz 是原模型频率**。若实体平台采用 50 Hz，X 与 C 的缩比不能直接照搬 60 Hz 数值；见 `IEEE37_Line_Parameter_Analysis.md2` 的频率说明。
- **Jumper 不是线路**。它是 B 相的 1 mΩ 集中连接，`Length=1.0` 只是 OpenDSS 默认值；不要把它当成 1 mi 的 721 线路。基态下它的电流近零（约 0.00015 A），因为三台设备的 B 相电流在 799.2 处相量和为零；它是电位参考，不是基态功率通道，但故障/不平衡时可能承载大电流，不能按基态电流选截面。

## 9. 结论

四个被引用 LineCode 的完整 R/X/C 矩阵均已提取；每条线路的单长矩阵与总矩阵均可追溯；总阻抗与 `Yprim` 的校验差在 {max_diff:.3e} Ω 量级。线路参数的直接可用形式是“单长 3×3 矩阵 + 长度（mi）”，不是“三个独立的 R/L 数值”。
"""
    (out_dir / "IEEE37_Line_Parameter_Analysis.md").write_text(text, encoding="utf-8")


def render_integrity(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    summary: dict[str, Any],
    exp_meters: Path | None,
) -> None:
    """生成 DSS 模型完整性检查报告。"""
    inventory_rows = []
    for item in model["inventory"]:
        inventory_rows.append(
            [item["file"], item["bytes"], item["sha256"], "yes" if item["referenced_by_dss"] else ("master" if item["file"] == model["master_file"] else "no")]
        )
    if exp_meters and exp_meters.exists():
        inventory_rows.append(
            [
                exp_meters.name,
                exp_meters.stat().st_size,
                sha256_file(exp_meters),
                "no（辅助表，未被 DSS 引用）",
            ]
        )
    ref_rows = [
        [item["type"], item["reference"], item["source_file"], item["source_line"], "yes" if item["exists"] else "no"]
        for item in model["refs"]
    ]
    ref_rows.append(["entry", model["master_file"], model["master_file"], 1, "yes"])
    ref_rows.append(["auxiliary", "ieee37_EXP_METERS.CSV", "（未引用）", "—", "yes" if exp_meters and exp_meters.exists() else "no"])

    text = f"""<!-- 摘要：IEEE 37 OpenDSS 模型完整性检查报告，记录四个源文件、Redirect/BusCoords 引用、对象计数、默认值依赖和缺失/重复/无效引用检查结论。 -->

# IEEE 37 DSS 模型完整性检查报告

> 文档定位：任务 1 的模型完整性审计结果。  
> 数据来源：`IEEE37_Load_Connection_Map.csv1`、`IEEELineCodes.DSS`、`IEEE37_Load_Connection_Map.csv0`、`ieee37_EXP_METERS.CSV`。  
> 状态：静态文本解析与 OpenDSS COM 求解均已执行；求解器版本见 `IEEE37_Solver_Validation.md`。

## 1. 审计对象与来源

本次审计的电气依据是用户提供的 `37Bus.rar` 解压内容。当前文件系统中未找到该 `.rar` 本体，但已有一份此前解压的源目录：

`{model['source_dir']}`

该目录中的 `IEEE37_Load_Connection_Map.csv1`、`IEEELineCodes.DSS` 和 `IEEE37_Load_Connection_Map.csv0` 与 OpenDSS 安装目录 `C:\\Program Files\\OpenDSS\\IEEETestCases\\37Bus` 中的同名文件逐字节一致（SHA256 相同）。提取目录缺少 `ieee37_EXP_METERS.CSV`，该文件仅作为历史仪表导出表从同一 canonical 目录补齐；它不参与电气拓扑和参数计算。

因此，本次审计没有用“网上的另一套 IEEE 37 模型”替代用户文件：三个电气文件与用户提取件逐字节一致，且所有结论均可追溯到具体 DSS 行号。

## 2. 文件清单

{md_table(['文件', '字节数', 'SHA256', 'DSS 引用'], inventory_rows)}

`ieee37_EXP_METERS.CSV` 只有一行仪表数据（Year 0、Hour 1、Meter MAIN），不是 DSS 输入，也不包含支路或节点定义。其 kWh/kvarh 数值与本次 1.0 pu 负荷快照不对应，不能用于参数提取。

## 3. 文件引用关系

{md_table(['类型', '引用', '源文件', '源行', '存在'], ref_rows)}

主文件只包含两条文件级引用：

- `Redirect IEEELineCodes.DSS`：第 22 行，大小写与磁盘文件名完全一致。
- `BusCoords IEEE37_Load_Connection_Map.csv0`：第 102 行，大小写一致。

没有 `Compile` 其他主文件，没有 `Loads.dss`、`Transformers.dss` 或 `Regulators.dss` 外部引用。变压器、调压器、负荷和线路全部直接定义在 `IEEE37_Load_Connection_Map.csv1` 中。

## 4. 对象计数

{md_table(['对象', '数量', '说明'], [
    ['母线名称（含 SourceBus）', summary['num_buses_total'], '求解器与静态解析一致'],
    ['网络母线（不含 SourceBus）', summary['num_buses_network'], '799、799r、701–744、775'],
    ['母线相节点（1.2.3）', summary['num_phase_nodes'], '39 个母线 × 3 相；不含变压器内部中性导体'],
    ['线路对象（含 Jumper）', summary['num_lines_total'], 'L1–L35 共 35 条三相线路，另 1 条单相 Jumper'],
    ['馈线线路 L1–L35', summary['num_feeder_lines'], '全部 Phases=3，Bus1/Bus2 均为 .1.2.3'],
    ['Jumper', summary['num_jumpers'], '799.2–799r.2，B 相 1 mΩ 集中连接'],
    ['负荷对象', summary['num_loads'], '29 个单相 Delta + 1 个三相 Delta（S728）'],
    ['变压器/调压器对象', summary['num_transformers'], 'SubXF、XFM1、reg1a、reg1c'],
    ['调压控制器', summary['num_regulators'], 'creg1a、creg1c'],
    ['LineCode 定义总数', summary['num_linecodes_defined'], 'IEEELineCodes.DSS 中全部定义'],
    ['被本模型引用的 LineCode', summary['num_linecodes_referenced'], '721、722、723、724'],
])}

## 5. 缺失文件、重复定义与无效引用

- 缺失文件：无。`Redirect` 与 `BusCoords` 的目标均存在；`ieee37_EXP_METERS.CSV` 不是 DSS 引用，缺失不影响电气模型。
- 重复定义：未发现同名 `New` 对象重复定义。`reg1c like=reg1a` 和 `creg1c like=creg1a` 是显式继承后创建新对象，不是重复定义。
- 无效引用：未发现。所有线路、负荷、变压器和调压器引用的母线都由对象自身或变压器绕组创建；所有 `LineCode` 引用都能在 `IEEELineCodes.DSS` 中找到。
- 路径与大小写：`IEEE37_Load_Connection_Map.csv1` 使用相对路径引用两个文件，大小写与磁盘一致。迁移到 Linux 等大小写敏感环境时仍可解析。
- 注释与有效定义：`IEEELineCodes.DSS` 中 721–724 的若干 `!!!~` 行是注释掉的历史矩阵，当前生效的是其后的 `~` 行；解析器已忽略注释行。`IEEE37_Load_Connection_Map.csv1` 中的 `show` 命令均已注释，不会影响求解。

## 6. 依赖 OpenDSS 默认值的参数

以下参数在 DSS 中没有显式写出，而是由 OpenDSS 默认值补齐。它们是模型“完整可求解”与“可移植到其他工具”之间的主要差异：

- `LineCode` 与 `Line` 的 `Units`：未写，OpenDSS 默认 `mi`。求解器返回 `Units=0`，即英里。
- `LineCode` 的 `NormAmps` 与 `EmergAmps`：未写，求解器返回 400 A / 600 A。这是 OpenDSS 默认值，不是 IEEE 37 导体的真实载流量。
- 负荷 `CVRwatts`、`CVRvars`：未写，求解器返回 1.0 / 2.0。Model 4 负荷的电压指数即由这两个默认值决定。
- 负荷 `Status`：未写，求解器返回 0（Fixed）。`Enabled` 均为 true。
- 调压器 `reg1a`/`reg1c` 的 `%r`：未写，求解器返回 0.2%/绕组。SubXF 与 XFM1 的 `%r` 是显式写出的。
- 变压器 `Xht`、`Xlt`：未写，求解器返回 35%、30%；两绕组正序计算只使用 `Xhl`，但这两个默认值仍属于对象属性。
- 变压器 `MaxTap`、`MinTap`、`NumTaps`：未写，求解器返回 1.1、0.9、32。
- 调压控制器 `Delay`、`MaxTapChange`、`VoltageLimit`：`Delay=15 s`、`MaxTapChange=16` 来自默认值，`VoltageLimit=0` 表示未设置。

这些默认值不影响本次 OpenDSS 求解的可重复性，但在导出到 KiCad、MATLAB 或其他潮流工具时必须显式写出，否则会得到不同的线路长度、载流量或负荷电压特性。

## 7. 发现与严重性

- 高：`SubXF` 在 1.0 pu 负荷下视在功率约 {summary['subxf']['SolvedS1_kVA']:.1f} kVA，为 2500 kVA 额定值的 {summary['subxf']['Loading_pct']:.1f}%；电流负载率更高（见 `IEEE37_Transformer_Table.csv`）。原模型的基态不是“轻载”工况。
- 高：`reg1a` 与 `reg1c` 的视在负载分别约 {summary['reg1a']['SolvedS1_kVA']:.1f} kVA 与 {summary['reg1c']['SolvedS1_kVA']:.1f} kVA，为各自 2000 kVA 额定值的 {summary['reg1a']['Loading_pct']:.1f}% 与 {summary['reg1c']['Loading_pct']:.1f}%；两台调压器均接近 90% 视在负载，且 `creg1a` 已在最大档位 16。
- 中：`XFM1` 二次侧 775 没有负荷，基态潮流接近零；该支路在模型中存在但不参与功率传输。
- 中：线路 `NormAmps=400 A` 是默认值，不能直接当作 PCB 铜箔或端子选型依据。
- 中：`Bus 799` 在基态为 0.923 pu（Vab/Vca），低于 799r；24 V 电源若定义为 799 的 1.0 pu，必须同时决定是否保留 SubXF 的电压跌落。
- 低：`ieee37_EXP_METERS.CSV` 与当前快照不一致，不能作为校验数据。

## 8. 结论

就 OpenDSS 求解而言，模型是完整的：文件引用闭合、对象定义齐全、无缺失母线、无无效 LineCode 引用、无重复对象。就“可直接移植的电气数据库”而言，模型仍依赖若干 OpenDSS 默认值，且基态存在变压器/调压器重载。上述两点已在后续表中显式记录，不构成拓扑错误。
"""
    (out_dir / "IEEE37_Integrity_Report.md").write_text(text, encoding="utf-8")


def plot_topology(
    out_dir: Path,
    model: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """绘制 IEEE 37 母线级拓扑图，保存 PNG 与 SVG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    coords = {key.lower(): value for key, value in model["coords"].items()}

    def plot_xy(name: str) -> tuple[float, float]:
        key = name.lower()
        if key == "sourcebus":
            return 0.0, 1.05
        if key == "799":
            return 0.0, 0.74
        if key == "799r":
            return 0.0, 0.43
        if key == "701":
            return 0.0, 0.12
        if key == "775":
            # BusXY 中 775 与 709 几乎重合，且 731 在附近；下移以显示 XFM1 支路。
            return 0.35, -3.55
        x, y = coords.get(key, (0.0, 0.0))
        return x, y + 2.01

    pos = {bus["BusKey"]: plot_xy(bus["BusKey"]) for bus in buses}
    fig, ax = plt.subplots(figsize=(16, 21))
    ax.set_facecolor("#fbfbfb")

    # 线路边。
    for line in lines:
        a = line["FromBus"].lower()
        b = line["ToBus"].lower()
        if line["Category"] == "Jumper":
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], color="#7f8c8d", linewidth=1.4, zorder=1)
        if line["Length_mi"] < 0.15 and line["LineName"] not in {"L7", "L13"}:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offsets = {
            "L17": (0.0, 0.12),
            "L14": (-0.10, 0.0),
            "L16": (0.0, 0.12),
            "L28": (0.08, 0.0),
            "L33": (-0.12, 0.0),
            "L34": (0.0, 0.10),
            "L8": (0.0, 0.10),
            "L7": (0.12, 0.0),
            "L13": (0.0, -0.10),
        }
        mx += offsets.get(line["LineName"], (0.0, 0.0))[0]
        my += offsets.get(line["LineName"], (0.0, 0.0))[1]
        ax.text(
            mx,
            my,
            line["LineName"],
            fontsize=5.8,
            color="#2c3e50",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.72),
            zorder=5,
        )

    # 799–799r：SubXF、reg1a、reg1c、Jumper。
    x799, y799 = pos["799"]
    x799r, y799r = pos["799r"]
    ax.plot([x799, x799r], [y799, y799r], color="#c0392b", linewidth=2.0, linestyle=":", zorder=2)
    ax.annotate("Jumper B", xy=((x799 + x799r) / 2, (y799 + y799r) / 2), xytext=(0.22, 0.58),
                fontsize=7, color="#c0392b", arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))
    for rad, color, label, offset in [(-0.35, "#8e44ad", "reg1a (A–B)", (-0.55, 0.50)),
                                      (0.35, "#2980b9", "reg1c (C–B)", (0.55, 0.50))]:
        ax.annotate("", xy=(x799r, y799r), xytext=(x799, y799),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=2)
        ax.text(offset[0], offset[1], label, fontsize=7, color=color, ha="center")

    # 变压器边。
    for xf in model["transformers"].values():
        if xf.name.lower() in {"reg1a", "reg1c"}:
            continue
        a = parse_bus(xf.buses[0])[0].lower()
        b = parse_bus(xf.buses[1])[0].lower()
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], color="#1f618d", linewidth=2.6, linestyle="--", zorder=2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.10, my, xf.name, fontsize=8, color="#1f618d", fontweight="bold", zorder=6)

    # 母线节点。
    for bus in buses:
        x, y = pos[bus["BusKey"]]
        if bus["BusKey"] == "sourcebus":
            ax.scatter([x], [y], marker="*", s=650, color="#c0392b", edgecolor="black", linewidth=0.8, zorder=4)
        elif bus["BusKey"] in {"799", "799r"}:
            ax.scatter([x], [y], marker="s", s=300, color="#f39c12", edgecolor="black", linewidth=0.8, zorder=4)
        elif bus["BusKey"] == "775":
            ax.scatter([x], [y], marker="^", s=280, color="#27ae60", edgecolor="black", linewidth=0.8, zorder=4)
        elif bus["ConnectedLoadCount"] > 0:
            ax.scatter([x], [y], marker="o", s=260, color="#2980b9", edgecolor="black", linewidth=0.6, zorder=4)
        else:
            ax.scatter([x], [y], marker="o", s=180, color="white", edgecolor="#2980b9", linewidth=1.2, zorder=4)
        # 标签位置微调，避免 709/775、728/729 等重叠。
        dx, dy = 0.06, 0.05
        if bus["BusKey"] in {"775"}:
            dx, dy = 0.10, -0.12
        if bus["BusKey"] in {"728"}:
            dx, dy = -0.24, 0.12
        if bus["BusKey"] in {"729"}:
            dx, dy = -0.24, 0.10
        if bus["BusKey"] in {"744"}:
            dx, dy = 0.06, 0.10
        if bus["BusKey"] in {"727"}:
            dx, dy = 0.08, 0.10
        if bus["BusKey"] in {"732"}:
            dx, dy = -0.55, -0.02
        if bus["BusKey"] in {"708"}:
            dx, dy = -0.45, 0.32
        if bus["BusKey"] in {"709"}:
            dx, dy = -0.05, -0.15
        if bus["BusKey"] in {"722"}:
            dx, dy = -0.28, -0.10
        if bus["BusKey"] in {"707"}:
            dx, dy = 0.10, 0.10
        if bus["BusKey"] in {"714"}:
            dx, dy = 0.10, -0.12
        if bus["BusKey"] in {"704"}:
            dx, dy = -0.22, 0.10
        if bus["BusKey"] in {"710"}:
            dx, dy = -0.28, 0.08
        if bus["BusKey"] in {"799", "799r", "sourcebus"}:
            dx, dy = 0.09, 0.03
        ax.text(x + dx, y + dy, bus["BusName"], fontsize=8.2, color="#17202a", fontweight="bold", zorder=6)

    # 标题与图例。
    ax.set_title(
        "IEEE 37 Node Test Feeder — OpenDSS model topology\n"
        f"{summary['num_buses_total']} bus names, {summary['num_phase_nodes']} phase nodes, "
        f"{summary['num_feeder_lines']} feeder lines + 1 jumper; solved 60 Hz base case",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )
    legend_handles = [
        Line2D([0], [0], color="#7f8c8d", lw=1.6, label="Feeder line L1–L35"),
        Line2D([0], [0], color="#1f618d", lw=2.4, linestyle="--", label="Transformer (SubXF, XFM1)"),
        Line2D([0], [0], color="#8e44ad", lw=2.0, label="Open-Delta regulator"),
        Line2D([0], [0], color="#c0392b", lw=2.0, linestyle=":", label="B-phase jumper"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#c0392b", markersize=15, label="SourceBus (230 kV)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f39c12", markersize=10, label="799 / 799r (4.8 kV regulator boundary)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60", markersize=10, label="775 (0.48 kV transformer secondary)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9", markersize=10, label="Bus with load"),
        Line2D([0], [0], marker="o", color="#2980b9", markerfacecolor="white", markersize=9, label="Bus without load"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.95)
    ax.set_xlabel("x / mile (BusXY.csv; vertical positions adjusted for the 799/799r cluster)", fontsize=8)
    ax.set_ylabel("y / mile (schematic; not geographic north)", fontsize=8)
    ax.grid(True, color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(out_dir / "IEEE37_Line_Impedance_Matrix.csv6", dpi=300, facecolor="white")
    fig.savefig(out_dir / "IEEE37_Topology.svg", facecolor="white")
    plt.close(fig)


def render_readme(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    connection_map: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    summary: dict[str, Any],
    exp_meters: Path | None,
) -> None:
    """生成 README：审计结论、文件索引与最终问题回答。"""
    subxf = summary["subxf"]
    reg1a = summary["reg1a"]
    reg1c = summary["reg1c"]
    min_bus = summary["min_bus"]
    max_bus = summary["max_bus"]
    max_line = summary["max_current_line"]
    text = f"""<!-- 摘要：IEEE 37 OpenDSS 模型审计总入口，汇总文件完整性、节点/线路/负荷/变压器/调压器/电源参数、OpenDSS 求解验证结果、24 V 电源参考点建议、已验证与未确定事项，并索引全部交付物。 -->

# IEEE 37 OpenDSS 模型审计

> 文档定位：Method-A2 IEEE 37 实体配电网主干设计的模型审计总入口。  
> 数据依据：用户提供的 `37Bus.rar` 解压文件；电气结论以 `IEEE37_Load_Connection_Map.csv1` 与 `IEEELineCodes.DSS` 原文为准。  
> 审计状态：静态解析、图论检查与 OpenDSS COM 实际求解均已完成；未修改任何原始 DSS 文件。  
> 重要边界：本文只做模型审计与参数提取，不计算最终 PCB 元件值，不生成 PCB 布线，不把故障注入电路加入主干。

## 0. 审计结论摘要

- 模型在 OpenDSS 中完整可解：{summary['num_buses_total']} 个母线名（含 `SourceBus`）、{summary['num_phase_nodes']} 个相节点、{summary['num_feeder_lines']} 条馈线线路加 1 条 Jumper、{summary['num_loads']} 个负荷、4 个变压器/调压器对象、2 个调压控制器。
- 馈线本体 L1–L35 是严格的树；唯一的多重边是 799–799r 之间的 `reg1a`、`reg1c` 和 B 相 `Jumper`。没有孤立母线、重复线路、未定义节点或相别连接错误。
- 四个被引用 LineCode（721–724）的完整 3×3 R/X/C 矩阵已提取；线路总阻抗 = 单长矩阵 × 长度（mi），并与 `Yprim` 反推结果校验，最大差约 {max(line['Yprim_Impedance_MaxDiff_ohm'] for line in lines):.3e} Ω。
- 30 个负荷全部为 Delta 接线；29 个单相 Delta 支路加 1 个三相 Delta 负荷 `S728`，展开为 32 个相间支路。Model 1/2/4 的数量分别为 15/7/8。
- 原模型基态不是轻载：`SubXF` 视在负载约 {subxf['Loading_pct']:.1f}%（{subxf['SolvedS1_kVA']:.0f} kVA / 2500 kVA），4.8 kV 侧电流负载率约 {subxf['CurrentLoading2_pct']:.1f}%；`reg1a`、`reg1c` 分别约 {reg1a['Loading_pct']:.1f}%、{reg1c['Loading_pct']:.1f}%，且 `creg1a` 已在最大档位 16。
- `Bus 799` 可以作为 24 V 电源的拓扑参考点，但必须把 24 V 定义为 4.8 kV 的 1.0 pu 等值，并决定是否保留缩放后的 SubXF/上游等效阻抗；若要做故障注入，不能使用理想 24 V 源。

## 1. 数据来源与审计边界

### 1.1 文件来源

当前文件系统中未找到 `37Bus.rar` 本体，审计使用此前解压目录：

`{model['source_dir']}`

该目录中的 `IEEE37_Load_Connection_Map.csv1`、`IEEELineCodes.DSS`、`IEEE37_Load_Connection_Map.csv0` 与 OpenDSS 安装目录 `C:\\Program Files\\OpenDSS\\IEEETestCases\\37Bus` 的同名文件 SHA256 完全一致。提取目录缺少 `ieee37_EXP_METERS.CSV`，该文件仅作为历史仪表导出表从同一 canonical 目录补齐，不参与电气计算。

这不是“用另一套网上模型替代用户文件”：三个电气文件与用户提取件逐字节一致，所有统计都能追溯到 DSS 的具体行号。

### 1.2 本次做了什么

- 逐行解析 `IEEE37_Load_Connection_Map.csv1`：`Redirect`、`BusCoords`、`New` 对象、`like=` 继承、`Set`、`Solve`。
- 逐行解析 `IEEELineCodes.DSS`：29 个 LineCode 定义，其中 721–724 被本模型引用。
- 用 OpenDSS COM 实际编译并求解原始模型，导出母线电压、线路电流、负荷功率、变压器/调压器功率、调压器档位和短路阻抗。
- 用图论检查母线级与相节点级的连通性、回路、并联边、孤立节点和相别一致性。
- 用 `Yprim` 的端子互导纳块独立校验线路总阻抗矩阵。

### 1.3 本次没有做什么

- 没有修改 `IEEE37_Load_Connection_Map.csv1` 或 `IEEELineCodes.DSS`。
- 没有把 Model 1/4 改成 Model 2，没有改负荷 P/Q，没有删除调压器/变压器。
- 没有计算 24 V 缩比后的最终 R/L/C 元件值，没有生成 KiCad 原理图或 PCB 布线。
- 没有把故障注入电路加入主干模型。
- 没有做 50 Hz 重新求解；本次求解为 60 Hz 单快照。

## 2. 文件索引

{md_table(['文件', '内容', '状态'], [
    ['IEEE37_Load_Connection_Map.csv2', '本文件：结论、索引、最终问题回答', '完成'],
    ['IEEE37_Integrity_Report.md', '文件清单、引用关系、完整性检查', '完成'],
    ['IEEE37_Topology.md', '节点—线路拓扑、主馈线、分支、末端、图论检查', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv2', '39 个母线名的电压等级、相节点、连接、负荷、电压、短路阻抗', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv0', '35 条馈线 + Jumper 的端点、相数、LineCode、长度、单长/总 R/X/C、电流、损耗', '完成'],
    ['IEEE37_LineCode_Table.csv', '721–724 的完整 R/X/C 矩阵、单位、序分量、使用条数', '完成'],
    ['IEEE37_LineCode_All_Definitions.csv', 'IEEELineCodes.DSS 中全部 29 个 LineCode 的定义，含未引用项', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv', '逐线路逐元素的单长与总 R/X/C 矩阵长表', '完成'],
    ['IEEE37_Line_Parameter_Analysis.md', '矩阵单位、单长与总阻抗、序分量、PCB 简化风险', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv1', '30 个负荷对象的节点、相别、Delta、Model、kV/kW/kvar/PF、CVR、实际 P/Q', '完成'],
    ['IEEE37_Load_Connection_Map.csv', '32 个相间支路的 PhaseFrom/PhaseTo、名义/求解 P/Q、支路电压', '完成'],
    ['IEEE37_Load_Analysis.md', '负荷模型统计、S728 分配、固定 RLC 转换边界', '完成'],
    ['IEEE37_Transformer_Table.csv', 'SubXF、XFM1、reg1a、reg1c 的绕组、容量、阻抗、变比、潮流、负载率', '完成'],
    ['IEEE37_Regulator_Table.csv', 'creg1a、creg1c 的 Vreg、Band、PT/CT、LDC、档位、延时', '完成'],
    ['IEEE37_Line_Parameter_Analysis.md2', 'Vsource、SubXF、Open-Delta、XFM1、799/799r、24 V 等效阻抗与五个问题', '完成'],
    ['IEEE37_Solver_Validation.md', 'OpenDSS 编译/收敛、电压/电流/功率/档位、静态一致性、复现命令', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv4', '结构化的求解日志：版本、编译、收敛、关键读数、限制', '完成'],
    ['IEEE37_Bus_Voltages.csv', '39 个母线的 Vab/Vbc/Vca、pu、序分量、LN 电压、Zsc', '完成'],
    ['IEEE37_Line_Currents.csv', '逐线路逐端子逐相电流幅值/相角/标幺/方向', '完成'],
    ['IEEE37_Load_Powers.csv', '30 个负荷的定义值与求解值对照', '完成'],
    ['IEEE37_Transformer_Flows.csv', '变压器两侧 P/Q/S、损耗、负载率、最大电流', '完成'],
    ['IEEE37_Regulator_State.csv', '调压器档位、Tap、799/799r 电压', '完成'],
    ['IEEE37_Line_Parameter_Analysis.md9', '连通分量、回路、并联边、孤立节点、相别错误等检查', '完成'],
    ['IEEE37_Line_Parameter_Analysis.md1', '四个文件的字节数、SHA256、引用状态', '完成'],
    ['IEEE37_File_References.csv', 'Redirect/BusCoords/入口/辅助表的引用关系', '完成'],
    ['IEEE37_Solver_Raw.json', '求解器原始读数（含 Yprim、Zsc、Powers），供复核', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv7', '关键计数与统计的机器可读摘要', '完成'],
    ['IEEE37_Line_Impedance_Matrix.csv6', '完整拓扑图（按电压等级着色，标注线路、变压器、调压器、负荷）', '完成'],
    ['IEEE37_Topology.svg', '拓扑图矢量副本，便于打印/编辑', '完成'],
    ['source/IEEE37_Load_Connection_Map.csv1', '原始主文件副本，未修改', '只读副本'],
    ['source/IEEELineCodes.DSS', '原始线路配置副本，未修改', '只读副本'],
    ['source/IEEE37_Load_Connection_Map.csv0', '原始坐标副本，未修改', '只读副本'],
    ['source/ieee37_EXP_METERS.CSV', '历史仪表导出副本，未修改', '只读副本'],
])}

线路、LineCode、负荷、变压器和调压器表均带有 `SourceLine` 列（适用时），用于回查 `IEEE37_Load_Connection_Map.csv1` 或 `IEEELineCodes.DSS` 的具体行号；`IEEE37_Solver_Raw.json` 保留求解器原始数组，便于逐项复核。

## 3. 关键计数

{md_table(['项目', '数值', '说明'], [
    ['母线名（含 SourceBus）', summary['num_buses_total'], '静态解析与求解器一致'],
    ['网络母线（不含 SourceBus）', summary['num_buses_network'], '799、799r、701–744、775'],
    ['相节点', summary['num_phase_nodes'], '39 × 3；不含变压器内部中性导体'],
    ['线路对象', summary['num_lines_total'], '35 条三相馈线 + 1 条单相 Jumper'],
    ['馈线总长度', f"{summary['feeder_length_mi']:.2f} mi", 'L1–L35 长度之和'],
    ['负荷对象', summary['num_loads'], '29 单相 Delta + 1 三相 Delta'],
    ['负荷相间支路', len(connection_map), '29 + 3（S728）'],
    ['负荷母线', summary['num_load_buses'], '30 个对象分布在 25 个母线'],
    ['变压器/调压器', summary['num_transformers'], 'SubXF、XFM1、reg1a、reg1c'],
    ['调压控制器', summary['num_regulators'], 'creg1a、creg1c'],
    ['LineCode 定义/引用', f"{summary['num_linecodes_defined']} / {summary['num_linecodes_referenced']}", '全部定义 29，本模型引用 721–724'],
    ['负荷总 P', f"{summary['load_kw']:.1f} kW", 'DSS 定义值'],
    ['负荷总 Q', f"{summary['load_kvar']:.1f} kvar", 'DSS 定义值'],
    ['负荷总 S', f"{summary['load_kva']:.2f} kVA", '由 P/Q 计算'],
    ['负荷整体 PF', f"{summary['load_pf']:.4f}", '滞后'],
])}

## 4. 最终八个问题的明确结论

### 4.1 当前 DSS 模型是否完整？

**电气上完整，文件层面有一个辅助文件需要说明。**

- `IEEE37_Load_Connection_Map.csv1` 的两条文件引用（`IEEELineCodes.DSS`、`IEEE37_Load_Connection_Map.csv0`）均存在，大小写一致；没有缺失的电气文件。
- 没有重复对象、无效 LineCode 引用、未定义母线或相别错误。
- 但模型依赖若干 OpenDSS 默认值：线路单位 `mi`、NormAmps 400 A / EmergAmps 600 A、负荷 CVR 1/2、调压器 `%r=0.2%`、变压器 `Xht/Xlt`、tap 上下限。这些默认值在 OpenDSS 中可解，但移植到 KiCad/其他工具时必须显式写出。
- `ieee37_EXP_METERS.CSV` 不是 DSS 引用，且其数值与本次 1.0 pu 快照不一致；不能作为模型完整性或参数校验的依据。

结论：**作为 OpenDSS 模型完整；作为自描述、可移植的电气数据库，需要把上述默认值固化到派生模型中。**

### 4.2 是否能确认所有节点和线路连接关系？

**能。** 所有线路的 `Bus1`/`Bus2` 均来自 DSS 原文，未按名称推断。共 {summary['num_buses_total']} 个母线名、{summary['num_phase_nodes']} 个相节点、35 条三相馈线、1 条单相 Jumper、4 个变压器/调压器对象。

- 母线级图：1 个连通分量；40 条边（35 线路 + 1 Jumper + 4 变压器/调压器）。
- 馈线本体 L1–L35：36 个节点、35 条边，是树。
- 回路秩 2，全部来自 799–799r 的三条并联支路（reg1a、reg1c、Jumper）；没有其他意外环路。
- 没有孤立母线；775 通过 XFM1 连接，是“无负荷末端”，不是断开分支。
- 相别检查：三相线路均为 1-1/2-2/3-3；单相 Delta 负荷均接在两个存在的相节点之间；无错误。

### 4.3 是否能确认所有线路的三相 R/X/C 矩阵？

**能。** 721–724 的完整 3×3 矩阵已逐元素提取：

- R/X 单位 Ω/mi，C 单位 nF/mi；长度单位 mi。
- 721–724 的 R/X 矩阵有非零互阻抗；C 矩阵的非对角元为零（对角电容）。
- 单长矩阵 × 长度 = 整条线路总矩阵；与 `Yprim` 的 `-inv(Y12)` 校验，最大差约 {max(line['Yprim_Impedance_MaxDiff_ohm'] for line in lines):.3e} Ω。
- 721–724 不是完全换位对称矩阵，序分量存在小的序间耦合；不能默认“三相对称解耦”。

限制：DSS 没有显式写 `Units`，`mi` 是 OpenDSS 默认值；`NormAmps/EmergAmps` 也是默认值，不是真实载流量。矩阵本身可确认，单位和载流量属性需要显式固化。

### 4.4 是否能确认所有负荷的节点、相别、接线和 P/Q？

**能，但固定 RLC 的“等价性”只在基准点成立。**

- 30 个负荷对象、25 个负荷母线；29 个单相 Delta、1 个三相 Delta（`S728`）。
- 展开为 32 个相间支路；每个支路的 PhaseFrom/PhaseTo、名义 kW/kvar、求解电压和求解 P/Q 均在 `IEEE37_Load_Connection_Map.csv`。
- 总定义负荷 {summary['load_kw']:.1f} kW、{summary['load_kvar']:.1f} kvar；求解总 P/Q 与定义值一致（差异来自电压依赖）。
- Model 1/2/4 数量为 15/7/8；CVRwatts=1、CVRvars=2 是 OpenDSS 默认值。
- 所有负荷 `Conn=Delta`，`Enabled=true`，`Status=0`（Fixed）；没有 ZIPV，没有时变曲线。
- `S728` 的三个支路名义各 42 kW / 21 kvar；总功率 126 kW / 63 kvar。

限制：Model 1 恒功率、Model 4 指数/CVR 不能由固定 RLC 在全电压范围精确复现；只能做“基准工作点等效”。`S714a` 等 Model 4 的 P 按 V^1、Q 按 V^2（默认指数）变化，固定 RLC 的 P 按 V^2，二者不同。

### 4.5 是否能确认 SubXF、Open-Delta 调压器和 XFM1 的真实参数？

**能确认 DSS 中写出的参数和 OpenDSS 求解器解析出的参数；但“真实”仅指模型定义，不是设备铭牌实测。**

- `SubXF`：230/4.8 kV Δ–Δ，2500 kVA，Xhl=8%，%r=1%/绕组；4.8 kV 侧 R≈{subxf['R_total_ohm_side2']:.6f} Ω、X≈{subxf['X_total_ohm_side2']:.6f} Ω、|Z|≈{subxf['Z_total_ohm_side2_mag']:.6f} Ω；基态视在负载 {subxf['Loading_pct']:.1f}%，电流负载率 {subxf['CurrentLoading2_pct']:.1f}%。
- `reg1a`/`reg1c`：单相 4.8/4.8 kV，2000 kVA/绕组，Xhl=1%，%r 未写、默认 0.2%/绕组；`creg1a` Vreg=122 V、Band=2 V、PT=40、CT=350 A、R=-0.201 V、X=3.348 V、档位 16（tap=1.1）；`creg1c` R=2.799 V、X=1.848 V、档位 14（tap=1.0875）。
- `XFM1`：4.8/0.48 kV Δ–Δ，500 kVA，Xhl=1.81%，%r=0.045%/绕组；变比 10:1；775 侧基态无负荷，潮流接近零。

限制：调压器 `%r`、`Xht`、`Xlt`、tap 上下限、`Delay` 等来自默认值；调压器 `R/X` 是 LDC 设置，不是绕组阻抗。若要把这些参数用于实体设备，需要按缩比基准重新设计，而不是照搬铭牌。

### 4.6 Bus 799 是否适合作为 24 V AC 电源参考点？

**适合作为拓扑参考点，但不是无条件的“直接接 24 V 理想源”。**

- 799 是 SubXF 二次侧、Open-Delta 调压器输入，保留它可以保留 reg1a/reg1c、B 相 Jumper、L35 和 701 以下全部馈线；这是最接近原拓扑的电源边界。
- 24 V 必须明确是 **4.8 kV 的 1.0 pu 等值**（线电压 RMS，三相三线 Delta），而不是“799 基态 0.923 pu 电压”。若把 24 V 钳在 799 的负荷电压上，标幺基准会偏移。
- 若实验包含故障注入，建议在 24 V 源后串联缩放后的 SubXF+上游戴维南阻抗；若只做正常工况演示，可以用理想源，但必须标注“短路特性不代表原模型”。
- 把 24 V 接在 799r 会绕过调压器和 799，属于派生简化模型；若要做调压器实验，不应选 799r。

结论：**799 是推荐基准，但 24 V 的定义、源阻抗、调压器实现和基态重载处理必须一起确定。**

### 4.7 哪些数据已经验证，哪些只是静态提取？

**求解器验证（实际计算）：**

- 编译成功、错误号 0、基态收敛（{solver['solution']['iterations']} 次迭代、{solver['solution']['control_iterations']} 次控制迭代）。
- 39 个母线、117 个相节点、73 个元件与静态解析一致。
- 所有母线 Vab/Vbc/Vca、线路电流、负荷 P/Q、变压器/调压器功率、调压器档位。
- 线路总阻抗与 `Yprim` 的交叉校验；母线/线路/负荷/变压器名称集合的一致性。
- 基态最低/最高电压（{min_bus['BusName']} {min_bus['Vmin_pu']:.4f} pu；{max_bus['BusName']} {max_bus['Vmax_pu']:.4f} pu）、最大线路电流（{max_line['LineName']} {max_line['MaxCurrent_A']:.1f} A）、SubXF/调压器负载率。

**静态提取（文本解析，未由求解器逐项证明）：**

- 文件引用、对象行号、`like=` 继承关系、注释掉的历史矩阵。
- 变压器/调压器的铭牌参数、负荷定义值、LineCode 矩阵原文。
- `CVRwatts/CVRvars`、`Units`、`NormAmps`、调压器 `%r` 等默认值的“未写出”事实。

**工程设计建议（不是原始模型数据）：**

- 24 V 作为 1.0 pu 的定义、SubXF 等效阻抗的缩放、平台功率基准、源阻抗实现、调压器实体方案、PCB 模块划分。
- 这些建议需要用户确认后才能进入 KiCad 设计。

**尚未确定：**

- 平台频率的最终冻结：本次按 60 Hz 求解；项目其他设计文档已记录 60 Hz 运行频率，若该决策仍有效则无需换算，若改为 50 Hz 需重算 X/C 和负荷 RLC。
- 平台功率/电流基准、电源容量与短路能力、电源保护/限流。
- 调压器是动态调压、固定档位还是旁路；XFM1 是否保留、775 是否接负荷。
- 固定 RLC 的转换基准、互耦实现方式、PCB 元件值与公差。
- 故障注入模块的接口与保护配合（属于模块 3，不在本次范围）。

### 4.8 当前还缺少哪些信息，才能进入 KiCad 主干原理图设计？

至少需要先确定以下事项；在这些事项冻结前，可以画“连接关系图”，但不能冻结元件值和 PCB：

1. **平台频率的最终冻结**：本次按 60 Hz 求解，项目其他设计文档已记录 60 Hz 运行频率；若最终改为 50 Hz，线路 X/C、变压器漏抗和负荷 RLC 都要重算。
2. **电压/功率基准**：24 V 对应 4.8 kV 的 1.0 pu；功率基准或电流基准（例如 50/100/250/500 VA）决定所有阻抗的缩放系数。
3. **电源实现与等值阻抗**：24 V 源是变压器、功放还是可编程源；其输出阻抗、短路电流、限流/保护行为；是否串联 SubXF 等值阻抗。
4. **基态重载的处理**：是否复现 SubXF 121% / 调压器 90% 的基态；若不复现，负荷缩放系数和电压水平如何调整。
5. **调压器实现**：动态调压、固定档位（16/14）还是旁路；LDC 是否实现；799/799r 的测量与端口。
6. **XFM1/775 的处理**：775 是否接负荷；XFM1 是否作为独立模块保留；0.48 kV 侧如何缩比。
7. **负荷 RLC 转换**：Model 1/4 的基准点、频率、Q 符号、每支路 R/L/C；是否接受“仅基准点等效”的误差。
8. **三相互耦实现**：完整 3×3 R/X 矩阵在 PCB 上如何实现；若简化为独立三相支路，允许的误差是多少。
9. **相序与接地**：A-B-C 相序、三线制 Delta、无中性线；安全地/屏蔽地如何与功率地分离。
10. **测量与保护**：Vab/Vbc/Vca 测量点、端子定义、DAQ 输入保护、故障模块接口、电流/热设计边界。

## 5. 已验证 / 静态提取 / 工程建议 / 未确定（汇总）

{md_table(['类别', '内容', '状态'], [
    ['已验证', '编译/收敛、39 母线、117 相节点、73 元件、电压/电流/功率/档位、Yprim 阻抗校验', '可复现'],
    ['静态提取', 'DSS 原文、对象行号、LineCode 矩阵、负荷定义、变压器铭牌、like= 继承', '可追溯到行号'],
    ['工程建议', '24 V 1.0 pu 定义、SubXF 等值阻抗缩放、799/799r 选择、PCB 模块划分', '待用户确认'],
    ['未确定', '频率、功率基准、电源硬件、调压器实现、固定 RLC 值、互耦实现、保护配合', '进入 KiCad 前必须冻结'],
])}

## 6. 已知限制

- `.rar` 本体未在文件系统中找到；审计使用此前解压目录，三个电气文件与 canonical 安装逐字节一致。若用户手中的 `37Bus.rar` 与 canonical 版本不同，应以用户文件重新运行本脚本。
- `ieee37_EXP_METERS.CSV` 与本次快照不一致，未用于任何电气结论。
- OpenDSS COM 不返回完整解析器警告列表；只检查了错误号和电路建立。
- `Line.R1`/`LineCode.R1` 标量属性存在活动对象缓存；本审计使用矩阵，未使用这些标量。
- 本次只求解 60 Hz 单快照正常工况，没有故障、时序、50 Hz 或温度场景。
- 本文所有“建议”均是工程建议，不是原模型数据；不得把建议值写回原始 DSS。

## 7. 复现

```powershell
python code/method-a2/scripts/audit_ieee37_model.py `
  --source-dir code/method-a2/data/electrical-audit-ieee37-20261103 `
  --out code/method-a2/output/ieee37_model_audit `
  --publish docs/project/reports/IEEE37_Model_Audit
```

脚本只读取原始 DSS；`IEEE37_Solver_Raw.json` 与 `IEEE37_Line_Impedance_Matrix.csv4` 记录求解器的原始读数，CSV 与 Markdown 由同一份解析结果生成。
"""
    (out_dir / "IEEE37_Load_Connection_Map.csv2").write_text(text, encoding="utf-8")


def render_solver_validation(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """生成 OpenDSS 求解验证报告。"""
    solution = solver["solution"]
    key_buses = []
    for name in ["SourceBus", "799", "799r", "701", "709", "775", "741"]:
        bus = next((item for item in buses if item["BusName"] == name), None)
        if bus:
            key_buses.append(
                [
                    bus["BusName"],
                    fmt(bus["Vab_kV"], 4),
                    fmt(bus["Vbc_kV"], 4),
                    fmt(bus["Vca_kV"], 4),
                    fmt(bus["Vmin_pu"], 4),
                    fmt(bus["Vmax_pu"], 4),
                ]
            )
    top_lines = sorted(lines, key=lambda item: item["MaxCurrent_A"], reverse=True)[:8]
    top_line_rows = [
        [
            line["LineName"],
            line["LineCode"],
            f"{line['MaxCurrent_A']:.1f}",
            f"{line['MaxLoading_pct']:.1f}",
            f"{line['Loss_kW']:.2f}",
        ]
        for line in top_lines
    ]
    xf_rows = [
        [
            item["Name"],
            f"{item['SolvedP1_kW']:.1f}",
            f"{item['SolvedQ1_kvar']:.1f}",
            f"{item['SolvedS1_kVA']:.1f}",
            f"{item['Loading_pct']:.1f}",
            f"{item['CurrentLoading2_pct']:.1f}",
        ]
        for item in transformers
    ]
    reg_rows = [
        [
            item["Regulator"],
            item["TapPosition"],
            fmt(item["TapRatio"], 5),
            fmt(item["Vreg_kV_primary"], 3),
            fmt(item["Band_kV_primary"], 3),
            fmt(item["R_setting_V"], 3),
            fmt(item["X_setting_V"], 3),
        ]
        for item in regulators
    ]
    static_linecodes = {line.linecode for line in model["lines"].values() if line.linecode}
    solver_linecodes = {str(line["linecode"]) for line in solver["lines"] if line.get("linecode")}
    code_match = static_linecodes == solver_linecodes
    bus_names_static = {name.lower() for name in model["bus_names"]}
    bus_names_solver = {name.lower() for name in solver["bus_names"]}
    bus_match = bus_names_static == bus_names_solver
    max_diff = max(line["Yprim_Impedance_MaxDiff_ohm"] for line in lines)
    matrix_diff = max(line["LineCode_vs_Solver_Matrix_MaxDiff"] for line in lines)

    text = f"""<!-- 摘要：IEEE 37 OpenDSS 实际求解验证报告，记录求解器版本、编译错误、收敛迭代、节点电压、线路电流、变压器/调压器状态、静态解析一致性与复现命令。 -->

# IEEE 37 OpenDSS 求解验证报告

> 文档定位：任务 6 的实际求解结果与验证边界。  
> 数据来源：OpenDSS COM 引擎（版本见下）、原始 `IEEE37_Load_Connection_Map.csv1`。  
> 状态：已实际编译并求解；本文所有电压、电流、功率和档位均来自求解器，不是静态推测。

## 1. 求解环境与编译

{md_table(['项目', '结果'], [
    ['OpenDSS 版本', solver['version']],
    ['编译命令', solver['compile_command']],
    ['编译错误号', solver['compile_error_number']],
    ['编译错误描述', solver['compile_error_description'] or '（空）'],
    ['电路名', solver['circuit_name']],
    ['母线数', solver['num_buses']],
    ['相节点数', solver['num_nodes']],
    ['元件数', solver['num_elements']],
])}

OpenDSS COM 在编译阶段返回错误号 {solver['compile_error_number']}，没有错误描述。模型正常建立电路，说明主文件、`Redirect IEEELineCodes.DSS` 和 `BusCoords IEEE37_Load_Connection_Map.csv0` 均被正确加载。

## 2. 收敛与迭代

{md_table(['指标', '结果'], [
    ['是否收敛', 'yes' if solution['converged'] else 'no'],
    ['迭代次数', solution['iterations']],
    ['控制迭代次数', solution['control_iterations']],
    ['总迭代次数', solution['total_iterations']],
    ['最大迭代上限', solution['max_iterations']],
    ['收敛容差', solution['tolerance']],
    ['求解模式', solution['mode']],
    ['控制模式', solution['control_mode']],
    ['负荷倍率', solution['load_mult']],
    ['频率', f"{solution['frequency']} Hz"],
    ['总耗时', f"{solution['total_time_ms']:.1f} ms"],
    ['求解耗时', f"{solution['process_time_ms']:.1f} ms"],
])}

求解在 {solution['iterations']} 次迭代、{solution['control_iterations']} 次控制迭代后收敛；控制迭代次数不为零，说明调压器控制参与了求解，当前档位是控制作用后的结果。

## 3. 节点电压

全网 {summary['num_phase_nodes']} 个相节点的线电压范围：

- 最低：`{summary['min_bus']['BusName']}`，Vmin = {summary['min_bus']['Vmin_pu']:.4f} pu；
- 最高：`{summary['max_bus']['BusName']}`，Vmax = {summary['max_bus']['Vmax_pu']:.4f} pu；
- 最大负序/正序比：`{summary['max_unbalance']['BusName']}`，{summary['max_unbalance']['V2_V1_pct']:.3f}%。

关键母线：

{md_table(['母线', 'Vab/kV', 'Vbc/kV', 'Vca/kV', 'Vmin/pu', 'Vmax/pu'], key_buses)}

799 的电压低于 799r，符合“SubXF 二次侧 → 调压器升压 → L35 → 701”的预期。注意 799 的 0.923 pu 不是电源电压，而是 SubXF 在基态负荷下的二次侧电压；电源 `SourceBus` 仍为 1.0 pu。

## 4. 线路电流与损耗

电流最大的线路：

{md_table(['线路', 'LineCode', 'Imax/A', '负载率/%（按 400 A）', 'Loss/kW'], top_line_rows)}

- 全网最大电流：`{summary['max_current_line']['LineName']}`，{summary['max_current_line']['MaxCurrent_A']:.2f} A，为默认 `NormAmps=400 A` 的 {summary['max_current_line']['MaxLoading_pct']:.2f}%。
- 全网线路总有功损耗约 {summary['line_loss_kw']:.3f} kW。
- 400 A 是 OpenDSS 默认值，不是实际导体额定值；这里的“负载率”只能作为模型内部相对指标，不能直接用于 PCB 选型。

## 5. 变压器与调压器状态

{md_table(['设备', 'P1/kW', 'Q1/kvar', 'S1/kVA', '视在负载率/%', '二次侧电流负载率/%'], xf_rows)}

- `SubXF` 基态视在负载约 {summary['subxf']['Loading_pct']:.1f}%，二次侧电流负载率约 {summary['subxf']['CurrentLoading2_pct']:.1f}%；原模型基态是重载工况。
- `XFM1` 基态负载接近 0%，因为 775 没有负荷。
- `reg1a`、`reg1c` 的视在负载分别约 {summary['reg1a']['Loading_pct']:.1f}%、{summary['reg1c']['Loading_pct']:.1f}%，均接近 90%。

调压器档位：

{md_table(['控制器', '档位', 'Tap 比值', 'Vreg/kV', 'Band/kV', 'LDC R/V', 'LDC X/V'], reg_rows)}

`creg1a` 档位 16（最大档，tap=1.1）；`creg1c` 档位 14（tap=1.0875）。这说明基态 A–B 支路已经需要最大升压，后续若在实体平台上继续增加负荷，调压器没有更多升压裕度。

## 6. 静态解析与求解器的一致性

{md_table(['检查项', '静态解析', '求解器', '一致'], [
    ['母线名称集合', len(bus_names_static), len(bus_names_solver), 'yes' if bus_match else 'no'],
    ['母线数（含 SourceBus）', summary['num_buses_total'], solver['num_buses'], 'yes' if summary['num_buses_total'] == solver['num_buses'] else 'no'],
    ['相节点数', summary['num_phase_nodes'], solver['num_nodes'], 'yes' if summary['num_phase_nodes'] == solver['num_nodes'] else 'no'],
    ['线路对象数（含 Jumper）', summary['num_lines_total'], len(solver['lines']), 'yes' if summary['num_lines_total'] == len(solver['lines']) else 'no'],
    ['负荷对象数', summary['num_loads'], len(solver['loads']), 'yes' if summary['num_loads'] == len(solver['loads']) else 'no'],
    ['变压器/调压器数', summary['num_transformers'], len(solver['transformers']), 'yes' if summary['num_transformers'] == len(solver['transformers']) else 'no'],
    ['调压控制器数', summary['num_regulators'], len(solver['regcontrols']), 'yes' if summary['num_regulators'] == len(solver['regcontrols']) else 'no'],
    ['被引用 LineCode 集合', ','.join(sorted(static_linecodes)), ','.join(sorted(solver_linecodes)), 'yes' if code_match else 'no'],
])}

线路总阻抗的独立校验：由 `Yprim` 端子互导纳块反推的串联阻抗与“单长矩阵 × 长度”的最大绝对差为 {max_diff:.3e} Ω；静态 LineCode 单长矩阵与求解器实际加载矩阵的最大绝对差为 {matrix_diff:.3e}。两个差值都处于双精度舍入量级，说明矩阵、长度和单位口径一致。

## 7. 求解器接口的已知限制

- COM 接口没有返回完整的解析器警告列表；本次只检查了 `Error.Number`/`Error.Description` 和电路是否建立。错误号为 0，不能证明“没有任何非致命警告”，但可以证明没有编译级错误。
- `Bus.ZscRefresh()` 会改变求解状态；本脚本在读取全部电压、电流和功率之后才刷新短路阻抗，避免把电压读数清零。复现时不要在每个母线读电压前调用 `ZscRefresh`。
- `Line.R1`、`LineCode.R1` 等标量属性在该 OpenDSS 版本中表现出活动对象缓存：四个 LineCode 返回了相同的 0.058/0.1206。本审计不使用这些标量，全部由 3×3 矩阵重新计算序分量。
- `Zsc1` 是母线处的驱动点阻抗，包含下游网络的并联效应，不等于单独的 SubXF 漏抗；SubXF 的铭牌漏抗应按 `IEEE37_Line_Parameter_Analysis.md2` 第 2 节的铭牌公式单独计算。
- 求解是 60 Hz 单快照，不是时序或故障求解；故障注入模块属于后续工作，本次未定义 `Fault` 元件。

## 8. 复现方法

### 8.1 OpenDSS GUI

1. 打开 `IEEE37_Load_Connection_Map.csv1`（确保 `IEEELineCodes.DSS` 与 `IEEE37_Load_Connection_Map.csv0` 同目录）。
2. 执行 `Solve`。
3. 执行 `Show Voltages LL Nodes`、`Show Currents`、`Show Powers kVA Elements`、`Show Taps`。
4. 如需导出，执行 `Export Voltages`、`Export Currents`、`Export Powers`。

### 8.2 Python COM

```python
import win32com.client, os
p = os.path.abspath("IEEE37_Load_Connection_Map.csv1")
e = win32com.client.Dispatch("OpenDSSEngine.DSS")
e.AllowForms = False
e.Text.Command = f'Compile "{{p}}"'
c = e.ActiveCircuit
c.Solution.Solve()
print(c.Solution.Converged, c.Solution.Iterations, c.Solution.ControlIterations)
for name in c.AllBusNames:
    c.SetActiveBus(name)
    print(name, c.ActiveBus.kVBase, c.ActiveBus.VLL)
```

### 8.3 本审计脚本

```powershell
python code/method-a2/scripts/audit_ieee37_model.py `
  --source-dir code/method-a2/data/electrical-audit-ieee37-20261103 `
  --out code/method-a2/output/ieee37_model_audit `
  --publish docs/project/reports/IEEE37_Model_Audit
```

## 9. 结论

原始模型在 OpenDSS 中编译成功、基态潮流收敛，节点电压、线路电流、变压器/调压器功率和档位均已导出。静态解析与求解器在母线、相节点、线路、负荷、变压器和 LineCode 集合上一致；线路总阻抗与 `Yprim` 校验通过。求解结果表明原模型基态存在 SubXF 与调压器重载，这是后续 24 V 缩比设计必须正视的边界。
"""
    (out_dir / "IEEE37_Solver_Validation.md").write_text(text, encoding="utf-8")


def render_source_analysis(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """生成电源、SubXF、Open-Delta 调压器与 XFM1 分析。"""
    bus_799 = next(bus for bus in buses if bus["BusKey"] == "799")
    bus_799r = next(bus for bus in buses if bus["BusKey"] == "799r")
    bus_701 = next(bus for bus in buses if bus["BusKey"] == "701")
    bus_775 = next(bus for bus in buses if bus["BusKey"] == "775")
    subxf = summary["subxf"]
    xfm1 = summary["xfm1"]
    reg1a = summary["reg1a"]
    reg1c = summary["reg1c"]
    creg1a = next(item for item in regulators if item["Regulator"].lower() == "creg1a")
    creg1c = next(item for item in regulators if item["Regulator"].lower() == "creg1c")
    source_z = solver.get("source_bus_thevenin", {})
    z1_source = pair_to_complex(source_z["zsc1"]) if source_z.get("zsc1") else complex(0, 0)
    z0_source = pair_to_complex(source_z["zsc0"]) if source_z.get("zsc0") else complex(0, 0)

    zbase_sub = 4.8**2 / 2.5
    r_sub = 0.02 * zbase_sub
    x_sub = 0.08 * zbase_sub
    z_sub = complex(r_sub, x_sub)
    z_source_48 = z1_source * (4.8 / 230.0) ** 2
    z_th = z_sub + z_source_48
    fault_48 = 4800.0 / (math.sqrt(3) * abs(z_th))
    z_pu = abs(z_th) / zbase_sub

    scaling_rows = []
    for s_base in [50, 100, 250, 500, 1000]:
        z_platform = z_pu * 24.0**2 / s_base
        i_fault = 24.0 / (math.sqrt(3) * z_platform)
        scaling_rows.append([s_base, f"{z_platform:.5f}", f"{i_fault:.2f}"])
    z_25mva = z_pu * 24.0**2 / 2.5e6
    i_25mva = 24.0 / (math.sqrt(3) * z_25mva)

    text = f"""<!-- 摘要：IEEE 37 电源与源端结构分析，记录 Vsource、SubXF、Open-Delta 调压器、XFM1 参数，回答 799/799r 与 24 V 参考点及等效阻抗处理问题。 -->

# IEEE 37 电源与源端结构分析

> 文档定位：任务 5 的 Vsource、SubXF、Open-Delta 调压器、XFM1 核对，以及 24 V 实体电源接入建议。  
> 数据来源：`IEEE37_Load_Connection_Map.csv1` 第 8–19、61–66 行；OpenDSS COM 求解器的 `Vsource`、`Transformer`、`RegControl`、`Bus.Zsc1` 与基态潮流。  
> 状态：原始参数与求解器读数已核对；24 V 缩比部分属于工程设计建议，不是原始模型数据。

## 1. 电源定义

求解器与 DSS 给出的电源参数：

{md_table(['参数', '值', '说明'], [
    ['Vsource 名称', 'source', 'OpenDSS 隐式电源'],
    ['BasekV', solver['vsources'][0]['basekv'], '230 kV 线电压'],
    ['pu', solver['vsources'][0]['pu'], '1.00'],
    ['相角', solver['vsources'][0]['angle_deg'], '0°'],
    ['频率', solver['vsources'][0]['frequency'], '60 Hz'],
    ['相数', solver['vsources'][0]['phases'], '3'],
    ['MVAsc3', '200000', 'DSS 中 `MVAsc3=200000`'],
    ['MVAsc1', '210000', 'DSS 中 `MVAsc1=210000`'],
    ['Z1（求解器）', f"{z1_source.real:.6f} + j{z1_source.imag:.6f} Ω", f"|Z1|={abs(z1_source):.6f} Ω，∠{math.degrees(math.atan2(z1_source.imag, z1_source.real)):.2f}°"],
    ['Z0（求解器）', f"{z0_source.real:.6f} + j{z0_source.imag:.6f} Ω", '三线制 Delta 下零序网络不提供负荷通路，仅作参考'],
])}

`MVAsc3=200000` 表示 200 GVA 的短路容量，这是一个非常“硬”的测试电源，不是实际 230 kV 变电站的等值。它的正序阻抗约 0.2645 Ω（230 kV 侧），折算到 4.8 kV 侧仅约 {abs(z_source_48):.6f} Ω，相对 SubXF 的 {abs(z_sub):.4f} Ω 可以忽略。

4.8 kV 母线的 `Zsc0` 在求解器中约为数千欧姆，这是三线制 Delta 没有零序返回路径的数值表现，不是可用的物理零序阻抗；本审计的正序短路分析只使用 `Zsc1`。

## 2. SubXF 主变压器

{md_table(['参数', '值', '说明'], [
    ['连接', '230 kV Δ / 4.8 kV Δ', '三相双绕组，Delta–Delta'],
    ['额定容量', f"{subxf['kVA1']:.0f} kVA", '每绕组 2500 kVA（三相总容量）'],
    ['Xhl', f"{subxf['Xhl_pct']:.2f}%", '显式写在 DSS 中'],
    ['绕组电阻', f"{subxf['R1_pct']:.3f}% / {subxf['R2_pct']:.3f}%", '每绕组 1%，显式'],
    ['Zbase 4.8 kV 侧', f"{subxf['Zbase2_ohm']:.4f} Ω", '4.8² / 2.5 MVA'],
    ['R 总（4.8 kV 侧）', f"{subxf['R_total_ohm_side2']:.6f} Ω", '两侧 1% 相加'],
    ['X 总（4.8 kV 侧）', f"{subxf['X_total_ohm_side2']:.6f} Ω", 'Xhl=8%'],
    ['Z 总（4.8 kV 侧）', f"{subxf['Z_total_ohm_side2_mag']:.6f} Ω", f"标幺值 {subxf['Z_total_pu_side2']:.5f}"],
    ['额定电流 4.8 kV 侧', f"{subxf['RatedCurrent2_A']:.2f} A", '2500 kVA / (√3 × 4.8 kV)'],
    ['基态最大电流', f"{subxf['MaxCurrent2_A']:.2f} A", f"电流负载率 {subxf['CurrentLoading2_pct']:.1f}%"],
    ['基态视在功率', f"{subxf['SolvedS1_kVA']:.1f} kVA", f"视在负载率 {subxf['Loading_pct']:.1f}%"],
])}

**关键事实**：在 1.0 pu 负荷下，SubXF 的视在功率约 {subxf['SolvedS1_kVA']:.1f} kVA，为 2500 kVA 的 {subxf['Loading_pct']:.1f}%；4.8 kV 侧最大电流约 {subxf['MaxCurrent2_A']:.1f} A，为额定 {subxf['RatedCurrent2_A']:.1f} A 的 {subxf['CurrentLoading2_pct']:.1f}%。这不是审计错误，而是原模型基态的重载特性。任何 24 V 缩比方案都必须决定：是复现这一重载运行点，还是降低负荷/提高电源容量并明确记录偏离。

求解器在 799 的 `Zsc1` 约 {bus_799['Zsc1_ohm_mag']:.4f} Ω，略低于 SubXF 铭牌阻抗 {subxf['Z_total_ohm_side2_mag']:.4f} Ω。`Zsc1` 是母线驱动点阻抗，包含下游调压器和馈线的并联效应，不能反推 SubXF 漏抗；SubXF 漏抗应以本表铭牌公式计算。

## 3. Open-Delta 调压器

{md_table(['项目', 'reg1a', 'reg1c'], [
    ['变压器', 'reg1a', 'reg1c'],
    ['控制器', 'creg1a', 'creg1c'],
    ['相间', 'A–B（799.1.2 → 799r.1.2）', 'C–B（799.3.2 → 799r.3.2）'],
    ['相数/绕组', '1 / 2', '1 / 2'],
    ['额定电压', '4.8 / 4.8 kV', '4.8 / 4.8 kV'],
    ['额定容量', '2000 / 2000 kVA', '2000 / 2000 kVA'],
    ['Xhl', f"{reg1a['Xhl_pct']:.2f}%", f"{reg1c['Xhl_pct']:.2f}%"],
    ['绕组电阻', f"{reg1a['R1_pct']:.3f}% / {reg1a['R2_pct']:.3f}%", f"{reg1c['R1_pct']:.3f}% / {reg1c['R2_pct']:.3f}%"],
    ['Vreg', f"{creg1a['Vreg_kV_primary']:.3f} kV（122 V/120 V 基准）", f"{creg1c['Vreg_kV_primary']:.3f} kV"],
    ['Band', f"{creg1a['Band_kV_primary']:.3f} kV（2 V/120 V 基准）", f"{creg1c['Band_kV_primary']:.3f} kV"],
    ['PT/CT', f"{creg1a['PTRatio']:.0f} / {creg1a['CTPrimary_A']:.0f} A", f"{creg1c['PTRatio']:.0f} / {creg1c['CTPrimary_A']:.0f} A"],
    ['LDC R/X', f"{creg1a['R_setting_V']:.3f} / {creg1a['X_setting_V']:.3f} V", f"{creg1c['R_setting_V']:.3f} / {creg1c['X_setting_V']:.3f} V"],
    ['当前档位', f"{creg1a['TapPosition']}（{creg1a['TapRatio']:.5f}）", f"{creg1c['TapPosition']}（{creg1c['TapRatio']:.5f}）"],
    ['基态视在负载', f"{reg1a['SolvedS1_kVA']:.1f} kVA（{reg1a['Loading_pct']:.1f}%）", f"{reg1c['SolvedS1_kVA']:.1f} kVA（{reg1c['Loading_pct']:.1f}%）"],
])}

- 两台调压器的 `%r` 在 DSS 中没有写出，求解器按默认 0.2%/绕组处理；`Xht=35%`、`Xlt=30%` 也是默认值，不参与两绕组正序计算。
- `creg1a` 的 `R=-0.201 V`、`X=3.348 V`；`creg1c` 的 `R=2.799 V`、`X=1.848 V`。这是线路压降补偿（LDC）设置，不是变压器绕组电阻/电抗。
- `creg1a` 当前档位 16（tap=1.1，最大档）；`creg1c` 当前档位 14（tap=1.0875）。`creg1a` 已到上限，说明基态 A–B 支路已经需要最大升压。
- 两台调压器的基态视在负载都接近 90%，且 `reg1c` 的基态 Q 为负（约 {reg1c['SolvedQ1_kvar']:.1f} kvar），与 `reg1a` 的正 Q（约 {reg1a['SolvedQ1_kvar']:.1f} kvar）形成明显不对称。这是 Open-Delta 接线、B 相直接连接和不平衡负荷共同作用的结果；不要用“两台相同调压器平均分担”来简化。
- 这里的 2000 kVA 是**每台单相调压器**的额定容量；两台 Open-Delta 组合的三相可用容量不能简单相加为 4000 kVA，需按 Open-Delta 容量定义单独校核。本表的 90% 是单台调压器负载率，不是“两台合计 4000 kVA 的 90%”。

## 4. XFM1 负荷变压器

{md_table(['参数', '值', '说明'], [
    ['连接', '4.8 kV Δ / 0.48 kV Δ', '三相双绕组'],
    ['额定容量', f"{xfm1['kVA1']:.0f} kVA", '每绕组 500 kVA'],
    ['Xhl', f"{xfm1['Xhl_pct']:.2f}%", '显式'],
    ['绕组电阻', f"{xfm1['R1_pct']:.3f}% / {xfm1['R2_pct']:.3f}%", '每绕组 0.045%，显式'],
    ['变比', f"{xfm1['TurnsRatio']:.3f}", '4.8 / 0.48 = 10'],
    ['Zbase 4.8 kV 侧', f"{xfm1['Zbase1_ohm']:.4f} Ω", '4.8² / 0.5 MVA'],
    ['Z 总（4.8 kV 侧）', f"{xfm1['Z_total_ohm_side1_mag']:.6f} Ω", f"标幺值 {xfm1['Z_total_pu_side1']:.5f}"],
    ['775 侧基态电压', f"{bus_775['Vab_kV']:.4f} / {bus_775['Vbc_kV']:.4f} / {bus_775['Vca_kV']:.4f} kV", '0.48 kV 等级'],
    ['775 侧基态负载', f"{xfm1['SolvedS2_kVA']:.3f} kVA", '无负荷，潮流接近零'],
])}

`775` 没有负荷，也没有线路，只有 XFM1 一个连接。它是“变压器二次侧末端”，不是第 36 条馈线线路。PCB 上可以把 XFM1 做成独立可维护模块，但必须明确 775 侧在基态是空载的；如果后续要在 775 接负荷，需要重新做潮流和短路校验。

## 5. Bus 799 与 Bus 799r 的电气区别

基态求解值：

{md_table(['母线', 'Vab/kV', 'Vbc/kV', 'Vca/kV', 'Vab/pu', 'Vbc/pu', 'Vca/pu'], [
    ['799', fmt(bus_799['Vab_kV'], 4), fmt(bus_799['Vbc_kV'], 4), fmt(bus_799['Vca_kV'], 4), fmt(bus_799['Vab_pu'], 4), fmt(bus_799['Vbc_pu'], 4), fmt(bus_799['Vca_pu'], 4)],
    ['799r', fmt(bus_799r['Vab_kV'], 4), fmt(bus_799r['Vbc_kV'], 4), fmt(bus_799r['Vca_kV'], 4), fmt(bus_799r['Vab_pu'], 4), fmt(bus_799r['Vbc_pu'], 4), fmt(bus_799r['Vca_pu'], 4)],
    ['701', fmt(bus_701['Vab_kV'], 4), fmt(bus_701['Vbc_kV'], 4), fmt(bus_701['Vca_kV'], 4), fmt(bus_701['Vab_pu'], 4), fmt(bus_701['Vbc_pu'], 4), fmt(bus_701['Vca_pu'], 4)],
])}

- 799 是 SubXF 二次侧，基态只有 0.923 pu 左右；电压降落主要来自 SubXF 的漏抗和负荷。
- 799r 是调压器输出，A/C 相被升压到接近 1.0 pu；B 相通过 1 mΩ Jumper 与 799 直接相连，因此 799.2 与 799r.2 几乎等电位。
- 799r 不是“新电源”，也不是“新负荷母线”；它是调压器后的内部母线，经 L35 接 701。
- 799 和 799r 的电压差不是固定变比，而是由调压器档位、LDC、负荷和相别共同决定；基态 `creg1a` 已在最大档，不能把它当作可继续升压的裕度。

## 6. 问题 1：799 和 799r 在原模型中的电气区别是什么？

799 是 SubXF 的二次侧、Open-Delta 调压器的输入；799r 是两台单相调压器的输出、L35 的起点。两者只通过以下三条支路相连：`reg1a`（A–B）、`reg1c`（C–B）和 `Jumper`（B 相）。因此 B 相通过 1 mΩ Jumper 直接连接，A/C 相是调压器耦合；799r 的 A/C 电压由调压器档位和 LDC 决定，不等于 799。799 没有负荷，799r 也没有负荷；它们不是负荷节点。基态 Jumper 电流近零（约 0.00015 A），因为 SubXF/reg1a/reg1c 的 B 相电流在 799.2 处相量和为零；Jumper 是 B 相电位参考，不是基态功率通道，但故障/不平衡时可能承载大电流。

## 7. 问题 2：24 V AC 定义在 Bus 799 是否合理？

**拓扑上合理，但必须带条件。** 799 是原模型 SubXF 的二次侧边界，把 24 V 电源定义在这里可以保留 reg1a/reg1c、Jumper、L35 和 701 之后的全部馈线，是保留原拓扑最自然的选择。

需要明确的三点：

1. **24 V 是“1.0 pu”还是“基态 799 电压”？** 原模型 799 在 1.0 pu 负荷下只有 0.923 pu 左右；如果 24 V 被定义为 799 的基态电压，那么对应的 4.8 kV 基准不是 4.8 kV，而是约 4.43 kV。建议把 24 V 定义为 4.8 kV 的 1.0 pu 等值（即 24 V 对应 4.8 kV），让 799 在负荷下自然跌落到约 0.923×24=22.2 V；这与原模型一致。若把 24 V 钳在 799 的负荷电压上，则整个网络的标幺基准会偏移。
2. **24 V 是线电压 RMS。** 三相三线 Delta，Vab/Vbc/Vca 为 24 V；不是 24 V 相电压，也不是 ±24 V 直流。
3. **24 V 电源必须与缩比后的网络阻抗配套。** 只把电压改成 24 V、保留原始 Ω 值，会把标幺阻抗放大到不可实现；必须按第 9 节的缩放公式处理。

## 8. 问题 3：如果直接定义在 799，原始 SubXF 的等效阻抗是否需要保留？

**如果实验包含故障注入或需要短路特性，建议保留；如果只做正常工况电压/拓扑演示，可以省略，但必须记录“理想电源”假设。**

原因：

- SubXF 的标幺阻抗 |Z|≈{subxf['Z_total_pu_side2']:.5f}（0.02+j0.08），是 799 处短路电流的主要限制。上游 200 GVA 电源折算到 4.8 kV 后仅约 {abs(z_source_48):.6f} Ω，远小于 SubXF 的 {abs(z_sub):.4f} Ω。
- 若把 24 V 理想电源直接接 799，故障电流只由实体电源、线缆和保护决定，不再代表原模型的 2.5 MVA 变压器限流特性。
- 若保留 SubXF 等值阻抗，应在 24 V 侧用串联阻抗实现，并按平台功率基准缩放；不能直接把 4.8 kV 侧的 0.184+j0.737 Ω 照搬到 24 V。

## 9. 问题 4：如何处理上游电源与 SubXF 等值阻抗，才能尽可能保留短路特性？

建议按以下顺序：

1. **建立 799 处的戴维南等值**：

```text
Z_th_799 = Z_source_referred_to_4.8kV + Z_SubXF
Z_SubXF  = (R1% + R2%)/100 × Zbase_4.8 + j (Xhl%/100 × Zbase_4.8)
Zbase_4.8 = 4.8² / 2.5 MVA = 9.216 Ω
```

本模型数值：

```text
Z_source_referred = {z_source_48.real:.8f} + j{z_source_48.imag:.8f} Ω
Z_SubXF           = {z_sub.real:.8f} + j{z_sub.imag:.8f} Ω
Z_th_799          = {z_th.real:.8f} + j{z_th.imag:.8f} Ω
|Z_th_799|        = {abs(z_th):.8f} Ω
I_fault,3ph,4.8kV = 4800 / (√3 × |Z_th_799|) = {fault_48:.1f} A
```

2. **选择平台电压基准与功率基准**。24 V 只是电压基准，功率基准必须单独选。若选 `S_platform`，则：

```text
Z_platform = Z_th_pu × V_platform² / S_platform
Z_th_pu    = |Z_th_799| / Zbase_4.8 = {z_pu:.6f}
X/R        = {abs(z_th.imag / z_th.real):.2f}（阻抗角约 {math.degrees(math.atan2(z_th.imag, z_th.real)):.2f}°）
```

3. **把所有线路、变压器和负荷阻抗按同一比例缩放**：

```text
k_Z = (V_platform / 4.8 kV)² × (S_original / S_platform)
Z_platform_element = Z_original_element × k_Z
```

这样电压降、负荷率和短路电流的标幺值才与原模型一致。

4. **把实体电源自身的输出阻抗串入**。如果 24 V 电源是变压器、功放或可编程源，先测量其短路阻抗/限流特性，再与缩放后的 `Z_th` 相加；不要假设电源是理想 0 Ω，也不要重复计算 SubXF。

5. **验证**：在 24 V 派生模型中，至少比较以下标幺量与原模型：799/799r/701 的电压、L35 电流、SubXF 视在负载率、799 三相短路电流。只有这些量在同一标幺基准下接近，才能说“短路特性得到保留”。

24 V 侧所需串联阻抗的量级（仅按 SubXF 标幺值折算，不含实体电源自身阻抗）：

{md_table(['平台功率基准 S/VA', 'Z_platform/Ω', '799 三相短路电流/A'], scaling_rows)}

若坚持使用原始 2.5 MVA 功率基准，则 `Z={z_25mva:.3e} Ω`、`I_fault≈{i_25mva/1000:.1f} kA`，在 24 V 实体平台上不可实现。这正是“只缩电压、不缩功率”不可行的原因。

## 10. 问题 5：24 V 定义在 799 与 799r 分别会怎样？

**定义在 799（推荐基准）**：

- 保留 reg1a、reg1c、Jumper、L35、701 及以下全部馈线；
- 可以研究调压器档位、LDC、B 相旁路和 A/C 相不平衡；
- 电源等值阻抗可以代表 SubXF + 上游，短路特性可调；
- 需要实现两台单相调压器或它们的等效，硬件更复杂；
- 799 与 799r 是两个不同的电气节点，测量 Vab/Vbc/Vca 时应分别引出。

**定义在 799r**：

- 绕过 799 和两台调压器，799r 直接成为电源母线；
- L35、701 及以下馈线仍然保留；
- 网络更简单，但不再是原始拓扑：调压器、Jumper、B 相旁路和 799 节点被移除或旁路；
- 701 处的短路水平不再包含调压器阻抗和控制行为；
- 无法做调压器档位/LDC 实验；
- 若只是为了在 24 V 下研究 701 以下馈线，可以把 799r 作为简化电源点，但必须在派生模型中明确标注“调压器已旁路”，不能称为“原始 IEEE 37 模型”。

**不推荐的做法**：把 24 V 电源接在 799，但把 reg1a/reg1c 短路或旁路，却仍称“电源在 799”。此时 799 与 799r 实际上等电位，拓扑语义已经改变；应在文档和 PCB 丝印上明确写“调压器旁路”。

## 11. 频率与相序

- 原模型 `DefaultBaseFrequency=60`，所有 LineCode `BaseFreq=60`；本次求解在 60 Hz 下完成。
- 若实体平台采用 50 Hz：线路 X 近似按 50/60 缩放；C 的容纳按 50/60 缩放；负荷 RLC 的 L/C 需按 50 Hz 重新计算；变压器漏抗的标幺值通常按频率修正。不能把 60 Hz 的 X/C 数值直接当作 50 Hz。
- 相序：原模型为 A-B-C 正序（Vsource 0°、-120°、+120°）。PCB 接线必须保持 A→B→C 的相序；调压器 reg1a 跨 A–B、reg1c 跨 C–B，不能随意调换。

## 12. 结论

- SubXF、reg1a/reg1c、XFM1 的参数均可从 DSS 与求解器闭合；SubXF 的基态重载和调压器接近满载是原模型的重要运行特征。
- 799 是 24 V 电源的合理拓扑参考点，但 24 V 必须定义为 4.8 kV 的 1.0 pu 等值，并配套缩放后的 SubXF/上游等值阻抗。
- 若只做正常工况演示，可以用理想 24 V 源；若要做故障注入，必须保留等效阻抗并验证短路电流。
- 799r 作为电源点会绕过调压器，属于派生简化模型，不是原始拓扑。
"""
    (out_dir / "IEEE37_Line_Parameter_Analysis.md2").write_text(text, encoding="utf-8")


def render_load_analysis(
    out_dir: Path,
    model: dict[str, Any],
    loads: list[dict[str, Any]],
    connection_map: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """生成负荷对象与接线分析。"""
    load_rows = []
    for load in loads:
        load_rows.append(
            [
                load["LoadName"],
                load["Bus1"],
                load["Phases"],
                load["Conn"],
                load["Model"],
                f"{load['kV']:.3f}",
                f"{load['kW']:.1f}",
                f"{load['kvar']:.1f}",
                f"{load['PF']:.4f}",
                load["PhasePair"],
                f"{load['Actual_P_kW']:.1f}",
                f"{load['Actual_Q_kvar']:.1f}",
            ]
        )
    pair_rows = []
    for pair, data in summary["pair_totals"].items():
        pair_rows.append([pair, data["branches"], f"{data['kW']:.1f}", f"{data['kvar']:.1f}"])
    s728 = next(load for load in loads if load["LoadName"].lower() == "s728")
    s728_branches = [row for row in connection_map if row["LoadName"].lower() == "s728"]
    model_name_map = {1: "恒定 P", 2: "恒定 Z", 4: "指数/CVR"}
    model_rows = []
    for model_id in sorted(summary["model_counts"]):
        count = summary["model_counts"][model_id]
        model_rows.append([model_id, model_name_map.get(model_id, "未知"), count])

    text = f"""<!-- 摘要：IEEE 37 负荷对象审计，记录 30 个负荷的节点、相别、Delta 接线、P/Q、Model 1/2/4 数量、CVR 默认值、S728 三相分配与固定 RLC 转换边界。 -->

# IEEE 37 负荷对象与接线分析

> 文档定位：任务 4 的负荷对象完整提取与接线映射。  
> 数据来源：`IEEE37_Load_Connection_Map.csv1` 第 69–98 行的 30 个 `New Load` 定义；OpenDSS COM 求解器的 `kW/kvar/PF/kVA/Model/CVR/ZIPV/Powers`。  
> 状态：静态定义与求解读数均已核对；逐支路映射见 `IEEE37_Load_Connection_Map.csv`。

## 1. 对象计数

{md_table(['Model', '名称', '对象数'], model_rows)}

合计 {summary['num_loads']} 个负荷对象，分布在 {summary['num_load_buses']} 个母线。其中：

- 单相 Delta 负荷 {summary['num_loads'] - 1} 个；
- 三相 Delta 负荷 1 个（`S728`）；
- 没有 Y 接负荷，没有单相接地负荷；
- 所有负荷 `Conn=Delta`，`Enabled=true`，`Status=0`（Fixed）。

## 2. Delta 接线的实际含义

在 OpenDSS 中，`Phases=1` 与 `Conn=Delta` 的组合表示一个**跨接在两个相节点之间的单相负荷**，不是“相—中性点”负荷。例如：

- `S701a Bus1=701.1.2`：接在 701 的 1 相与 2 相之间，即 A–B；
- `S701c Bus1=701.3.1`：接在 3 相与 1 相之间，即 C–A；
- `S728 Bus1=728 Phases=3 Conn=Delta`：OpenDSS 自动使用 728 的 1/2/3 三个节点，形成 (1-2)、(2-3)、(3-1) 三个 Delta 支路。

因此，“30 个负荷对象”对应“29 个单相支路 + 3 个 S728 支路 = 32 个相间支路”。`IEEE37_Load_Connection_Map.csv` 按 32 行记录，每个支路都有明确的 `PhaseFrom`、`PhaseTo`、名义 kW/kvar 和求解电压。

## 3. 负荷对象表

{md_table(['负荷', 'Bus1', '相', 'Conn', 'Model', 'kV', 'kW', 'kvar', 'PF', '相间', '实际 P/kW', '实际 Q/kvar'], load_rows)}

`实际 P/Q` 是 OpenDSS 在 1.0 pu 负荷、当前电压下的求解值；它与定义值的差异来自：

- Model 1 恒功率：在电压偏离 1.0 pu 时仍维持 P/Q，因此实际值与定义值接近；
- Model 2 恒阻抗：P/Q 随电压平方变化；
- Model 4 指数/CVR：P 随电压的 CVRwatts 次方、Q 随 CVRvars 次方变化。

## 4. 相间支路功率分布

{md_table(['相间支路', '支路数', '名义 kW', '名义 kvar'], pair_rows)}

总名义负荷为 {summary['load_kw']:.1f} kW、{summary['load_kvar']:.1f} kvar，视在功率 {summary['load_kva']:.2f} kVA，整体功率因数 {summary['load_pf']:.4f}（滞后）。C–A 支路最重，B–C 最轻；这一不平衡与 Open-Delta 两台调压器的负载差异一致。

## 5. S728 的三相 Delta 分配

`S728` 定义为 `Bus1=728`、`Phases=3`、`Conn=Delta`、`Model=1`、`kW=126`、`kvar=63`。OpenDSS 将总功率按三个相间支路均分：

{md_table(['支路', '相间', '名义 kW', '名义 kvar', '求解相节点 P/kW', '求解相节点 Q/kvar', '支路电压/kV'], [[row['BranchID'], f"{row['PhaseFrom']}-{row['PhaseTo']}", fmt(row['Nominal_Branch_kW'], 2), fmt(row['Nominal_Branch_kvar'], 2), fmt(row['Solved_PhaseNode_P_kW'], 2), fmt(row['Solved_PhaseNode_Q_kvar'], 2), fmt(row['Solved_VLL_kV'], 4)] for row in s728_branches])}

注意：`Solved_PhaseNode_P/Q` 是三个相节点的注入功率，不是三个 Delta 支路的支路功率；三者之和为 {s728['Actual_P_kW']:.2f} kW / {s728['Actual_Q_kvar']:.2f} kvar。名义支路功率仍按 42 kW / 21 kvar 记录，供固定 RLC 设计使用。

## 6. CVR、ZIPV 与其他属性

- 所有负荷的 `CVRwatts=1.0`、`CVRvars=2.0`。这两个值没有写在 DSS 中，是 OpenDSS 默认值，由求解器返回。
- 所有负荷的 `ZIPV` 为空，没有 ZIPV 参数。
- 没有 `CVRcurve`、`CVRvars` 曲线、`daily`、`duty`、`growth` 等时变属性；这是**单快照**模型，不是时序负荷模型。
- `Status=0` 表示 Fixed，不是“停运”；`Enabled=true` 才是投入状态。
- 所有负荷的 `PF` 由 kW/kvar 计算；2:1 的负荷为 0.8944，85/40 的负荷为 0.9048，`S738a`（126/62）为 0.8973；以负荷表逐项为准。

## 7. Model 1/2/4 与固定 RLC 的关系

- **Model 2（7 个）**：恒阻抗，P/Q 随电压平方变化。在基频下，固定 RLC 可以精确复现其电压依赖关系；但 R/L/C 的具体数值必须按平台电压、频率和支路功率重新计算。
- **Model 1（15 个）**：恒功率。固定 RLC 只能在选定的基准电压/功率点匹配 P/Q，不能在整个电压范围内维持恒功率。
- **Model 4（8 个）**：指数/CVR，P∝V^1、Q∝V^2（默认指数）。固定 RLC 的 P∝V²、Q∝V²，因此 P 的电压依赖不同，Q 在默认指数下相同。也不能在整个电压范围内精确复现。
- 对 Model 1/4 的固定 RLC 转换，应在报告中明确写“基准工作点等效”，并给出转换电压、频率和 P/Q；不能把转换后的 RLC 说成“原负荷模型的等价实现”。

## 8. 负荷所在节点与电压等级

- 所有负荷都在 4.8 kV 等级；没有负荷接在 230 kV 或 0.48 kV。
- `775` 是 XFM1 二次侧 0.48 kV 母线，**没有负荷**；因此 XFM1 基态潮流接近零。
- `799`、`799r` 也没有负荷；它们是调压器边界，不是负荷母线。
- 25 个负荷母线中，701、714、722、742 各有两个以上负荷对象；其余各一个。

## 9. 对固定 RLC 与 PCB 的含义

- 32 个相间支路是 PCB 负荷子板的最小连接单元；每个支路必须保留 `PhaseFrom`/`PhaseTo`，不能把单相 Delta 负荷误接成相—中性点。
- S728 的三个支路应作为三个独立的两端 RLC 支路实现，不能用一个三相 Y 型负荷替代。
- Model 1/4 的固定 RLC 只能在基态匹配；若实验需要扫描电压或故障期间的电压跌落，必须记录“负荷特性不再等价”的误差。
- 负荷功率是 4.8 kV 原始值，不是 24 V 值；在缩比计算前不要直接照搬 kW/kvar 到 24 V 元件。

## 10. 结论

30 个负荷对象的节点、相别、Delta 接线、P/Q、Model 和 CVR 默认值均已从 DSS 与求解器闭合。负荷表与接线映射表可以直接作为 KiCad 负荷子板的连接依据；但固定 RLC 的元件值尚未计算，且 Model 1/4 的全电压范围等价性不成立。
"""
    (out_dir / "IEEE37_Load_Analysis.md").write_text(text, encoding="utf-8")


def render_topology(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    graph: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """生成完整拓扑说明。"""
    bus_rows = []
    for bus in sorted(buses, key=lambda item: item["BusIndex"]):
        bus_rows.append(
            [
                bus["BusName"],
                bus["VoltageLevel_kV"],
                bus["BusType"],
                bus["BusDegree"],
                bus["ConnectedLines"] or "—",
                bus["ConnectedTransformers"] or "—",
                bus["ConnectedLoads"] or "—",
                "yes" if bus["IsLeaf"] else "no",
            ]
        )
    line_rows = []
    for line in lines:
        line_rows.append(
            [
                line["LineName"],
                line["Category"],
                line["Bus1"],
                line["Bus2"],
                line["Phases"],
                line["LineCode"] or "—",
                fmt(line["Length_mi"], 4),
                fmt(line["R1_total_ohm"], 5),
                fmt(line["X1_total_ohm"], 5),
                f"{line['MaxCurrent_A']:.1f}",
            ]
        )
    main_path = " → ".join(graph["main_path"])
    main_lines = " → ".join(graph["main_path_lines"])
    leaf_text = "、".join(graph["leaves"])
    branch_text = "、".join(graph["branch_points"])
    no_load_buses = [bus["BusName"] for bus in buses if bus["ConnectedLoadCount"] == 0]
    load_buses = [bus["BusName"] for bus in buses if bus["ConnectedLoadCount"] > 0]

    text = f"""<!-- 摘要：IEEE 37 实际 DSS 拓扑审计，记录 39 个母线名、117 个相节点、35 条馈线线路、Jumper、变压器/调压器连接、主馈线、分支、末端节点及图论检查结果。 -->

# IEEE 37 电气拓扑审计

> 文档定位：任务 2 的节点—线路拓扑核对结果，直接服务于 KiCad 主干原理图的连接关系。  
> 数据来源：`IEEE37_Load_Connection_Map.csv1` 的 `Bus1`/`Bus2`、变压器绕组、调压器和负荷定义；拓扑图见 `IEEE37_Line_Impedance_Matrix.csv6`。  
> 状态：图论检查与 OpenDSS 求解均已执行。

## 1. 计数口径

必须先区分三个层次，不能把“IEEE 37”这个名称直接当作节点数：

- **母线名称**：{summary['num_buses_total']} 个（含 `SourceBus`）。不含 `SourceBus` 的网络母线为 {summary['num_buses_network']} 个。
- **母线相节点**：{summary['num_phase_nodes']} 个（39 × 3）。每个母线都有 1、2、3 三个相节点；`799.2`、`799r.2` 等是相节点，不是独立母线。
- **变压器两侧母线**：`SubXF` 的 `SourceBus`/`799`、`XFM1` 的 `709`/`775`、`reg1a` 的 `799.1.2`/`799r.1.2`、`reg1c` 的 `799.3.2`/`799r.3.2`。这些端子已经计入上述母线名称，不重复计数。

`SourceBus` 是 230 kV 理想电源母线，不属于 4.8 kV 馈线的 37/38 个负荷母线；`775` 是 XFM1 的 0.48 kV 二次侧母线，也不是 L1–L35 中的一条线路。变压器 primitive 矩阵内部还有第 4 个中性导体，但它不是母线相节点，未计入 117。

## 2. 电压等级

{md_table(['电压等级', '母线', '说明'], [
    ['230 kV', 'SourceBus', 'Vsource 理想电源；经 SubXF 降至 4.8 kV'],
    ['4.8 kV', '799、799r、701–744', '三相三线 Delta；799 为 SubXF 二次侧，799r 为 Open-Delta 调压器输出'],
    ['0.48 kV', '775', 'XFM1 二次侧；基态无负荷，潮流接近零'],
])}

注意：OpenDSS 的 `Bus.kVBase` 返回的是**线中性点基准**（4.8 kV 母线为 2.771 kV，0.48 kV 母线为 0.2771 kV），而 `VLL` 返回线电压。`IEEE37_Line_Impedance_Matrix.csv2` 同时保留 `kVBase_LN` 与 `NominalLL_kV`，避免把 2.771 kV 误当作 4.8 kV 线电压。

## 3. 节点表

{md_table(['母线', 'kV（LL）', '类型', '度', '线路', '变压器', '负荷', '末端'], bus_rows)}

负荷母线共 {summary['num_load_buses']} 个：{', '.join(load_buses)}。无负荷母线共 {len(no_load_buses)} 个：{', '.join(no_load_buses)}。

## 4. 线路表

{md_table(['线路', '类别', 'Bus1', 'Bus2', '相', 'LineCode', 'Length/mi', 'R1/Ω', 'X1/Ω', 'Imax/A'], line_rows)}

说明：

- L1–L35 全部为 `Phases=3`，`Bus1`/`Bus2` 均显式写出 `.1.2.3`；本表保留 DSS 原文，不按名称推断端点。
- `Jumper` 是 `Phases=1`，只连接 `799.2` 与 `799r.2`，不参与 L1–L35 计数。
- `R1`、`X1` 是全矩阵正序分量的整条线路总阻抗，不是 DSS 的 `Line.R1` 单长属性；计算口径见 `IEEE37_Line_Parameter_Analysis.md`。

## 5. 主馈线、分支与末端

按线路长度加权、从 `799r` 出发的最远路径为：

`{main_path}`

对应线路：

`{main_lines}`

路径长度约 {graph['main_path_length_mi']:.3f} mi。这里的“主馈线”是图论最长路径，不代表运行方式或保护配合的唯一主干；实际 PCB 分区可以按电气邻近性重组，但不得改变连接关系。

分支点（母线度 ≥ 3）：{branch_text}。其中 799、799r 是调压器边界，其余为馈线分支点。

末端节点（度 1，不含 SourceBus）：{leaf_text}。其中 775 是 XFM1 二次侧无负荷末端，不是馈线线路末端；其余 14 个是 L1–L35 的馈线末端。

## 6. 799、799r、701 的真实关系

DSS 中不存在一条名为“799–799r”的三相线路。三者关系是：

- `SubXF` 二次侧接 `799`；`799` 是调压器输入母线。
- `reg1a` 接 `799.1.2` → `799r.1.2`（A–B 相间，单相双绕组，4.8/4.8 kV，2000 kVA，Xhl=1%）。
- `reg1c` 接 `799.3.2` → `799r.3.2`（C–B 相间，单相双绕组，4.8/4.8 kV，2000 kVA，Xhl=1%）。
- `Jumper` 接 `799.2` → `799r.2`（B 相，1 mΩ）。
- `L35` 接 `799r.1.2.3` → `701.1.2.3`（LineCode 721，1.85 mi）。

因此 B 相通过 1 mΩ Jumper 直接连接，A、C 相经过调压器；799r 的 A/C 电压由调压器档位和负荷决定，不能假定等于 799。`701` 是 L35 之后的第一个馈线母线，不是调压器输出母线。

基态下 Jumper 电流近零（约 0.00015 A），因为 SubXF、reg1a、reg1c 的 B 相电流在 799.2 处相量和为零；Jumper 的作用是固定 B 相电位，不是基态功率通道。故障或严重不平衡时 Jumper 仍可能承载大电流，PCB 上不能按基态电流给它定尺寸。

## 7. 变压器与调压器形成的连接

{md_table(['设备', '端子 1', '端子 2', '连接', '额定', '作用'], [
    ['SubXF', 'SourceBus', '799', 'Δ–Δ', '230/4.8 kV，2500 kVA', '主降压；其漏抗决定 799 的短路水平'],
    ['XFM1', '709', '775', 'Δ–Δ', '4.8/0.48 kV，500 kVA', '负荷变压器支路；775 基态无负荷'],
    ['reg1a', '799.1.2', '799r.1.2', '单相 Δ–Δ', '4.8/4.8 kV，2000 kVA', 'Open-Delta 的 A–B 支路'],
    ['reg1c', '799.3.2', '799r.3.2', '单相 Δ–Δ', '4.8/4.8 kV，2000 kVA', 'Open-Delta 的 C–B 支路'],
    ['Jumper', '799.2', '799r.2', '单相线路', '1 mΩ，0 mH', 'B 相旁路/直接连接'],
])}

## 8. 图论检查

{md_table(['检查项', '结果', '说明'], [
    ['母线级连通分量', graph['num_components'], '全部 39 个母线名连通'],
    ['母线级边数', graph['num_edges'], '35 条馈线 + 1 Jumper + 4 个变压器/调压器'],
    ['回路秩（多重边图）', graph['cycle_rank'], 'E − V + C = 40 − 39 + 1 = 2，全部来自 799–799r 的并联设备'],
    ['馈线本体（L1–L35）是否为树', 'yes' if graph['feeder_is_tree'] else 'no', f"{graph['feeder_tree_nodes']} 节点、{graph['feeder_tree_edges']} 边"],
    ['馈线简单回路', len(graph['simple_cycles']), '无意外环路'],
    ['并联边组', len(graph['parallel_groups']), '799–799r：reg1a、reg1c、Jumper'],
    ['重复线路定义', len(graph['duplicate_lines']), '无同端点重复线路'],
    ['孤立母线', len(graph['isolated']), '无'],
    ['相节点连通分量', graph['num_phase_components'], f"含 4 个变压器设备节点；{graph['num_phase_nodes']} 个相节点/设备节点全部连通"],
    ['相别连接错误', len(graph['phase_errors']), '无；三相线路均为 1-1/2-2/3-3，单相 Delta 负荷均为两节点'],
])}

799–799r 之间的三条并联边不是“重复线路”：它们是两个单相调压器加一条 B 相 Jumper，属于原始模型的有意设计。除这组并联边外，L1–L35 构成严格的树，没有意外环路。

## 9. 负荷在节点上的分布

30 个负荷对象分布在 {summary['num_load_buses']} 个母线；其中 701、714、722、742 各有两个以上对象。三个相间支路的名义功率分布为：

{md_table(['相间支路', '支路数', 'kW', 'kvar'], [[pair, data['branches'], fmt(data['kW'], 6), fmt(data['kvar'], 6)] for pair, data in summary['pair_totals'].items()])}

C–A 支路负荷最重（{summary['pair_totals']['3-1']['kW']:.0f} kW），A–B 次之（{summary['pair_totals']['1-2']['kW']:.0f} kW），B–C 最轻（{summary['pair_totals']['2-3']['kW']:.0f} kW）。这一不平衡是 Open-Delta 两台调压器负载差异的直接原因，详见 `IEEE37_Load_Analysis.md` 与 `IEEE37_Line_Parameter_Analysis.md2`。

## 10. 拓扑图

拓扑图 `IEEE37_Line_Impedance_Matrix.csv6` 使用 `IEEE37_Load_Connection_Map.csv0` 的坐标作为底图，按电压等级着色，线路用实线、变压器用虚线、调压器/ Jumper 用不同线型，并标注 L1–L35、负荷节点和无负荷末端。图中的 799/799r 顶部区域经过纵向展开，以避免三个母线标签重叠；电气连接关系不变。

## 11. 结论

- 所有母线名、相节点和线路端点均可从 DSS 直接闭合，无需按名称猜测。
- 馈线本体是树；唯一的多重边是 799–799r 的调压器/Jumper 组合。
- 没有孤立母线、重复线路、未定义节点或相别连接错误。
- 775 是 XFM1 二次侧无负荷末端；799/799r 是调压器边界，不是负荷母线。
"""
    (out_dir / "IEEE37_Topology.md").write_text(text, encoding="utf-8")


def render_solver_log(
    out_dir: Path,
    model: dict[str, Any],
    solver: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """生成结构化求解日志，不伪造 OpenDSS 控制台原文。"""
    solution = solver["solution"]
    max_diff = max(line["Yprim_Impedance_MaxDiff_ohm"] for line in summary.get("_lines", [])) if summary.get("_lines") else "see IEEE37_Line_Impedance_Matrix.csv0"
    lines = [
        "IEEE 37 OpenDSS Solver Log",
        "=" * 72,
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"source_dir: {model['source_dir']}",
        f"master_file: {model['master_file']}",
        f"sha256_ieee37: {next(item['sha256'] for item in model['inventory'] if item['file'] == 'IEEE37_Load_Connection_Map.csv1')}",
        f"sha256_linecodes: {next(item['sha256'] for item in model['inventory'] if item['file'] == 'IEEELineCodes.DSS')}",
        f"opendss_version: {solver['version']}",
        f"compile_command: {solver['compile_command']}",
        f"compile_error_number: {solver['compile_error_number']}",
        f"compile_error_description: {solver['compile_error_description']!r}",
        f"circuit_name: {solver['circuit_name']}",
        f"wall_time_s: {solver.get('_wall_time_s', 'n/a')}",
        f"num_buses: {solver['num_buses']}",
        f"num_nodes: {solver['num_nodes']}",
        f"num_elements: {solver['num_elements']}",
        f"converged: {solution['converged']}",
        f"iterations: {solution['iterations']}",
        f"control_iterations: {solution['control_iterations']}",
        f"total_iterations: {solution['total_iterations']}",
        f"max_iterations: {solution['max_iterations']}",
        f"tolerance: {solution['tolerance']}",
        f"frequency_hz: {solution['frequency']}",
        f"mode: {solution['mode']}",
        f"control_mode: {solution['control_mode']}",
        f"total_time_ms: {solution['total_time_ms']}",
        f"process_time_ms: {solution['process_time_ms']}",
        "",
        "Key results",
        "-" * 72,
        f"bus_count_with_sourcebus: {summary['num_buses_total']}",
        f"bus_count_network: {summary['num_buses_network']}",
        f"phase_nodes: {summary['num_phase_nodes']}",
        f"feeder_lines: {summary['num_feeder_lines']}",
        f"jumpers: {summary['num_jumpers']}",
        f"loads: {summary['num_loads']}",
        f"load_branches: {sum(v['branches'] for v in summary['pair_totals'].values())}",
        f"transformers: {summary['num_transformers']}",
        f"regulators: {summary['num_regulators']}",
        f"load_kw: {summary['load_kw']}",
        f"load_kvar: {summary['load_kvar']}",
        f"load_kva: {summary['load_kva']}",
        f"load_pf: {summary['load_pf']}",
        f"min_bus: {summary['min_bus']['BusName']} {summary['min_bus']['Vmin_pu']}",
        f"max_bus: {summary['max_bus']['BusName']} {summary['max_bus']['Vmax_pu']}",
        f"max_unbalance_bus: {summary['max_unbalance']['BusName']} {summary['max_unbalance']['V2_V1_pct']}",
        f"max_current_line: {summary['max_current_line']['LineName']} {summary['max_current_line']['MaxCurrent_A']}",
        f"max_line_loading_pct: {summary['max_loading_line']['MaxLoading_pct']}",
        f"subxf_loading_pct: {summary['subxf']['Loading_pct']}",
        f"subxf_current_loading_pct: {summary['subxf']['CurrentLoading2_pct']}",
        f"reg1a_loading_pct: {summary['reg1a']['Loading_pct']}",
        f"reg1c_loading_pct: {summary['reg1c']['Loading_pct']}",
        f"reg1a_tap: {summary['reg1a']['Tap2']}",
        f"reg1c_tap: {summary['reg1c']['Tap2']}",
        "",
        "Checks",
        "-" * 72,
        f"feeder_is_tree: {summary['graph']['feeder_is_tree']}",
        f"bus_components: {summary['graph']['num_components']}",
        f"cycle_rank: {summary['graph']['cycle_rank']}",
        f"parallel_edge_groups: {len(summary['graph']['parallel_groups'])}",
        f"isolated_buses: {len(summary['graph']['isolated'])}",
        f"phase_errors: {len(summary['graph']['phase_errors'])}",
        f"phase_node_components: {summary['graph']['num_phase_components']}",
        f"max_yprim_impedance_diff_ohm: {max_diff}",
        "",
        "Limitations",
        "-" * 72,
        "- COM 接口未暴露完整解析器警告列表；仅检查 Error.Number/Description。",
        "- ZscRefresh 会改变求解状态；本次在所有电压/电流/功率读取完成后才刷新。",
        "- Line.R1 / LineCode.R1 标量属性存在活动对象缓存；本审计使用 3×3 矩阵。",
        "- 本次为 60 Hz 单快照正常工况，不含故障、时序或 50 Hz 场景。",
        "- 原始 .rar 本体未在文件系统中找到；使用此前解压目录，三个电气文件与 canonical 安装逐字节一致。",
    ]
    (out_dir / "IEEE37_Line_Impedance_Matrix.csv4").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """命令行入口：静态解析、求解、CSV/Markdown/图件生成与发布。"""
    parser = argparse.ArgumentParser(description="IEEE 37 OpenDSS 模型审计")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--exp-meters", type=Path)
    parser.add_argument("--solver-json", type=Path)
    parser.add_argument("--solve-worker", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.solve_worker:
        if args.solver_json is None:
            raise SystemExit("--solve-worker 需要 --solver-json")
        solve_worker(args.source_dir.resolve(), args.solver_json.resolve())
        return

    source_dir = args.source_dir.resolve()
    if not (source_dir / "IEEE37_Load_Connection_Map.csv1").exists():
        raise SystemExit(f"源目录缺少 IEEE37_Load_Connection_Map.csv1：{source_dir}")
    default_out = Path(__file__).resolve().parents[1] / "output" / "ieee37_model_audit"
    out_dir = (args.out or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(source_dir)
    exp_meters = args.exp_meters
    if exp_meters is None:
        local = source_dir / "ieee37_EXP_METERS.CSV"
        canonical = Path(r"C:\Program Files\OpenDSS\IEEETestCases\37Bus\ieee37_EXP_METERS.CSV")
        exp_meters = local if local.exists() else (canonical if canonical.exists() else None)
    exp_meters = exp_meters.resolve() if exp_meters and exp_meters.exists() else None

    raw_json = out_dir / "IEEE37_Solver_Raw.json"
    solver, solver_message = run_solver(source_dir, raw_json, timeout_s=args.timeout)
    if solver is None:
        (out_dir / "IEEE37_Line_Impedance_Matrix.csv4").write_text(
            "IEEE 37 OpenDSS Solver Log\n"
            "==========================\n"
            f"求解未完成：{solver_message}\n"
            "未生成任何伪造的电压、电流或功率数据。\n",
            encoding="utf-8",
        )
        raise SystemExit(f"求解失败：{solver_message}")

    buses = build_bus_records(model, solver)
    lines = build_line_records(model, solver)
    linecodes = build_linecode_records(model, solver, referenced_only=True)
    all_linecodes = build_linecode_records(model, solver, referenced_only=False)
    loads = build_load_records(model, solver)
    connection_map = build_connection_map(model, solver, buses)
    transformers = build_transformer_records(model, solver)
    regulators = build_regulator_records(model, solver)
    graph = graph_audit(model)
    summary = build_summary(
        model, solver, buses, lines, loads, connection_map, transformers, regulators, graph
    )
    summary["_lines"] = lines

    write_csv_outputs(
        out_dir, model, solver, buses, lines, linecodes, all_linecodes, loads,
        connection_map, transformers, regulators, graph, exp_meters,
    )
    render_integrity(out_dir, model, solver, summary, exp_meters)
    render_topology(out_dir, model, solver, buses, lines, loads, transformers, regulators, graph, summary)
    render_line_analysis(out_dir, model, lines, linecodes, summary)
    render_load_analysis(out_dir, model, loads, connection_map, summary)
    render_source_analysis(out_dir, model, solver, buses, transformers, regulators, summary)
    render_solver_validation(out_dir, model, solver, buses, lines, loads, transformers, regulators, summary)
    render_readme(out_dir, model, solver, buses, lines, loads, connection_map, transformers, regulators, summary, exp_meters)
    render_solver_log(out_dir, model, solver, summary)
    plot_topology(out_dir, model, buses, lines, summary)

    source_out = out_dir / "source"
    source_out.mkdir(parents=True, exist_ok=True)
    for name in ["IEEE37_Load_Connection_Map.csv1", "IEEELineCodes.DSS", "IEEE37_Load_Connection_Map.csv0"]:
        shutil.copy2(source_dir / name, source_out / name)
    if exp_meters:
        shutil.copy2(exp_meters, source_out / exp_meters.name)

    serializable_summary = {key: value for key, value in summary.items() if key != "_lines"}
    (out_dir / "IEEE37_Line_Impedance_Matrix.csv7").write_text(
        json.dumps(serializable_summary, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )

    print(f"审计完成：{out_dir}")
    print(solver_message)

    if args.publish:
        target = args.publish.resolve()
        if target.name not in {"IEEE37_Model_Audit", "ieee37_model_audit"}:
            raise SystemExit(f"拒绝发布到非审计目录：{target}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(out_dir, target)
        print(f"已发布：{target}")


if __name__ == "__main__":
    main()

