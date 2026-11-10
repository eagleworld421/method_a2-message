"""基于 OpenDSS COM 接口的故障场景仿真器。"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np

FAULT_CLASSES = {0: "LG", 1: "LL", 2: "LLG", 3: "LLLG", 4: "LLL"}
_DSS_CASES = {
    "ieee13": r"C:\Program Files\OpenDSS\IEEETestCases\13Bus\IEEE13Nodeckt.dss",
    "ieee37": r"C:\Program Files\OpenDSS\IEEETestCases\37Bus\ieee37.dss",
    "ieee123": r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss",
}


@dataclass(frozen=True)
class FaultSpec:
    """一个物理可行的故障类型与相别组合。"""

    fault_class: int
    phases: tuple[int, ...]


def enumerate_fault_specs(available_phases) -> list[FaultSpec]:
    """按母线实际相别枚举五类故障中的物理可行组合。"""
    phases = tuple(sorted({int(value) for value in available_phases if int(value) > 0}))
    result = [FaultSpec(0, (phase,)) for phase in phases]
    for pair in combinations(phases, 2):
        result.append(FaultSpec(1, pair))
        result.append(FaultSpec(2, pair))
    if len(phases) >= 3:
        for triple in combinations(phases, 3):
            result.append(FaultSpec(3, triple))
            result.append(FaultSpec(4, triple))
    return result


def build_fault_command(
    bus_name: str,
    fault_class: int,
    z_fault: float,
    phases,
    floating_node: int = 9,
) -> str:
    """构造具有明确相别和接地语义的 OpenDSS 故障命令。"""
    if fault_class not in FAULT_CLASSES:
        raise ValueError(f"未知故障类型：{fault_class}")
    phase_tuple = tuple(int(value) for value in phases)
    required = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3}[int(fault_class)]
    if len(phase_tuple) != required or len(set(phase_tuple)) != len(phase_tuple):
        raise ValueError(f"故障类型 {FAULT_CLASSES[fault_class]} 的相别数量必须为 {required}")
    phase_text = ".".join(str(value) for value in phase_tuple)
    if fault_class == 0:
        bus1 = f"{bus_name}.{phase_text}"
        bus2 = f"{bus_name}.0"
        element_phases = 1
    elif fault_class == 1:
        bus1 = f"{bus_name}.{phase_tuple[0]}"
        bus2 = f"{bus_name}.{phase_tuple[1]}"
        element_phases = 1
    elif fault_class == 2:
        bus1 = f"{bus_name}.{phase_text}"
        bus2 = f"{bus_name}.0.0"
        element_phases = 2
    elif fault_class == 3:
        bus1 = f"{bus_name}.{phase_text}"
        bus2 = f"{bus_name}.0.0.0"
        element_phases = 3
    else:
        bus1 = f"{bus_name}.{phase_text}"
        floating = ".".join([str(int(floating_node))] * 3)
        bus2 = f"{bus_name}.{floating}"
        element_phases = 3
    return (
        f"New Fault.F1 bus1={bus1} bus2={bus2} "
        f"phases={element_phases} r={float(z_fault):g}"
    )


class OpenDSSUnavailableError(RuntimeError):
    """表示当前 Python 环境无法加载 OpenDSS COM 接口。"""


def _try_import_com():
    """延迟导入 COM 客户端，保证纯单元测试不依赖 OpenDSS。"""
    try:
        import win32com.client
    except Exception:
        return None
    return win32com.client


@dataclass
class FaultConfig:
    """单次故障场景配置。"""

    fault_class: int
    fault_bus: int
    z_fault: float
    load_multipliers: Optional[dict] = None
    fault_phases: Optional[tuple[int, ...]] = None


class FaultSimulator:
    """驱动 OpenDSS IEEE 测试馈线并读取六维母线电压相量。

    母线与候选轴不包含电源母线：电源母线由理想电压源固定，其电压不随故障位置变化，
    保留在候选轴里会让「故障发生在电源母线」这类物理上不成立的样本进入数据，并使
    该候选与无故障候选几乎重合。剔除逻辑见 `exclude_source_buses`。
    """

    def __init__(self, case_name: str = "ieee13"):
        if case_name not in _DSS_CASES:
            raise ValueError(f"未知电网：{case_name}，可选值为 {list(_DSS_CASES)}")
        self.case_name = case_name
        self._master_path = _DSS_CASES[case_name]
        self._dss = None
        self._bus_names = []
        self._n_nodes = 0
        self._node_counts = []
        self._bus_phase_nodes = []
        self._base_loads = None
        self.adj_matrix = None
        self.line_params = {}
        self._topology_loaded = False
        self._com_initialized = False
        self.excluded_bus_names: list[str] = []
        """被剔除出母线与候选轴的电源母线名，供数据元信息留档。"""

    @staticmethod
    def exclude_source_buses(all_names, source_buses) -> tuple[list[str], list[str]]:
        """从母线序列中剔除电源母线，返回（保留列表，剔除列表）。

        比较时统一转小写并去除相别后缀，避免 OpenDSS 返回的大小写差异影响判断。
        """
        source = {
            str(name).split(".")[0].strip().lower() for name in source_buses
        }
        kept = [
            name for name in all_names
            if str(name).split(".")[0].strip().lower() not in source
        ]
        dropped = [
            name for name in all_names
            if str(name).split(".")[0].strip().lower() in source
        ]
        return kept, dropped

    def _read_source_buses(self) -> set[str]:
        """读取全部电压源元件连接的母线名。"""
        circuit = self._dss.ActiveCircuit
        names: set[str] = set()
        collection = circuit.Vsources
        index = collection.First
        while index > 0:
            circuit.SetActiveElement(f"Vsource.{collection.Name}")
            for bus in circuit.ActiveCktElement.BusNames:
                main = str(bus).split(".")[0].strip().lower()
                if main:
                    names.add(main)
            index = collection.Next
        return names

    def close(self) -> None:
        """释放 OpenDSS 电路与 COM 引用。"""
        dss = self._dss
        self._dss = None
        if dss is not None:
            clear = getattr(dss, "ClearAll", None)
            if callable(clear):
                clear()
        del dss
        gc.collect()
        if self._com_initialized:
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            finally:
                self._com_initialized = False

    def _resolve_bus(self, bus_str: str):
        """将带相位后缀的母线名解析为母线索引。"""
        main = bus_str.strip().lower().split(".")[0]
        return self._bus_name_to_idx.get(main)

    def _load_topology(self) -> None:
        """读取母线邻接关系及线路/变压器参数。"""
        dss = self._dss
        n = self._n_nodes
        adj = np.zeros((n, n), dtype=np.int32)
        params = {}
        self._bus_name_to_idx = {
            name.strip().lower(): idx
            for idx, name in enumerate(self._bus_names)
        }
        i = dss.ActiveCircuit.Lines.First
        while i > 0:
            names = dss.ActiveCircuit.ActiveCktElement.BusNames
            if len(names) >= 2:
                bi, bj = self._resolve_bus(names[0]), self._resolve_bus(names[1])
                if bi is not None and bj is not None:
                    r = float(dss.ActiveCircuit.Lines.R1)
                    x = float(dss.ActiveCircuit.Lines.X1)
                    length = float(dss.ActiveCircuit.Lines.Length)
                    if length > 0:
                        r *= length
                        x *= length
                    z = float(np.hypot(r, x))
                    adj[bi, bj] = adj[bj, bi] = 1
                    params[(bi, bj)] = (r, x, z)
                    params[(bj, bi)] = (r, x, z)
            i = dss.ActiveCircuit.Lines.Next
        i = dss.ActiveCircuit.Transformers.First
        while i > 0:
            names = dss.ActiveCircuit.ActiveCktElement.BusNames
            if len(names) >= 2:
                bi, bj = self._resolve_bus(names[0]), self._resolve_bus(names[1])
                if bi is not None and bj is not None:
                    x = float(dss.ActiveCircuit.Transformers.Xhl)
                    adj[bi, bj] = adj[bj, bi] = 1
                    params[(bi, bj)] = (0.0, x, x)
                    params[(bj, bi)] = (0.0, x, x)
            i = dss.ActiveCircuit.Transformers.Next
        self.adj_matrix = adj
        self.line_params = params

    def _compile_and_solve_base(self, load_multipliers: Optional[dict] = None) -> None:
        """编译电路、应用负荷缩放并求解无故障基态。"""
        com = _try_import_com()
        if com is None:
            raise OpenDSSUnavailableError(
                f"无法加载 OpenDSS COM：case={self.case_name}, path={self._master_path}"
            )
        try:
            if not self._com_initialized:
                import pythoncom

                pythoncom.CoInitialize()
                self._com_initialized = True
            dss = com.Dispatch("OpenDSSEngine.DSS")
            self._dss = dss
            dss.Text.Command = f'Compile "{self._master_path}"'
            if not dss.ActiveCircuit:
                raise RuntimeError("OpenDSS 电路对象为空")
            if self._base_loads is None:
                base = {}
                i = dss.ActiveCircuit.Loads.First
                while i > 0:
                    name = dss.ActiveCircuit.Loads.Name
                    base[name] = (
                        float(dss.ActiveCircuit.Loads.kW),
                        float(dss.ActiveCircuit.Loads.kvar),
                    )
                    i = dss.ActiveCircuit.Loads.Next
                self._base_loads = base
            if load_multipliers:
                i = dss.ActiveCircuit.Loads.First
                while i > 0:
                    name = dss.ActiveCircuit.Loads.Name
                    if name in self._base_loads:
                        mult = float(load_multipliers.get(name, 1.0))
                        kw, kvar = self._base_loads[name]
                        dss.ActiveCircuit.Loads.kW = kw * mult
                        dss.ActiveCircuit.Loads.kvar = kvar * mult
                    i = dss.ActiveCircuit.Loads.Next
            dss.ActiveCircuit.Solution.Solve()
            # 电源母线由理想电压源钳位，其电压不随故障位置变化，因此从母线与候选轴
            # 中剔除；若剔除后为空（异常算例），退回全部母线以免生成空数据集。
            kept, dropped = self.exclude_source_buses(
                list(dss.ActiveCircuit.AllBusNames), self._read_source_buses()
            )
            if not kept:
                kept, dropped = list(dss.ActiveCircuit.AllBusNames), []
            self._bus_names = kept
            self.excluded_bus_names = dropped
            self._n_nodes = len(self._bus_names)
            self._bus_phase_nodes = self._read_bus_phase_nodes()
            self._node_counts = [len(nodes) for nodes in self._bus_phase_nodes]
            if not self._topology_loaded:
                self._load_topology()
                self._topology_loaded = True
        except OpenDSSUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 基态求解失败：case={self.case_name}, path={self._master_path}；"
                f"底层异常 {type(exc).__name__}: {exc}"
            ) from exc

    def _read_node_counts(self) -> list:
        """读取每个母线的相节点数。"""
        return [len(nodes) for nodes in self._read_bus_phase_nodes()]

    def _read_bus_phase_nodes(self) -> list[tuple[int, ...]]:
        """读取每个母线实际存在的一至三相节点编号。"""
        phases = []
        for name in self._bus_names:
            self._dss.ActiveCircuit.SetActiveBus(name)
            nodes = tuple(
                int(node)
                for node in self._dss.ActiveCircuit.ActiveBus.Nodes
                if 1 <= int(node) <= 3
            )
            phases.append(nodes)
        return phases

    def _apply_fault(self, config: FaultConfig) -> tuple[bool, tuple[int, ...] | None]:
        """在目标母线注入故障并重新求解，返回是否可行与实际使用的相别。

        故障相别必须落在该母线实际存在的相别上：母线相别不足时，仅指定相数的
        命令会被 OpenDSS 静默忽略，使故障不生效。此处先按母线可用相别筛出物理
        可行的相别组合，再交给 build_fault_command 构造带明确相别的命令。
        若该母线不存在支持该故障类型的相别组合，则返回 (False, None)，由调用方
        按数据契约处理，不在此处抛出异常。
        """
        if not 0 <= config.fault_bus < len(self._bus_names):
            raise IndexError(f"故障母线越界：{config.fault_bus}")
        try:
            bus_name = self._bus_names[config.fault_bus]
            phases = config.fault_phases
            if phases is None:
                specs = [
                    spec for spec in enumerate_fault_specs(
                        self._bus_phase_nodes[config.fault_bus]
                    )
                    if spec.fault_class == config.fault_class
                ]
                if not specs:
                    return False, None
                phases = specs[0].phases
            command = build_fault_command(
                bus_name,
                config.fault_class,
                config.z_fault,
                phases,
            )
            self._dss.Text.Command = command
            self._dss.ActiveCircuit.Solution.Solve()
            return True, tuple(int(value) for value in phases)
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 故障求解失败：case={self.case_name}, "
                f"bus={config.fault_bus}, class={config.fault_class}, z={config.z_fault}；"
                f"底层异常 {type(exc).__name__}: {exc}"
            ) from exc

    def _read_voltages(self) -> np.ndarray:
        """读取母线三相幅值与相角，返回 `[N,6]`。"""
        try:
            circuit = self._dss.ActiveCircuit
            out = np.zeros((self._n_nodes, 6), dtype=np.float32)
            for bus_index, name in enumerate(self._bus_names):
                circuit.SetActiveBus(name)
                nodes = [int(node) for node in circuit.ActiveBus.Nodes]
                values = list(circuit.ActiveBus.puVmagAngle)
                for position, node in enumerate(nodes):
                    if not 1 <= node <= 3 or 2 * position + 1 >= len(values):
                        continue
                    phase_index = node - 1
                    out[bus_index, phase_index] = float(values[2 * position])
                    out[bus_index, 3 + phase_index] = float(values[2 * position + 1])
            return out
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 电压读取失败：case={self.case_name}, bus_count={self._n_nodes}；"
                f"底层异常 {type(exc).__name__}: {exc}"
            ) from exc

    def generate_scenario(self, config: FaultConfig) -> dict:
        """生成单个故障场景并返回预故障/故障后相量及标签。"""
        if config.fault_class not in FAULT_CLASSES:
            raise ValueError(f"未知故障类型：{config.fault_class}")
        self._compile_and_solve_base(config.load_multipliers)
        pre_v = self._read_voltages()
        applied, used_phases = self._apply_fault(config)
        if not applied:
            # 该母线无可用相别组合支持此故障类型：故障后仍取无故障基态，
            # 由数据生成层依据 fault_feasible 标记把该候选置为无效
            return {
                "pre_v": pre_v,
                "post_v": pre_v.copy(),
                "fault_feasible": False,
                "fault_phases": None,
                "y_detect": 1,
                "y_loc": config.fault_bus,
                "y_class": config.fault_class,
                "y_resist": float(config.z_fault),
            }
        post_v = self._read_voltages()
        return {
            "pre_v": pre_v,
            "post_v": post_v,
            "fault_feasible": True,
            "fault_phases": used_phases,
            "y_detect": 1,
            "y_loc": config.fault_bus,
            "y_class": config.fault_class,
            "y_resist": float(config.z_fault),
        }
