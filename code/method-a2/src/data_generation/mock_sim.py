"""不依赖 OpenDSS 的确定性 mock 故障仿真器，用于 smoke 与契约测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .opendss_sim import FAULT_CLASSES, FaultConfig


@dataclass
class _MockResult:
    """保存 mock 场景的预故障与故障后相量。"""

    pre_v: np.ndarray
    post_v: np.ndarray


class MockFaultSimulator:
    """提供与 FaultSimulator 兼容的最小接口。

    该仿真器只用于单元测试和 mock smoke，不参与正式 OpenDSS 数据生成。
    """

    def __init__(self, case_name: str = "mock", n_nodes: int = 4):
        if n_nodes < 2:
            raise ValueError("mock 仿真器至少需要两个母线")
        self.case_name = str(case_name)
        self._n_nodes = int(n_nodes)
        self._base_loads = {
            f"load{i}": (10.0 + i, 2.0 + 0.5 * i) for i in range(max(1, n_nodes // 2))
        }
        self.line_params = {}
        for i in range(n_nodes - 1):
            r = 0.05 + 0.01 * i
            x = 0.10 + 0.02 * i
            self.line_params[(i, i + 1)] = (r, x, float(np.hypot(r, x)))
            self.line_params[(i + 1, i)] = (r, x, float(np.hypot(r, x)))
        self._load_multipliers = {}
        self._last_config: Optional[FaultConfig] = None

    def _load_scale(self, name: str) -> float:
        """返回指定负荷在当前工况下的倍率。"""
        return float(self._load_multipliers.get(name, 1.0))

    def _compile_and_solve_base(self, load_multipliers: Optional[dict] = None) -> None:
        """记录当前负荷工况，不执行外部求解。"""
        self._load_multipliers = {
            str(name): float(value) for name, value in (load_multipliers or {}).items()
        }

    def _read_voltages(self) -> np.ndarray:
        """返回无故障基态相量 `[N,6]`。"""
        out = np.zeros((self._n_nodes, 6), dtype=np.float32)
        for node in range(self._n_nodes):
            scale = 1.0
            for name in self._base_loads:
                scale += 0.005 * (self._load_scale(name) - 1.0)
            out[node, 0] = 1.0 * scale - 0.004 * node
            out[node, 1] = 1.0 * scale - 0.003 * node
            out[node, 2] = 1.0 * scale - 0.002 * node
            out[node, 3] = -0.5 * node
            out[node, 4] = -120.0 - 0.5 * node
            out[node, 5] = 120.0 + 0.5 * node
        return out

    def _apply_fault(self, config: FaultConfig, base: np.ndarray) -> np.ndarray:
        """根据故障类型、位置和阻抗生成确定性故障后相量。"""
        post = base.copy()
        bus = int(config.fault_bus)
        strength = 1.0 / (1.0 + float(config.z_fault))
        if config.fault_class in (0, 2, 3, 4):
            phases = (0,) if config.fault_class == 0 else (0, 1, 2)
        else:
            phases = (0, 1)
        for phase in phases:
            post[bus, phase] *= max(0.05, 1.0 - 0.8 * strength)
            post[bus, 3 + phase] -= 5.0 * strength
        for neighbor in (bus - 1, bus + 1):
            if 0 <= neighbor < self._n_nodes:
                for phase in range(3):
                    post[neighbor, phase] *= 1.0 - 0.15 * strength
                    post[neighbor, 3 + phase] -= 1.0 * strength
        return post.astype(np.float32)

    def generate_scenario(self, config: FaultConfig) -> dict:
        """返回单个 mock 故障场景的相量与标签。"""
        if config.fault_class not in FAULT_CLASSES:
            raise ValueError(f"未知故障类型：{config.fault_class}")
        self._compile_and_solve_base(config.load_multipliers)
        base = self._read_voltages()
        post = self._apply_fault(config, base)
        self._last_config = config
        return {
            "pre_v": base,
            "post_v": post,
            "y_detect": 1,
            "y_loc": int(config.fault_bus),
            "y_class": int(config.fault_class),
            "y_resist": float(config.z_fault),
        }
