"""构建 S0 全候选 OpenDSS 动态签名数据集。"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .opendss_sim import FAULT_CLASSES, FaultConfig, FaultSimulator
from .topology import build_observation_mask
from .waveform import build_dynamic_window


@dataclass
class EventConfig:
    """一组候选共享的物理事件参数。"""

    fault_class: int
    z_fault: float
    load_multipliers: dict


def _edge_arrays(simulator: FaultSimulator):
    """从仿真器线路参数构造双向边及五维边属性。"""
    undirected = sorted({(i, j) if i < j else (j, i) for i, j in simulator.line_params})
    edges = [edge for pair in undirected for edge in (pair, pair[::-1])]
    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_attr = np.asarray([
        [
            (simulator.line_params[(i, j)] if (i, j) in simulator.line_params else simulator.line_params[(j, i)])[0],
            (simulator.line_params[(i, j)] if (i, j) in simulator.line_params else simulator.line_params[(j, i)])[1],
            (simulator.line_params[(i, j)] if (i, j) in simulator.line_params else simulator.line_params[(j, i)])[2],
            1.0,
            1.0,
        ]
        for i, j in edges
    ], dtype=np.float32)
    if not edges:
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_attr = np.empty((0, 5), dtype=np.float32)
    return edge_index, edge_attr


def generate_signature_bank(
    simulator: FaultSimulator,
    event_config: EventConfig,
    n_nodes: int,
    fs: float,
    pre_cycles: float,
    post_cycles: float,
    sample_seed: int,
) -> np.ndarray:
    """对所有母线和无故障候选生成 `[N+1,N,T,6]` 签名库。"""
    signatures = []
    for candidate_idx in range(n_nodes):
        result = simulator.generate_scenario(FaultConfig(
            fault_class=event_config.fault_class,
            fault_bus=candidate_idx,
            z_fault=event_config.z_fault,
            load_multipliers=event_config.load_multipliers,
        ))
        rng = np.random.default_rng(sample_seed + 1009 * candidate_idx)
        signatures.append(build_dynamic_window(
            result["pre_v"], result["post_v"], fs=fs,
            pre_cycles=pre_cycles, post_cycles=post_cycles, rng=rng,
        ))

    simulator._compile_and_solve_base(event_config.load_multipliers)
    normal_pre = simulator._read_voltages()
    rng = np.random.default_rng(sample_seed + 1009 * n_nodes)
    signatures.append(build_dynamic_window(
        normal_pre, normal_pre, fs=fs,
        pre_cycles=pre_cycles, post_cycles=post_cycles, rng=rng,
    ))
    return np.stack(signatures, axis=0).astype(np.float32)


def build_dataset(
    output_dir: Path,
    case_name: str = "ieee13",
    samples_per_bus: int = 1,
    fs: float = 200.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
    res_min: float = 0.1,
    res_max: float = 100.0,
    normal_ratio: float = 1.0,
    seed: int = 42,
    s0_only: bool = True,
) -> dict:
    """生成 S0 数据集并写入数组与元数据文件。"""
    if not s0_only:
        raise NotImplementedError("首轮仅支持 S0；S1/S2 留待后续实验")
    if samples_per_bus < 1:
        raise ValueError("samples_per_bus 必须为正整数")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    simulator = FaultSimulator(case_name)

    warmup = simulator.generate_scenario(
        FaultConfig(fault_class=0, fault_bus=0, z_fault=res_min)
    )
    del warmup
    n_nodes = simulator._n_nodes
    if n_nodes <= 0:
        raise RuntimeError("OpenDSS 未返回任何母线")
    edge_index, edge_attr = _edge_arrays(simulator)
    n_fault = n_nodes * samples_per_bus
    n_normal = max(1, int(round(n_fault * normal_ratio)))
    total = n_fault + n_normal

    x_obs_list, x_full_list, bank_list = [], [], []
    mask_list, edge_mask_list = [], []
    y_detect, y_loc, y_class, y_resist = [], [], [], []
    simulation_calls = 1
    simulation_seconds = 0.0

    for bus in range(n_nodes):
        for _ in range(samples_per_bus):
            event = EventConfig(
                fault_class=int(rng.integers(0, len(FAULT_CLASSES))),
                z_fault=float(rng.uniform(res_min, res_max)),
                load_multipliers={
                    name: float(rng.uniform(0.8, 1.2))
                    for name in (simulator._base_loads or {})
                },
            )
            started = time.perf_counter()
            bank = generate_signature_bank(
                simulator, event, n_nodes, fs, pre_cycles, post_cycles,
                sample_seed=int(rng.integers(0, 2**31 - 1)),
            )
            simulation_seconds += time.perf_counter() - started
            simulation_calls += n_nodes + 1
            full = bank[bus]
            x_obs_list.append(full.copy())
            x_full_list.append(full)
            bank_list.append(bank)
            mask_list.append(build_observation_mask(
                n_nodes, "full", 1.0, rng).astype(np.float32)
            )
            edge_mask_list.append(np.ones(len(edge_index), dtype=np.float32))
            y_detect.append(1)
            y_loc.append(bus)
            y_class.append(event.fault_class)
            y_resist.append(event.z_fault)

    for _ in range(n_normal):
        event = EventConfig(
            fault_class=0,
            z_fault=float(res_min),
            load_multipliers={
                name: float(rng.uniform(0.8, 1.2))
                for name in (simulator._base_loads or {})
            },
        )
        started = time.perf_counter()
        bank = generate_signature_bank(
            simulator, event, n_nodes, fs, pre_cycles, post_cycles,
            sample_seed=int(rng.integers(0, 2**31 - 1)),
        )
        simulation_seconds += time.perf_counter() - started
        simulation_calls += n_nodes + 1
        full = bank[n_nodes]
        x_obs_list.append(full.copy())
        x_full_list.append(full)
        bank_list.append(bank)
        mask_list.append(build_observation_mask(
            n_nodes, "full", 1.0, rng).astype(np.float32)
        )
        edge_mask_list.append(np.ones(len(edge_index), dtype=np.float32))
        y_detect.append(0)
        y_loc.append(-1)
        y_class.append(-1)
        y_resist.append(0.0)

    x_obs = np.stack(x_obs_list).astype(np.float32)
    x_full = np.stack(x_full_list).astype(np.float32)
    signature_bank = np.stack(bank_list).astype(np.float32)
    mask = np.stack(mask_list).astype(np.float32)
    edge_mask = np.stack(edge_mask_list).astype(np.float32)
    y_detect = np.asarray(y_detect, dtype=np.int64)
    y_loc = np.asarray(y_loc, dtype=np.int64)
    y_class = np.asarray(y_class, dtype=np.int64)
    y_resist = np.asarray(y_resist, dtype=np.float32)

    indices = np.arange(total, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)
    n_train = max(1, min(total - 1, int(round(total * 0.8)))) if total > 1 else 1
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    # 只使用训练样本的全候选签名统计量，避免测试集信息泄漏。
    train_values = signature_bank[train_idx].reshape(-1, signature_bank.shape[-1])
    feature_mean = train_values.mean(axis=0).astype(np.float32)
    feature_std = train_values.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

    def standardize(values: np.ndarray) -> np.ndarray:
        """使用训练集统计量对最后一个特征轴做 z-score 标准化。"""
        return ((values - feature_mean) / feature_std).astype(np.float32)

    x_obs = standardize(x_obs)
    x_full = standardize(x_full)
    signature_bank = standardize(signature_bank)

    arrays = {
        "X_obs": x_obs,
        "X_full": x_full,
        "mask": mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "signature_bank": signature_bank,
        "y_detect": y_detect,
        "y_loc": y_loc,
        "y_class": y_class,
        "y_resist": y_resist,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    np.savez(output_dir / "feature_scaler.npz", mean=feature_mean, std=feature_std)

    meta = {
        "case": case_name,
        "scenario": "S0",
        "s0_only": True,
        "n_nodes": int(n_nodes),
        "n_candidates": int(n_nodes + 1),
        "n_edges": int(len(edge_index)),
        "directed_edge_count": int(len(edge_index)),
        "undirected_edge_count": int(len(edge_index) // 2),
        "n_samples": int(total),
        "n_fault": int(n_fault),
        "n_normal": int(n_normal),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "fs": float(fs),
        "pre_cycles": float(pre_cycles),
        "post_cycles": float(post_cycles),
        "window_len": int(x_obs.shape[2]),
        "feature_dim": 6,
        "feature_format": "real_imag_standardized",
        "feature_channels": ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"],
        "feature_scaler": "feature_scaler.npz",
        "res_min": float(res_min),
        "res_max": float(res_max),
        "seed": int(seed),
        "simulation_calls": int(simulation_calls),
        "simulation_seconds": round(float(simulation_seconds), 4),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def load_dataset(output_dir: Path) -> dict:
    """加载 A1 数据数组及元数据。"""
    output_dir = Path(output_dir)
    names = (
        "X_obs", "X_full", "mask", "edge_index", "edge_attr", "edge_mask",
        "signature_bank", "y_detect", "y_loc", "y_class", "y_resist",
        "train_idx", "test_idx",
    )
    result = {name: np.load(output_dir / f"{name}.npy", allow_pickle=False) for name in names}
    result["meta"] = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    return result
