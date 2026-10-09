"""构建 S0 全候选 OpenDSS 动态签名数据集。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .opendss_sim import (
    FAULT_CLASSES,
    FaultConfig,
    FaultSimulator,
    enumerate_fault_specs,
)
from .topology import build_observation_mask
from .waveform import build_dynamic_window
from ..progress import ProgressWriter


@dataclass
class EventConfig:
    """一组候选共享的物理事件参数。"""

    fault_class: int
    z_fault: float
    load_multipliers: dict


def _sample_event(
    rng: np.random.Generator,
    simulator: FaultSimulator,
    res_min: float,
    res_max: float,
    min_feasible_buses: int = 3,
    fault_classes: tuple[int, ...] = (0,),
) -> EventConfig:
    """采样一组供同一反事实族共享的事件参数。

    故障类型按 `fault_classes` 限定。限定原因是母线相别不足时故障命令不会生效：
    两相与三相故障无法在只存在两相或单相的母线上承载，若不加限定，事件的候选
    集合会随故障类型退化，真实故障位置也可能落在物理不可行的母线上。默认只使用
    单相接地故障，使全部母线都能承载该故障类型，候选集合恒为全节点集合。
    `min_feasible_buses` 用于在显式放宽故障类型时，过滤掉可行母线数不足的类型。
    """
    feasible_classes = [
        fault_class
        for fault_class in fault_classes
        if fault_class in FAULT_CLASSES
        and sum(
            any(spec.fault_class == fault_class for spec in enumerate_fault_specs(phases))
            for phases in simulator._bus_phase_nodes
        ) >= min_feasible_buses
    ]
    if not feasible_classes:
        raise RuntimeError(
            f"故障类型 {fault_classes} 在当前拓扑上没有满足最小可行母线数的选项"
        )
    return EventConfig(
        fault_class=int(rng.choice(feasible_classes)),
        z_fault=float(rng.uniform(res_min, res_max)),
        load_multipliers={
            name: float(rng.uniform(0.8, 1.2))
            for name in (simulator._base_loads or {})
        },
    )


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
) -> tuple[np.ndarray, np.ndarray, list]:
    """对所有母线和无故障候选生成 `[N+1,N,T,6]` 签名库并返回候选可获得性。

    返回三元组：签名库、故障候选可获得性掩码 `[N]`（1 表示该母线存在支持当前
    故障类型的相别组合）、每个候选实际使用的故障相别。相别不足的候选其响应取
    无故障基态，并依据掩码在训练与评价中被排除。
    """
    signatures = []
    feasible = np.zeros(n_nodes, dtype=np.float32)
    phase_usage = []
    for candidate_idx in range(n_nodes):
        result = simulator.generate_scenario(FaultConfig(
            fault_class=event_config.fault_class,
            fault_bus=candidate_idx,
            z_fault=event_config.z_fault,
            load_multipliers=event_config.load_multipliers,
        ))
        feasible[candidate_idx] = 1.0 if result["fault_feasible"] else 0.0
        phase_usage.append(result["fault_phases"])
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
    phase_usage.append(None)
    return np.stack(signatures, axis=0).astype(np.float32), feasible, phase_usage


def group_hardest_pairs(
    signature_bank: np.ndarray,
    observation: np.ndarray,
    scale: np.ndarray,
) -> list[tuple[int, int]]:
    """计算一个反事实族内每个事实位置的（真实位置, 最难错误候选）候选对。

    signature_bank：`[N+1,N,T,6]` 签名库，前 `N` 项为故障候选；
    observation：`[N,T,6]` 事实观测；
    scale：逐通道尺度 `[F]`。返回去重后的有序候选对列表，用于困难候选定向采样。
    """
    n_fault = signature_bank.shape[0] - 1
    pairs = set()
    for true_location in range(n_fault):
        normalized = (
            signature_bank[:n_fault] - observation[None, :, :, :]
        ) / scale.reshape(1, 1, 1, -1)
        scores = (normalized.astype(np.float64) ** 2).mean(axis=(1, 2, 3))
        order = np.argsort(scores, kind="stable")
        hardest = int(order[0]) if int(order[0]) != true_location else int(order[1])
        pairs.add(tuple(sorted((true_location, hardest))))
    return sorted(pairs)


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
    val_ratio: float | None = None,
    test_ratio: float = 0.15,
    fault_classes: tuple[int, ...] = (0,),
    hard_pairs: set[tuple[int, int]] | None = None,
    hard_pair_max_attempts: int = 12,
    progress: ProgressWriter | None = None,
) -> dict:
    """生成 S0 数据集并写入数组与元数据文件。

    `val_ratio` 为 None 时按训练/测试两段划分；给出时按训练/验证/测试三段划分，
    其中验证集只包含故障样本。`fault_classes` 限定事件可用的故障类型，默认只用
    单相接地故障，以保证全部母线都能承载该故障类型。`hard_pairs` 给出时进入主动
    困难候选采样：每个事件族重新采样负荷、阻抗与事件随机状态，直到该族包含给定
    困难候选对之一，或达到 `hard_pair_max_attempts` 次上限。
    """
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
    if progress is not None:
        progress.stage_start("data_generation", total)
    completed = 0

    x_obs_list, x_full_list, bank_list = [], [], []
    mask_list, edge_mask_list = [], []
    candidate_mask_list, phase_usage_list = [], []
    y_detect, y_loc, y_class, y_resist = [], [], [], []
    simulation_calls = 1
    simulation_seconds = 0.0

    group_ids = []
    rejected_groups = 0
    for group_id in range(samples_per_bus):
        # 同一个反事实族只生成一次签名库：族内共享事件参数与窗口生成实现，
        # 每一条事实观测都取自该族签名库中与其真实位置对应的那一项
        started = time.perf_counter()
        for attempt in range(hard_pair_max_attempts if hard_pairs else 1):
            event = _sample_event(
                rng, simulator, res_min, res_max, fault_classes=fault_classes
            )
            candidate_seed = int(rng.integers(0, 2**31 - 1))
            bank, feasible, phase_usage = generate_signature_bank(
                simulator, event, n_nodes, fs, pre_cycles, post_cycles,
                sample_seed=candidate_seed,
            )
            if not hard_pairs:
                break
            observed = group_hardest_pairs(
                bank[:n_nodes + 1], bank[0], np.ones(bank.shape[-1])
            )
            if any(pair in hard_pairs for pair in observed):
                break
            rejected_groups += 1
        else:
            attempt = hard_pair_max_attempts - 1
        simulation_seconds += time.perf_counter() - started
        simulation_calls += 1
        group_candidate_mask = np.concatenate([feasible, np.ones(1, dtype=np.float32)])
        group_phase_usage = phase_usage
        for bus in range(n_nodes):
            full = bank[bus]
            x_obs_list.append(full.copy())
            x_full_list.append(full)
            bank_list.append(bank)
            candidate_mask_list.append(group_candidate_mask.copy())
            phase_usage_list.append(group_phase_usage)
            mask_list.append(build_observation_mask(
                n_nodes, "full", 1.0, rng).astype(np.float32)
            )
            edge_mask_list.append(np.ones(len(edge_index), dtype=np.float32))
            y_detect.append(1)
            y_loc.append(bus)
            y_class.append(event.fault_class)
            y_resist.append(event.z_fault)
            group_ids.append(group_id)
            completed += 1
            if progress is not None:
                progress.update(
                    "data_generation", "scenario", completed, total,
                    note=f"fault group={group_id} bus={bus}",
                    unit_seconds=(time.perf_counter() - started) / n_nodes, force=True,
                )

    for _ in range(n_normal):
        event = _sample_event(
            rng, simulator, res_min, res_max, fault_classes=fault_classes
        )
        started = time.perf_counter()
        bank, feasible, phase_usage = generate_signature_bank(
            simulator, event, n_nodes, fs, pre_cycles, post_cycles,
            sample_seed=int(rng.integers(0, 2**31 - 1)),
        )
        simulation_seconds += time.perf_counter() - started
        simulation_calls += 1
        full = bank[n_nodes]
        x_obs_list.append(full.copy())
        x_full_list.append(full)
        bank_list.append(bank)
        candidate_mask_list.append(
            np.concatenate([feasible, np.ones(1, dtype=np.float32)])
        )
        phase_usage_list.append(phase_usage)
        mask_list.append(build_observation_mask(
            n_nodes, "full", 1.0, rng).astype(np.float32)
        )
        edge_mask_list.append(np.ones(len(edge_index), dtype=np.float32))
        y_detect.append(0)
        y_loc.append(-1)
        y_class.append(-1)
        y_resist.append(0.0)
        group_ids.append(-1)
        completed += 1
        if progress is not None:
            progress.update(
                "data_generation", "scenario", completed, total,
                note="normal candidate", unit_seconds=time.perf_counter() - started, force=True,
            )

    x_obs = np.stack(x_obs_list).astype(np.float32)
    x_full = np.stack(x_full_list).astype(np.float32)
    signature_bank = np.stack(bank_list).astype(np.float32)
    mask = np.stack(mask_list).astype(np.float32)
    edge_mask = np.stack(edge_mask_list).astype(np.float32)
    candidate_mask = np.stack(candidate_mask_list).astype(np.float32)
    y_detect = np.asarray(y_detect, dtype=np.int64)
    y_loc = np.asarray(y_loc, dtype=np.int64)
    y_class = np.asarray(y_class, dtype=np.int64)
    y_resist = np.asarray(y_resist, dtype=np.float32)
    group_ids = np.asarray(group_ids, dtype=np.int64)

    # 按反事实族划分，避免同一 group_id 的不同事实位置跨越数据集合。
    # 给出 val_ratio 时按训练/验证/测试三段划分，未给出时保持训练/测试两段划分。
    split_rng = np.random.default_rng(seed)
    fault_groups = np.arange(samples_per_bus, dtype=np.int64)
    split_rng.shuffle(fault_groups)
    val_idx = np.empty(0, dtype=np.int64)
    if val_ratio is None:
        if len(fault_groups) > 1:
            n_train_groups = max(1, min(len(fault_groups) - 1, int(round(len(fault_groups) * 0.8))))
        else:
            n_train_groups = len(fault_groups)
        train_groups = set(fault_groups[:n_train_groups].tolist())
        val_groups: set[int] = set()
    else:
        # 先按比例确定验证集与测试集分组数，剩余分组全部作为训练集
        n_val_groups = max(1, int(round(len(fault_groups) * float(val_ratio))))
        n_test_groups = max(1, int(round(len(fault_groups) * float(test_ratio))))
        n_train_groups = len(fault_groups) - n_val_groups - n_test_groups
        if n_train_groups < 1:
            raise ValueError("分组数不足，无法同时给出训练、验证与测试集合")
        train_groups = set(fault_groups[:n_train_groups].tolist())
        val_groups = set(
            fault_groups[n_train_groups:n_train_groups + n_val_groups].tolist()
        )
        test_groups = set(fault_groups[n_train_groups + n_val_groups:].tolist())
    train_mask = np.asarray(
        [(group_id in train_groups) or group_id == -1 for group_id in group_ids],
        dtype=bool,
    )
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    if val_ratio is None:
        test_idx = np.flatnonzero(~train_mask).astype(np.int64)
    else:
        # 验证集只含故障样本，正常样本全部划入训练集参与响应族重构
        val_mask = np.asarray(
            [(group_id in val_groups) for group_id in group_ids], dtype=bool
        )
        test_mask = np.asarray(
            [
                group_id >= 0 and group_id not in train_groups and group_id not in val_groups
                for group_id in group_ids
            ],
            dtype=bool,
        )
        val_idx = np.flatnonzero(val_mask).astype(np.int64)
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
    if test_idx.size == 0 and val_ratio is None and total > 1:
        normal_indices = np.flatnonzero(group_ids == -1).astype(np.int64)
        if normal_indices.size > 0:
            test_idx = np.asarray([normal_indices[-1]], dtype=np.int64)
            train_idx = train_idx[train_idx != normal_indices[-1]]

    # 只使用训练样本的有效候选签名统计量，避免测试集信息泄漏，
    # 同时排除相别不可行候选（其响应等于无故障基态）对尺度统计的污染。
    train_rows = signature_bank[train_idx]
    train_valid = candidate_mask[train_idx].astype(bool)
    train_values = train_rows[train_valid].reshape(-1, signature_bank.shape[-1])
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
        "candidate_mask": candidate_mask,
        "y_detect": y_detect,
        "y_loc": y_loc,
        "y_class": y_class,
        "y_resist": y_resist,
        "group_id": group_ids,
        "train_idx": train_idx,
        "val_idx": val_idx,
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
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "split_groups": {
            "train": sorted(int(value) for value in train_groups),
            "val": sorted(int(value) for value in val_groups),
            "test": sorted(
                int(value)
                for value in set(group_ids.tolist()) - train_groups - val_groups
                if value >= 0
            ),
        },
        "fault_classes_used": [int(value) for value in fault_classes],
        "hard_pair_sampling": {
            "enabled": bool(hard_pairs),
            "n_hard_pairs": len(hard_pairs) if hard_pairs else 0,
            "max_attempts_per_group": int(hard_pair_max_attempts),
            "rejected_groups": int(rejected_groups),
        },
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
        "counterfactual_perturbation": "disabled",
        "group_id_semantics": "同一非负 group_id 下共享事件参数和无扰动窗口生成",
        "fault_command": "按母线实际相别构造带明确相别的故障命令",
        "candidate_mask_semantics": "1 表示该候选母线存在支持本事件故障类型的相别组合",
        "candidate_mask_mean_fault_rows": round(float(candidate_mask[y_loc >= 0].mean()), 4),
        "phase_usage_first_event": [
            None if phases is None else list(phases)
            for phases in phase_usage_list[0]
        ],
        "bus_phase_nodes": [
            list(phases) for phases in simulator._bus_phase_nodes
        ],
        "simulation_calls": int(simulation_calls),
        "simulation_seconds": round(float(simulation_seconds), 4),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if progress is not None:
        progress.stage_done("data_generation", total)
    return meta


def load_dataset(output_dir: Path) -> dict:
    """加载 A1 数据数组及元数据。"""
    output_dir = Path(output_dir)
    names = (
        "X_obs", "X_full", "mask", "edge_index", "edge_attr", "edge_mask",
        "signature_bank", "candidate_mask", "y_detect", "y_loc", "y_class",
        "y_resist", "group_id", "train_idx", "val_idx", "test_idx",
    )
    result = {}
    for name in names:
        path = output_dir / f"{name}.npy"
        if name == "group_id" and not path.exists():
            result[name] = np.full(result["y_loc"].shape, -1, dtype=np.int64)
        elif name == "candidate_mask" and not path.exists():
            # 旧数据没有候选可获得性字段，按“全部候选有效”的兼容口径补齐
            result[name] = np.ones(
                (result["signature_bank"].shape[0], result["signature_bank"].shape[1]),
                dtype=np.float32,
            )
        elif name == "val_idx" and not path.exists():
            # 旧数据只有训练/测试两段划分，验证集置空，由训练脚本按分组重新划分
            result[name] = np.empty(0, dtype=np.int64)
        else:
            result[name] = np.load(path, allow_pickle=False)
    result["meta"] = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    return result
