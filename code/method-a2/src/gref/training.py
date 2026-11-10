"""GReF 的分组训练、统一评价、门控审计与可复算产物写出。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..progress import ProgressWriter
from ..training.evaluation import rank_metrics, split_by_group
from .data import (
    apply_edge_drop,
    build_undirected_edge_groups,
    fit_normalization,
    load_arrays,
    make_batch,
    sample_node_masks,
)
from .losses import (
    gref_losses,
    region_term_losses,
    response_energy,
    teacher_gate,
    three_term_losses,
)
from .model import GrefPredictor
from .region import graph_distance_hops, region_kernel

METHOD_DIR = Path(__file__).resolve().parents[2]

# 两种多分项损失模式各自记录的分项名；gref 模式仍使用门控交叉熵口径
TERM_KEYS = {"three-term": ("energy", "rank", "smooth"), "region": ("energy", "rank", "region")}


def _write_json(path: Path, value) -> None:
    """写出严格 JSON，禁止以非标准无穷值掩盖数值异常。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _forward(model, batch):
    """只将部署可见字段传入模型，监督字段留在训练函数中。"""
    return model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                 edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                 candidate_mask=batch["candidate_mask"])


def _losses(output, batch, args):
    """统一训练、验证和测试的能量与目标定义。"""
    loss_mode = getattr(args, "loss_mode", "gref")
    if loss_mode == "three-term":
        return three_term_losses(
            output, batch["observation"], batch["target"],
            batch["candidate_mask"], batch["location"], batch["edge_index"],
            rank_temperature=args.rank_temperature,
            lambda_energy=args.lambda_energy, lambda_rank=args.lambda_rank,
            lambda_smooth=args.lambda_smooth, mask=batch["node_mask"],
        )
    if loss_mode == "region":
        return region_term_losses(
            output, batch["observation"], batch["target"],
            batch["candidate_mask"], batch["location"], args._region_kernel,
            rank_temperature=args.rank_temperature,
            lambda_energy=args.lambda_energy, lambda_rank=args.lambda_rank,
            lambda_region=args.lambda_region, beta_obs=args.beta_obs,
            beta_miss=args.beta_miss, mask=batch["node_mask"],
        )
    return gref_losses(output, batch["observation"], batch["target"],
                       batch["candidate_mask"], batch["location"],
                       tolerance=args.energy_tolerance, temperature=args.temperature,
                       lambda_family=args.lambda_family, mask=batch["node_mask"])


@torch.no_grad()
def calibrate_rank_temperature(data, indices, normalization, batch_size, device, progress=None):
    """在训练集上按真实能量两两绝对差的中位数标定列表排序温度。

    逐事件取全部有效故障候选两两真实能量差的中位数，再对全部故障事件取中位数；
    该规则只使用训练集，不访问验证集或测试集。
    """
    per_event = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = make_batch(data, chunk, normalization, torch.device(device))
        true_energy = response_energy(batch["observation"], batch["target"], batch["node_mask"])
        fault = batch["candidate_mask"].bool().clone()
        fault[:, -1] = False
        for row in torch.where(batch["location"] >= 0)[0].tolist():
            values = true_energy[row][fault[row]]
            if values.numel() < 2:
                continue
            gaps = (values[:, None] - values[None, :]).abs()
            upper = torch.triu_indices(values.numel(), values.numel(), offset=1)
            per_event.append(float(gaps[upper[0], upper[1]].median()))
        if progress is not None:
            progress.update("rank_calibration", "rows", min(start + batch_size, len(indices)), len(indices))
    if not per_event:
        raise ValueError("无法从训练集标定排序温度：没有可用的故障候选对")
    values = np.asarray(per_event, dtype=np.float64)
    return {
        "temperature": max(float(np.median(values)), 1e-12),
        "n_events": int(len(per_event)),
        "quantiles": np.quantile(values, [0, .05, .5, .95, 1]).tolist(),
        "source": "训练集逐事件真实能量两两绝对差的中位数的中位数",
    }


def _strata(correct, values, metric="top1") -> list[dict]:
    """按离散节点、事件族或门控状态汇总全部故障事件。"""
    return [{"value": int(value), "n_events": int((values == value).sum()),
             metric: float(correct[values == value].mean())} for value in np.unique(values)]


@torch.no_grad()
def evaluate(model, data, indices, normalization, args, progress=None, stage="validation", capture=False):
    """始终采用评价模式，以全局分母汇总损失，完整集合不受门控筛选。"""
    model.eval()
    term_keys = TERM_KEYS.get(getattr(args, "loss_mode", "gref"))
    term_mode = term_keys is not None
    parts = {key: [] for key in ("scores", "oracle", "weights", "margins", "location", "mask")}
    payload = {key: [] for key in ("observation", "response", "target", "node_mask")}
    loc_num = weight_sum = family_num = family_den = 0.0
    term_numerator = {name: 0.0 for name in (term_keys or ())}
    term_denominator = {name: 0.0 for name in (term_keys or ())}
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start:start + args.batch_size]
        batch = make_batch(data, chunk, normalization, torch.device(args.device))
        output = _forward(model, batch)
        losses = _losses(output, batch, args)
        if term_mode:
            for name in term_keys:
                term_numerator[name] += float(losses[f"{name}_numerator"])
                term_denominator[name] += float(losses[f"{name}_denominator"])
            fault_mask = batch["candidate_mask"].bool() & (
                output["candidate_types"].squeeze(-1) == 0
            )
            weight, margins = teacher_gate(
                losses["teacher_energy"], batch["location"], fault_mask, args.energy_tolerance
            )
            oracle = losses["teacher_energy"]
        else:
            denominator = float((batch["node_mask"].sum(1) * batch["candidate_mask"].sum(1)).sum())
            denominator *= batch["observation"].shape[2] * batch["observation"].shape[3]
            loc_num += float(losses["loc_numerator"])
            weight_sum += float(losses["weight_sum"])
            family_num += float(losses["family"]) * denominator
            family_den += denominator
            weight, margins, oracle = losses["gate_weights"], losses["teacher_margins"], losses["teacher_energy"]
        items = {"scores": output["candidate_score"], "oracle": oracle,
                 "weights": weight, "margins": margins,
                 "location": batch["location"], "mask": batch["candidate_mask"]}
        for key, value in items.items():
            parts[key].append(value.cpu().numpy())
        if capture:
            for key in payload:
                value = output["response"] if key == "response" else batch[key]
                payload[key].append(value.cpu().numpy())
        if progress is not None:
            progress.update(stage, "rows", min(start + args.batch_size, len(indices)), len(indices))
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    n_nodes = data["X_obs"].shape[1]
    fault = arrays["location"] >= 0
    labels = arrays["location"][fault]
    valid = arrays["mask"][fault, :n_nodes].astype(bool)
    score = np.where(valid, arrays["scores"][fault, :n_nodes], np.inf)
    oracle = np.where(valid, arrays["oracle"][fault, :n_nodes], np.inf)
    correct = score.argmin(1) == labels
    active = arrays["weights"][fault] > 0
    if term_mode:
        values = {
            name: (term_numerator[name] / term_denominator[name]
                   if term_denominator[name] > 0 else 0.0)
            for name in term_keys
        }
        weights = {"energy": args.lambda_energy, "rank": args.lambda_rank,
                   "smooth": getattr(args, "lambda_smooth", 0.0),
                   "region": getattr(args, "lambda_region", 0.0)}
        loss_report = dict(values)
        loss_report["total"] = sum(weights[name] * values[name] for name in term_keys)
    else:
        family = family_num / max(family_den, 1.0)
        loc = loc_num / (weight_sum + 1e-12)
        loss_report = {"loc": loc, "family": family, "total": loc + args.lambda_family * family}
    weight_sum = float(arrays["weights"].sum())
    metrics = {
        "n_events": int(len(indices)), "n_fault_events": int(fault.sum()),
        "losses": loss_report,
        "score": rank_metrics(score, labels), "oracle": rank_metrics(oracle, labels),
        "gate_coverage": float(active.mean()), "gate_weight_sum": weight_sum,
        "by_node": _strata(correct, labels),
        "by_group": _strata(correct, np.asarray(data["group_id"][indices])[fault]),
        "by_gate": _strata(correct, active.astype(int)),
        "resistance_events": [
            {"group_id": int(group), "resistance": float(data["y_resist"][indices][fault][rows][0]),
             "n_events": int(rows.sum()), "top1": float(correct[rows].mean())}
            for group in np.unique(data["group_id"][indices][fault])
            for rows in [data["group_id"][indices][fault] == group]
        ],
    }
    if capture:
        arrays.update({key: np.concatenate(values) for key, values in payload.items()})
    return metrics, arrays


@torch.no_grad()
def audit_training(data, indices, normalization, args, progress):
    """在训练侧核验事实配对并记录按节点、电阻和事件族的门控覆盖。"""
    records, positive_margins = [], []
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start:start + args.batch_size]
        batch = make_batch(data, chunk, normalization, torch.device("cpu"))
        teacher = response_energy(batch["observation"], batch["target"], batch["node_mask"])
        fault_mask = batch["candidate_mask"].clone()
        fault_mask[:, -1] = 0
        weights, margins = teacher_gate(teacher, batch["location"], fault_mask, args.energy_tolerance)
        for row, index in enumerate(chunk):
            label = int(batch["location"][row])
            if label < 0:
                continue
            if not torch.equal(batch["observation"][row], batch["target"][row, label]):
                raise ValueError("首版要求事实观测逐元素等于真实候选响应")
            margin = float(margins[row])
            if margin > 0:
                positive_margins.append(margin)
            records.append({"index": int(index), "group_id": int(data["group_id"][index]),
                            "location": label, "resistance": float(data["y_resist"][index]),
                            "margin": margin, "weight": float(weights[row])})
        progress.update("audit", "rows", min(start + args.batch_size, len(indices)), len(indices))
    active = np.array([record["weight"] > 0 for record in records])
    if not active.any():
        raise ValueError("全部训练故障事件均被门控关闭；请核验容差来源及能量尺度")
    return {"scope": "最终训练集", "tolerance": args.energy_tolerance,
            "tolerance_source": args.tolerance_source, "coverage": float(active.mean()),
            "by_node": _strata(active, np.array([r["location"] for r in records]), "coverage"),
            "by_group": _strata(active, np.array([r["group_id"] for r in records]), "coverage"),
            "positive_margin_quantiles": np.quantile(positive_margins, [0, .05, .5, .95, 1]).tolist(),
            "events": records}


@torch.no_grad()
def shortcut_audit(model, data, indices, normalization, args):
    """实际执行候选置换与响应屏蔽，记录结构核验和屏蔽后的完整测试指标。"""
    chunk = indices[:min(args.batch_size, len(indices))]
    batch = make_batch(data, chunk, normalization, torch.device(args.device))
    model.eval()
    original = _forward(model, batch)
    permutation = torch.arange(original["response"].shape[1] - 1, -1, -1, device=args.device)
    permuted = model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                     edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                     candidate_markers=original["candidate_markers"][:, permutation],
                     candidate_types=original["candidate_types"][:, permutation],
                     candidate_mask=batch["candidate_mask"][:, permutation])
    expected = original["candidate_score"][:, permutation]
    difference = float((permuted["candidate_score"] - expected).abs().max())
    follows = bool(torch.allclose(permuted["candidate_score"], expected, atol=1e-6, rtol=1e-5))
    if not follows:
        raise RuntimeError("候选置换评分不满足等变性")
    return {"candidate_permutation_passed": follows, "permutation_max_abs_error": difference,
            "permutation_n_events": len(chunk),
            "free_scoring_parameters": [name for name, _ in model.named_parameters() if "score_head" in name],
            "response_ablation_scope": "完整测试集，候选均值响应，严格并列按索引稳定排序",
            "limitation": "结构检查不能替代物理重构质量、事实屏蔽及多种子机制实验"}


def run(args: argparse.Namespace) -> Path:
    """执行独立训练并将全部交付物写入新的 GReF 输出目录。"""
    for name in ("energy_tolerance", "temperature", "lambda_family", "learning_rate"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为正有限数")
    three_term = args.loss_mode == "three-term"
    region_mode = args.loss_mode == "region"
    if three_term:
        for name in ("lambda_energy", "lambda_rank", "lambda_smooth"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        if args.rank_temperature is not None and (
            not math.isfinite(args.rank_temperature) or args.rank_temperature <= 0
        ):
            raise ValueError("显式给出的排序温度必须为正有限数")
    if region_mode:
        for name in ("lambda_energy", "lambda_rank", "lambda_region"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        if args.rank_temperature is None or not math.isfinite(args.rank_temperature) \
                or args.rank_temperature <= 0:
            raise ValueError("RaReF 区域概率损失必须显式给出正有限排序温度")
        if not math.isfinite(args.region_sigma) or args.region_sigma <= 0:
            raise ValueError("region_sigma 必须为正有限数")
        for name in ("beta_obs", "beta_miss"):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if not 0.0 <= args.node_mask_full_probability <= 1.0:
            raise ValueError("node_mask_full_probability 必须位于 [0,1]")
        if not 0.0 <= args.node_mask_force_observed_probability <= 1.0:
            raise ValueError("node_mask_force_observed_probability 必须位于 [0,1]")
    if not args.tolerance_source.strip() or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("必须提供容差来源及正的轮数、批大小")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("验证、测试比例必须为正且总和小于一")
    edge_drop_prob = float(getattr(args, "edge_drop_prob", 0.0))
    edge_drop_max = int(getattr(args, "edge_drop_max", 3))
    if not 0.0 <= edge_drop_prob <= 1.0:
        raise ValueError("edge_drop_prob 必须位于 [0, 1]")
    if edge_drop_max < 1:
        raise ValueError("edge_drop_max 必须为正整数")
    output_dir = Path(args.output_dir).resolve()
    output_root = (METHOD_DIR / "output" / "gref").resolve()
    if output_root not in output_dir.parents:
        raise ValueError("输出必须位于本方法 output/gref/<run-id>/ 下")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_arrays(Path(args.data_dir))
    n_fault = int(data["X_obs"].shape[1])
    region_distance = region_kernel_values = None
    if region_mode:
        if int(data["signature_bank"].shape[1]) != n_fault + 1:
            raise ValueError("候选轴必须为故障候选加末尾正常候选")
        if int(np.asarray(data["edge_index"]).max()) >= n_fault:
            raise ValueError("边端点超出故障候选轴，候选身份与拓扑节点不一致")
        # 小规模合成数据允许把观测节点区间收窄到节点总数以内，实际取值写入配置
        args.node_mask_min_observed = min(int(args.node_mask_min_observed), n_fault - 1)
        args.node_mask_max_observed = min(int(args.node_mask_max_observed), n_fault)
        if args.node_mask_min_observed < 1:
            raise ValueError("观测节点数下限必须为正整数")
        if args.node_mask_min_observed > args.node_mask_max_observed:
            raise ValueError("观测节点数区间无效：min 大于 max")
        region_distance = graph_distance_hops(np.asarray(data["edge_index"]), n_fault)
        region_kernel_values = region_kernel(region_distance, args.region_sigma)
        args._region_kernel = torch.as_tensor(region_kernel_values, dtype=torch.float32,
                                              device=args.device)
    edge_groups = build_undirected_edge_groups(np.asarray(data["edge_index"]))
    split = split_by_group(data["group_id"], data["y_loc"], args.seed, args.val_ratio, args.test_ratio)
    run_id = output_dir.name
    progress_path = METHOD_DIR / "logs" / run_id / "progress.jsonl"
    if progress_path.exists():
        raise FileExistsError("运行日志已存在，请使用新的运行标识")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(progress_path, run_id, terminal=not args.no_terminal_progress)
    try:
        source_root = Path(__file__).resolve().parents[2]
        config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
        loss_definitions = {
            "gref": "门控能量交叉熵 + 全候选 Smooth L1 重构",
            "three-term": "三项损失：逐元素掩码均方响应重构能量 + 部署能量软列表 KL + 误差场图拉普拉斯平滑",
            "region": "RaReF 区域概率损失：逐元素掩码均方响应重构能量 + 部署能量软列表 KL + 拓扑区域概率交叉熵（替换图平滑）",
        }
        loss_definition = loss_definitions[args.loss_mode]
        loss_weights = (
            {"energy": args.lambda_energy, "rank": args.lambda_rank,
             "smooth": args.lambda_smooth}
            if three_term else
            {"energy": args.lambda_energy, "rank": args.lambda_rank,
             "region": args.lambda_region}
            if region_mode else {"loc": 1.0, "family": args.lambda_family}
        )
        config.update({"method": "GReF", "dataset_meta": data["meta"],
                       "loss_mode": args.loss_mode, "loss_definition": loss_definition,
                       "loss_weights": loss_weights,
                       "rank_temperature": args.rank_temperature,
                       "rank_temperature_source": getattr(args, "rank_temperature_source", None),
                       "edge_drop": {
                           "enabled": edge_drop_prob > 0.0,
                           "probability": edge_drop_prob,
                           "max_edges_per_sample": edge_drop_max,
                           "undirected_edge_groups": len(edge_groups),
                           "seed_offset": 1,
                       },
                       "score_definition": "最终训练侧标准化坐标中的掩码均方响应距离",
                       "normalization_definition": "仅最终训练有效候选，逐节点通道，现有输入坐标的仿射重拟合",
                       "software": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__)},
                       "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [*Path(__file__).parent.glob("*.py"),
                                                   source_root / "src/model/marker_response_predictor.py",
                                                   source_root / "src/model/losses.py", source_root / "src/training/evaluation.py",
                                                   source_root / "src/progress.py", source_root / "src/model/gnn.py",
                                                   source_root / "src/model/temporal.py", source_root / "scripts/train_gref.py"]}})
        if region_mode:
            config["region_kernel"] = {
                "distance_metric": "候选拓扑无向跳数",
                "sigma": args.region_sigma,
                "distance_diameter": int(region_distance.max()),
                "kernel_sha256": hashlib.sha256(region_kernel_values.tobytes()).hexdigest(),
                "candidate_axis": "故障候选索引与拓扑节点索引一一对应，末尾正常候选不参与区域计算",
                "beta_obs": args.beta_obs, "beta_miss": args.beta_miss,
                "temperature": args.rank_temperature,
                "temperature_source": "沿用排序温度，首轮不额外调节",
            }
            config["node_mask_augmentation"] = {
                "full_probability": args.node_mask_full_probability,
                "min_observed": args.node_mask_min_observed,
                "max_observed": args.node_mask_max_observed,
                "force_observed_probability": args.node_mask_force_observed_probability,
                "seed_offset": 2,
                "coverage": ["完整观测", "真实故障节点被观测", "真实故障节点未被观测", "随机部分节点观测"],
            }
        _write_json(output_dir / "config.json", config)
        progress.stage_start("normalization", len(split["train_idx"]))
        normalization = fit_normalization(data, split["train_idx"], progress)
        np.savez_compressed(output_dir / "normalization.npz", **normalization)
        np.savez_compressed(output_dir / "split.npz", **split)
        if region_mode:
            np.savez_compressed(output_dir / "region_kernel.npz",
                                kernel=region_kernel_values.astype(np.float32),
                                distance=region_distance.astype(np.int64))
        progress.stage_done("normalization", len(split["train_idx"]))
        rank_calibration = None
        if three_term and args.rank_temperature is None:
            progress.stage_start("rank_calibration", len(split["train_idx"]))
            rank_calibration = calibrate_rank_temperature(
                data, split["train_idx"], normalization, args.batch_size, args.device, progress
            )
            args.rank_temperature = rank_calibration["temperature"]
            args.rank_temperature_source = rank_calibration["source"]
            progress.stage_done("rank_calibration", len(split["train_idx"]))
            config.update({"rank_temperature": args.rank_temperature,
                           "rank_temperature_source": args.rank_temperature_source,
                           "rank_calibration": rank_calibration})
            _write_json(output_dir / "config.json", config)
        elif three_term or region_mode:
            args.rank_temperature_source = "命令行显式给出"
            config.update({"rank_temperature_source": args.rank_temperature_source})
            _write_json(output_dir / "config.json", config)
        progress.stage_start("audit", len(split["train_idx"]))
        gate = audit_training(data, split["train_idx"], normalization, args, progress)
        _write_json(output_dir / "gate_audit.json", gate)
        progress.stage_done("audit", len(split["train_idx"]))
        model_config = {"n_nodes": data["X_obs"].shape[1], "time_steps": data["X_obs"].shape[2],
                        "feature_dim": data["X_obs"].shape[3]}
        model = GrefPredictor(**model_config).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        rng = np.random.default_rng(args.seed)
        edge_drop_rng = np.random.default_rng(args.seed + 1)
        mask_rng = np.random.default_rng(args.seed + 2)
        term_keys = TERM_KEYS.get(args.loss_mode, ())
        term_mode = bool(term_keys)
        history, best_key, best_epoch = [], None, None
        save_epochs = {
            int(item) for item in str(getattr(args, "save_epochs", "")).split(",") if item.strip()
        }
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(split["train_idx"])
            batch_count = math.ceil(len(order) / args.batch_size)
            progress.stage_start("training", batch_count)
            total_sum = 0.0
            term_sum = {name: 0.0 for name in term_keys}
            mask_counts = {"full": 0, "partial_observed": 0, "partial_missing": 0,
                           "partial_normal": 0}
            for start in range(0, len(order), args.batch_size):
                started = time.perf_counter()
                batch = make_batch(data, order[start:start + args.batch_size], normalization, torch.device(args.device))
                if edge_drop_prob > 0.0:
                    apply_edge_drop(
                        batch, edge_groups, edge_drop_prob, edge_drop_max, edge_drop_rng
                    )
                if region_mode:
                    locations = batch["location"].detach().cpu().numpy()
                    masks = sample_node_masks(
                        locations, n_fault, mask_rng,
                        full_probability=args.node_mask_full_probability,
                        min_observed=args.node_mask_min_observed,
                        max_observed=args.node_mask_max_observed,
                        force_observed_probability=args.node_mask_force_observed_probability,
                    )
                    batch["node_mask"] = torch.as_tensor(
                        masks, dtype=torch.float32, device=args.device)
                    rows = np.arange(locations.size)
                    observed = masks[rows, np.clip(locations, 0, n_fault - 1)] > 0
                    full = masks.sum(axis=1) == n_fault
                    fault_rows = locations >= 0
                    mask_counts["full"] += int(full.sum())
                    mask_counts["partial_observed"] += int((fault_rows & observed & ~full).sum())
                    mask_counts["partial_missing"] += int((fault_rows & ~observed).sum())
                    mask_counts["partial_normal"] += int((~fault_rows & ~full).sum())
                optimizer.zero_grad(set_to_none=True)
                losses = _losses(_forward(model, batch), batch, args)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("训练损失出现非有限值")
                losses["total"].backward()
                optimizer.step()
                total_sum += float(losses["total"].detach()) * len(batch["location"])
                if term_mode:
                    for name in term_keys:
                        term_sum[name] += float(losses[name].detach()) * len(batch["location"])
                progress.update("training", "batch", start // args.batch_size + 1, batch_count,
                                note=f"epoch={epoch}/{args.epochs}", unit_seconds=time.perf_counter() - started)
            progress.stage_done("training", batch_count)
            progress.stage_start("validation", len(split["val_idx"]))
            validation, _ = evaluate(model, data, split["val_idx"], normalization, args, progress)
            progress.stage_done("validation", len(split["val_idx"]))
            if term_mode:
                record = {
                    "epoch": epoch, "train_batch_total_mean": total_sum / len(order),
                    **{f"train_{name}_mean": term_sum[name] / len(order) for name in term_keys},
                    "val_top1": validation["score"]["top1"], "val_top3": validation["score"]["top3"],
                    **{f"val_{name}": validation["losses"][name] for name in term_keys},
                    "val_total": validation["losses"]["total"],
                    "val_gate_coverage": validation["gate_coverage"],
                }
                if region_mode:
                    record.update({
                        "train_n_full": mask_counts["full"],
                        "train_n_partial_observed": mask_counts["partial_observed"],
                        "train_n_partial_missing": mask_counts["partial_missing"],
                        "train_n_partial_normal": mask_counts["partial_normal"],
                    })
                key = (record["val_top1"], -record["val_energy"])
            else:
                record = {"epoch": epoch, "train_batch_total_mean": total_sum / len(order),
                          "val_top1": validation["score"]["top1"], "val_top3": validation["score"]["top3"],
                          "val_family": validation["losses"]["family"], "val_loc": validation["losses"]["loc"],
                          "val_gate_coverage": validation["gate_coverage"]}
                key = (record["val_top1"], -record["val_family"])
            history.append(record)
            with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(history)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                            "best_epoch": epoch, "loss_mode": args.loss_mode,
                            "rank_temperature": args.rank_temperature}, output_dir / "model.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch}, output_dir / "optimizer.pt")
            if epoch in save_epochs:
                torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                            "epoch": epoch, "loss_mode": args.loss_mode,
                            "rank_temperature": args.rank_temperature},
                           output_dir / f"checkpoint-epoch{epoch:03d}.pt")
        checkpoint = torch.load(output_dir / "model.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation, _ = evaluate(model, data, split["val_idx"], normalization, args)
        dataset_audit = {
            "split_groups": {key: split[key].tolist() for key in ("train_groups", "val_groups", "test_groups")},
            "n_train": len(split["train_idx"]), "n_val": len(split["val_idx"]), "n_test": len(split["test_idx"]),
            "fact_matches_true_response_train": True, "normalization_fit_on_final_train_only": True,
            "normal_samples_in_evaluation": False,
        }
        config.update({"best_epoch": best_epoch, "model_config": model_config,
                       "best_validation_losses": validation["losses"],
                       "trainable_parameters": sum(p.numel() for p in model.parameters())})
        mask_text = (
            f"完整观测概率 {args.node_mask_full_probability}，观测节点数 "
            f"[{args.node_mask_min_observed},{args.node_mask_max_observed}]，"
            f"强制包含真实节点概率 {args.node_mask_force_observed_probability}"
            if region_mode else "未启用"
        )
        if args.skip_test_evaluation:
            # 扫描模式：只写验证集产物，测试集留给最终配置一次性评价
            config.update({"test_evaluation": "skipped_for_sweep",
                           "test_evaluation_reason": "第二阶段权重扫描，测试集仅在最终配置评价一次"})
            _write_json(output_dir / "config.json", config)
            _write_json(output_dir / "validation_metrics.json", validation)
            _write_json(output_dir / "dataset_audit.json", dataset_audit)
            (output_dir / "report.md").write_text(
                "<!-- 摘要：RaReF 区域概率损失权重扫描单次运行记录，测试集未评价。 -->\n\n"
                f"# GReF 扫描运行记录\n\n- 损失模式：{args.loss_mode}\n- 损失定义：{loss_definition}\n"
                f"- 能量项权重：{args.lambda_energy}\n- 排序项权重：{args.lambda_rank}\n"
                f"- 区域损失权重：{args.lambda_region}\n"
                f"- 区域核 sigma：{args.region_sigma}\n"
                f"- 训练期节点缺测：{mask_text}\n"
                f"- 排序温度：{args.rank_temperature}\n"
                f"- 最佳轮次：{best_epoch}\n"
                f"- 验证 Top-1：{validation['score']['top1']}\n"
                f"- 验证 Top-3：{validation['score']['top3']}\n"
                f"- 验证损失：{validation['losses']}\n"
                f"- 测试评价：跳过（扫描模式，最终配置另行评价一次）\n"
                f"- 冒烟标记：{args.smoke}\n\n本次结果不构成方法有效性结论。\n", encoding="utf-8")
            progress.stage_done("writeout", 1)
            progress.finish()
            return output_dir
        progress.stage_start("test", len(split["test_idx"]))
        metrics, predictions = evaluate(model, data, split["test_idx"], normalization, args,
                                        progress, "test", capture=True)
        progress.stage_done("test", len(split["test_idx"]))
        progress.stage_start("writeout", 1)
        shortcut = shortcut_audit(model, data, split["test_idx"], normalization, args)
        # 用已保存的完整测试响应执行屏蔽，避免另作测试集选模。
        response = predictions["response"]
        mean_response = np.broadcast_to(response.mean(axis=1, keepdims=True), response.shape)
        ablated = response_energy(torch.from_numpy(predictions["observation"]),
                                   torch.from_numpy(np.array(mean_response)),
                                   torch.from_numpy(predictions["node_mask"])).numpy()
        valid_fault = predictions["mask"][:, :-1].astype(bool)
        shortcut["response_mean_ablation"] = rank_metrics(
            np.where(valid_fault, ablated[:, :-1], np.inf), predictions["location"])
        _write_json(output_dir / "config.json", config)
        _write_json(output_dir / "validation_metrics.json", validation)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "shortcut_audit.json", shortcut)
        _write_json(output_dir / "dataset_audit.json", dataset_audit)
        np.savez_compressed(output_dir / "predictions.npz", candidate_score=predictions["scores"],
                            oracle_energy=predictions["oracle"], gate_weights=predictions["weights"],
                            teacher_margins=predictions["margins"], candidate_mask=predictions["mask"],
                            test_indices=split["test_idx"], group_id=data["group_id"][split["test_idx"]],
                            location=predictions["location"], observation=predictions["observation"],
                            response=predictions["response"], target=predictions["target"],
                            node_mask=predictions["node_mask"])
        (output_dir / "report.md").write_text(
            "<!-- 摘要：GReF 单次运行产物摘要，不构成多种子方法有效性结论。 -->\n\n"
            f"# GReF 单次运行记录\n\n- 损失模式：{args.loss_mode}\n- 损失定义：{loss_definition}\n"
            f"- 能量项权重：{args.lambda_energy if (three_term or region_mode) else 1.0}\n"
            f"- 排序项权重：{args.lambda_rank if (three_term or region_mode) else 0.0}\n"
            f"- 图平滑项权重：{args.lambda_smooth if three_term else 0.0}\n"
            f"- 区域损失权重：{args.lambda_region if region_mode else 0.0}\n"
            f"- 区域核 sigma：{args.region_sigma if region_mode else '未启用'}\n"
            f"- 训练期节点缺测：{mask_text}\n"
            f"- 排序温度：{args.rank_temperature}\n"
            f"- 训练期边删除：概率 {edge_drop_prob}，每样本最多 {edge_drop_max} 条，"
            f"无向边组数 {len(edge_groups)}\n"
            f"- 最佳轮次：{best_epoch}\n"
            f"- 测试 Top-1：{metrics['score']['top1']}\n- 测试 Top-3：{metrics['score']['top3']}\n"
            f"- 测试门控覆盖率：{metrics['gate_coverage']}\n- 容差来源：{args.tolerance_source}\n"
            f"- 冒烟标记：{args.smoke}\n\n本次结果不证明噪声鲁棒性、跨拓扑泛化或因果机制。\n", encoding="utf-8")
        progress.stage_done("writeout", 1)
        progress.finish()
    except Exception as exc:
        progress.error("gref", f"{type(exc).__name__}: {exc}")
        raise
    return output_dir


def parse_args(argv=None) -> argparse.Namespace:
    """解析独立入口参数；容差、容差来源与温度必须显式给出。

    `--loss-mode three-term` 启用能量、排序与图平滑三项损失；`--loss-mode region`
    以区域概率交叉熵替换图平滑项，并在训练期施加节点观测掩码；`--loss-mode gref`
    保持首轮门控交叉熵与重构口径。
    """
    parser = argparse.ArgumentParser(description="训练 GReF 门控响应能量定位模型")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--energy-tolerance", required=True, type=float)
    parser.add_argument("--tolerance-source", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--lambda-family", type=float, default=0.1)
    parser.add_argument("--loss-mode", choices=("gref", "three-term", "region"), default="gref")
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-region", type=float, default=0.1)
    parser.add_argument("--region-sigma", type=float, default=1.0,
                        help="区域核的跳数尺度 sigma，训练前固定")
    parser.add_argument("--beta-obs", type=float, default=0.1,
                        help="真实故障节点被观测时的区域目标扩散系数")
    parser.add_argument("--beta-miss", type=float, default=0.4,
                        help="真实故障节点未被观测时的区域目标扩散系数")
    parser.add_argument("--node-mask-full-probability", type=float, default=0.25,
                        help="训练批中保留完整观测的样本比例")
    parser.add_argument("--node-mask-min-observed", type=int, default=15,
                        help="随机部分观测时保留的最少节点数")
    parser.add_argument("--node-mask-max-observed", type=int, default=37,
                        help="随机部分观测时保留的最多节点数")
    parser.add_argument("--node-mask-force-observed-probability", type=float, default=0.5,
                        help="部分观测故障样本中强制包含真实故障节点的比例")
    parser.add_argument("--rank-temperature", type=float, default=None)
    parser.add_argument("--smooth-edge-weights", choices=("unit",), default="unit")
    parser.add_argument("--save-epochs", default="",
                        help="逗号分隔的轮次，用于保存分段检查点，例如 1,50,100,200")
    parser.add_argument("--edge-drop-prob", type=float, default=0.0,
                        help="训练时每个样本随机删除无向边的概率；0 表示不启用")
    parser.add_argument("--edge-drop-max", type=int, default=3,
                        help="每个启用删除的样本最多删除的无向边数")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test-evaluation", action="store_true",
                        help="扫描模式：训练后跳过测试集评价，只写验证集产物")
    parser.add_argument("--no-terminal-progress", action="store_true")
    return parser.parse_args(argv)
