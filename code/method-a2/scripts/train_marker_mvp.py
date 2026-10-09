"""训练 Method-A2 Node Marker MVP 并保存可核验产物。

本脚本按《Method-A2 排除既有方法后的准确率提升验证计划》要求组织训练与评价：

- 按 `group_id` 划分训练、验证与测试三段，测试集仅在配置冻结后评估一次；
- 支持鲁棒池化（`mean`、`trimmed_mean`、`mean_median_concat`）、双尺度时间编码、
  派生观测通道与候选响应不确定性四个可开关项；
- 只使用验证集选择最佳轮次与部署评分口径；
- 输出规定产物：config、dataset_audit、train_log、validation_metrics、metrics、
  predictions、oracle_metrics、hard_negative_pairs、shortcut_audit、report。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.dataset_builder import load_dataset
from src.model.marker_response_predictor import MarkerResponsePredictor
from src.progress import ProgressWriter
from src.training.evaluation import (
    oracle_rank_metrics,
    rank_metrics,
    ridge_probe,
    split_by_group,
)
from src.training.marker_training import compute_marker_losses, train_step


def _set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch(
    data: dict,
    indices: np.ndarray,
    device: torch.device,
    candidate_mask: np.ndarray,
    n_fault_candidates: int,
) -> dict[str, torch.Tensor]:
    """从数组数据构造 student 可见训练批次。"""
    mask = torch.as_tensor(candidate_mask[indices], device=device)
    fault_mask = mask.clone()
    fault_mask[:, n_fault_candidates:] = 0.0
    return {
        "observation": torch.as_tensor(data["X_obs"][indices], device=device),
        "target": torch.as_tensor(data["signature_bank"][indices], device=device),
        "edge_index": torch.as_tensor(data["edge_index"], device=device),
        "edge_attr": torch.as_tensor(data["edge_attr"], device=device),
        "edge_mask": torch.as_tensor(data["edge_mask"][indices], device=device),
        "node_mask": torch.as_tensor(data["mask"][indices], device=device),
        "candidate_mask": mask,
        "fault_mask": fault_mask,
        "group_id": torch.as_tensor(data["group_id"][indices], device=device),
        "location": torch.as_tensor(data["y_loc"][indices], device=device),
    }


def _forward_all(
    model: MarkerResponsePredictor,
    data: dict,
    indices: np.ndarray,
    device: torch.device,
    candidate_mask: np.ndarray,
    n_fault_candidates: int,
    batch_size: int = 64,
) -> dict[str, np.ndarray | None]:
    """在给定索引上分批前向，返回响应、方差与事件级共享表示。"""
    responses, scores, variances, pooled, nodes = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = _batch(data, chunk, device, candidate_mask, n_fault_candidates)
            node_repr = model.encode_shared(
                batch["observation"], batch["edge_index"], batch["edge_attr"],
                batch["edge_mask"], batch["node_mask"],
            )
            output = model(
                batch["observation"], batch["edge_index"], batch["edge_attr"],
                edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                candidate_mask=batch["candidate_mask"],
            )
            responses.append(output["response"].cpu().numpy())
            scores.append(output["candidate_score"].cpu().numpy())
            pooled.append(output["z_shared"].cpu().numpy())
            nodes.append(node_repr.cpu().numpy())
            if "response_logvar" in output:
                variances.append(
                    torch.exp(output["response_logvar"]).cpu().numpy()
                )
    result = {
        "response": np.concatenate(responses, axis=0) if responses else None,
        "candidate_score": np.concatenate(scores, axis=0) if scores else None,
        "z_shared": np.concatenate(pooled, axis=0) if pooled else None,
        "node_repr": np.concatenate(nodes, axis=0) if nodes else None,
        "variance": np.concatenate(variances, axis=0) if variances else None,
    }
    return result


def _evaluate_split(
    model: MarkerResponsePredictor,
    data: dict,
    indices: np.ndarray,
    device: torch.device,
    candidate_mask: np.ndarray,
    n_fault_candidates: int,
    scale: np.ndarray,
    lambda_rank: float,
    lambda_response: float,
    rank_margin: float,
    rank_temperature: float,
    uncertainty_weight: float,
    logvar_clamp: float,
) -> dict:
    """在给定划分上计算损失、部署评分指标与 Oracle 指标。"""
    batch = _batch(data, indices, device, candidate_mask, n_fault_candidates)
    with torch.no_grad():
        output = model(
            batch["observation"], batch["edge_index"], batch["edge_attr"],
            edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
            candidate_mask=batch["candidate_mask"],
        )
        losses = compute_marker_losses(
            output, batch["observation"], batch["target"], batch["candidate_mask"],
            batch["fault_mask"], batch["location"],
            lambda_rank=lambda_rank, lambda_response=lambda_response,
            rank_margin=rank_margin,
            rank_temperature=rank_temperature,
            uncertainty_weight=uncertainty_weight,
            logvar_clamp=logvar_clamp,
        )
    forward = _forward_all(
        model, data, indices, device, candidate_mask, n_fault_candidates
    )
    response = forward["response"]
    observation = data["X_obs"][indices]
    locations = data["y_loc"][indices]
    target = data["signature_bank"][indices]

    fault_rows = locations >= 0
    result: dict = {
        "losses": {name: float(value.cpu()) for name, value in losses.items()},
        "n_events": int(indices.size),
        "n_fault_events": int(fault_rows.sum()),
    }
    if not fault_rows.any():
        return result

    fault_scores = forward["candidate_score"][fault_rows, :n_fault_candidates]
    fault_observation = observation[fault_rows]
    fault_locations = locations[fault_rows]
    fault_target = target[fault_rows][:, :n_fault_candidates]
    oracle = oracle_rank_metrics(
        fault_observation, fault_target, fault_locations, n_fault_candidates, scale=scale
    )
    margins = oracle.pop("margins")

    # 候选评分与响应距离不在同一量纲，不使用物理响应间隔计算误差比值。
    result["score"] = rank_metrics(fault_scores, fault_locations)
    result["oracle"] = oracle
    if forward["variance"] is not None:
        variance = forward["variance"][fault_rows][:, :n_fault_candidates]
        difference = response[fault_rows, :n_fault_candidates] - fault_observation[:, None]
        absolute_error = np.abs(difference.astype(np.float64)).mean(axis=(2, 3, 4))
        predicted_variance = variance.astype(np.float64).mean(axis=(2, 3, 4))
        flat_error = absolute_error.reshape(-1)
        flat_variance = predicted_variance.reshape(-1)
        if flat_error.std() > 0 and flat_variance.std() > 0:
            result["variance_error_correlation"] = float(
                np.corrcoef(flat_error, flat_variance)[0, 1]
            )
        else:
            result["variance_error_correlation"] = None
    return result


def _hard_negative_pairs(
    data: dict,
    indices: np.ndarray,
    candidate_scores: np.ndarray,
    scale: np.ndarray,
    n_fault_candidates: int,
) -> list[dict]:
    """汇总验证集上评分最难的候选对，并附带真实响应间隔诊断。"""
    locations = data["y_loc"][indices]
    target = data["signature_bank"][indices][:, :n_fault_candidates]
    observation = data["X_obs"][indices]
    difference = target - observation[:, None]
    normalized = difference / scale.reshape(1, 1, 1, 1, -1)
    oracle_scores = (normalized.astype(np.float64) ** 2).mean(axis=(2, 3, 4))
    rows = []
    for position, sample_index in enumerate(indices):
        true_location = int(locations[position])
        order = np.argsort(oracle_scores[position])
        hardest = int(order[0]) if int(order[0]) != true_location else int(order[1])
        predicted = int(np.argmin(candidate_scores[position]))
        rows.append({
            "sample_index": int(sample_index),
            "group_id": int(data["group_id"][sample_index]),
            "true_location": true_location,
            "predicted_location": predicted,
            "oracle_hardest_location": hardest,
            "true_margin": float(
                oracle_scores[position, hardest] - oracle_scores[position, true_location]
            ),
            "predicted_margin": float(
                candidate_scores[position, hardest] - candidate_scores[position, true_location]
            ),
            "top1_correct": bool(predicted == true_location),
            "candidate_pair": sorted([true_location, hardest]),
        })
    rows.sort(key=lambda row: (row["top1_correct"], row["true_margin"]))
    return rows


def _save_predictions(
    path: Path,
    data: dict,
    indices: np.ndarray,
    response: np.ndarray,
    candidate_score: np.ndarray,
    variance: np.ndarray | None,
    pooled: np.ndarray,
) -> None:
    """保存测试集预测、观测、真实位置与事件级共享表示。"""
    payload = {
        "response": response,
        "candidate_score": candidate_score,
        "observation": data["X_obs"][indices],
        "target": data["signature_bank"][indices],
        "location": data["y_loc"][indices],
        "group_id": data["group_id"][indices],
        "candidate_mask": data["candidate_mask"][indices],
        "test_indices": indices,
        "z_shared": pooled,
    }
    if variance is not None:
        payload["variance"] = variance
    np.savez_compressed(path, **payload)


def run(args: argparse.Namespace) -> Path:
    """执行训练与评价并返回产物目录。"""
    _set_seed(args.seed)
    data = load_dataset(args.data_dir)
    meta = data["meta"]
    n_nodes = int(meta["n_nodes"])
    n_fault_candidates = n_nodes
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"train-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-seed{args.seed}"
    progress_path = args.progress or (METHOD_DIR / "logs" / run_id / "progress.jsonl")
    progress = ProgressWriter(progress_path, run_id, terminal=not args.no_terminal_progress)

    candidate_mask = data["candidate_mask"]
    scaler = np.load(Path(args.data_dir) / meta.get("feature_scaler", "feature_scaler.npz"))
    scale = scaler["std"].astype(np.float64)
    scale = np.where(scale < 1e-8, 1.0, scale)

    if "val_idx" in data and data["val_idx"].size > 0 and not args.resplit:
        split = {
            "train_idx": data["train_idx"],
            "val_idx": data["val_idx"],
            "test_idx": data["test_idx"],
        }
        split_source = "数据目录内置划分"
    else:
        split = split_by_group(
            data["group_id"], data["y_loc"], args.seed,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        )
        split_source = "训练脚本按分组重新划分"
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    model = MarkerResponsePredictor(
        n_nodes=n_nodes,
        time_steps=int(meta["window_len"]),
        feature_dim=int(meta["feature_dim"]),
        pooling=args.pooling,
        temporal_mode=args.temporal_mode,
        use_derived_channels=args.derived_channels,
        uncertainty=args.uncertainty,
        logvar_clamp=args.logvar_clamp,
        temporal_layers=args.temporal_layers,
        encoder_layers=args.encoder_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log_rows: list[dict] = []
    history: list[dict] = []
    try:
        batches_per_epoch = (len(train_idx) + args.batch_size - 1) // args.batch_size
        if args.max_batches > 0:
            batches_per_epoch = min(batches_per_epoch, args.max_batches)
        total_batches = args.epochs * batches_per_epoch
        progress.stage_start("training", total_batches + args.epochs)
        completed = 0
        best_state = None
        best_val = None
        for epoch in range(args.epochs):
            shuffled = np.random.default_rng(args.seed + epoch).permutation(train_idx)
            epoch_rows = []
            for start in range(0, len(shuffled), args.batch_size):
                if args.max_batches > 0 and len(epoch_rows) >= args.max_batches:
                    break
                indices = shuffled[start:start + args.batch_size]
                started = time.perf_counter()
                values = train_step(
                    model, optimizer,
                    _batch(data, indices, device, candidate_mask, n_fault_candidates),
                    lambda_rank=args.lambda_rank,
                    lambda_response=args.lambda_response,
                    rank_margin=args.rank_margin,
                    rank_temperature=args.rank_temperature,
                    uncertainty_weight=args.uncertainty_weight,
                    logvar_clamp=args.logvar_clamp,
                )
                values["epoch"] = epoch
                epoch_rows.append(values)
                log_rows.append(values)
                completed += 1
                progress.update(
                    "training", "batch", completed, total_batches + args.epochs,
                    note=f"epoch={epoch + 1}/{args.epochs}",
                    unit_seconds=time.perf_counter() - started, force=True,
                )
            if not epoch_rows:
                raise RuntimeError("训练批次为空")
            started = time.perf_counter()
            validation = _evaluate_split(
                model, data, val_idx, device, candidate_mask, n_fault_candidates,
                scale, args.lambda_rank, args.lambda_response, args.rank_margin,
                args.rank_temperature,
                args.uncertainty_weight, args.logvar_clamp,
            )
            completed += 1
            progress.update(
                "training", "validation", completed, total_batches + args.epochs,
                note=f"epoch={epoch + 1}/{args.epochs} 验证集评估完成",
                unit_seconds=time.perf_counter() - started, force=True,
            )
            record = {
                "epoch": epoch + 1,
                "train_family": float(np.mean([row["family"] for row in epoch_rows])),
                "train_loc": float(np.mean([row["loc"] for row in epoch_rows])),
                "train_rank": float(np.mean([row["rank"] for row in epoch_rows])),
                "train_total": float(np.mean([row["total"] for row in epoch_rows])),
                "val_family": validation["losses"]["family"],
                "val_loc": validation["losses"]["loc"],
                "val_rank": validation["losses"]["rank"],
                "val_total": validation["losses"]["total"],
                "val_top1": validation.get("score", {}).get("top1"),
                "val_top3": validation.get("score", {}).get("top3"),
                "val_mean_rank": validation.get("score", {}).get("mean_rank"),
            }
            history.append(record)
            score = (record["val_top1"] or 0.0, -(record["val_family"] or 0.0))
            if best_val is None or score > best_val:
                best_val = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        progress.stage_done("training", total_batches + args.epochs)
        if best_state is not None:
            model.load_state_dict(best_state)

        progress.stage_start("evaluation", 2)
        started = time.perf_counter()
        validation = _evaluate_split(
            model, data, val_idx, device, candidate_mask, n_fault_candidates,
            scale, args.lambda_rank, args.lambda_response, args.rank_margin,
            args.rank_temperature,
            args.uncertainty_weight, args.logvar_clamp,
        )
        progress.update("evaluation", "validation", 1, 2, note="验证集指标完成",
                        unit_seconds=time.perf_counter() - started, force=True)
        started = time.perf_counter()
        test = _evaluate_split(
            model, data, test_idx, device, candidate_mask, n_fault_candidates,
            scale, args.lambda_rank, args.lambda_response, args.rank_margin,
            args.rank_temperature,
            args.uncertainty_weight, args.logvar_clamp,
        )
        progress.update("evaluation", "test", 2, 2, note="测试集指标完成",
                        unit_seconds=time.perf_counter() - started, force=True)
        progress.stage_done("evaluation", 2)

        progress.stage_start("probes", 4)
        forward_train = _forward_all(
            model, data, train_idx, device, candidate_mask, n_fault_candidates
        )
        forward_val = _forward_all(
            model, data, val_idx, device, candidate_mask, n_fault_candidates
        )
        # 只在故障样本上做探针，训练与验证按分组划分，避免同一反事实族泄漏
        fault_train = data["y_loc"][train_idx] >= 0
        train_rows = np.flatnonzero(fault_train)
        train_features = forward_train["z_shared"][train_rows]
        train_labels = data["y_loc"][train_idx][train_rows]
        val_features = forward_val["z_shared"]
        val_labels = data["y_loc"][val_idx]
        train_selector = np.concatenate([
            np.ones(train_rows.size, dtype=bool), np.zeros(val_idx.size, dtype=bool)
        ])
        val_selector = ~train_selector
        combined_features = np.concatenate([train_features, val_features], axis=0)
        combined_labels = np.concatenate([train_labels, val_labels], axis=0)

        started = time.perf_counter()
        probes = {
            "z_shared_fault_location": ridge_probe(
                combined_features, combined_labels, train_selector, val_selector,
                task="classification",
            )
        }
        probes["z_shared_fault_location_permuted_label_control"] = ridge_probe(
            combined_features,
            np.random.default_rng(args.seed).permutation(combined_labels),
            train_selector, val_selector, task="classification",
        )
        progress.update("probes", "logistic", 1, 4, note="故障位置 probe 完成",
                        unit_seconds=time.perf_counter() - started, force=True)

        # 节点级表示的可恢复性对照：展平全部节点后做同一任务
        started = time.perf_counter()
        node_train = forward_train["node_repr"][train_rows].reshape(train_rows.size, -1)
        node_val = forward_val["node_repr"].reshape(val_idx.size, -1)
        probes["node_repr_fault_location"] = ridge_probe(
            np.concatenate([node_train, node_val], axis=0),
            combined_labels, train_selector, val_selector, task="classification",
        )
        node_within_group = []
        for group in sorted(set(data["group_id"][train_idx].tolist()) - {-1}):
            rows = np.flatnonzero(data["group_id"][train_idx] == group)
            if rows.size < 2:
                continue
            vectors = forward_train["node_repr"][rows].reshape(rows.size, -1)
            node_within_group.append(float(((vectors - vectors.mean(axis=0)) ** 2).mean()))
        probes["node_repr_within_group_squared_distance_mean"] = (
            float(np.mean(node_within_group)) if node_within_group else None
        )
        progress.update("probes", "logistic", 2, 4, note="节点级对照完成",
                        unit_seconds=time.perf_counter() - started, force=True)

        started = time.perf_counter()
        if "event_parameters" in data:
            parameters = np.asarray(data["event_parameters"], dtype=np.float64)
            probes["z_shared_event_parameters"] = ridge_probe(
                combined_features,
                np.concatenate([
                    parameters[train_idx][train_rows], parameters[val_idx]
                ], axis=0),
                train_selector, val_selector, task="regression",
            )
        progress.update("probes", "ridge", 3, 4, note="事件参数 probe 完成",
                        unit_seconds=time.perf_counter() - started, force=True)

        # 同组表示一致性：同一分组内不同事实位置的共享表示距离
        started = time.perf_counter()
        within_group = []
        for group in sorted(set(data["group_id"][train_idx].tolist()) - {-1}):
            rows = np.flatnonzero(data["group_id"][train_idx] == group)
            if rows.size < 2:
                continue
            vectors = forward_train["z_shared"][rows]
            within_group.append(float(((vectors - vectors.mean(axis=0)) ** 2).mean()))
        probes["z_shared_within_group_squared_distance_mean"] = (
            float(np.mean(within_group)) if within_group else None
        )
        probes["z_shared_across_event_variance"] = float(
            forward_train["z_shared"].var(axis=0).mean()
        )
        progress.update("probes", "ridge", 4, 4, note="表示一致性统计完成",
                        unit_seconds=time.perf_counter() - started, force=True)
        progress.stage_done("probes", 4)
        progress.finish("writeout")
    except Exception as exc:
        progress.error("training", f"{type(exc).__name__}: {exc}")
        raise

    progress.stage_start("writeout", 1)
    forward_test = _forward_all(
        model, data, test_idx, device, candidate_mask, n_fault_candidates
    )
    _save_predictions(
        output_dir / "predictions.npz", data, test_idx,
        forward_test["response"], forward_test["candidate_score"],
        forward_test["variance"], forward_test["z_shared"],
    )
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "family", "loc", "rank", "total", "logvar_clamped_ratio"],
        )
        writer.writeheader()
        for row in log_rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    with (output_dir / "validation_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update({
        "meta": meta,
        "run_id": run_id,
        "progress_file": str(progress_path),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "split_source": split_source,
        "split": {
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_test": int(test_idx.size),
            "train_groups": sorted(set(data["group_id"][train_idx].tolist()) - {-1}),
            "val_groups": sorted(set(data["group_id"][val_idx].tolist()) - {-1}),
            "test_groups": sorted(set(data["group_id"][test_idx].tolist()) - {-1}),
        },
        "best_epoch": int(max(history, key=lambda row: (row["val_top1"] or 0.0, -(row["val_family"] or 0.0)))["epoch"]),
        "score_definition": "候选评分头输出，分数越低越可能是真实故障位置",
    })
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_groups = set(data["group_id"][train_idx].tolist()) - {-1}
    test_groups = set(data["group_id"][test_idx].tolist()) - {-1}
    val_groups = set(data["group_id"][val_idx].tolist()) - {-1}
    (output_dir / "dataset_audit.json").write_text(json.dumps({
        "group_id_present": bool(np.any(data["group_id"] >= 0)),
        "train_val_group_overlap": len(train_groups & val_groups),
        "train_test_group_overlap": len(train_groups & test_groups),
        "val_test_group_overlap": len(val_groups & test_groups),
        "counterfactual_perturbation": meta.get("counterfactual_perturbation", "unknown"),
        "candidate_mask_min_coverage": float(candidate_mask[data["y_loc"] >= 0].sum(axis=1).min()),
        "candidate_mask_mean_coverage": float(candidate_mask[data["y_loc"] >= 0].mean()),
        "fault_classes_used": meta.get("fault_classes_used"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    hard_pairs = _hard_negative_pairs(
        data, val_idx,
        forward_val["candidate_score"][:, :n_fault_candidates],
        scale, n_fault_candidates,
    )
    (output_dir / "hard_negative_pairs.json").write_text(
        json.dumps({
            "source_split": "validation",
            "n_pairs": len(hard_pairs),
            "pairs": hard_pairs,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "oracle_metrics.json").write_text(
        json.dumps({
            "validation": validation.get("oracle"),
            "test": test.get("oracle"),
            "scale_definition": "训练集逐通道标准差",
        }, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    (output_dir / "probes.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    (output_dir / "shortcut_audit.json").write_text(json.dumps({
        "teacher_used": False,
        "global_repr_used": False,
        "qkv_or_edge_scoring_used": False,
        "derived_channels_from_observation_only": True,
        "note": "置换与信息边界审计由 tests/test_accuracy_upgrade.py 覆盖",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "unit_tests.txt").write_text(
        "python -m pytest -q code/method-a2/tests/test_progress.py code/method-a2/tests/test_marker_mvp.py "
        "code/method-a2/tests/test_accuracy_upgrade.py\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "<!-- 摘要：Node Marker MVP 准确率提升验证的单次运行记录。 -->\n\n"
        "# Node Marker MVP 训练记录\n\n"
        f"- 运行标识：`{run_id}`\n"
        f"- 数据目录：`{args.data_dir}`\n"
        f"- 池化方式：`{args.pooling}`；时间编码：`{args.temporal_mode}`；"
        f"派生通道：`{args.derived_channels}`；不确定性：`{args.uncertainty}`\n"
        f"- 划分：训练 {train_idx.size}、验证 {val_idx.size}、测试 {test_idx.size}\n"
        f"- 测试 Top-1：{test.get('score', {}).get('top1')}；"
        f"Top-3：{test.get('score', {}).get('top3')}\n"
        f"- Oracle Top-1：{test.get('oracle', {}).get('oracle_top1')}；"
        f"Top-3：{test.get('oracle', {}).get('oracle_top3')}\n\n"
        "本产物未使用 teacher 路径、QKV、边打分与全局表示。\n",
        encoding="utf-8",
    )
    progress.update("writeout", "artifact", 1, 1, note="产物写入完成", force=True)
    progress.stage_done("writeout", 1)
    progress.finish("writeout")
    return output_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 Method-A2 Node Marker MVP")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--lambda-response", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--logvar-clamp", type=float, default=8.0)
    parser.add_argument("--temporal-layers", type=int, default=3,
                        help="时间编码器残差块层数")
    parser.add_argument("--encoder-layers", type=int, default=2,
                        help="共享图编码器消息传递层数")
    parser.add_argument("--pooling", choices=["mean", "trimmed_mean", "mean_median_concat"],
                        default="mean")
    parser.add_argument("--temporal-mode", choices=["single", "dual"], default="single")
    parser.add_argument("--derived-channels", action="store_true")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resplit", action="store_true",
                        help="忽略数据目录内置划分，按分组重新划分")
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--no-terminal-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
