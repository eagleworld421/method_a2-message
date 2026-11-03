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
from .data import fit_normalization, load_arrays, make_batch
from .losses import gref_losses, response_energy, teacher_gate
from .model import GrefPredictor

METHOD_DIR = Path(__file__).resolve().parents[2]


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
    return gref_losses(output, batch["observation"], batch["target"],
                       batch["candidate_mask"], batch["location"],
                       tolerance=args.energy_tolerance, temperature=args.temperature,
                       lambda_family=args.lambda_family, mask=batch["node_mask"])


def _strata(correct, values, metric="top1") -> list[dict]:
    """按离散节点、事件族或门控状态汇总全部故障事件。"""
    return [{"value": int(value), "n_events": int((values == value).sum()),
             metric: float(correct[values == value].mean())} for value in np.unique(values)]


@torch.no_grad()
def evaluate(model, data, indices, normalization, args, progress=None, stage="validation", capture=False):
    """始终采用评价模式，以全局分母汇总损失，完整集合不受门控筛选。"""
    model.eval()
    parts = {key: [] for key in ("scores", "oracle", "weights", "margins", "location", "mask")}
    payload = {key: [] for key in ("observation", "response", "target", "node_mask")}
    loc_num = weight_sum = family_num = family_den = 0.0
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start:start + args.batch_size]
        batch = make_batch(data, chunk, normalization, torch.device(args.device))
        output = _forward(model, batch)
        losses = _losses(output, batch, args)
        denominator = float((batch["node_mask"].sum(1) * batch["candidate_mask"].sum(1)).sum())
        denominator *= batch["observation"].shape[2] * batch["observation"].shape[3]
        loc_num += float(losses["loc_numerator"])
        weight_sum += float(losses["weight_sum"])
        family_num += float(losses["family"]) * denominator
        family_den += denominator
        items = {"scores": output["candidate_score"], "oracle": losses["teacher_energy"],
                 "weights": losses["gate_weights"], "margins": losses["teacher_margins"],
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
    family = family_num / max(family_den, 1.0)
    loc = loc_num / (weight_sum + 1e-12)
    metrics = {
        "n_events": int(len(indices)), "n_fault_events": int(fault.sum()),
        "losses": {"loc": loc, "family": family, "total": loc + args.lambda_family * family},
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
    if not args.tolerance_source.strip() or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("必须提供容差来源及正的轮数、批大小")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("验证、测试比例必须为正且总和小于一")
    output_dir = Path(args.output_dir).resolve()
    output_root = (METHOD_DIR / "output" / "gref").resolve()
    if output_root not in output_dir.parents:
        raise ValueError("输出必须位于本方法 output/gref/<run-id>/ 下")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_arrays(Path(args.data_dir))
    split = split_by_group(data["group_id"], data["y_loc"], args.seed, args.val_ratio, args.test_ratio)
    run_id = output_dir.name
    progress_path = METHOD_DIR / "logs" / run_id / "progress.jsonl"
    if progress_path.exists():
        raise FileExistsError("运行日志已存在，请使用新的运行标识")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(progress_path, run_id, terminal=not args.no_terminal_progress)
    try:
        source_root = Path(__file__).resolve().parents[2]
        config = dict(vars(args))
        config.update({"method": "GReF", "dataset_meta": data["meta"],
                       "score_definition": "最终训练侧标准化坐标中的掩码均方响应距离",
                       "normalization_definition": "仅最终训练有效候选，逐节点通道，现有输入坐标的仿射重拟合",
                       "software": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__)},
                       "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [*Path(__file__).parent.glob("*.py"),
                                                   source_root / "src/model/marker_response_predictor.py",
                                                   source_root / "src/model/losses.py", source_root / "src/training/evaluation.py",
                                                   source_root / "src/progress.py", source_root / "src/model/gnn.py",
                                                   source_root / "src/model/temporal.py", source_root / "scripts/train_gref.py"]}})
        _write_json(output_dir / "config.json", config)
        progress.stage_start("normalization", len(split["train_idx"]))
        normalization = fit_normalization(data, split["train_idx"], progress)
        np.savez_compressed(output_dir / "normalization.npz", **normalization)
        np.savez_compressed(output_dir / "split.npz", **split)
        progress.stage_done("normalization", len(split["train_idx"]))
        progress.stage_start("audit", len(split["train_idx"]))
        gate = audit_training(data, split["train_idx"], normalization, args, progress)
        _write_json(output_dir / "gate_audit.json", gate)
        progress.stage_done("audit", len(split["train_idx"]))
        model_config = {"n_nodes": data["X_obs"].shape[1], "time_steps": data["X_obs"].shape[2],
                        "feature_dim": data["X_obs"].shape[3]}
        model = GrefPredictor(**model_config).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        rng = np.random.default_rng(args.seed)
        history, best_key, best_epoch = [], None, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(split["train_idx"])
            batch_count = math.ceil(len(order) / args.batch_size)
            progress.stage_start("training", batch_count)
            total_sum = 0.0
            for start in range(0, len(order), args.batch_size):
                started = time.perf_counter()
                batch = make_batch(data, order[start:start + args.batch_size], normalization, torch.device(args.device))
                optimizer.zero_grad(set_to_none=True)
                losses = _losses(_forward(model, batch), batch, args)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("训练损失出现非有限值")
                losses["total"].backward()
                optimizer.step()
                total_sum += float(losses["total"].detach()) * len(batch["location"])
                progress.update("training", "batch", start // args.batch_size + 1, batch_count,
                                note=f"epoch={epoch}/{args.epochs}", unit_seconds=time.perf_counter() - started)
            progress.stage_done("training", batch_count)
            progress.stage_start("validation", len(split["val_idx"]))
            validation, _ = evaluate(model, data, split["val_idx"], normalization, args, progress)
            progress.stage_done("validation", len(split["val_idx"]))
            record = {"epoch": epoch, "train_batch_total_mean": total_sum / len(order),
                      "val_top1": validation["score"]["top1"], "val_top3": validation["score"]["top3"],
                      "val_family": validation["losses"]["family"], "val_loc": validation["losses"]["loc"],
                      "val_gate_coverage": validation["gate_coverage"]}
            history.append(record)
            with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(history)
            key = (record["val_top1"], -record["val_family"])
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                            "best_epoch": epoch}, output_dir / "model.pt")
        checkpoint = torch.load(output_dir / "model.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation, _ = evaluate(model, data, split["val_idx"], normalization, args)
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
        config.update({"best_epoch": best_epoch, "model_config": model_config,
                       "trainable_parameters": sum(p.numel() for p in model.parameters())})
        _write_json(output_dir / "config.json", config)
        _write_json(output_dir / "validation_metrics.json", validation)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "shortcut_audit.json", shortcut)
        _write_json(output_dir / "dataset_audit.json", {
            "split_groups": {key: split[key].tolist() for key in ("train_groups", "val_groups", "test_groups")},
            "n_train": len(split["train_idx"]), "n_val": len(split["val_idx"]), "n_test": len(split["test_idx"]),
            "fact_matches_true_response_train": True, "normalization_fit_on_final_train_only": True,
            "normal_samples_in_evaluation": False,
        })
        np.savez_compressed(output_dir / "predictions.npz", candidate_score=predictions["scores"],
                            oracle_energy=predictions["oracle"], gate_weights=predictions["weights"],
                            teacher_margins=predictions["margins"], candidate_mask=predictions["mask"],
                            test_indices=split["test_idx"], group_id=data["group_id"][split["test_idx"]],
                            location=predictions["location"], observation=predictions["observation"],
                            response=predictions["response"], target=predictions["target"],
                            node_mask=predictions["node_mask"])
        (output_dir / "report.md").write_text(
            "<!-- 摘要：GReF 单次运行产物摘要，不构成多种子方法有效性结论。 -->\n\n"
            f"# GReF 单次运行记录\n\n- 最佳轮次：{best_epoch}\n"
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
    """解析独立入口参数；容差、容差来源与温度必须显式给出。"""
    parser = argparse.ArgumentParser(description="训练 GReF 门控响应能量定位模型")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--energy-tolerance", required=True, type=float)
    parser.add_argument("--tolerance-source", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--lambda-family", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-terminal-progress", action="store_true")
    return parser.parse_args(argv)
