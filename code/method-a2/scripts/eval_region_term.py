"""在冻结区域概率模型上评价完整观测与部分节点观测的定位及区域指标。

测试集只在冻结检查点上评价一次：先用验证集完整观测标定单节点概率温度，再在
完整观测与固定的 35/30/25/20/15 个观测节点各 10 组共享掩码上计算排名指标、
单节点概率统计和区域覆盖率/区域大小/拓扑半径/中心距离，并按真实故障节点是否
被观测做条件分解。不更新参数、不依据测试结果选择任何配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.data import load_arrays, make_batch
from src.gref.model import GrefPredictor
from src.gref.region import (
    calibrate_temperature,
    graph_distance_hops,
    probability_statistics,
    region_event_metrics,
    region_kernel,
    region_probabilities,
)
from src.training.evaluation import rank_metrics

METRIC_KEYS = ("top1", "top3", "top5", "mean_rank", "median_rank",
               "signed_margin_median", "signed_margin_negative_ratio")


def parse_args() -> argparse.Namespace:
    """解析区域指标评价参数。"""
    parser = argparse.ArgumentParser(description="评价区域概率模型的定位与区域指标")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--observed-nodes", type=int, nargs="+", default=[35, 30, 25, 20, 15])
    parser.add_argument("--masks-per-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--masks-file", type=Path, default=None,
                        help="用于一致性核对的既有 node_masks.npz；给出时要求逐元素一致")
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="评价测试集，或在验证集上做选模评价")
    parser.add_argument("--mask-seed-offset", type=int, default=0,
                        help="验证集掩码使用的种子偏移；测试集保持 0")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="非区域模式对照运行使用的区域核尺度")
    parser.add_argument("--no-merge-predictions", dest="merge_predictions",
                        action="store_false",
                        help="对照运行只写 region_predictions.npz，不覆盖原 predictions.npz")
    parser.set_defaults(merge_predictions=True)
    return parser.parse_args()


def build_node_masks(n_nodes: int, observed_nodes: int, count: int, seed: int) -> np.ndarray:
    """按 S2 协议生成去重的节点观测掩码，返回 `[count,n_nodes]`。

    第 i 个观测规模使用 `seed+1009*i`，从节点中无放回抽取指定数量，索引升序记录。
    与 S2 冻结推理使用完全相同的生成规则，便于跨实验对齐。
    """
    if not 1 <= observed_nodes <= n_nodes:
        raise ValueError("observed_nodes 必须位于 [1,n_nodes]")
    if count < 1:
        raise ValueError("masks_per_level 必须为正整数")
    rng = np.random.default_rng(seed)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    attempts = 0
    while len(unique) < count and attempts < count * 100:
        attempts += 1
        selected = tuple(sorted(rng.choice(n_nodes, size=observed_nodes, replace=False).tolist()))
        if selected in unique:
            continue
        mask = np.zeros(n_nodes, dtype=np.float32)
        mask[list(selected)] = 1.0
        unique[selected] = mask
    if len(unique) != count:
        raise RuntimeError("无法生成足够的不重复节点掩码")
    return np.stack(list(unique.values())).astype(np.float32)


@torch.no_grad()
def model_energies(
    model: GrefPredictor, data: dict, indices: np.ndarray, normalization: dict,
    device: torch.device, node_mask: np.ndarray, batch_size: int,
) -> np.ndarray:
    """在固定节点掩码下返回部署能量 `[E,C]`（候选响应掩码均方距离）。"""
    values = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = make_batch(data, chunk, normalization, device, graph_view="true")
        batch["node_mask"] = torch.as_tensor(
            np.repeat(node_mask[None, :], len(chunk), axis=0),
            dtype=torch.float32, device=device,
        )
        output = model(
            batch["observation"], batch["edge_index"], batch["edge_attr"],
            edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
            candidate_mask=batch["candidate_mask"],
        )
        values.append(output["candidate_score"].cpu().numpy())
    return np.concatenate(values)


def _stats(values: list) -> dict:
    """给出掩码级读数的均值、最小值与最大值。"""
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}


def _aggregate(records: list[dict], path: tuple) -> dict:
    """按嵌套字段路径汇总掩码级标量，返回均值与范围。"""
    values = []
    for record in records:
        current = record
        for key in path:
            current = current[key]
        values.append(float(current))
    return _stats(values)


def _condition_metrics(
    energies: np.ndarray, locations: np.ndarray, node_mask: np.ndarray,
    kernel: np.ndarray, distance: np.ndarray, temperature: float,
    alphas: list[float], n_fault: int,
) -> tuple[dict, dict, dict]:
    """计算单组掩码下的排名、概率与区域指标，并返回概率与区域数组。"""
    fault = locations >= 0
    scores = np.where(fault[:, None], energies[:, :n_fault], np.inf)
    rank = rank_metrics(scores[fault], locations[fault])
    logits = -energies[:, :n_fault] / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    statistics = probability_statistics(probabilities[fault])
    region = region_probabilities(probabilities, np.asarray(kernel, dtype=np.float64))
    record = {
        "rank": rank,
        "probability": {
            "entropy_mean": float(statistics["entropy"].mean()),
            "max_probability_mean": float(statistics["max_probability"].mean()),
            "top_gap_median": float(np.median(statistics["top_gap"])),
        },
        "region": {},
    }
    observed = np.asarray(node_mask).reshape(-1)[
        np.clip(locations, 0, n_fault - 1)] > 0
    for alpha in alphas:
        metrics = region_event_metrics(region[fault], locations[fault], distance, alpha)
        record["region"][str(alpha)] = {
            "coverage": metrics["coverage"],
            "size_mean": float(metrics["size"].mean()),
            "size_median": float(np.median(metrics["size"])),
            "size_p90": float(np.percentile(metrics["size"], 90)),
            "radius_mean": float(metrics["radius"].mean()),
            "radius_median": float(np.median(metrics["radius"])),
            "radius_p90": float(np.percentile(metrics["radius"], 90)),
            "center_distance_mean": float(metrics["center_distance"].mean()),
            "center_distance_median": float(np.median(metrics["center_distance"])),
            "mass_mean": float(metrics["mass"].mean()),
        }
    conditional = {}
    for name, select in (("observed", observed), ("missing", ~observed)):
        subset = fault & select
        if not subset.any():
            conditional[name] = None
            continue
        entry = {"n_events": int(subset.sum()),
                 "top1": float((scores[subset].argmin(axis=1) == locations[subset]).mean())}
        for alpha in alphas:
            metrics = region_event_metrics(region[subset], locations[subset], distance, alpha)
            entry[f"region_{alpha}"] = {
                "coverage": metrics["coverage"],
                "size_mean": float(metrics["size"].mean()),
                "size_p90": float(np.percentile(metrics["size"], 90)),
                "radius_mean": float(metrics["radius"].mean()),
                "center_distance_mean": float(metrics["center_distance"].mean()),
            }
        conditional[name] = entry
    record["conditional"] = conditional
    return record, probabilities, region


def main() -> int:
    """执行区域指标评价并写出 region_metrics.json 与合并后的 predictions.npz。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = (args.data_dir or Path(config["data_dir"])).resolve()
    output_dir = (args.output_dir or run_dir).resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = load_arrays(data_dir)
    n_fault = int(data["X_obs"].shape[1])
    n_nodes = n_fault
    if int(data["signature_bank"].shape[1]) != n_fault + 1:
        raise ValueError("候选轴必须为故障候选加末尾正常候选")
    if int(np.asarray(data["edge_index"]).max()) >= n_fault:
        raise ValueError("边端点超出故障候选轴")
    distance = graph_distance_hops(np.asarray(data["edge_index"]), n_fault)
    region_config = config.get("region_kernel")
    sigma = float(region_config["sigma"]) if region_config else float(args.sigma)
    kernel = region_kernel(distance, sigma)
    kernel_sha256 = hashlib.sha256(kernel.tobytes()).hexdigest()
    if region_config and kernel_sha256 != region_config["kernel_sha256"]:
        raise ValueError("重建的区域核与训练配置不一致")
    normalization_file = np.load(run_dir / "normalization.npz")
    normalization = {key: normalization_file[key] for key in ("mean", "std")}
    split = np.load(run_dir / "split.npz")
    split_name = args.split
    test_idx = split["test_idx"]
    val_idx = split["val_idx"]
    indices = np.asarray(test_idx if split_name == "test" else val_idx)
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = GrefPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 验证集完整观测上标定单节点概率温度，只用验证集
    val_energies = model_energies(model, data, val_idx, normalization, device,
                                  np.ones(n_nodes, dtype=np.float32), args.batch_size)
    val_locations = np.asarray(data["y_loc"][val_idx])
    val_fault_mask = np.asarray(data["candidate_mask"][val_idx][:, :n_fault], dtype=bool)
    calibration = calibrate_temperature(val_energies[:, :n_fault], val_fault_mask, val_locations)
    temperature = calibration["temperature"]
    print(json.dumps({"calibrated_temperature": temperature,
                      "validation_nll": calibration["negative_log_likelihood"]}, ensure_ascii=False))

    locations = np.asarray(data["y_loc"][indices])
    conditions: list[dict] = []
    masks_by_level: dict[str, np.ndarray] = {}
    arrays = {"probabilities": [], "regions": [], "masks": [], "keys": []}

    # 完整观测
    full_mask = np.ones(n_nodes, dtype=np.float32)
    energies = model_energies(model, data, indices, normalization, device, full_mask, args.batch_size)
    record, probabilities, region = _condition_metrics(
        energies, locations, full_mask, kernel, distance, temperature, args.alphas, n_fault)
    record.update({"condition": "full", "observed_nodes": None, "mask_index": 0})
    conditions.append(record)
    arrays["probabilities"].append(probabilities)
    arrays["regions"].append(region)
    arrays["masks"].append(np.repeat(full_mask[None, :], len(indices), axis=0))
    arrays["keys"].append("full")

    # 部分观测：与 S2 相同的掩码规则；验证集使用独立种子偏移
    for level_index, level in enumerate(args.observed_nodes):
        masks = build_node_masks(n_nodes, int(level), args.masks_per_level,
                                 args.seed + args.mask_seed_offset + 1009 * level_index)
        masks_by_level[str(level)] = masks
        for mask_index, mask in enumerate(masks):
            energies = model_energies(model, data, indices, normalization, device, mask,
                                      args.batch_size)
            record, probabilities, region = _condition_metrics(
                energies, locations, mask, kernel, distance, temperature, args.alphas, n_fault)
            record.update({"condition": "partial", "observed_nodes": int(level),
                           "mask_index": int(mask_index)})
            conditions.append(record)
            arrays["probabilities"].append(probabilities)
            arrays["regions"].append(region)
            arrays["masks"].append(np.repeat(mask[None, :], len(indices), axis=0))
            arrays["keys"].append(f"{level}:{mask_index}")

    if args.masks_file is not None:
        if split_name != "test":
            raise ValueError("掩码一致性核对只适用于测试集评价")
        reference = np.load(args.masks_file)
        for level, masks in masks_by_level.items():
            if level in reference.files:
                if not np.array_equal(masks, reference[level]):
                    raise ValueError(f"观测规模 {level} 的掩码与对照文件不一致")
        print(json.dumps({"mask_file_checked": str(args.masks_file)}, ensure_ascii=False))

    full_record = conditions[0]
    partial_summary: dict[str, dict] = {}
    for level in args.observed_nodes:
        subset = [row for row in conditions
                  if row["condition"] == "partial" and row["observed_nodes"] == int(level)]
        summary = {
            "n_masks": len(subset),
            "rank": {key: _aggregate(subset, ("rank", key)) for key in METRIC_KEYS},
            "probability": {key: _aggregate(subset, ("probability", key))
                            for key in ("entropy_mean", "max_probability_mean", "top_gap_median")},
            "region": {
                str(alpha): {
                    key: _aggregate(subset, ("region", str(alpha), key))
                    for key in subset[0]["region"][str(alpha)]
                } for alpha in args.alphas
            },
            "conditional": {},
        }
        for name in ("observed", "missing"):
            group = [row["conditional"][name] for row in subset if row["conditional"][name]]
            if not group:
                summary["conditional"][name] = None
                continue
            entry = {"n_events_mean": float(np.mean([item["n_events"] for item in group]))}
            entry["top1"] = _stats([item["top1"] for item in group])
            for alpha in args.alphas:
                entry[f"region_{alpha}"] = {
                    key: _stats([item[f"region_{alpha}"][key] for item in group])
                    for key in group[0][f"region_{alpha}"]
                }
            summary["conditional"][name] = entry
        partial_summary[str(level)] = summary

    payload = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "loss_mode": config["loss_mode"],
        "loss_weights": config["loss_weights"],
        "region_kernel": {
            "distance_metric": "候选拓扑无向跳数",
            "sigma": sigma,
            "beta_obs": (region_config or {}).get("beta_obs"),
            "beta_miss": (region_config or {}).get("beta_miss"),
            "source": "运行配置" if region_config else "评价期按统一核重建",
        },
        "temperature_calibration": calibration,
        "used_temperature": temperature,
        "rank_temperature": config.get("rank_temperature"),
        "alphas": args.alphas,
        "seed": args.seed,
        "split": split_name,
        "masks_per_level": args.masks_per_level,
        "observed_nodes": args.observed_nodes,
        "n_events": int(len(indices)),
        "n_test": int(len(test_idx)),
        "n_nodes": n_nodes,
        "mask_sha256": {level: hashlib.sha256(masks.tobytes()).hexdigest()
                        for level, masks in masks_by_level.items()},
        "full": full_record,
        "partial": partial_summary,
        "conditions": conditions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = ("region_metrics.json" if split_name == "test"
                    else "validation_region_metrics.json")
    (output_dir / metrics_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # 将单节点概率、区域概率与区域集合合并进 predictions.npz，保持原有数组不变
    proba = np.stack(arrays["probabilities"]).astype(np.float32)
    region = np.stack(arrays["regions"]).astype(np.float32)
    membership: dict[str, np.ndarray] = {}
    positions = np.arange(region.shape[2])[None, None, :]
    for alpha in args.alphas:
        order = np.argsort(-region, axis=2, kind="stable")
        ordered = np.take_along_axis(region, order, axis=2)
        cumulative = np.cumsum(ordered, axis=2)
        size = (cumulative < (1.0 - alpha) - 1e-12).sum(axis=2) + 1
        rank_of_node = np.empty_like(order)
        np.put_along_axis(rank_of_node, order,
                          np.broadcast_to(positions, order.shape), axis=2)
        membership[str(alpha)] = rank_of_node < size[..., None]
    if split_name != "test":
        # 验证集评价只写独立产物，不触碰测试集 predictions.npz
        np.savez_compressed(
            output_dir / "validation_region_predictions.npz",
            condition_keys=np.asarray(arrays["keys"]),
            condition_masks=np.stack(arrays["masks"]).astype(np.float32),
            single_node_probability=proba,
            region_probability=region,
            **{f"region_membership_alpha{str(alpha).replace('.', '').zfill(2)}": value
               for alpha, value in membership.items()},
            calibrated_temperature=np.asarray([temperature], dtype=np.float64),
            alphas=np.asarray(args.alphas, dtype=np.float64),
            location=locations, val_indices=indices,
        )
    else:
        existing = dict(np.load(run_dir / "predictions.npz"))
        existing.update({
            "condition_keys": np.asarray(arrays["keys"]),
            "condition_masks": np.stack(arrays["masks"]).astype(np.float32),
            "single_node_probability": proba,
            "region_probability": region,
            **{f"region_membership_alpha{str(alpha).replace('.', '').zfill(2)}": value
               for alpha, value in membership.items()},
            "calibrated_temperature": np.asarray([temperature], dtype=np.float64),
            "alphas": np.asarray(args.alphas, dtype=np.float64),
        })
        if args.merge_predictions:
            np.savez_compressed(run_dir / "predictions.npz", **existing)
        else:
            np.savez_compressed(output_dir / "region_predictions.npz", **existing)

    summary_out = {"full_top1": full_record["rank"]["top1"], "partial": {}}
    for level in partial_summary:
        item = partial_summary[level]
        missing = item["conditional"]["missing"] or {}
        summary_out["partial"][level] = {
            "top1_mean": item["rank"]["top1"]["mean"],
            "missing_top1_mean": (missing.get("top1") or {}).get("mean"),
            "missing_coverage": {str(alpha): ((missing.get(f"region_{alpha}") or {})
                                              .get("coverage") or {}).get("mean")
                                 for alpha in args.alphas},
            "missing_size_mean": {str(alpha): ((missing.get(f"region_{alpha}") or {})
                                               .get("size_mean") or {}).get("mean")
                                  for alpha in args.alphas},
        }
    print(json.dumps(summary_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
