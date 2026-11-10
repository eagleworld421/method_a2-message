"""RaReF 区域概率损失三项分量的梯度范数、两两余弦与权重反推分析。

只读取已训练的区域概率模型检查点、训练集划分与区域核，按固定种子重建训练批次，
在完整观测与训练期节点掩码两种输入分布下分别测量能量、排序、区域三项的加权梯度
范数、两两余弦、合成投影，并据此反推达到给定梯度份额所需的权重。不修改模型、
数据或任何既有运行产物。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.data import make_batch, sample_node_masks
from src.gref.losses import region_term_losses
from src.gref.model import GrefPredictor
from src.loss_ablation.analysis import gradient_report

MODULE_NAMES = {"shared": "编码器", "response": "解码器", "other": "其他"}
TERMS = ("energy", "rank", "region")


def parse_args(argv=None) -> argparse.Namespace:
    """解析梯度分析参数。"""
    parser = argparse.ArgumentParser(description="分析RaReF 区域概率损失三项的梯度与余弦")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", required=True, help="分析产物目录，必须为新目录")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=36)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _median(values: list) -> float | None:
    """过滤缺失值后返回中位数。"""
    usable = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.median(usable)) if usable else None


def _quantiles(values: list) -> list[float] | None:
    """过滤缺失值后返回五数概括。"""
    usable = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not usable:
        return None
    return [float(item) for item in np.quantile(usable, [0.0, 0.25, 0.5, 0.75, 1.0])]


def _pairwise_gram(group_report: dict, weights: dict[str, float]) -> dict:
    """由逐参数组的加权范数与余弦重建三项的无权 Gram 矩阵元素。"""
    norms = group_report["gradient_norm"]
    unweighted = {name: norms[name] / float(weights.get(name, 1.0)) for name in TERMS}
    dot = {(left, right): 0.0 for left in TERMS for right in TERMS}
    for left in TERMS:
        for right in TERMS:
            if left == right:
                dot[(left, right)] += unweighted[left] ** 2
                continue
            cosine = group_report["cosine"].get(f"{left}__{right}")
            if cosine is None:
                cosine = group_report["cosine"].get(f"{right}__{left}")
            if cosine is None:
                continue
            dot[(left, right)] += unweighted[left] * unweighted[right] * float(cosine)
    return dot


def _gram_from_records(records: list[dict], weights: dict[str, float]) -> np.ndarray:
    """把同一批次各参数组的无权点积汇总为三项 Gram 矩阵。"""
    gram = np.zeros((len(TERMS), len(TERMS)))
    for record in records:
        dot = _pairwise_gram(record, weights)
        for left_index, left in enumerate(TERMS):
            for right_index, right in enumerate(TERMS):
                gram[left_index, right_index] += dot[(left, right)]
    return gram


def _weights_for_norm_share(gram: np.ndarray, weights: dict[str, float],
                            target: float, free: str) -> float:
    """在其余权重固定时，反推使指定项占加权梯度范数份额 target 的权重。"""
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    index = {name: position for position, name in enumerate(TERMS)}
    fixed = sum(float(weights[name]) * norms[index[name]]
                for name in TERMS if name != free)
    denominator = max((1.0 - target) * norms[index[free]], 1e-30)
    return float(target * fixed / denominator)


def _projection_shares(gram: np.ndarray, weights: dict[str, float]) -> dict:
    """给定权重下的合成投影份额 P_i = λ_i (Gλ)_i / λ^T Gλ。"""
    vector = np.asarray([float(weights[name]) for name in TERMS])
    product = gram @ vector
    quadratic = float(vector @ product)
    if quadratic <= 1e-30:
        return {name: None for name in TERMS}
    return {name: float(vector[index] * product[index] / quadratic)
            for index, name in enumerate(TERMS)}


def _summarize(records: list[dict]) -> dict:
    """按输入分布与参数组汇总梯度范数、份额、余弦与投影。"""
    summary = {}
    for mode in sorted({record["mode"] for record in records}):
        subset = [record for record in records if record["mode"] == mode]
        mode_summary = {"term_values": {}, "groups": {}}
        for term in TERMS:
            mode_summary["term_values"][term] = _median(
                [record["term_values"][term] for record in subset])
        for group in sorted({record["group"] for record in subset}):
            group_records = [record for record in subset if record["group"] == group]
            metrics = {"n_batches": len(group_records)}
            for term in TERMS:
                metrics[f"gradient_norm_{term}"] = _median(
                    [record["gradient_norm"].get(term) for record in group_records])
                metrics[f"norm_share_{term}"] = _median(
                    [record["norm_share"].get(term) for record in group_records])
                metrics[f"projection_{term}"] = _median(
                    [record["projection"].get(term) for record in group_records])
            for pair in ("energy__rank", "energy__region", "rank__region"):
                values = [record["cosine"].get(pair) for record in group_records]
                metrics[f"cosine_{pair}"] = _median(values)
                metrics[f"cosine_{pair}_negative_ratio"] = (
                    float(np.mean([value < 0 for value in values
                                   if value is not None and np.isfinite(value)]))
                    if any(value is not None for value in values) else None)
                metrics[f"cosine_{pair}_quantiles"] = _quantiles(values)
            mode_summary["groups"][group] = metrics
        summary[mode] = mode_summary
    return summary


def _load_run(run_dir: Path, device: str) -> tuple[dict, dict, np.ndarray, GrefPredictor]:
    """读取运行配置、标准化、训练索引与检查点。"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("loss_mode") != "region":
        raise ValueError("该入口只分析RaReF 区域概率损失运行")
    with np.load(run_dir / "normalization.npz") as payload:
        normalization = {key: payload[key] for key in ("mean", "std")}
    with np.load(run_dir / "split.npz") as payload:
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    model = GrefPredictor(**config["model_config"]).to(device)
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return config, normalization, train_idx, model


def main(argv=None) -> int:
    """执行两种输入分布下的梯度分析并写出 JSON。"""
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    config, normalization, train_idx, model = _load_run(run_dir, args.device)
    data_dir = Path(args.data_dir or config["data_dir"]).resolve()
    data = {name: np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("X_obs", "signature_bank", "mask", "candidate_mask", "edge_index",
                         "edge_attr", "edge_mask", "group_id", "y_loc", "y_resist")}
    n_fault = int(data["X_obs"].shape[1])
    kernel = torch.as_tensor(np.load(run_dir / "region_kernel.npz")["kernel"],
                             dtype=torch.float32, device=args.device)
    weights = {name: float(config["loss_weights"][name]) for name in TERMS}
    temperature = float(config["rank_temperature"])
    beta_obs = float(config["region_kernel"]["beta_obs"])
    beta_miss = float(config["region_kernel"]["beta_miss"])
    mask_config = config["node_mask_augmentation"]

    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError("分析输出目录已存在，请使用新目录")
    output_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(train_idx)
    batches = [order[start:start + args.batch_size]
               for start in range(0, len(order), args.batch_size)][: args.max_batches]
    records, grams = [], {mode: [] for mode in ("full", "masked")}
    for mode in ("full", "masked"):
        model.eval()
        mask_rng = np.random.default_rng(args.seed + 2)
        for index, chunk in enumerate(batches):
            batch = make_batch(data, chunk, normalization, torch.device(args.device))
            if mode == "masked":
                masks = sample_node_masks(
                    batch["location"].detach().cpu().numpy(), n_fault, mask_rng,
                    full_probability=float(mask_config["full_probability"]),
                    min_observed=int(mask_config["min_observed"]),
                    max_observed=int(mask_config["max_observed"]),
                    force_observed_probability=float(mask_config["force_observed_probability"]),
                )
                batch["node_mask"] = torch.as_tensor(
                    masks, dtype=torch.float32, device=args.device)
            output = model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                           edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                           candidate_mask=batch["candidate_mask"])
            losses = region_term_losses(
                output, batch["observation"], batch["target"], batch["candidate_mask"],
                batch["location"], kernel, rank_temperature=temperature,
                lambda_energy=weights["energy"], lambda_rank=weights["rank"],
                lambda_region=weights["region"], beta_obs=beta_obs,
                beta_miss=beta_miss, mask=batch["node_mask"])
            terms = {name: losses[name] for name in TERMS}
            report = gradient_report(terms, model, weights)
            event = {
                "mode": mode, "batch": index, "n_events": int(len(chunk)),
                "term_values": {name: float(terms[name].detach()) for name in TERMS},
            }
            for group, values in report["groups"].items():
                record = dict(event)
                record["group"] = group
                record["group_name"] = MODULE_NAMES.get(group, group)
                for key in ("parameter_count", "gradient_norm", "norm_share",
                            "total_gradient_norm", "cosine", "projection"):
                    record[key] = values.get(key)
                records.append(record)
            grams[mode].append(_gram_from_records(records[-len(report["groups"]):], weights))

    summary = _summarize(records)
    derived = {}
    for mode in ("full", "masked"):
        gram = np.mean(np.stack(grams[mode]), axis=0)
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        targets = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        region_grid = [_weights_for_norm_share(gram, weights, target, "region")
                       for target in targets]
        rank_grid = [_weights_for_norm_share(gram, weights, target, "rank")
                     for target in targets]
        scan = []
        for region_weight in sorted(set([0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 1.0, 2.0, 4.0]
                                        + [round(value, 6) for value in region_grid])):
            candidate = dict(weights, region=region_weight)
            scan.append({"lambda_region": region_weight,
                         "norm_share_region": float(
                             region_weight * norms[2] /
                             sum(candidate[name] * norms[index]
                                 for index, name in enumerate(TERMS))),
                         "projection": _projection_shares(gram, candidate)})
        derived[mode] = {
            "gram": gram.tolist(),
            "unweighted_norm": {name: float(norms[index])
                                for index, name in enumerate(TERMS)},
            "current": {
                "norm_share": {name: float(weights[name] * norms[index] /
                                           sum(weights[other] * norms[position]
                                               for position, other in enumerate(TERMS)))
                               for index, name in enumerate(TERMS)},
                "projection": _projection_shares(gram, weights),
            },
            "lambda_region_for_norm_share": dict(zip(targets, region_grid)),
            "lambda_rank_for_norm_share": dict(zip(targets, rank_grid)),
            "region_weight_scan": scan,
        }
    result = {
        "run_dir": str(run_dir), "data_dir": str(data_dir),
        "loss_weights": weights, "rank_temperature": temperature,
        "beta_obs": beta_obs, "beta_miss": beta_miss,
        "n_batches": len(batches), "batch_size": args.batch_size, "seed": args.seed,
        "summary": summary, "derived": derived, "batches": records,
    }
    (output_dir / "region_gradient_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    report = {"run_dir": str(run_dir), "output": str(output_dir)}
    for mode in ("full", "masked"):
        item = derived[mode]
        report[mode] = {
            "unweighted_norm": {name: round(value, 6) for name, value in item["unweighted_norm"].items()},
            "current_norm_share": {name: round(value, 4) for name, value in item["current"]["norm_share"].items()},
            "current_projection": {name: (None if value is None else round(value, 4))
                                   for name, value in item["current"]["projection"].items()},
            "lambda_region_norm_share": {str(key): round(value, 4)
                                         for key, value in item["lambda_region_for_norm_share"].items()},
            "lambda_rank_norm_share": {str(key): round(value, 4)
                                       for key, value in item["lambda_rank_for_norm_share"].items()},
            "cosines": {
                group: {pair: (None if values[f"cosine_{pair}"] is None
                               else round(values[f"cosine_{pair}"], 4))
                        for pair in ("energy__rank", "energy__region", "rank__region")}
                for group, values in summary[mode]["groups"].items()
            },
        }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
