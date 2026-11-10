"""GRReF 的梯度范数、两两余弦与误差图频分解分析。

本脚本只读取已训练的最佳检查点与训练集划分，按固定种子构建混合批次，分别
在关闭与启用 Dropout 两种模式下测量能量、排序、图平滑三项的加权梯度，输出
逐批记录与汇总统计。它不修改模型、数据或其他运行产物。
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

from src.gref.data import make_batch
from src.gref.losses import three_term_losses, unique_undirected_edges
from src.gref.model import GrefPredictor
from src.loss_ablation.analysis import gradient_report

MODULE_NAMES = {"shared": "编码器", "response": "解码器", "other": "其他"}


def parse_args(argv=None) -> argparse.Namespace:
    """解析分析入口参数；运行目录、数据目录与输出目录必须显式给出。"""
    parser = argparse.ArgumentParser(description="分析 GRReF的梯度与余弦")
    parser.add_argument("--run-dir", required=True, help="包含 model.pt 与 config.json 的运行目录")
    parser.add_argument("--data-dir", default=None, help="缺省时读取运行配置中记录的数据目录")
    parser.add_argument("--output", required=True, help="分析产物目录，必须为新目录")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=36)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _write_json(path: Path, value) -> None:
    """写出严格 JSON，禁止非标准无穷值。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _median(values: list[float]) -> float | None:
    """过滤缺失值后返回中位数。"""
    usable = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(usable)) if usable else None


def _quantiles(values: list[float]) -> list[float] | None:
    """过滤缺失值后返回五数概括。"""
    usable = [value for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return None
    return [float(item) for item in np.quantile(usable, [0.0, 0.25, 0.5, 0.75, 1.0])]


def _load_run(run_dir: Path, device: str) -> tuple[dict, dict, np.ndarray, GrefPredictor]:
    """读取运行配置、训练侧归一化、训练索引与最佳检查点。"""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "normalization.npz") as payload:
        normalization = {key: payload[key] for key in ("mean", "std")}
    with np.load(run_dir / "split.npz") as payload:
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    model = GrefPredictor(**config["model_config"]).to(device)
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return config, normalization, train_idx, model


def _forward(model: GrefPredictor, batch: dict) -> dict:
    """只把部署可见字段送入模型。"""
    return model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                 edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                 candidate_mask=batch["candidate_mask"])


def _graph_spectrum(batch: dict, output: dict) -> dict:
    """计算当前批次的图频统计：Rayleigh 商、高频能量占比与常数模态占比。"""
    edge_index = batch["edge_index"].detach().cpu()
    edges, weights = unique_undirected_edges(edge_index)
    n_nodes = int(output["response"].shape[2])
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for (u, v), weight in zip(edges.tolist(), weights.tolist()):
        adjacency[u, v] = weight
        adjacency[v, u] = weight
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    field = (output["response"] - batch["target"]).detach().cpu().numpy()
    candidate = batch["candidate_mask"].detach().cpu().numpy().astype(bool)
    field = field[candidate]
    if field.size == 0:
        return {"eigenvalues": eigenvalues.tolist(), "n_candidates": 0}
    flat = field.reshape(field.shape[0], n_nodes, -1)
    coefficients = np.einsum("ni,bnd->bid", eigenvectors.T, flat)
    mode_energy = np.square(coefficients).sum(axis=2)
    total = mode_energy.sum(axis=1)
    positive = eigenvalues > 1e-12
    threshold = float(np.median(eigenvalues[positive])) if positive.any() else 0.0
    high = eigenvalues > threshold
    high_share = np.divide(
        mode_energy[:, high].sum(axis=1), total,
        out=np.zeros_like(total), where=total > 1e-12,
    )
    rayleigh = np.divide(
        (mode_energy * eigenvalues).sum(axis=1), total,
        out=np.zeros_like(total), where=total > 1e-12,
    )
    constant = coefficients[:, 0, :]
    common_share = np.divide(
        np.square(constant).sum(axis=1), total,
        out=np.zeros_like(total), where=total > 1e-12,
    )
    return {
        "eigenvalues": eigenvalues.tolist(),
        "high_threshold": threshold,
        "rayleigh_median": float(np.median(rayleigh)),
        "high_mode_share_median": float(np.median(high_share)),
        "common_mode_share_median": float(np.median(common_share)),
        "n_candidates": int(flat.shape[0]),
    }


def _summarize(records: list[dict]) -> dict:
    """按模式、模块与统计量汇总逐批结果的中位数与分位。"""
    summary = {}
    for mode in sorted({record["mode"] for record in records}):
        subset = [record for record in records if record["mode"] == mode]
        mode_summary = {}
        for group in sorted({record["group"] for record in subset}):
            group_records = [record for record in subset if record["group"] == group]
            metrics = {}
            for key in ("total_gradient_norm",):
                metrics[key] = _median([record[key] for record in group_records])
                metrics[f"{key}_quantiles"] = _quantiles([record[key] for record in group_records])
            for term in ("energy", "rank", "smooth"):
                values = [record["gradient_norm"].get(term) for record in group_records]
                metrics[f"gradient_norm_{term}"] = _median(values)
                metrics[f"gradient_norm_{term}_quantiles"] = _quantiles(values)
                shares = [record["norm_share"].get(term) for record in group_records]
                metrics[f"norm_share_{term}"] = _median(shares)
                projections = [record["projection"].get(term) for record in group_records]
                metrics[f"projection_{term}"] = _median(projections)
            for pair, values in _collect_pairs(group_records).items():
                metrics[f"cosine_{pair}"] = _median(values)
                metrics[f"cosine_{pair}_negative_ratio"] = float(
                    np.mean([value < 0 for value in values if value is not None])
                ) if any(value is not None for value in values) else None
                metrics[f"cosine_{pair}_quantiles"] = _quantiles(values)
            mode_summary[group] = metrics
        summary[mode] = mode_summary
    return summary


def _collect_pairs(records: list[dict]) -> dict[str, list[float | None]]:
    """从逐批余弦记录中收集每条损失对的取值。"""
    pairs: dict[str, list[float | None]] = {}
    for record in records:
        for pair, value in record["cosine"].items():
            pairs.setdefault(pair, []).append(value)
    return pairs


def main(argv=None) -> int:
    """执行两项 Dropout 模式下的梯度分析并写出 JSON 产物。"""
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    config, normalization, train_idx, model = _load_run(run_dir, args.device)
    data_dir = Path(args.data_dir or config["data_dir"]).resolve()
    data = {name: np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("X_obs", "signature_bank", "mask", "candidate_mask", "edge_index",
                         "edge_attr", "edge_mask", "group_id", "y_loc", "y_resist")}
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError("分析输出目录已存在，请使用新目录")
    output_dir.mkdir(parents=True)
    weights = config.get("loss_weights", {"energy": 1.0, "rank": 0.1, "smooth": 0.1})
    if not {"energy", "rank", "smooth"} <= set(weights):
        raise ValueError("运行配置不是三项损失设置，无法执行本分析")
    temperature = config.get("rank_temperature")
    if temperature is None:
        raise ValueError("运行配置缺少排序温度")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(train_idx)
    batches = [order[start:start + args.batch_size]
               for start in range(0, len(order), args.batch_size)][: args.max_batches]
    records, spectrum = [], []
    for mode in ("eval", "train"):
        model.train(mode == "train")
        for index, chunk in enumerate(batches):
            batch = make_batch(data, chunk, normalization, torch.device(args.device))
            output = _forward(model, batch)
            losses = three_term_losses(
                output, batch["observation"], batch["target"],
                batch["candidate_mask"], batch["location"], batch["edge_index"],
                rank_temperature=float(temperature), mask=batch["node_mask"],
            )
            terms = {name: losses[name] for name in ("energy", "rank", "smooth")}
            report = gradient_report(terms, model, weights)
            event = {
                "mode": mode, "batch": index, "n_events": int(len(chunk)),
                "term_values": {name: float(terms[name].detach()) for name in terms},
                "rank_events": int(losses["rank_events"]),
            }
            for group, values in report["groups"].items():
                record = dict(event)
                record["group"] = group
                record["group_name"] = MODULE_NAMES.get(group, group)
                for key in ("parameter_count", "gradient_norm", "gradient_rms", "norm_share",
                            "total_gradient_norm", "cosine", "projection"):
                    record[key] = values.get(key)
                records.append(record)
            if mode == "eval":
                spectrum.append(_graph_spectrum(batch, output))
    summary = _summarize(records)
    spectrum_summary = {
        "rayleigh_median": _median([item["rayleigh_median"] for item in spectrum if "rayleigh_median" in item]),
        "high_mode_share_median": _median([item["high_mode_share_median"] for item in spectrum if "high_mode_share_median" in item]),
        "common_mode_share_median": _median([item["common_mode_share_median"] for item in spectrum if "common_mode_share_median" in item]),
        "n_batches": len(spectrum),
        "eigenvalues": spectrum[0]["eigenvalues"] if spectrum else [],
    }
    result = {
        "run_dir": str(run_dir), "data_dir": str(data_dir),
        "loss_weights": weights, "rank_temperature": float(temperature),
        "n_batches": len(batches), "batch_size": args.batch_size,
        "seed": args.seed, "modules": MODULE_NAMES,
        "summary": summary, "graph_spectrum": spectrum_summary,
        "batches": records, "spectrum_batches": spectrum,
    }
    _write_json(output_dir / "gradient_analysis.json", result)
    print(f"已写出 {output_dir / 'gradient_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
