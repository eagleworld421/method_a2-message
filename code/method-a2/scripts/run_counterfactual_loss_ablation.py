"""运行反事实监督候选评分的损失消融与梯度分析。"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.counterfactual_scoring.evaluation import distance_gap_metrics
from src.counterfactual_scoring.gradient import gradient_report
from src.counterfactual_scoring.losses import compute_losses, true_distances
from src.counterfactual_scoring.model import CounterfactualScoringModel
from src.data_generation.dataset_builder import load_dataset


WEIGHTS = {"family": 0.1, "list": 1.0, "value": 1.0, "gap": 1.0}
CONFIGS = {
    "base": ("family", "list"),
    "value": ("family", "list", "value"),
    "gap": ("family", "list", "gap"),
    "both": ("family", "list", "value", "gap"),
}


def set_seed(seed: int) -> None:
    """固定实验随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(data: dict, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """从已有数据构造反事实评分批次。"""
    return {
        "observation": torch.as_tensor(data["X_obs"][indices], dtype=torch.float32, device=device),
        "target": torch.as_tensor(data["signature_bank"][indices], dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(data["candidate_mask"][indices], dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(data["edge_attr"], dtype=torch.float32, device=device),
        "edge_mask": torch.as_tensor(data["edge_mask"][indices], dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(data["mask"][indices], dtype=torch.float32, device=device),
    }


def evaluate(model, data, indices, device, distance_scale, gap_threshold, temperature, ablation, n_fault):
    """计算损失、距离一致性和定位 Top-K。"""
    model.eval()
    with torch.no_grad():
        batch = make_batch(data, indices, device)
        output = model(batch)
        losses = compute_losses(output, batch, ablation, distance_scale, gap_threshold, temperature)
        true = losses["true_distances"].cpu().numpy()
        predicted = true_distances(batch["observation"], output["response"]).cpu().numpy()
        result = {name: float(value.cpu()) for name, value in losses.items() if name != "true_distances"}
        result.update(distance_gap_metrics(output["candidate_score"].cpu().numpy(), true, predicted, n_fault))
        scores = output["candidate_score"].cpu().numpy()[:, :n_fault]
        locations = data["y_loc"][indices]
        valid = locations >= 0
        if np.any(valid):
            ranks = (np.argsort(scores[valid], axis=1, kind="stable") == locations[valid, None]).nonzero()[1] + 1
            result["top1"] = float(np.mean(ranks <= 1))
            result["top3"] = float(np.mean(ranks <= 3))
            result["top5"] = float(np.mean(ranks <= 5))
            result["mean_rank"] = float(np.mean(ranks))
        else:
            result.update({"top1": None, "top3": None, "top5": None, "mean_rank": None})
        return result


def main() -> None:
    """运行一个反事实监督消融配置并保存结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGS), default="base")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_dataset(Path(args.data_dir))
    n_fault = int(data["X_obs"].shape[1])
    train_idx = data["train_idx"]
    with torch.no_grad():
        train_batch = make_batch(data, train_idx, device)
        train_distances = true_distances(train_batch["observation"], train_batch["target"])[:, :n_fault]
    distance_scale = float(torch.median(train_distances[train_distances > 0]).item())
    sorted_distances = torch.sort(train_distances, dim=1).values
    gap_threshold = float(torch.quantile(torch.diff(sorted_distances, dim=1).abs().reshape(-1), 0.25).item()) / max(distance_scale, 1e-8)
    model = CounterfactualScoringModel(n_fault, int(data["X_obs"].shape[2]), int(data["X_obs"].shape[3])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    active = set(CONFIGS[args.config])
    active_weights = {name: WEIGHTS[name] if name in active else 0.0 for name in WEIGHTS}
    gradient_epochs = {1, max(1, args.epochs // 2), args.epochs}
    history, gradient_history = [], []
    best_state, best_val, best_epoch = None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        values = []
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = make_batch(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_losses(output, batch, args.config, distance_scale, gap_threshold, args.temperature)
            losses["total"].backward()
            optimizer.step()
            values.append({name: float(value.detach().cpu()) for name, value in losses.items() if name != "true_distances"})
        history.append({"epoch": epoch, **{name: float(np.mean([row[name] for row in values])) for name in values[0]}})
        if epoch in gradient_epochs:
            model.eval()
            probe_indices = train_idx[:min(args.batch_size, len(train_idx))]
            probe_batch = make_batch(data, probe_indices, device)
            probe_output = model(probe_batch)
            probe_losses = compute_losses(probe_output, probe_batch, args.config, distance_scale, gap_threshold, args.temperature)
            gradient_history.append({"epoch": epoch, **gradient_report({name: probe_losses[name] for name in WEIGHTS}, model, active_weights)})
        validation = evaluate(model, data, data["val_idx"], device, distance_scale, gap_threshold, args.temperature, args.config, n_fault)
        if validation["top1"] > best_val:
            best_val, best_epoch = validation["top1"], epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "config": args.config,
        "active_losses": sorted(active),
        "weights": WEIGHTS,
        "active_weights": active_weights,
        "seed": args.seed,
        "epochs": args.epochs,
        "distance_scale": distance_scale,
        "gap_threshold": gap_threshold,
        "best_epoch": best_epoch,
        "train": evaluate(model, data, train_idx, device, distance_scale, gap_threshold, args.temperature, args.config, n_fault),
        "val": evaluate(model, data, data["val_idx"], device, distance_scale, gap_threshold, args.temperature, args.config, n_fault),
        "test": evaluate(model, data, data["test_idx"], device, distance_scale, gap_threshold, args.temperature, args.config, n_fault),
        "history": history,
        "gradient_history": gradient_history,
    }
    output_dir = Path(args.output_root) / args.config
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "distance_scale": distance_scale, "gap_threshold": gap_threshold}, output_dir / "model.pt")


if __name__ == "__main__":
    main()

