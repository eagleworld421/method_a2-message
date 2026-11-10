"""训练并行的真实反事实监督候选评分实验。"""

from __future__ import annotations

import argparse
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
from src.counterfactual_scoring.losses import compute_losses, true_distances
from src.counterfactual_scoring.model import CounterfactualScoringModel
from src.data_generation.dataset_builder import load_dataset


def set_seed(seed: int) -> None:
    """固定实验随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(data: dict, indices: np.ndarray, device: torch.device, n_fault: int) -> dict[str, torch.Tensor]:
    """从已有数据构造新路径的只读批次。"""
    candidate_mask = torch.as_tensor(data["candidate_mask"][indices], dtype=torch.float32, device=device)
    return {
        "observation": torch.as_tensor(data["X_obs"][indices], dtype=torch.float32, device=device),
        "target": torch.as_tensor(data["signature_bank"][indices], dtype=torch.float32, device=device),
        "candidate_mask": candidate_mask,
        "edge_index": torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(data["edge_attr"], dtype=torch.float32, device=device),
        "edge_mask": torch.as_tensor(data["edge_mask"][indices], dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(data["mask"][indices], dtype=torch.float32, device=device),
    }


def evaluate(model: CounterfactualScoringModel, data: dict, indices: np.ndarray, device: torch.device,
             distance_scale: float, gap_threshold: float, temperature: float, ablation: str, n_fault: int) -> dict:
    """计算一个数据划分上的损失和距离间距指标。"""
    model.eval()
    with torch.no_grad():
        batch = make_batch(data, indices, device, n_fault)
        output = model(batch)
        losses = compute_losses(output, batch, ablation, distance_scale, gap_threshold, temperature)
        true = losses["true_distances"].cpu().numpy()
        predicted = true_distances(batch["observation"], output["response"]).cpu().numpy()
        metrics = {name: float(value.cpu()) for name, value in losses.items() if name != "true_distances"}
        normalized = losses["true_distances"][:, :n_fault] / max(float(distance_scale), 1e-8)
        target_prob = torch.softmax(-normalized / temperature, dim=1)
        metrics["list_target_entropy"] = float((-(target_prob * target_prob.clamp_min(1e-12).log()).sum(dim=1)).mean().cpu())
        metrics.update(distance_gap_metrics(output["candidate_score"].cpu().numpy(), true, predicted, n_fault))
        ranked = np.argsort(output["candidate_score"].cpu().numpy()[:, :n_fault], axis=1, kind="stable")
        locations = data["y_loc"][indices]
        ranks = (ranked == locations[:, None]).nonzero()[1] + 1
        metrics["top1"] = float(np.mean(ranks <= 1))
        metrics["top3"] = float(np.mean(ranks <= 3))
        metrics["top5"] = float(np.mean(ranks <= 5))
        metrics["mean_rank"] = float(np.mean(ranks))
        return metrics


def main() -> None:
    """运行一个指定消融配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation", choices=("base", "value", "gap", "both"), default="base")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_dataset(Path(args.data_dir))
    n_nodes = int(data["X_obs"].shape[1])
    n_fault = n_nodes
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    train_batch = make_batch(data, train_idx, device, n_fault)
    with torch.no_grad():
        train_distances = true_distances(train_batch["observation"], train_batch["target"])[:, :n_fault]
    distance_scale = float(torch.median(train_distances[train_distances > 0]).item())
    sorted_distances = torch.sort(train_distances, dim=1).values
    gap_threshold = float(torch.quantile(torch.diff(sorted_distances, dim=1).abs().reshape(-1), 0.25).item()) / max(distance_scale, 1e-8)
    model = CounterfactualScoringModel(n_nodes, int(data["X_obs"].shape[2]), int(data["X_obs"].shape[3])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        epoch_values = []
        for start in range(0, len(order), args.batch_size):
            batch = make_batch(data, order[start:start + args.batch_size], device, n_fault)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_losses(output, batch, args.ablation, distance_scale, gap_threshold, args.temperature)
            losses["total"].backward()
            optimizer.step()
            epoch_values.append({name: float(value.detach().cpu()) for name, value in losses.items() if name != "true_distances"})
        history.append({"epoch": epoch + 1, **{name: float(np.mean([row[name] for row in epoch_values])) for name in epoch_values[0]}})
    result = {
        "ablation": args.ablation,
        "seed": args.seed,
        "distance_scale": distance_scale,
        "gap_threshold": gap_threshold,
        "train": evaluate(model, data, train_idx, device, distance_scale, gap_threshold, args.temperature, args.ablation, n_fault),
        "val": evaluate(model, data, val_idx, device, distance_scale, gap_threshold, args.temperature, args.ablation, n_fault),
        "test": evaluate(model, data, test_idx, device, distance_scale, gap_threshold, args.temperature, args.ablation, n_fault),
        "history": history,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "distance_scale": distance_scale, "gap_threshold": gap_threshold}, output_dir / "model.pt")


if __name__ == "__main__":
    main()
