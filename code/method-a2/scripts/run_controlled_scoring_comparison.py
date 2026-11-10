"""执行旧评分监督与反事实评分监督的受控对比实验。"""

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

from src.counterfactual_scoring.controlled_losses import CONFIGS, WEIGHTS, compute_controlled_losses
from src.counterfactual_scoring.evaluation import distance_gap_metrics
from src.counterfactual_scoring.gradient import gradient_report
from src.counterfactual_scoring.losses import true_distances
from src.counterfactual_scoring.model import CounterfactualScoringModel
from src.data_generation.dataset_builder import load_dataset


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(data: dict, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """从已有数据构造受控实验批次。"""
    return {
        "observation": torch.as_tensor(data["X_obs"][indices], dtype=torch.float32, device=device),
        "target": torch.as_tensor(data["signature_bank"][indices], dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(data["candidate_mask"][indices], dtype=torch.float32, device=device),
        "location": torch.as_tensor(data["y_loc"][indices], dtype=torch.long, device=device),
        "edge_index": torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(data["edge_attr"], dtype=torch.float32, device=device),
        "edge_mask": torch.as_tensor(data["edge_mask"][indices], dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(data["mask"][indices], dtype=torch.float32, device=device),
    }


def evaluate(model, data, indices, device, distance_scale, gap_threshold, config, temperature, n_fault):
    """评估损失、物理距离结构和定位 Top-K。"""
    model.eval()
    with torch.no_grad():
        batch = make_batch(data, indices, device)
        output = model(batch)
        losses = compute_controlled_losses(output, batch, config, distance_scale, gap_threshold, temperature)
        scores = output["candidate_score"].cpu().numpy()[:, :n_fault]
        true = losses["true_distances"].cpu().numpy()[:, :n_fault]
        predicted = true_distances(batch["observation"], output["response"]).cpu().numpy()[:, :n_fault]
    result = {name: float(losses[name].cpu()) for name in (*WEIGHTS, "total")}
    result.update(distance_gap_metrics(scores, true, predicted, n_fault))
    locations = data["y_loc"][indices]
    valid = locations >= 0
    if np.any(valid):
        order = np.argsort(scores[valid], axis=1, kind="stable")
        ranks = (order == locations[valid, None]).nonzero()[1] + 1
        result.update({
            "top1": float(np.mean(ranks <= 1)),
            "top3": float(np.mean(ranks <= 3)),
            "top5": float(np.mean(ranks <= 5)),
            "mean_rank": float(np.mean(ranks)),
            "margin_median": float(np.median(np.sort(scores[valid], axis=1)[:, 1] - np.sort(scores[valid], axis=1)[:, 0])),
        })
    else:
        result.update({"top1": None, "top3": None, "top5": None, "mean_rank": None, "margin_median": None})
    return result


def save_predictions(model, data, indices, device, output_path, n_fault):
    """保存逐事件评分、真实距离、预测距离和候选排序。"""
    model.eval()
    with torch.no_grad():
        batch = make_batch(data, indices, device)
        output = model(batch)
        scores = output["candidate_score"].cpu().numpy()[:, :n_fault]
        true = true_distances(batch["observation"], batch["target"]).cpu().numpy()[:, :n_fault]
        predicted = true_distances(batch["observation"], output["response"]).cpu().numpy()[:, :n_fault]
    np.savez_compressed(output_path, scores=scores, true_distances=true, predicted_distances=predicted,
                        locations=data["y_loc"][indices], score_order=np.argsort(scores, axis=1, kind="stable"),
                        true_order=np.argsort(true, axis=1, kind="stable"))


def main() -> None:
    """运行一个配置和一个种子的受控实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
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
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    with torch.no_grad():
        train_batch = make_batch(data, train_idx, device)
        train_distances = true_distances(train_batch["observation"], train_batch["target"])[:, :n_fault]
    distance_scale = float(torch.median(train_distances[train_distances > 0]).item())
    sorted_distances = torch.sort(train_distances, dim=1).values
    gap_threshold = float(torch.quantile(torch.diff(sorted_distances, dim=1).abs().reshape(-1), 0.25).item()) / max(distance_scale, 1e-8)
    model = CounterfactualScoringModel(n_fault, int(data["X_obs"].shape[2]), int(data["X_obs"].shape[3])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    active_weights = {name: WEIGHTS[name] if name in CONFIGS[args.config] else 0.0 for name in WEIGHTS}
    gradient_epochs = {1, max(1, args.epochs // 2), args.epochs}
    history, gradient_history = [], []
    best_state, best_key, best_epoch = None, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        epoch_values = []
        for start in range(0, len(order), args.batch_size):
            batch = make_batch(data, order[start:start + args.batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_controlled_losses(output, batch, args.config, distance_scale, gap_threshold, args.temperature)
            losses["total"].backward()
            optimizer.step()
            epoch_values.append({name: float(losses[name].detach().cpu()) for name in (*WEIGHTS, "total")})
        history.append({"epoch": epoch, **{name: float(np.mean([row[name] for row in epoch_values])) for name in epoch_values[0]}})
        if epoch in gradient_epochs:
            model.eval()
            probe_batch = make_batch(data, train_idx[:min(args.batch_size, len(train_idx))], device)
            probe_output = model(probe_batch)
            probe_losses = compute_controlled_losses(probe_output, probe_batch, args.config, distance_scale, gap_threshold, args.temperature)
            gradient_history.append({"epoch": epoch, **gradient_report({name: probe_losses[name] for name in WEIGHTS}, model, active_weights)})
        validation = evaluate(model, data, val_idx, device, distance_scale, gap_threshold, args.config, args.temperature, n_fault)
        key = (validation["top1"], validation["top3"], -validation["family"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    output_dir = Path(args.output_root) / args.config / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": args.config, "seed": args.seed, "epochs": args.epochs,
        "weights": WEIGHTS, "active_weights": active_weights,
        "distance_scale": distance_scale, "gap_threshold": gap_threshold,
        "best_epoch": best_epoch,
        "train": evaluate(model, data, train_idx, device, distance_scale, gap_threshold, args.config, args.temperature, n_fault),
        "val": evaluate(model, data, val_idx, device, distance_scale, gap_threshold, args.config, args.temperature, n_fault),
        "test": evaluate(model, data, test_idx, device, distance_scale, gap_threshold, args.config, args.temperature, n_fault),
        "history": history, "gradient_history": gradient_history,
    }
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "gradient_history.npz", payload=np.asarray([json.dumps(row) for row in gradient_history], dtype=object))
    save_predictions(model, data, test_idx, device, output_dir / "prediction_results.npz", n_fault)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "distance_scale": distance_scale, "gap_threshold": gap_threshold}, output_dir / "model.pt")


if __name__ == "__main__":
    main()
