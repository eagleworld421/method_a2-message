"""在不改动旧代码的前提下重训并分析旧直接评分路径的损失消融。"""

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

from src.data_generation.dataset_builder import load_dataset
from src.loss_ablation.analysis import gradient_report
from src.model.losses import candidate_classification_loss, masked_response_loss, pairwise_candidate_ranking_loss
from src.model.marker_response_predictor import MarkerResponsePredictor
from src.training.evaluation import rank_metrics


WEIGHTS = {"loc": 1.0, "rank": 0.1, "response": 0.1}
CONFIGS = {
    "full": ("loc", "rank", "response"),
    "without_loc": ("rank", "response"),
    "without_rank": ("loc", "response"),
    "without_response": ("loc", "rank"),
}


def set_seed(seed: int) -> None:
    """固定训练随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(data: dict, indices: np.ndarray, device: torch.device, n_fault: int) -> dict[str, torch.Tensor]:
    """构造与旧训练入口相同语义的批次。"""
    mask = torch.as_tensor(data["candidate_mask"][indices], dtype=torch.float32, device=device)
    fault_mask = mask.clone()
    fault_mask[:, n_fault:] = 0.0
    return {
        "observation": torch.as_tensor(data["X_obs"][indices], dtype=torch.float32, device=device),
        "target": torch.as_tensor(data["signature_bank"][indices], dtype=torch.float32, device=device),
        "candidate_mask": mask,
        "fault_mask": fault_mask,
        "location": torch.as_tensor(data["y_loc"][indices], dtype=torch.long, device=device),
        "edge_index": torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(data["edge_attr"], dtype=torch.float32, device=device),
        "edge_mask": torch.as_tensor(data["edge_mask"][indices], dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(data["mask"][indices], dtype=torch.float32, device=device),
    }


def forward_losses(model: MarkerResponsePredictor, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """计算旧直接评分路径的三项独立损失。"""
    output = model(
        batch["observation"], batch["edge_index"], batch["edge_attr"],
        edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
        candidate_mask=batch["candidate_mask"],
    )
    losses = {
        "loc": candidate_classification_loss(output["candidate_score"], batch["location"], batch["fault_mask"]),
        "rank": pairwise_candidate_ranking_loss(output["candidate_score"], batch["location"], batch["fault_mask"], margin=0.1, temperature=1.0),
        "response": masked_response_loss(output["response"], batch["target"], batch["candidate_mask"]),
    }
    return output, losses


def evaluate(model: MarkerResponsePredictor, data: dict, indices: np.ndarray, device: torch.device, n_fault: int) -> dict:
    """评估定位 Top-K 和三项损失。"""
    model.eval()
    batch = make_batch(data, indices, device, n_fault)
    with torch.no_grad():
        output, losses = forward_losses(model, batch)
    scores = output["candidate_score"].cpu().numpy()[:, :n_fault]
    locations = data["y_loc"][indices]
    result = {name: float(value.cpu()) for name, value in losses.items()}
    result.update(rank_metrics(scores, locations))
    return result


def main() -> None:
    """运行一个损失消融配置并保存梯度统计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGS), default="full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_dataset(Path(args.data_dir))
    n_nodes = int(data["X_obs"].shape[1])
    n_fault = n_nodes
    model = MarkerResponsePredictor(
        n_nodes=n_nodes,
        time_steps=int(data["X_obs"].shape[2]),
        feature_dim=int(data["X_obs"].shape[3]),
        pooling="mean", temporal_mode="single", temporal_layers=3, encoder_layers=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    active = set(CONFIGS[args.config])
    active_weights = {name: WEIGHTS[name] if name in active else 0.0 for name in WEIGHTS}
    gradient_epochs = {1, max(1, args.epochs // 2), args.epochs}
    gradient_history = []
    history = []
    best_state = None
    best_val = -1.0
    best_epoch = 0
    train_idx = data["train_idx"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        values = []
        for start in range(0, len(order), args.batch_size):
            batch = make_batch(data, order[start:start + args.batch_size], device, n_fault)
            optimizer.zero_grad(set_to_none=True)
            output, losses = forward_losses(model, batch)
            total = sum(WEIGHTS[name] * losses[name] for name in active)
            total.backward()
            optimizer.step()
            values.append({name: float(loss.detach().cpu()) for name, loss in losses.items()})
        history.append({"epoch": epoch, **{name: float(np.mean([row[name] for row in values])) for name in values[0]}})
        if epoch in gradient_epochs:
            model.eval()
            probe_indices = train_idx[: min(args.batch_size, len(train_idx))]
            probe_batch = make_batch(data, probe_indices, device, n_fault)
            probe_output, probe_losses = forward_losses(model, probe_batch)
            gradient_history.append({"epoch": epoch, **gradient_report(probe_losses, model, active_weights)})
        validation = evaluate(model, data, data["val_idx"], device, n_fault)
        if validation["top1"] > best_val or (validation["top1"] == best_val and validation["response"] < (best_state or {}).get("val_response", float("inf"))):
            best_val = validation["top1"]
            best_epoch = epoch
            best_state = {"state_dict": copy.deepcopy(model.state_dict()), "val_response": validation["response"]}
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])
    result = {
        "config": args.config,
        "active_losses": sorted(active),
        "weights": WEIGHTS,
        "active_weights": active_weights,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train": evaluate(model, data, data["train_idx"], device, n_fault),
        "val": evaluate(model, data, data["val_idx"], device, n_fault),
        "test": evaluate(model, data, data["test_idx"], device, n_fault),
        "history": history,
        "gradient_history": gradient_history,
    }
    output_dir = Path(args.output_root) / args.config
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, output_dir / "model.pt")


if __name__ == "__main__":
    main()
