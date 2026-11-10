"""在冻结 GReF 上评估 S1 边虚增，保持真实拓扑与观测拓扑分离。"""

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

from src.gref.data import load_arrays, make_batch
from src.gref.model import GrefPredictor
from src.training.evaluation import rank_metrics


def parse_args() -> argparse.Namespace:
    """解析边虚增评估参数。"""
    parser = argparse.ArgumentParser(description="评估 GReF 在 S1 边虚增下的定位表现")
    parser.add_argument("--run-dir", type=Path, required=True, help="冻结模型运行目录")
    parser.add_argument("--data-dir", type=Path, default=None, help="含候选边全集的数据目录")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-events", type=int, default=None,
                        help="仅评估前若干测试事件；缺省时评估完整测试集")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate_mask(model, data, indices, normalization, device, obs_mask, batch_size, n_fault):
    """在固定逐样本观测边掩码下计算定位指标。"""
    scores, locations = [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = make_batch(data, chunk, normalization, device, graph_view="universe")
        batch["edge_mask"] = torch.as_tensor(obs_mask[chunk], dtype=torch.float32, device=device)
        output = model(
            batch["observation"], batch["edge_index"], batch["edge_attr"],
            edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
            candidate_mask=batch["candidate_mask"],
        )
        valid = batch["candidate_mask"][:, :n_fault].bool()
        value = torch.where(valid, output["candidate_score"][:, :n_fault], torch.inf)
        scores.append(value.cpu().numpy())
        locations.append(batch["location"].cpu().numpy())
    return rank_metrics(np.concatenate(scores), np.concatenate(locations))


def main() -> int:
    """执行边虚增评估并写出可复核结果。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = (args.data_dir or Path(config["data_dir"])).resolve()
    output_dir = (args.output_dir or (run_dir / "s1-edge-universe-eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = load_arrays(data_dir)
    if data["edge_index_universe"].shape[0] <= data["edge_index"].shape[0]:
        raise ValueError("数据没有多于真实拓扑的候选边，无法评估边虚增")
    normalization_file = np.load(run_dir / "normalization.npz")
    normalization = {key: normalization_file[key] for key in ("mean", "std")}
    split = np.load(run_dir / "split.npz")
    test_idx = split["test_idx"]
    if args.max_test_events is not None:
        if args.max_test_events < 1:
            raise ValueError("max-test-events 必须为正整数")
        test_idx = test_idx[:args.max_test_events]
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = GrefPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    true_mask = np.asarray(data["edge_mask_true"], dtype=np.float32)
    base_obs = np.asarray(data["edge_mask_obs"], dtype=np.float32)
    pair_id = np.asarray(data["edge_pair_id"], dtype=np.int64)
    candidate_pairs = [
        int(pair) for pair in np.unique(pair_id)
        if np.all(true_mask[:, pair_id == pair] == 0)
    ]
    n_fault = data["X_obs"].shape[1]
    clean = evaluate_mask(model, data, test_idx, normalization, device, base_obs,
                          args.batch_size, n_fault)
    records = []
    for pair in candidate_pairs:
        obs = base_obs.copy()
        obs[:, pair_id == pair] = 1.0
        metrics = evaluate_mask(model, data, test_idx, normalization, device, obs,
                                args.batch_size, n_fault)
        records.append({
            "edge_pair_id": pair,
            "edge_rows": np.flatnonzero(pair_id == pair).tolist(),
            "edge_index": data["edge_index_universe"][pair_id == pair].tolist(),
            "edge_type": int(data["edge_type"][pair_id == pair][0]),
            "edge_attr_source": int(data["edge_attr_source"][pair_id == pair][0]),
            "metrics": metrics,
            "delta_top1": float(metrics["top1"] - clean["top1"]),
        })
    payload = {
        "run_dir": str(run_dir), "data_dir": str(data_dir),
        "n_test": int(len(test_idx)), "n_fault_candidates": int(n_fault),
        "n_universe_edges": int(len(pair_id)), "n_virtual_edge_pairs": len(candidate_pairs),
        "clean": clean, "virtual_edges": records,
    }
    (output_dir / "s1_edge_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# GReF S1 边虚增评估", "", f"- 数据目录：`{data_dir}`",
        f"- 测试事件数：{len(test_idx)}", f"- 虚增候选无向边数：{len(candidate_pairs)}",
        f"- 干净 Top-1/3/5：{clean['top1']:.4f}/{clean['top3']:.4f}/{clean['top5']:.4f}", "",
        "## 候选虚增边", "",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"- pair={record['edge_pair_id']}，Top-1/3/5="
            f"{metrics['top1']:.4f}/{metrics['top3']:.4f}/{metrics['top5']:.4f}，"
            f"相对干净 Top-1={record['delta_top1']:.4f}"
        )
    (output_dir / "s1_edge_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "n_virtual_edge_pairs": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
