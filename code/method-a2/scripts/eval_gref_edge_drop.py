"""在 IEEE37 GReF 模型上评估 S1-EN 无向边删除。

本脚本只做推理评估，不训练、不修改模型。它读取训练目录中的配置、标准化、
划分与模型权重，对测试集依次施加干净掩码、逐条单边删除与随机多边删除，
并输出 Top-K、平均名次与相对干净基线的变化。
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

from src.gref.data import build_undirected_edge_groups, load_arrays, make_batch
from src.gref.model import GrefPredictor
from src.training.evaluation import rank_metrics


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""
    parser = argparse.ArgumentParser(description="评估 GReF 在 IEEE37 上的 S1-EN 边删除")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="GReF 训练产物目录，需包含 model.pt、normalization.npz 与 split.npz")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="数据目录；缺省时读取 config.json 中的 data_dir")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="评估输出目录；缺省为 <run-dir>/s1en-eval")
    parser.add_argument("--random-masks", type=int, default=20,
                        help="每个删除条数随机采样的掩码数")
    parser.add_argument("--max-drop", type=int, default=3,
                        help="随机掩码的最大删除条数")
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_mask(edge_count: int, groups: list[np.ndarray], chosen: list[int]) -> np.ndarray:
    """由选中的无向边组构造双向边掩码。"""
    mask = np.ones(edge_count, dtype=np.float32)
    for index in chosen:
        mask[groups[index]] = 0.0
    return mask


def sample_masks(edge_count: int, groups: list[np.ndarray], count: int,
                 drop: int, rng: np.random.Generator) -> list[np.ndarray]:
    """采样若干条无向边的随机删除掩码，并去重。"""
    unique: dict[tuple[int, ...], np.ndarray] = {}
    attempts = 0
    while len(unique) < count and attempts < count * 50:
        attempts += 1
        chosen = tuple(sorted(rng.choice(len(groups), size=drop, replace=False).tolist()))
        if chosen not in unique:
            unique[chosen] = build_mask(edge_count, groups, list(chosen))
    return list(unique.values())


@torch.no_grad()
def evaluate_mask(
    model: GrefPredictor,
    data: dict,
    indices: np.ndarray,
    normalization: dict,
    device: torch.device,
    mask: np.ndarray,
    batch_size: int,
    n_fault: int,
) -> dict:
    """在固定边掩码下评估全部测试事件。"""
    scores_all, locations_all = [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = make_batch(data, chunk, normalization, device)
        batch["edge_mask"] = torch.as_tensor(
            np.repeat(mask[None, :], len(chunk), axis=0),
            dtype=torch.float32, device=device,
        )
        output = model(
            batch["observation"], batch["edge_index"], batch["edge_attr"],
            edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
            candidate_mask=batch["candidate_mask"],
        )
        scores = output["candidate_score"][:, :n_fault]
        valid = batch["candidate_mask"][:, :n_fault].bool()
        scores = torch.where(valid, scores, torch.inf)
        scores_all.append(scores.detach().cpu().numpy())
        locations_all.append(batch["location"].detach().cpu().numpy())
    return rank_metrics(np.concatenate(scores_all), np.concatenate(locations_all))


def summarize(records: list[dict], keys: tuple[str, ...]) -> dict:
    """对一组记录求均值、最小值与最大值。"""
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "min": float(np.min([record[key] for record in records])),
            "max": float(np.max([record[key] for record in records])),
        }
        for key in keys
    }


def main() -> int:
    """执行评估并写出 JSON 与 Markdown。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = args.data_dir or Path(config["data_dir"])
    output_dir = args.output_dir or (run_dir / "s1en-eval")
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = load_arrays(Path(data_dir))
    normalization = {key: np.load(run_dir / "normalization.npz")[key]
                     for key in ("mean", "std")}
    split = np.load(run_dir / "split.npz")
    test_idx = split["test_idx"]
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = GrefPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    groups = build_undirected_edge_groups(edge_index)
    edge_count = edge_index.shape[0]
    n_fault = data["X_obs"].shape[1]
    clean = np.ones(edge_count, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    clean_metrics = evaluate_mask(model, data, test_idx, normalization, device,
                                  clean, args.batch_size, n_fault)
    single = []
    for group_index in range(len(groups)):
        mask = build_mask(edge_count, groups, [group_index])
        metrics = evaluate_mask(model, data, test_idx, normalization, device,
                                mask, args.batch_size, n_fault)
        single.append({
            "edge_group": group_index,
            "edge_rows": groups[group_index].tolist(),
            "top1": metrics["top1"],
            "top3": metrics["top3"],
            "top5": metrics["top5"],
            "mean_rank": metrics["mean_rank"],
            "delta_top1": metrics["top1"] - clean_metrics["top1"],
        })

    random_records = {}
    for drop in range(1, args.max_drop + 1):
        records = []
        for mask_index, mask in enumerate(
            sample_masks(edge_count, groups, args.random_masks, drop, rng)
        ):
            metrics = evaluate_mask(model, data, test_idx, normalization, device,
                                    mask, args.batch_size, n_fault)
            metrics = dict(metrics)
            metrics.update({"mask_index": mask_index, "drop": drop,
                            "delta_top1": metrics["top1"] - clean_metrics["top1"]})
            records.append(metrics)
        random_records[str(drop)] = {
            "records": records,
            "summary": summarize(records, ("top1", "top3", "top5", "mean_rank", "delta_top1")),
        }

    payload = {
        "run_dir": str(run_dir),
        "data_dir": str(Path(data_dir).resolve()),
        "model": {"best_epoch": checkpoint.get("best_epoch"),
                  "loss_mode": checkpoint.get("loss_mode")},
        "n_test": int(len(test_idx)),
        "n_fault_candidates": int(n_fault),
        "n_undirected_edges": int(len(groups)),
        "clean": clean_metrics,
        "single_edge": {
            "records": single,
            "summary": summarize(single, ("top1", "top3", "top5", "mean_rank", "delta_top1")),
        },
        "random": random_records,
    }
    (output_dir / "s1en_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# GReF S1-EN 边删除评估",
        "",
        f"- 运行目录：`{run_dir}`",
        f"- 测试事件数：{len(test_idx)}",
        f"- 故障候选数：{n_fault}",
        f"- 无向边数：{len(groups)}",
        f"- 干净测试 Top-1/3/5：{clean_metrics['top1']:.4f}/"
        f"{clean_metrics['top3']:.4f}/{clean_metrics['top5']:.4f}",
        "",
        "## 单边删除",
        "",
        f"- 平均 Top-1/3/5：{payload['single_edge']['summary']['top1']['mean']:.4f}/"
        f"{payload['single_edge']['summary']['top3']['mean']:.4f}/"
        f"{payload['single_edge']['summary']['top5']['mean']:.4f}",
        f"- Top-1 最小值/最大值：{payload['single_edge']['summary']['top1']['min']:.4f}/"
        f"{payload['single_edge']['summary']['top1']['max']:.4f}",
        f"- 相对干净 Top-1 平均变化：{payload['single_edge']['summary']['delta_top1']['mean']:.4f}",
        "",
        "## 随机多边删除",
        "",
    ]
    for drop in range(1, args.max_drop + 1):
        summary = random_records[str(drop)]["summary"]
        lines.append(
            f"- 删除 {drop} 条：平均 Top-1/3/5 为 "
            f"{summary['top1']['mean']:.4f}/{summary['top3']['mean']:.4f}/"
            f"{summary['top5']['mean']:.4f}，Top-1 最小值 {summary['top1']['min']:.4f}，"
            f"相对干净平均变化 {summary['delta_top1']['mean']:.4f}"
        )
    lines.extend(["", "本评估只覆盖边删除，不覆盖边虚增、开关假闭合或相别错误。"])
    (output_dir / "s1en_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "clean_top1": clean_metrics["top1"],
        "single_mean_top1": payload["single_edge"]["summary"]["top1"]["mean"],
        "random_summary": {drop: random_records[str(drop)]["summary"]["top1"]
                           for drop in range(1, args.max_drop + 1)},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
