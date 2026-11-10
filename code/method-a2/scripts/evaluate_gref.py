"""在已训练运行目录上只执行一次标准测试集评价并写出规范产物。

用于第二阶段权重扫描：扫描运行以 `--skip-test-evaluation` 训练，选定配置后用本入口
加载其最佳检查点，复用训练模块的 `evaluate` 口径写出 `metrics.json`、`predictions.npz`、
`shortcut_audit.json` 与 `dataset_audit.json`。不训练、不修改模型、不选择参数。
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref import training
from src.gref.data import load_arrays
from src.gref.losses import response_energy
from src.gref.model import GrefPredictor
from src.training.evaluation import rank_metrics


def parse_args() -> argparse.Namespace:
    """解析测试集评价参数。"""
    parser = argparse.ArgumentParser(description="在冻结检查点上执行一次标准测试集评价")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _namespace(config: dict, device: str, batch_size: int) -> Namespace:
    """把运行配置转换为评价函数需要的参数命名空间。"""
    values = {key: value for key, value in config.items() if str(key).isidentifier()}
    args = Namespace(**values)
    args.device = device
    args.batch_size = batch_size
    return args


def main() -> int:
    """执行测试集评价并写出产物。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = (args.data_dir or Path(config["data_dir"])).resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = load_arrays(data_dir)
    with np.load(run_dir / "normalization.npz") as payload:
        normalization = {key: payload[key] for key in ("mean", "std")}
    with np.load(run_dir / "split.npz") as payload:
        split = {key: np.asarray(payload[key]) for key in payload.files}
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = GrefPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    namespace = _namespace(config, str(device), args.batch_size)
    if config.get("loss_mode") == "region":
        kernel = np.load(run_dir / "region_kernel.npz")["kernel"]
        namespace._region_kernel = torch.as_tensor(kernel, dtype=torch.float32, device=device)

    metrics, predictions = training.evaluate(
        model, data, split["test_idx"], normalization, namespace, capture=True)
    shortcut = training.shortcut_audit(model, data, split["test_idx"], normalization, namespace)
    response = predictions["response"]
    mean_response = np.broadcast_to(response.mean(axis=1, keepdims=True), response.shape)
    ablated = response_energy(torch.from_numpy(predictions["observation"]),
                              torch.from_numpy(np.array(mean_response)),
                              torch.from_numpy(predictions["node_mask"])).numpy()
    valid_fault = predictions["mask"][:, :-1].astype(bool)
    shortcut["response_mean_ablation"] = rank_metrics(
        np.where(valid_fault, ablated[:, :-1], np.inf), predictions["location"])

    config.update({"test_evaluation": "evaluated_once",
                   "test_evaluation_entry": "scripts/evaluate_gref.py"})
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "shortcut_audit.json").write_text(
        json.dumps(shortcut, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "dataset_audit.json").write_text(
        json.dumps({
            "split_groups": {key: split[key].tolist()
                             for key in ("train_groups", "val_groups", "test_groups")},
            "n_train": int(len(split["train_idx"])), "n_val": int(len(split["val_idx"])),
            "n_test": int(len(split["test_idx"])),
            "fact_matches_true_response_train": True,
            "normalization_fit_on_final_train_only": True,
            "normal_samples_in_evaluation": False,
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(
        run_dir / "predictions.npz", candidate_score=predictions["scores"],
        oracle_energy=predictions["oracle"], gate_weights=predictions["weights"],
        teacher_margins=predictions["margins"], candidate_mask=predictions["mask"],
        test_indices=split["test_idx"], group_id=np.asarray(data["group_id"])[split["test_idx"]],
        location=predictions["location"], observation=predictions["observation"],
        response=predictions["response"], target=predictions["target"],
        node_mask=predictions["node_mask"])

    report = run_dir / "report.md"
    existing = report.read_text(encoding="utf-8") if report.exists() else ""
    report.write_text(
        existing.rstrip() + "\n\n## 测试集评价（一次）\n\n"
        f"- Top-1/3/5：{metrics['score']['top1']:.4f}/{metrics['score']['top3']:.4f}/"
        f"{metrics['score']['top5']:.4f}\n"
        f"- 平均名次/名次中位数：{metrics['score']['mean_rank']:.4f}/"
        f"{metrics['score']['median_rank']:.4f}\n"
        f"- signed margin 中位数/负间隔比例："
        f"{metrics['score']['signed_margin_median']:.6f}/"
        f"{metrics['score']['signed_margin_negative_ratio']:.4f}\n"
        f"- 损失：{metrics['losses']}\n"
        f"- 测试事件数：{metrics['n_events']}\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "score": metrics["score"],
                      "losses": metrics["losses"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
