"""复核 GReF 最佳模型在完整训练批次上的加权梯度和响应误差结构。"""

from __future__ import annotations

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
from src.gref.losses import gref_losses
from src.gref.model import GrefPredictor
from src.loss_ablation.analysis import gradient_report
from src.progress import ProgressWriter


def _summary(values):
    """以批次为单位报告分布，缺失或近零梯度统计不强行归一化。"""
    values = np.array([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {"n_valid": 0}
    return {"n_valid": len(values), "mean": float(values.mean()),
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def _aggregate(records):
    """汇总每个模块的范数、方向投影与一阶局部作用。"""
    result = {}
    for module in records[0]["gradients"]["groups"]:
        groups = [row["gradients"]["groups"][module] for row in records]
        ratios = [g["gradient_norm"]["loc"] / g["gradient_norm"]["family"]
                  if g["gradient_norm"]["family"] > 1e-12 else None for g in groups]
        result[module] = {
            "loc_norm": _summary([g["gradient_norm"]["loc"] for g in groups]),
            "weighted_family_norm": _summary([g["gradient_norm"]["family"] for g in groups]),
            "loc_to_weighted_family_ratio": _summary(ratios),
            "loc_norm_share": _summary([g["norm_share"]["loc"] for g in groups]),
            "cosine": _summary([g["cosine"]["loc__family"] for g in groups]),
            "negative_cosine_fraction": float(np.mean([g["cosine"]["loc__family"] < 0
                                                        for g in groups if g["cosine"]["loc__family"] is not None])),
            "loc_projection": _summary([g["projection"]["loc"] for g in groups]),
            "family_projection": _summary([g["projection"]["family"] for g in groups]),
            "family_increase_under_negative_total_gradient_fraction": float(np.mean([
                g["projection"]["family"] < 0 for g in groups if g["projection"]["family"] is not None])),
        }
    return result


def _response_geometry(run_dir):
    """在已保存测试响应上分解候选共有误差与候选变化误差，不执行选参。"""
    with np.load(run_dir / "predictions.npz") as saved:
        prediction = saved["response"][:, :-1].astype(np.float64)
        target = saved["target"][:, :-1].astype(np.float64)
        observation = saved["observation"].astype(np.float64)
        if not saved["candidate_mask"][:, :-1].all() or not saved["node_mask"].all():
            raise ValueError("首轮误差分解仅适用于完整候选与完整观测")
        error = prediction - target
        offset = error.mean(axis=1, keepdims=True)
        total = float(np.mean(error ** 2))
        common = float(np.mean(offset ** 2))
        centered = float(np.mean((error - offset) ** 2))
        assert np.isclose(total, common + centered, rtol=1e-10, atol=1e-12)
        predicted_energy = np.mean((prediction - observation[:, None]) ** 2, axis=(2, 3, 4))
        teacher_energy = np.mean((target - observation[:, None]) ** 2, axis=(2, 3, 4))
        energy_error = predicted_energy - teacher_energy
        energy_common = energy_error.mean(axis=1, keepdims=True)
        return {
            "fault_candidates_only": True, "response_mse": total,
            "candidate_common_error_mse": common, "candidate_centered_error_mse": centered,
            "candidate_common_error_fraction": common / total,
            "energy_error_common_component_fraction": float(np.mean(energy_common ** 2) / np.mean(energy_error ** 2)),
            "huber_saturated_element_fraction": float(np.mean(np.abs(error) >= 1)),
            "limitation": "共有响应偏移不保证评分排序不变；只有评分的候选共有加性常数严格不改变排序和交叉熵。",
        }


def main():
    """对两个固定最佳检查点运行分析，不重新训练或修改模型参数。"""
    out = METHOD_DIR / "output/gref/gradient-analysis-20261103"
    out.mkdir(exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress = ProgressWriter(METHOD_DIR / "logs/gref-gradient-analysis-20261103/progress.jsonl",
                              "gref-gradient-analysis-20261103", terminal=False)
    summaries = {}
    for label in ("a", "b"):
        run_dir = METHOD_DIR / f"output/gref/ieee37-{label}-seed342-20261103"
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
        model = GrefPredictor(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}
        data = load_arrays(Path(config["data_dir"]))
        with np.load(run_dir / "normalization.npz") as saved:
            normalization = {name: saved[name] for name in saved.files}
        with np.load(run_dir / "split.npz") as saved:
            indices = np.random.default_rng(342).permutation(saved["train_idx"])
        modes = {}
        for mode in ("eval", "train_dropout"):
            model.train(mode == "train_dropout")
            records = []
            progress.stage_start(label + "_" + mode, len(indices))
            for start in range(0, len(indices), 64):
                torch.manual_seed(10000 + start)
                chunk = indices[start:start + 64]
                batch = make_batch(data, chunk, normalization, device)
                output = model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                               edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
                               candidate_mask=batch["candidate_mask"])
                losses = gref_losses(output, batch["observation"], batch["target"],
                                     batch["candidate_mask"], batch["location"],
                                     tolerance=config["energy_tolerance"], temperature=config["temperature"],
                                     lambda_family=config["lambda_family"], mask=batch["node_mask"])
                gradients = gradient_report({"loc": losses["loc"], "family": losses["family"]},
                                             model, {"loc": 1., "family": config["lambda_family"]})
                for group in gradients["groups"].values():
                    assert abs(sum(group["projection"].values()) - 1) < 2e-5
                records.append({"indices": chunk.tolist(), "fault_events": int(losses["fault_events"]),
                                "active_events": int(losses["active_events"]),
                                "loc": float(losses["loc"].detach()), "family": float(losses["family"].detach()),
                                "gradients": gradients})
                progress.update(label + "_" + mode, "rows", min(start + 64, len(indices)), len(indices))
                del output, losses, gradients
            modes[mode] = {"n_batches": len(records), "summary": _aggregate(records), "batches": records}
            progress.stage_done(label + "_" + mode, len(indices))
        assert all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items())
        result = {"checkpoint_epoch": checkpoint["best_epoch"], "device": str(device),
                  "weights": {"loc": 1., "family": config["lambda_family"]},
                  "temperature": config["temperature"], "batch_size": 64, "n_training_events": len(indices),
                  "protocol": "固定最佳检查点，覆盖最终训练集的随机固定批次；分别关闭和启用 Dropout；同一次前向上分项求导；不执行优化器更新。",
                  "boundary": "不是训练全过程轨迹；优化器状态缺失，投影与一阶作用不等同于 Adam 更新或准确率贡献。",
                  "modes": modes, "response_geometry": _response_geometry(run_dir),
                  "model_state_unchanged": True,
                  "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out / f"analysis-{label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summaries[label] = {"modes": {name: value["summary"] for name, value in modes.items()},
                            "response_geometry": result["response_geometry"]}
        print(label, json.dumps(summaries[label], ensure_ascii=True), flush=True)
        del model, frozen, data
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.finish()


if __name__ == "__main__":
    main()
