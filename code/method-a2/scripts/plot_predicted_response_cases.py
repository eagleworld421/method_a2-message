"""固定真实响应九宫格案例，以模型预测候选响应替换后重新作图。"""

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from scripts.analyze_oracle_distance import classify_oracle, DISTANCES
from scripts.analyze_gref_electrical_distance import relationship, savefig
from src.gref.data import load_arrays, make_batch
from src.gref.model import GrefPredictor


def physical_candidate_error(response, observation, node_mean, node_std, channel_std):
    """先撤销模型逐节点归一化，再还原通道物理尺度，保留真实观测作参照。"""
    stored_prediction = np.asarray(response, dtype=np.float64) * node_std[:, None, :] + node_mean[:, None, :]
    difference = (stored_prediction-np.asarray(observation, dtype=np.float64)) * channel_std
    return np.mean(difference**2, axis=(1, 2, 3))


def main():
    """仅对已选事件执行冻结推理，并复核物理坐标及测试预测一致性。"""
    torch.set_num_threads(1)
    torch.manual_seed(342)
    out = METHOD / "output/gref/paired-oracle-prediction-20261103"
    out.mkdir(exist_ok=False)
    oracle_dir = METHOD / "output/oracle-distance/ieee37-20261103/final"
    meta = json.loads((oracle_dir / "analysis.json").read_text(encoding="utf-8"))
    oracle_events = json.loads((oracle_dir / "events.json").read_text(encoding="utf-8"))
    selected = meta["selected"]["decrease_present"]
    with np.load(Path(meta["distance_source"]) / "distances.npz") as saved:
        matrices, buses = {k: saved[k] for k in DISTANCES}, saved["buses"].tolist()
    assert [(r["dataset"], r["test_index"]) for r in selected] == [("a", 647), ("b", 336), ("a", 1466)]
    cache, records, csv_rows, arrays, hashes = {}, [], [], {}, {}
    for case in selected:
        dataset, index = case["dataset"], case["test_index"]
        if dataset not in cache:
            run = METHOD / f"output/gref/ieee37-{dataset}-seed342-20261103"
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            data = load_arrays(Path(config["data_dir"]))
            checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
            model = GrefPredictor(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with np.load(run / "normalization.npz") as saved:
                norm = {k: saved[k] for k in saved.files}
            with np.load(run / "split.npz") as saved:
                split = {k: saved[k] for k in ("train_idx", "val_idx", "test_idx")}
            scaler = np.load(Path(config["data_dir"]) / "feature_scaler.npz")["std"].astype(np.float64)
            cache[dataset] = run, data, model, norm, split, scaler
            for file in (run / "model.pt", run / "normalization.npz", run / "split.npz", Path(config["data_dir"]) / "feature_scaler.npz"):
                hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        run, data, model, norm, split, scaler = cache[dataset]
        original = next(r for r in oracle_events if r["coordinate"] == "pu" and r["dataset"] == dataset and r["test_index"] == index)
        root = int(data["y_loc"][index])
        assert root == original["true_location"] and int(data["group_id"][index]) == original["group_id"]
        assert np.array_equal(data["X_obs"][index], data["signature_bank"][index, root])
        subset = next(k.removesuffix("_idx") for k, v in split.items() if index in v)
        batch = make_batch(data, np.array([index]), norm, torch.device("cpu"))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            output = model(batch["observation"], batch["edge_index"], batch["edge_attr"],
                           edge_mask=batch["edge_mask"], candidate_mask=batch["candidate_mask"], node_mask=batch["node_mask"])
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        response = output["response"][0, :38].numpy()
        predicted = physical_candidate_error(response, data["X_obs"][index], norm["mean"].astype(float), norm["std"].astype(float), scaler)
        true = np.mean(((np.asarray(data["signature_bank"][index, :38], dtype=float)-data["X_obs"][index])*scaler)**2, axis=(1, 2, 3))
        np.testing.assert_allclose(true, original["values"], rtol=1e-12, atol=1e-15)
        # 用已存测试预测独立核对检查点恢复和前向入口。
        saved_error = None
        if subset == "test":
            with np.load(run / "predictions.npz") as saved:
                position = int(np.flatnonzero(saved["test_indices"] == index)[0])
                saved_error = float(np.max(np.abs(response-saved["response"][position, :38])))
                np.testing.assert_allclose(response, saved["response"][position, :38], rtol=2e-4, atol=2e-5)
        classified = classify_oracle(__import__("networkx").Graph(original["outward_edges"]), root, predicted)
        relations = {k: relationship(m[root], predicted, root, classified["tolerance"]) for k, m in matrices.items()}
        record = dict(dataset=dataset, index=index, group_id=original["group_id"], true_location=root, true_bus=buses[root], subset=subset,
                      values=predicted.tolist(), oracle_values=true.tolist(), original_relations=original["relations"], relations=relations,
                      model_score_top1=int(output["candidate_score"][0, :38].argmin()), physical_distance_top1=int(predicted.argmin()),
                      true_candidate_error=float(predicted[root]), saved_prediction_max_error=saved_error,
                      **{k: classified[k] for k in ("outward_edges", "edge_deltas", "tolerance", "worst_edge", "decrease_present", "plateau_present")})
        records.append(record)
        arrays[f"{dataset}_{index}_predicted_response_model_coordinates"] = response
        arrays[f"{dataset}_{index}_reference_stored_coordinates"] = data["X_obs"][index]
        for k in range(38):
            csv_rows.append(dict(dataset=dataset, index=index, group_id=record["group_id"], subset=subset, fault_bus=buses[root], candidate_index=k,
                                 candidate_bus=buses[k], oracle_error=float(true[k]), predicted_error=float(predicted[k]),
                                 **{name: float(m[root, k]) for name, m in matrices.items()}))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    leaf = buses.index("775")
    for detail in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11.7), squeeze=False)
        for i, row in enumerate(records):
            root, values = row["true_location"], np.array(row["values"])
            retained = np.arange(38) != leaf if detail else np.ones(38, bool)
            span = max(float(np.ptp(values[retained])), 1e-12)
            minimum, maximum = float(values[retained].min()), float(values[retained].max())
            for j, key in enumerate(DISTANCES):
                ax = axes[i, j]
                distance = matrices[key][root]/matrices[key].max()
                for (u, v), delta in zip(row["outward_edges"], row["edge_deltas"]):
                    if not retained[u] or not retained[v]:
                        continue
                    color = "#d73935" if delta < -row["tolerance"] else "#e59b26" if abs(delta) <= row["tolerance"] else "#b9c4d0"
                    ax.plot(distance[[u, v]], values[[u, v]], color=color, lw=1, zorder=1)
                ax.scatter(distance[retained], values[retained], c=values[retained], cmap="viridis", s=29, zorder=3)
                ax.scatter([0], [values[root]], marker="*", color="red", s=130, zorder=5)
                for n in np.flatnonzero(retained):
                    ax.annotate(str(n), (distance[n], values[n]), xytext=(2, 3), textcoords="offset points", fontsize=6)
                ax.set_xlim(-.03, 1.03)
                ax.set_ylim(minimum-span*.08, maximum+span*.14)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
                ax.set_xlabel(DISTANCES[key]+" / 全网对应直径")
                ax.set_ylabel("真实参照与预测候选的均方差（pu²）")
                subset = {"train": "训练", "val": "验证", "test": "测试"}[row["subset"]]
                ax.set_title(f"{row['dataset'].upper()} 族 {row['group_id']} / 故障 {row['true_bus']}（{subset}）\nρ：真实 {row['original_relations'][key]['spearman']:.3f} → 预测 {row['relations'][key]['spearman']:.3f}", fontsize=9)
                ax.grid(alpha=.2)
        suffix = "detail" if detail else "full"
        note = "仅展示与原放大图相同的候选（775 不展示）；ρ 与阈值按全部 38 个候选计算" if detail else "全部 38 个故障候选；红色星号为真实故障候选，其预测误差可以非零"
        fig.suptitle("固定原三个案例：保留真实故障响应作参照，仅将候选响应换成模型预测\n"+note+"；红/橙/灰线=下降/平台/增长", fontsize=11)
        savefig(fig, "predicted-response-"+suffix, out)
    for file, digest in hashes.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    result = dict(records=records, input_sha256=hashes, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  oracle_source=str(oracle_dir), reference="true observation", score="MSE in physical voltage pu", model_mode="eval, frozen, CPU")
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(out / "predicted-responses.npz", **arrays)
    with (out / "candidate-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    html = "<meta charset='utf-8'><title>同事件预测响应九宫格</title><style>body{max-width:1500px;margin:25px auto;font-family:Arial,'Microsoft YaHei'}img{width:100%}</style><h1>同事件预测响应九宫格</h1><p>真实故障参照固定；同一三个事件与候选；第一、二行为训练事件，第三行为测试事件；纵轴为物理标幺均方差。</p>"
    for name in ("detail", "full"):
        html += f"<h2>{name}</h2><a href='predicted-response-{name}.svg'>SVG</a><img src='predicted-response-{name}.png'>"
    (out / "gallery.html").write_text(html, encoding="utf-8")
    with zipfile.ZipFile(out / "figures-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.iterdir()):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.name)
    print(json.dumps([dict(dataset=r["dataset"], index=r["index"], subset=r["subset"], root_error=r["true_candidate_error"],
                           rho_true=r["original_relations"], rho_predicted=r["relations"], physical_top1=r["physical_distance_top1"]) for r in records], indent=2))


if __name__ == "__main__":
    main()
