"""固定真实响应九宫格案例，用GRReF 模型的预测候选响应重新作图。

本脚本是 `plot_predicted_response_cases.py` 的可参数化副本，不修改原脚本：

- 通过 `--a-run-dir` 与 `--b-run-dir` 指定两套新模型的运行目录，读取各自的
  `config.json`、`model.pt`、`normalization.npz` 与 `split.npz`；
- 案例选择、物理标幺尺度恢复、分类阈值、图件布局与原脚本一致；
- 输出目录必须为新建目录，避免覆盖首轮 CE 的九宫格产物。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

METHOD = Path(__file__).resolve().parents[1]
if str(METHOD) not in sys.path:
    sys.path.insert(0, str(METHOD))

from scripts.analyze_gref_electrical_distance import relationship, savefig
from scripts.analyze_oracle_distance import DISTANCES, classify_oracle
from scripts.plot_predicted_response_cases import physical_candidate_error
from src.gref.data import load_arrays, make_batch
from src.gref.model import GrefPredictor

SUBSET_NAMES = {"train": "训练", "val": "验证", "test": "测试"}


def parse_args(argv=None) -> argparse.Namespace:
    """解析运行目录、模型标签与输出目录。"""
    parser = argparse.ArgumentParser(description="用GRReF 模型重绘同事件预测响应九宫格")
    parser.add_argument("--a-run-dir", default=str(METHOD / "output/gref/three-term-ieee37-a-seed342"))
    parser.add_argument("--b-run-dir", default=str(METHOD / "output/gref/three-term-ieee37-b-seed342"))
    parser.add_argument("--a-label", default="GRReF A")
    parser.add_argument("--b-label", default="三项损失 B")
    parser.add_argument("--oracle-dir", default=str(METHOD / "output/oracle-distance/ieee37-20261103/final"))
    parser.add_argument("--output", required=True, help="新输出目录，必须尚不存在")
    parser.add_argument("--title", default="GRReF 模型的预测响应九宫格")
    return parser.parse_args(argv)


def _resolve(value: str) -> Path:
    """把命令行路径解析为绝对路径，相对路径优先按当前目录，其次按项目根目录。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, METHOD.parents[1] / path, METHOD / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def main(argv=None) -> int:
    """对固定三个案例执行冻结推理，写出九宫格、数值与校验信息。"""
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(342)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    oracle_dir = _resolve(args.oracle_dir)
    meta = json.loads((oracle_dir / "analysis.json").read_text(encoding="utf-8"))
    oracle_events = json.loads((oracle_dir / "events.json").read_text(encoding="utf-8"))
    selected = meta["selected"]["decrease_present"]
    with np.load(Path(meta["distance_source"]) / "distances.npz") as saved:
        matrices, buses = {key: saved[key] for key in DISTANCES}, saved["buses"].tolist()
    assert [(record["dataset"], record["test_index"]) for record in selected] == [
        ("a", 647), ("b", 336), ("a", 1466)
    ], "案例集合与首轮九宫格不一致"
    runs = {"a": _resolve(args.a_run_dir), "b": _resolve(args.b_run_dir)}
    labels = {"a": args.a_label, "b": args.b_label}
    cache, records, csv_rows, arrays, hashes = {}, [], [], {}, {}
    for case in selected:
        dataset, index = case["dataset"], case["test_index"]
        if dataset not in cache:
            run = runs[dataset]
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            data = load_arrays(_resolve(config["data_dir"]))
            checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
            model = GrefPredictor(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with np.load(run / "normalization.npz") as saved:
                norm = {key: saved[key] for key in saved.files}
            with np.load(run / "split.npz") as saved:
                split = {key: saved[key] for key in ("train_idx", "val_idx", "test_idx")}
            scaler = np.load(_resolve(config["data_dir"]) / "feature_scaler.npz")["std"].astype(np.float64)
            cache[dataset] = run, data, model, norm, split, scaler
            for file in (run / "model.pt", run / "normalization.npz", run / "split.npz",
                         _resolve(config["data_dir"]) / "feature_scaler.npz"):
                hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        run, data, model, norm, split, scaler = cache[dataset]
        original = next(record for record in oracle_events
                        if record["coordinate"] == "pu" and record["dataset"] == dataset
                        and record["test_index"] == index)
        root = int(data["y_loc"][index])
        assert root == original["true_location"]
        assert int(data["group_id"][index]) == original["group_id"]
        assert np.array_equal(data["X_obs"][index], data["signature_bank"][index, root])
        subset = next(key.removesuffix("_idx") for key, value in split.items() if index in value)
        batch = make_batch(data, np.array([index]), norm, torch.device("cpu"))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            output_tensor = model(
                batch["observation"], batch["edge_index"], batch["edge_attr"],
                edge_mask=batch["edge_mask"], candidate_mask=batch["candidate_mask"],
                node_mask=batch["node_mask"],
            )
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
        response = output_tensor["response"][0, :38].numpy()
        predicted = physical_candidate_error(
            response, data["X_obs"][index], norm["mean"].astype(float),
            norm["std"].astype(float), scaler,
        )
        true = np.mean(
            ((np.asarray(data["signature_bank"][index, :38], dtype=float) - data["X_obs"][index]) * scaler) ** 2,
            axis=(1, 2, 3),
        )
        np.testing.assert_allclose(true, original["values"], rtol=1e-12, atol=1e-15)
        saved_error = None
        if subset == "test":
            with np.load(run / "predictions.npz") as saved:
                position = int(np.flatnonzero(saved["test_indices"] == index)[0])
                saved_error = float(np.max(np.abs(response - saved["response"][position, :38])))
                np.testing.assert_allclose(response, saved["response"][position, :38], rtol=2e-4, atol=2e-5)
        classified = classify_oracle(nx.Graph(original["outward_edges"]), root, predicted)
        relations = {
            key: relationship(matrix[root], predicted, root, classified["tolerance"])
            for key, matrix in matrices.items()
        }
        record = dict(
            dataset=dataset, index=index, group_id=original["group_id"], true_location=root,
            true_bus=buses[root], subset=subset, model_run=str(run), model_label=labels[dataset],
            values=predicted.tolist(), oracle_values=true.tolist(),
            original_relations=original["relations"], relations=relations,
            model_score_top1=int(output_tensor["candidate_score"][0, :38].argmin()),
            physical_distance_top1=int(predicted.argmin()),
            true_candidate_error=float(predicted[root]),
            saved_prediction_max_error=saved_error,
            **{key: classified[key] for key in (
                "outward_edges", "edge_deltas", "tolerance", "worst_edge",
                "decrease_present", "plateau_present")},
        )
        records.append(record)
        arrays[f"{dataset}_{index}_predicted_response_model_coordinates"] = response
        arrays[f"{dataset}_{index}_reference_stored_coordinates"] = data["X_obs"][index]
        for candidate in range(38):
            csv_rows.append(dict(
                dataset=dataset, index=index, group_id=record["group_id"], subset=subset,
                fault_bus=buses[root], candidate_index=candidate, candidate_bus=buses[candidate],
                oracle_error=float(true[candidate]), predicted_error=float(predicted[candidate]),
                **{name: float(matrix[root, candidate]) for name, matrix in matrices.items()},
            ))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    leaf = buses.index("775")
    for detail in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11.7), squeeze=False)
        for row_index, row in enumerate(records):
            root, values = row["true_location"], np.array(row["values"])
            retained = np.arange(38) != leaf if detail else np.ones(38, bool)
            span = max(float(np.ptp(values[retained])), 1e-12)
            minimum, maximum = float(values[retained].min()), float(values[retained].max())
            for column, key in enumerate(DISTANCES):
                ax = axes[row_index, column]
                distance = matrices[key][root] / matrices[key].max()
                for (u, v), delta in zip(row["outward_edges"], row["edge_deltas"]):
                    if not retained[u] or not retained[v]:
                        continue
                    color = "#d73935" if delta < -row["tolerance"] else "#e59b26" if abs(delta) <= row["tolerance"] else "#b9c4d0"
                    ax.plot(distance[[u, v]], values[[u, v]], color=color, lw=1, zorder=1)
                ax.scatter(distance[retained], values[retained], c=values[retained],
                           cmap="viridis", s=29, zorder=3)
                ax.scatter([0], [values[root]], marker="*", color="red", s=130, zorder=5)
                for node in np.flatnonzero(retained):
                    ax.annotate(str(node), (distance[node], values[node]), xytext=(2, 3),
                                textcoords="offset points", fontsize=6)
                ax.set_xlim(-.03, 1.03)
                ax.set_ylim(minimum - span * .08, maximum + span * .14)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
                ax.set_xlabel(DISTANCES[key] + " / 全网对应直径")
                ax.set_ylabel("真实参照与预测候选的均方差（pu²）")
                ax.set_title(
                    f"{row['dataset'].upper()} 族 {row['group_id']} / 故障 {row['true_bus']}"
                    f"（{SUBSET_NAMES[row['subset']]}；{row['model_label']}）\n"
                    f"ρ：真实 {row['original_relations'][key]['spearman']:.3f} → "
                    f"预测 {row['relations'][key]['spearman']:.3f}",
                    fontsize=9,
                )
                ax.grid(alpha=.2)
        suffix = "detail" if detail else "full"
        note = ("仅展示与原放大图相同的候选（775 不展示）；ρ 与阈值按全部 38 个候选计算"
                if detail else "全部 38 个故障候选；红色星号为真实故障候选，其预测误差可以非零")
        fig.suptitle(
            f"{args.title}\n{note}；红/橙/灰线=下降/平台/增长"
            f"（A：{labels['a']}；B：{labels['b']}）",
            fontsize=11,
        )
        savefig(fig, "predicted-response-" + suffix, output)
    for file, digest in hashes.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    model_info = {}
    for dataset, run in runs.items():
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        model_info[dataset] = {
            "run_dir": str(run), "label": labels[dataset],
            "loss_mode": config.get("loss_mode"), "loss_weights": config.get("loss_weights"),
            "rank_temperature": config.get("rank_temperature"), "best_epoch": config.get("best_epoch"),
        }
    result = dict(
        records=records, input_sha256=hashes, models=model_info,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_source=str(oracle_dir), reference="true observation",
        score="MSE in physical voltage pu", model_mode="eval, frozen, CPU",
        base_script="scripts/plot_predicted_response_cases.py",
    )
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "predicted-responses.npz", **arrays)
    with (output / "candidate-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    html = (
        "<meta charset='utf-8'><title>GRReF 模型同事件预测响应九宫格</title>"
        "<style>body{max-width:1500px;margin:25px auto;font-family:Arial,'Microsoft YaHei'}"
        "img{width:100%}</style><h1>GRReF 模型同事件预测响应九宫格</h1>"
        f"<p>真实故障参照固定；同一三个事件与候选；A：{labels['a']}；B：{labels['b']}；"
        "第一、二行为训练事件，第三行为测试事件；纵轴为物理标幺均方差。</p>"
    )
    for name in ("detail", "full"):
        html += (f"<h2>{name}</h2><a href='predicted-response-{name}.svg'>SVG</a>"
                 f"<img src='predicted-response-{name}.png'>")
    (output / "gallery.html").write_text(html, encoding="utf-8")
    with zipfile.ZipFile(output / "figures-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.name)
    print(json.dumps([dict(
        dataset=record["dataset"], index=record["index"], subset=record["subset"],
        model=record["model_label"], root_error=record["true_candidate_error"],
        physical_top1=record["physical_distance_top1"],
    ) for record in records], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
