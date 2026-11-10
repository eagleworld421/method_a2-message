"""用与三项损失预测图相同的三个案例，改画真实反事实响应差异（Oracle）。

本脚本只读取已有九宫格的 `analysis.json` 中保存的 `oracle_values`、案例身份与
树边，不为模型重新推理；纵轴为真实候选响应与真实故障观测的物理标幺均方差。
图件格式与 `predicted-response-detail/full` 保持一致，包括 775 的展示层排除、
红★位置、节点编号、外向边红/橙/灰分类与 ρ 标注。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

METHOD = Path(__file__).resolve().parents[1]
if str(METHOD) not in sys.path:
    sys.path.insert(0, str(METHOD))

from scripts.analyze_gref_electrical_distance import relationship, savefig
from scripts.analyze_oracle_distance import DISTANCES, classify_oracle


def parse_args(argv=None) -> argparse.Namespace:
    """解析预测九宫格目录与 Oracle 距离来源。"""
    parser = argparse.ArgumentParser(description="按预测九宫格案例重绘真实响应差异图")
    parser.add_argument("--run-dir", default=str(METHOD / "output/gref/paired-oracle-prediction-three-term-20261104"))
    parser.add_argument("--oracle-dir", default=str(METHOD / "output/oracle-distance/ieee37-20261103/final"))
    return parser.parse_args(argv)


def _resolve(value: str) -> Path:
    """把相对路径解析为绝对路径。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), METHOD.parents[1], METHOD):
        if (base / path).exists():
            return (base / path).resolve()
    return (Path.cwd() / path).resolve()


def _plot(axis, record, matrices, values, relations, tolerance, edge_deltas, retained, key, root):
    """在单个子图上绘制距离—真实差异散点、外向边与候选编号。"""
    distance = matrices[key][root] / matrices[key].max()
    for (u, v), delta in zip(record["outward_edges"], edge_deltas):
        if not retained[u] or not retained[v]:
            continue
        color = "#d73935" if delta < -tolerance else ("#e59b26" if abs(delta) <= tolerance else "#b9c4d0")
        axis.plot(distance[[u, v]], values[[u, v]], color=color, lw=1, zorder=1)
    axis.scatter(distance[retained], values[retained], c=values[retained], cmap="viridis", s=30, zorder=3)
    axis.scatter([0], [0], c="red", marker="*", s=100, zorder=4)
    for node in np.flatnonzero(retained):
        axis.annotate(str(node), (distance[node], values[node]), xytext=(2, 3),
                      textcoords="offset points", fontsize=6)
    axis.set_xlim(-.03, 1.03)
    axis.set_xlabel(DISTANCES[key] + " / 全网对应直径")
    axis.set_ylabel("真实响应均方差（pu²）")
    axis.set_title(
        f"{record['dataset'].upper()} 族 {record['group_id']} / 故障 {root}"
        f" | 全候选 ρ={relations[key]['spearman']:.3f}",
        fontsize=10,
    )
    axis.grid(alpha=.2)


def main(argv=None) -> int:
    """读取预测案例与真实差异，写出 subject 放大与全候选两版图件。"""
    args = parse_args(argv)
    run_dir = _resolve(args.run_dir)
    oracle_dir = _resolve(args.oracle_dir)
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    oracle_meta = json.loads((oracle_dir / "analysis.json").read_text(encoding="utf-8"))
    with np.load(Path(oracle_meta["distance_source"]) / "distances.npz") as saved:
        matrices = {key: saved[key] for key in DISTANCES}
        buses = [str(bus) for bus in saved["buses"].tolist()]
    leaf = buses.index("775")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    records, csv_rows = [], []
    for record in analysis["records"]:
        values = np.asarray(record["oracle_values"], dtype=np.float64)
        root = int(record["true_location"])
        graph = nx.Graph(record["outward_edges"])
        classified = classify_oracle(graph, root, values)
        relations = {key: relationship(matrices[key][root], values, root, classified["tolerance"])
                     for key in DISTANCES}
        for key in DISTANCES:
            recorded = record["original_relations"][key]
            if abs(recorded["spearman"] - relations[key]["spearman"]) > 1e-9:
                raise ValueError("真实响应相关性与既有 Oracle 分析不一致")
        if abs(values[root]) > 1e-15:
            raise ValueError("真实故障候选的真实响应差异必须为零")
        item = dict(record)
        item["oracle_classification"] = {k: classified[k] for k in
                                         ("tolerance", "outward_edges", "edge_deltas", "decrease_present", "plateau_present")}
        item["oracle_relations"] = relations
        records.append(item)
        for candidate in range(38):
            csv_rows.append(dict(
                dataset=record["dataset"], index=record["index"], group_id=record["group_id"],
                subset=record["subset"], fault_bus=buses[root], candidate_index=candidate,
                candidate_bus=buses[candidate], oracle_error=float(values[candidate]),
                **{name: float(matrix[root, candidate]) for name, matrix in matrices.items()},
            ))

    for detail in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11.7), squeeze=False)
        for row_index, record in enumerate(records):
            values = np.asarray(record["oracle_values"], dtype=np.float64)
            root = int(record["true_location"])
            retained = (np.arange(38) != leaf) if detail else np.ones(38, bool)
            limit = float(values[retained].max()) * 1.12
            for column, key in enumerate(DISTANCES):
                axis = axes[row_index, column]
                _plot(axis, record, matrices, values, record["oracle_relations"],
                      record["oracle_classification"]["tolerance"],
                      record["oracle_classification"]["edge_deltas"], retained, key, root)
                axis.set_ylim(-limit * .04, limit)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
        suffix = "detail" if detail else "full"
        note = ("仅展示与原放大图相同的候选（775 不展示）；ρ 与阈值按全部 38 个候选计算"
                if detail else "全部 38 个故障候选；红色星号为真实故障候选，其真实差异为零")
        fig.suptitle(
            "真实响应（Oracle）同事件九宫格：与三项损失预测图相同的三个案例\n"
            f"{note}；红/橙/灰线=下降/平台/增长",
            fontsize=11,
        )
        savefig(fig, "oracle-response-" + suffix, run_dir)

    with (run_dir / "oracle-response-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "source_run_dir": str(run_dir),
        "oracle_dir": str(oracle_dir),
        "source_analysis_sha256": hashlib.sha256((run_dir / "analysis.json").read_bytes()).hexdigest(),
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "真实候选响应与真实故障观测的物理标幺均方差；红★为真实故障候选（差异为零）",
        "records": [
            dict(dataset=record["dataset"], index=record["index"], group_id=record["group_id"],
                 subset=record["subset"], true_location=record["true_location"],
                 true_bus=buses[int(record["true_location"])],
                 oracle_relations=record["oracle_relations"],
                 tolerance=record["oracle_classification"]["tolerance"],
                 oracle_values=record["oracle_values"])
            for record in records
        ],
    }
    (run_dir / "oracle-response-analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "README-oracle-response.md").write_text(
        "# 真实响应（Oracle）同事件九宫格\n\n"
        "- `oracle-response-detail.png/.svg`：与 `predicted-response-detail` 相同的三个案例，"
        "纵轴改为真实候选响应与真实观测的物理标幺均方差；775 仅在展示层排除。\n"
        "- `oracle-response-full.png/.svg`：全部 38 个候选的同一组案例。\n"
        "- `oracle-response-values.csv`：逐候选真实差异与三种距离。\n"
        "- `oracle-response-analysis.json`：案例身份、真实响应相关、容差与源文件哈希。\n"
        "- 真实故障候选的真实差异恒为零，因此红★落在 (0,0)；预测图中的红★可以高于零，"
        "因为预测响应存在误差。\n",
        encoding="utf-8")
    print(f"已写出 {run_dir / 'oracle-response-detail.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
