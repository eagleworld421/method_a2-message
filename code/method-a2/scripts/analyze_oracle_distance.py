"""直接检验真实反事实响应差异的路径单调性、远近排序与距离相关性。"""

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

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from scripts.analyze_gref_topology import build_graph, classify_values, select_examples
from scripts.analyze_gref_electrical_distance import relationship, savefig
from src.progress import ProgressWriter

DISTANCES = {"hop": "跳数", "length_m": "线路长度", "impedance_pu": "模态阻抗"}
CATEGORIES = {"strict_increase": "严格逐边增长", "nondecreasing_with_plateau": "非递减含平台",
              "plateau_present": "存在平台", "decrease_present": "存在向外下降", "branch_difference": "同跳数分支差异"}
HISTORICAL_TEST_GROUPS = {1, 3, 24, 26, 28, 38}


def response_distances(bank, std):
    """直接相减计算两两均方差，避免内积展开造成近邻差异的消减误差。"""
    bank = np.asarray(bank, dtype=np.float64)
    scale = np.asarray(std, dtype=np.float64)
    result = np.empty((len(bank), len(bank)))
    for root in range(len(bank)):
        difference = (bank-bank[root]) * scale
        result[root] = np.mean(difference*difference, axis=tuple(range(1, bank.ndim)))
    if not np.array_equal(np.diag(result), np.zeros(len(bank))) or not np.array_equal(result, result.T):
        raise ValueError("真实响应差异必须满足零对角及对称性")
    return result


def classify_oracle(graph, root, values, relative=.01):
    """先作同比例变换以复用分类器，避免固定绝对下限吞没小物理差异。"""
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values))) or np.finfo(np.float64).tiny
    row = classify_values(graph, root, values/scale, relative)
    for key in ("tolerance", "span", "branch_threshold", "min_edge_delta"):
        row[key] *= scale
    row["edge_deltas"] = (np.array(row["edge_deltas"])*scale).tolist()
    row["branch_witness"][0] *= scale
    return row


def audit_group(data, indices):
    """核验同族条件、完整候选及共享响应库，不将不同工况误作位置干预。"""
    roots = np.asarray(data["y_loc"][indices])
    if len(indices) != 38 or set(roots.tolist()) != set(range(38)):
        raise ValueError("每个参数族必须完整覆盖 38 个事实位置")
    if len(np.unique(data["y_resist"][indices])) != 1 or len(np.unique(data["y_class"][indices])) != 1:
        raise ValueError("同族故障条件不一致")
    if not data["candidate_mask"][indices].all() or not data["mask"][indices].all():
        raise ValueError("本次全节点全候选检验不支持缺测")
    bank = np.asarray(data["signature_bank"][indices[0]])
    maximum = 0.
    for i in indices:
        if not np.array_equal(data["signature_bank"][i], bank):
            raise ValueError("同族的候选响应库不一致")
        maximum = max(maximum, float(np.max(np.abs(data["X_obs"][i]-bank[data["y_loc"][i]]))))
    if maximum != 0:
        raise ValueError("事实观测与真实候选响应不相等，应另行分析观测噪声")
    return bank[:38], maximum


def summarize(rows):
    """分别汇总全数据和历史测试族，保留逐族结果与阈值敏感性。"""
    result = {}
    for coord in ("pu", "stored"):
        result[coord] = {}
        for dataset in ("a", "b"):
            result[coord][dataset] = {}
            for subset in ("all", "historical_test"):
                items = [r for r in rows if r["coordinate"] == coord and r["dataset"] == dataset
                         and (subset == "all" or r["group_id"] in HISTORICAL_TEST_GROUPS)]
                flags = [*CATEGORIES, "numerical_nondecreasing", "numerical_strict_increase"]
                counts = {key: sum(r[key] for r in items) for key in flags}
                distances = {}
                for key in DISTANCES:
                    values = np.array([r["relations"][key]["spearman"] for r in items])
                    distances[key] = dict(spearman_median=float(np.median(values)),
                                          spearman_q1=float(np.quantile(values, .25)), spearman_q3=float(np.quantile(values, .75)),
                                          positive_count=int((values > 0).sum()),
                                          pearson_median=float(np.median([r["relations"][key]["pearson"] for r in items])),
                                          pair_violation_median=float(np.median([r["relations"][key]["pair_violation"] for r in items])),
                                          completely_ordered_count=sum(r["relations"][key]["pair_violation"] == 0 for r in items))
                groups = {str(g): {key: float(np.median([r["relations"][key]["spearman"] for r in items if r["group_id"] == g]))
                                   for key in DISTANCES} for g in sorted({r["group_id"] for r in items})}
                result[coord][dataset][subset] = dict(n_events=len(items), n_groups=len(groups), counts=counts,
                    percent={key: 100*value/len(items) for key, value in counts.items()}, distances=distances, group_medians=groups,
                    threshold_sensitivity={str(t): sum(r["nondecreasing_at"][str(t)] for r in items) for t in (0., .001, .01, .05)},
                    min_wrong_energy_median=float(np.median([r["min_wrong_energy"] for r in items])),
                    exact_duplicate_events=sum(r["exact_duplicate_candidates"] > 0 for r in items))
    return result


def leaf_sensitivity(rows, graph, matrices, leaf):
    """剔除特殊低压叶节点的事实与候选，检查其对范围阈值和排序的影响。"""
    retained = [n for n in graph if n != leaf]
    remap = {n: i for i, n in enumerate(retained)}
    reduced = nx.relabel_nodes(graph.subgraph(retained), remap, copy=True)
    result = {}
    for dataset in ("a", "b"):
        items = [r for r in rows if r["dataset"] == dataset and r["coordinate"] == "pu" and r["true_location"] != leaf]
        classified, relations = [], {k: [] for k in DISTANCES}
        for row in items:
            values = np.array(row["values"])[retained]
            root = remap[row["true_location"]]
            classified.append(classify_oracle(reduced, root, values))
            for key, matrix in matrices.items():
                relations[key].append(relationship(matrix[row["true_location"]][retained], values, root, classified[-1]["tolerance"]))
        result[dataset] = dict(n_events=len(items), n_wrong_candidates=len(retained)-1,
            decrease_count=sum(r["decrease_present"] for r in classified), plateau_count=sum(r["plateau_present"] for r in classified),
            branch_difference_count=sum(r["branch_difference"] for r in classified),
            distances={k: dict(spearman_median=float(np.median([r["spearman"] for r in items])),
                              pair_violation_median=float(np.median([r["pair_violation"] for r in items]))) for k, items in relations.items()})
    return result


def figures(rows, summary, graph, matrices, buses, out):
    """绘制物理尺度的总体统计和每类至多三个真实案例。"""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    physical = [r for r in rows if r["coordinate"] == "pu"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, dataset in zip(axes[:2], ("a", "b")):
        items = [r for r in physical if r["dataset"] == dataset]
        values = [[r["relations"][k]["spearman"] for r in items] for k in DISTANCES]
        ax.boxplot(values, labels=list(DISTANCES.values()), showfliers=False)
        for x, v in enumerate(values, 1):
            ax.text(x, np.median(v)+.04, f"{np.median(v):.3f}", ha="center")
        ax.set_ylim(-1.05, 1.1)
        ax.axhline(0, color="grey", ls="--", lw=.8)
        ax.set_title(f"{dataset.upper()} | 40 族，1520 个故障事件")
        ax.set_ylabel("错误候选 Oracle 差异与距离的 Spearman ρ")
        ax.grid(axis="y", alpha=.2)
    ax = axes[2]
    x = np.arange(len(CATEGORIES))
    for offset, dataset in ((-.19, "a"), (.19, "b")):
        v = [summary["pu"][dataset]["all"]["percent"][k] for k in CATEGORIES]
        bars = ax.bar(x+offset, v, .36, label=dataset.upper())
        for bar, value in zip(bars, v):
            ax.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(CATEGORIES.values()), rotation=27, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("占全部故障事件（%）")
    ax.set_title("1% 范围容差；平台与分支差异可重叠")
    ax.legend()
    fig.suptitle("直接使用真实响应库：恢复电压标幺尺度，完全排除模型输出")
    savefig(fig, "oracle-summary", out)
    selected = {}
    positions = nx.kamada_kawai_layout(graph)
    for category, title in CATEGORIES.items():
        cases = select_examples(physical, category)
        selected[category] = [{k: r[k] for k in ("dataset", "test_index", "group_id", "true_location")} for r in cases]
        if not cases:
            continue
        fig, axes = plt.subplots(len(cases), 4, figsize=(19, 3.8*len(cases)), squeeze=False)
        for i, case in enumerate(cases):
            root, values = case["true_location"], np.array(case["values"])
            ax = axes[i, 0]
            nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#ced5dd", width=1)
            artist = nx.draw_networkx_nodes(graph, positions, node_color=values, cmap="viridis", node_size=100, ax=ax)
            nx.draw_networkx_labels(graph, positions, ax=ax, font_size=5.5)
            nx.draw_networkx_nodes(graph, positions, nodelist=[root], node_color="none", edgecolors="red", linewidths=2, node_size=220, ax=ax)
            fig.colorbar(artist, ax=ax, fraction=.035, pad=.01).set_label("Oracle 均方差（pu²）", fontsize=8)
            ax.set_title(f"{case['dataset'].upper()} | 族 {case['group_id']} | 故障 {root} ({buses[root]})", fontsize=9)
            ax.axis("off")
            for j, key in enumerate(DISTANCES, 1):
                ax = axes[i, j]
                distance = matrices[key][root]/matrices[key].max()
                for (u, v), delta in zip(case["outward_edges"], case["edge_deltas"]):
                    color = "#d83e39" if delta < -case["tolerance"] else ("#e7a22b" if abs(delta) <= case["tolerance"] else "#bfc8d2")
                    ax.plot(distance[[u, v]], values[[u, v]], color=color, lw=1, zorder=1)
                ax.scatter(distance, values, c=values, cmap="viridis", s=23, zorder=3)
                ax.scatter([0], [0], c="red", marker="*", s=110, zorder=4)
                for n in range(38):
                    ax.annotate(str(n), (distance[n], values[n]), xytext=(2, 2), textcoords="offset points", fontsize=5.5)
                ax.set_xlim(-.03, 1.03)
                ax.set_xlabel(DISTANCES[key]+" / 全网对应直径", fontsize=8)
                ax.set_ylabel("真实响应均方差（pu²）", fontsize=8)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
                ax.set_title(f"ρ={case['relations'][key]['spearman']:.3f}；逆序对={case['relations'][key]['pair_violation']:.1%}", fontsize=9)
                ax.grid(alpha=.2)
        fig.suptitle(f"真实候选响应差异：{title}的 {len(cases)} 个实际案例；灰/橙/红线表示向外增长/平台/下降")
        savefig(fig, "oracle-"+category, out)
    return selected


def main():
    """审计全部真实数据，再计算和保存无模型依赖的统计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=METHOD / "output/oracle-distance/ieee37-20261103")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("拒绝覆盖既有审计目录")
    out.mkdir(parents=True)
    directory = out / "figures"
    directory.mkdir()
    progress = ProgressWriter(METHOD / "logs/oracle-distance-20261103/progress.jsonl", out.name, terminal=False)
    source = METHOD / "output/gref/electrical-distance-20261103/final"
    source_meta = json.loads((source / "analysis.json").read_text(encoding="utf-8"))
    with np.load(source / "distances.npz") as saved:
        matrices = {k: saved[k] for k in DISTANCES}
        buses = saved["buses"].tolist()
    rows, audit, candidate_rows, pair_matrices, hashes = [], {}, [], {}, {}
    for dataset, name in (("a", "s0-ieee37-r0p01-20-loguniform"), ("b", "s0-ieee37-r10-100-loguniform")):
        path = METHOD / "data" / name
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        assert meta["counterfactual_perturbation"] == "disabled"
        assert meta["phase_usage_first_event"][:38] == [[1]]*38
        data = {k: np.load(path / (k+".npy"), mmap_mode="r") for k in
                ("signature_bank", "group_id", "y_loc", "y_resist", "y_class", "X_obs", "mask", "candidate_mask", "edge_index")}
        graph = build_graph(38, data["edge_index"])
        for edge in source_meta["edge_parameters"]:
            assert graph.has_edge(edge["u"], edge["v"])
        expected_hops = np.array([[nx.shortest_path_length(graph, u, v) for v in graph] for u in graph])
        assert np.array_equal(expected_hops, matrices["hop"])
        std = np.load(path / "feature_scaler.npz")["std"].astype(np.float64)
        groups = sorted(set(data["group_id"].tolist())-{-1})
        assert len(groups) == 40
        maximum, total = 0., 0
        progress.stage_start(dataset, len(groups))
        for step, group in enumerate(groups, 1):
            indices = np.flatnonzero(data["group_id"] == group)
            bank, error = audit_group(data, indices)
            maximum = max(maximum, error)
            total += len(indices)
            for coord, scale in (("pu", std), ("stored", np.ones(6))):
                pairwise = response_distances(bank, scale)
                pair_matrices[f"{dataset}_{coord}_g{group}"] = pairwise
                for index in indices:
                    root = int(data["y_loc"][index])
                    values = pairwise[root]
                    detail = classify_oracle(graph, root, values)
                    numerical = classify_oracle(graph, root, values, 0.)
                    relations = {k: relationship(m[root], values, root, detail["tolerance"]) for k, m in matrices.items()}
                    wrong = np.arange(38) != root
                    row = dict(dataset=dataset, coordinate=coord, group_id=int(group), test_index=int(index), true_location=root,
                               fault_resistance=float(data["y_resist"][index]), values=values.tolist(), **detail,
                               numerical_nondecreasing=not numerical["decrease_present"], numerical_strict_increase=numerical["strict_increase"],
                               nondecreasing_at={str(t): not classify_oracle(graph, root, values, t)["decrease_present"] for t in (0., .001, .01, .05)},
                               min_wrong_energy=float(values[wrong].min()), exact_duplicate_candidates=int((values[wrong] == 0).sum()), relations=relations)
                    rows.append(row)
                    for k, value in enumerate(values):
                        candidate_rows.append(dict(dataset=dataset, coordinate=coord, group_id=int(group), index=int(index), true_location=root,
                            candidate=k, bus=buses[k], oracle_energy=float(value), **{name: float(m[root, k]) for name, m in matrices.items()}))
            progress.update(dataset, "groups", step, len(groups))
        progress.stage_done(dataset, len(groups))
        audit[dataset] = dict(n_groups=len(groups), n_fault_events=total, observation_signature_max_error=maximum,
                              shared_banks_identical=True, complete_locations=True, phase_usage=meta["phase_usage_first_event"][:38], std=std.tolist())
        for file in sorted(path.glob("*.npy")):
            if file.stem in data:
                with file.open("rb") as handle:
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(8*1024*1024), b""):
                        digest.update(block)
                hashes[str(file)] = digest.hexdigest()
        for file in (path / "meta.json", path / "feature_scaler.npz"):
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    summary = summarize(rows)
    sensitivity = leaf_sensitivity(rows, graph, matrices, buses.index("775"))
    progress.stage_start("figures", len(CATEGORIES)+1)
    selected = figures(rows, summary, graph, matrices, buses, directory)
    progress.stage_done("figures", len(CATEGORIES)+1)
    result = dict(summary=summary, audit=audit, selected=selected, input_sha256=hashes,
                  exclude_low_voltage_leaf_sensitivity=sensitivity,
                  physical_coordinate="per-bus voltage pu, real/imag channels; mean squared difference over all nodes/times/6 channels",
                  model_outputs_used=False, distance_source=str(source),
                  distance_source_sha256=hashlib.sha256((source / "distances.npz").read_bytes()).hexdigest(),
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name, value in (("analysis.json", result), ("events.json", rows)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(out / "pairwise-oracle.npz", **pair_matrices)
    with (out / "candidate-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    sections = ["<meta charset='utf-8'><title>真实候选响应差异检验</title><style>body{max-width:1600px;margin:25px auto;font-family:Arial,'Microsoft YaHei'}img{width:100%}</style>",
                "<h1>真实候选响应差异与距离关系</h1><p>全部真实响应数据；每组 40 族、1520 个故障事件；未使用模型预测。图中采用恢复后的电压标幺尺度。</p>"]
    for file in sorted(directory.glob("*.png")):
        sections.append(f"<h2>{file.stem}</h2><a href='figures/{file.stem}.svg'>SVG</a><img src='figures/{file.name}'>")
    sections.append("<p><a href='analysis.json'>完整统计</a> · <a href='candidate-values.csv'>全部候选数值</a> · <a href='pairwise-oracle.npz'>各族真实响应差异矩阵</a></p>")
    (out / "gallery.html").write_text("".join(sections), encoding="utf-8")
    with zipfile.ZipFile(out / "figures-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.rglob("*")):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.relative_to(out))
    progress.finish()
    print(json.dumps({k: {s: summary["pu"][k][s]["counts"] for s in ("all", "historical_test")} for k in ("a", "b")}, indent=2))


if __name__ == "__main__":
    main()
