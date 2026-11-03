"""检查候选能量和响应重构误差的拓扑单调性，并仅绘制分类案例。"""

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
from scipy.stats import spearmanr

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
from src.progress import ProgressWriter

METRICS = {"energy": "候选能量", "reconstruction": "响应重构误差（Smooth L1）"}
CATEGORIES = {"strict_increase": "严格逐边增长", "plateau_present": "存在平台",
              "decrease_present": "存在向外下降", "branch_difference": "同跳数分支差异"}


def build_graph(n_nodes, edges):
    """建立与候选轴一致的无向树，防止在不连通图上误算距离。"""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    if not nx.is_tree(graph):
        raise ValueError("本次检查要求候选拓扑为连通无向树")
    return graph


def classify_values(graph, fault, values, relative_tolerance=.01):
    """以外向边变化分类，并保留平台与分支现象的可重叠标志。"""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.shape != (len(graph),):
        raise ValueError("候选数值必须为逐节点有限向量")
    distance = np.array([nx.shortest_path_length(graph, fault, k) for k in range(len(graph))])
    edges = [(u, v) if distance[u] < distance[v] else (v, u) for u, v in graph.edges()]
    delta = np.array([values[v] - values[u] for u, v in edges])
    span = float(np.ptp(values))
    floor = max(1e-8, 16 * np.finfo(np.float32).eps * float(np.abs(values).max()))
    tolerance = max(floor, relative_tolerance * span)
    down = delta < -tolerance
    flat = np.abs(delta) <= tolerance
    strict = bool(np.all(delta > tolerance))
    primary = "decrease_present" if down.any() else ("strict_increase" if strict else "nondecreasing_with_plateau")
    layer_pairs = []
    layer_means = []
    for hop in np.unique(distance):
        nodes = np.flatnonzero(distance == hop)
        layer_means.append(float(values[nodes].mean()))
        if hop > 0 and len(nodes) > 1:
            low, high = nodes[np.argmin(values[nodes])], nodes[np.argmax(values[nodes])]
            layer_pairs.append((float(values[high] - values[low]), int(low), int(high), int(hop)))
    branch = max(layer_pairs, default=(0., -1, -1, -1))
    branch_threshold = max(tolerance, .1 * span)
    pair_mask = (distance[None, :] > distance[:, None]) & (distance[:, None] > 0)
    pair_delta = values[None, :] - values[:, None]
    wrong = distance > 0
    rho = None if np.ptp(values[wrong]) == 0 or np.ptp(distance[wrong]) == 0 else float(spearmanr(distance[wrong], values[wrong]).statistic)
    worst = int(np.argmin(delta))
    return {
        "distance": distance.tolist(), "tolerance": tolerance, "span": span,
        "strict_increase": strict, "nondecreasing_with_plateau": primary == "nondecreasing_with_plateau",
        "primary": primary, "plateau_present": bool(flat.any()), "decrease_present": bool(down.any()),
        "branch_difference": bool(branch[0] > branch_threshold), "branch_threshold": branch_threshold,
        "branch_witness": list(branch), "outward_edges": [list(pair) for pair in edges],
        "edge_deltas": delta.tolist(), "decrease_edge_fraction": float(down.mean()),
        "plateau_edge_fraction": float(flat.mean()), "min_edge_delta": float(delta.min()),
        "worst_edge": list(edges[worst]), "max_relative_drop": float(max(0., -delta.min()) / max(span, floor)),
        "max_relative_branch_spread": float(branch[0] / max(span, floor)),
        "wrong_candidate_spearman": rho,
        "wrong_candidate_pair_violation_fraction": float(np.mean(pair_delta[pair_mask] < -tolerance)) if pair_mask.any() else None,
        "layer_mean_nondecreasing": bool(np.all(np.diff(layer_means) >= -tolerance)),
    }


def select_examples(events, category):
    """优先选择两组的中位严重度案例，再补一个较高分位的不同参数族案例。"""
    key = {"strict_increase": "min_edge_delta", "plateau_present": "plateau_edge_fraction",
           "decrease_present": "max_relative_drop", "branch_difference": "max_relative_branch_spread",
           "nondecreasing_with_plateau": "min_edge_delta", "numerical_nondecreasing": "min_edge_delta"}[category]
    pool = [event for event in events if event[category]]
    selected = []
    for label in ("a", "b"):
        rows = sorted([r for r in pool if r["dataset"] == label], key=lambda r: (r[key], r["test_index"]))
        if rows:
            selected.append(rows[len(rows) // 2])
    remaining = [r for r in pool if r not in selected]
    diverse = [r for r in remaining if (r["dataset"], r["group_id"]) not in
               {(x["dataset"], x["group_id"]) for x in selected}]
    if diverse:
        remaining = diverse
    remaining.sort(key=lambda r: (r[key], r["dataset"], r["test_index"]))
    if remaining:
        selected.append(remaining[int(.75 * (len(remaining) - 1))])
    # 若某一组没有该现象，仍在实际案例足够时补齐三个不重复事件。
    for row in pool:
        if len(selected) >= 3:
            break
        if row not in selected:
            selected.append(row)
    return selected[:3]


def summarize(events):
    """同时报告互斥分类和可重叠现象，以全部测试事件为分母。"""
    result = {}
    for dataset in ("a", "b", "pooled"):
        rows = [r for r in events if dataset == "pooled" or r["dataset"] == dataset]
        keys = [*CATEGORIES, "nondecreasing_with_plateau", "layer_mean_nondecreasing"]
        result[dataset] = {"n_events": len(rows), "counts": {k: sum(r[k] for r in rows) for k in keys}}
        result[dataset]["percent"] = {k: 100 * value / len(rows) for k, value in result[dataset]["counts"].items()}
        for key in ("wrong_candidate_spearman", "wrong_candidate_pair_violation_fraction", "decrease_edge_fraction"):
            values = [r[key] for r in rows if r[key] is not None]
            result[dataset][key + "_median"] = float(np.median(values)) if values else None
    return result


def plot_examples(graph, positions, cases, metric, category, directory):
    """每个实际情况绘制三组案例，每例并列展示拓扑分布与跳数曲线。"""
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 4.6 * len(cases)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.15]})
    for row, case in enumerate(cases):
        values = np.array(case["values"])
        distance = np.array(case["distance"])
        root = case["true_location"]
        vmin, vmax = float(values.min()), float(values.max())
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-8)
        ax = axes[row, 0]
        nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#c6ccd3", width=1.2)
        nodes = nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=values, cmap="viridis",
                                       vmin=norm.vmin, vmax=norm.vmax, node_size=180, edgecolors="white")
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=6.5)
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[root], node_size=350,
                               node_color="none", edgecolors="#e23b36", linewidths=2.5)
        if case["decrease_present"]:
            nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=[case["worst_edge"]],
                                    edge_color="#d93025", width=3)
        if category == "branch_difference":
            _, low, high, _ = case["branch_witness"]
            nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[low, high], node_size=290,
                                   node_color="none", edgecolors="#a12dba", linewidths=2)
        fig.colorbar(nodes, ax=ax, fraction=.036, pad=.02).set_label(METRICS[metric], fontsize=9)
        ax.set_title(f"{case['dataset'].upper()} | 测试索引 {case['test_index']} | 故障点 {root} | 参数族 {case['group_id']}", fontsize=10)
        ax.axis("off")
        ax = axes[row, 1]
        jitter = ((np.arange(len(values)) * 17 % 13) / 12 - .5) * .27
        jitter[root] = 0
        xx = distance + jitter
        for (u, v), delta in zip(case["outward_edges"], case["edge_deltas"]):
            color = "#d93025" if delta < -case["tolerance"] else ("#f39c12" if abs(delta) <= case["tolerance"] else "#b7c0ca")
            ax.plot(xx[[u, v]], values[[u, v]], color=color, alpha=.55, lw=.9, zorder=1)
        ax.scatter(xx, values, c=values, cmap="viridis", norm=norm, s=35, edgecolors="white", linewidths=.5, zorder=3)
        ax.scatter([0], [values[root]], marker="*", s=180, c="#e23b36", zorder=5, label="真实故障点")
        for node in range(len(values)):
            ax.annotate(str(node), (xx[node], values[node]), xytext=(2, 3), textcoords="offset points", fontsize=6, alpha=.8)
        hops = np.unique(distance)
        ax.plot(hops, [values[distance == h].mean() for h in hops], "k--", lw=1.3, label="同跳数均值")
        ax.set_xticks(hops)
        ax.set_xlabel("距真实故障点的最短路径跳数（同层节点轻微横向错开）", fontsize=9)
        ax.set_ylabel(METRICS[metric], fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
        rho = case["wrong_candidate_spearman"]
        title = f"平台边 {case['plateau_edge_fraction']:.1%} | 下降边 {case['decrease_edge_fraction']:.1%} | 错误候选 ρ={rho:.3f}" if rho is not None else "候选数值恒定"
        ax.set_title(title + f"\nτ={case['tolerance']:.3g}；最小外向变化={case['min_edge_delta']:.3g}；灰/橙/红=增长/平台/下降", fontsize=8)
    labels = {**CATEGORIES, "nondecreasing_with_plateau": "1%容差内非递减（含平台）",
              "numerical_nondecreasing": "仅数值容差下逐边非递减"}
    fig.suptitle(f"{METRICS[metric]}：{labels[category]}的 {len(cases)} 个实际案例", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .975])
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{metric}-{category}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    """对全部测试事件统计，仅对预先规定的分类例子作图并保存原始数值。"""
    parser = argparse.ArgumentParser(description="检查 GReF 候选量与拓扑距离的关系")
    parser.add_argument("--output-dir", type=Path, default=METHOD_DIR / "output/gref/topology-analysis-20261103")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (METHOD_DIR / "output/gref").resolve() not in out.parents:
        raise ValueError("输出必须位于本方法的 output/gref/ 下")
    out.mkdir(exist_ok=False)
    figures = out / "figures"
    figures.mkdir()
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    progress = ProgressWriter(METHOD_DIR / "logs" / out.name / "progress.jsonl", out.name, terminal=False)
    all_events = {key: [] for key in METRICS}
    candidate_rows, graphs, verification = [], {}, {}
    for label in ("a", "b"):
        run = METHOD_DIR / f"output/gref/ieee37-{label}-seed342-20261103"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        graph = build_graph(38, np.load(Path(config["data_dir"]) / "edge_index.npy"))
        graphs[label] = graph
        with np.load(run / "predictions.npz") as saved:
            prediction = saved["response"][:, :38].astype(np.float64)
            target = saved["target"][:, :38].astype(np.float64)
            observation = saved["observation"].astype(np.float64)
            if not saved["candidate_mask"][:, :38].all() or not saved["node_mask"].all():
                raise ValueError("本轮固定完整观测与完整故障候选")
            recomputed = ((prediction - observation[:, None]) ** 2).mean(axis=(2, 3, 4))
            energy = saved["candidate_score"][:, :38].astype(np.float64)
            assert np.allclose(recomputed, energy, rtol=2e-5, atol=2e-6)
            error = np.abs(prediction - target)
            reconstruction = np.where(error < 1, .5 * error ** 2, error - .5).mean(axis=(2, 3, 4))
            verification[label] = {"energy_max_abs_recompute_error": float(np.abs(recomputed - energy).max()),
                                   "graph_nodes": len(graph), "graph_edges": graph.number_of_edges()}
            progress.stage_start(label, len(energy))
            for event in range(len(energy)):
                true = int(saved["location"][event])
                base = {"dataset": label, "test_index": int(saved["test_indices"][event]),
                        "group_id": int(saved["group_id"][event]), "true_location": true,
                        "predicted_location": int(energy[event].argmin())}
                for metric, matrix in (("energy", energy), ("reconstruction", reconstruction)):
                    row = {**base, **classify_values(graph, true, matrix[event]), "values": matrix[event].tolist()}
                    all_events[metric].append(row)
                    for node, value in enumerate(matrix[event]):
                        candidate_rows.append({**base, "metric": metric, "candidate": node,
                                               "hop_distance": row["distance"][node], "value": float(value)})
                progress.update(label, "events", event + 1, len(energy))
            progress.stage_done(label, len(energy))
    assert set(graphs["a"].edges()) == set(graphs["b"].edges())
    positions = nx.kamada_kawai_layout(graphs["a"])
    summary = {metric: summarize(rows) for metric, rows in all_events.items()}
    sensitivity = {}
    for relative in (0., .001, .01, .05):
        sensitivity[str(relative)] = {}
        for metric, rows in all_events.items():
            classified = [{**row, **classify_values(graphs[row["dataset"]], row["true_location"], row["values"], relative)} for row in rows]
            sensitivity[str(relative)][metric] = summarize(classified)
    selected = {}
    progress.stage_start("figures", 8)
    for index, (metric, rows) in enumerate(all_events.items()):
        selected[metric] = {}
        for j, category in enumerate(CATEGORIES):
            cases = select_examples(rows, category)
            selected[metric][category] = cases
            plot_examples(graphs["a"], positions, cases, metric, category, figures)
            progress.update("figures", "panels", index * 4 + j + 1, 8, force=True)
    supplemental = {}
    for metric, rows in all_events.items():
        numerical = [{**row, **classify_values(graphs[row["dataset"]], row["true_location"], row["values"] , 0.)} for row in rows]
        for row in numerical:
            row["numerical_nondecreasing"] = not row["decrease_present"]
        supplemental[metric] = {}
        for category, source in (("nondecreasing_with_plateau", rows), ("numerical_nondecreasing", numerical)):
            cases = select_examples(source, category)
            supplemental[metric][category] = cases
            plot_examples(graphs["a"], positions, cases, metric, category, figures)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (metric, data) in zip(axes, summary.items()):
        x = np.arange(len(CATEGORIES))
        for offset, label, color in ((-.18, "a", "#2c7fb8"), (.18, "b", "#f28e2b")):
            values = [data[label]["percent"][k] for k in CATEGORIES]
            bars = ax.bar(x + offset, values, .34, label=label.upper(), color=color)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(CATEGORIES.values()), fontsize=9)
        ax.set_ylim(0, 113)
        ax.set_ylabel("占全部测试事件的比例（%）")
        ax.set_title(METRICS[metric])
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("每组 n=228；平台、下降与分支差异可重叠；严格逐边增长单独判定", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .92])
    for extension in ("png", "svg"):
        fig.savefig(figures / f"category-prevalence.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    result = {"summary": summary, "verification": verification,
              "classification": {"relative_plateau_tolerance": .01, "relative_branch_threshold": .1,
                                  "absolute_floor": "max(1e-8,16*float32_eps*max_abs_value)", "overlap_allowed": True},
              "sensitivity": sensitivity, "selected": selected, "supplemental": supplemental,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for metric, rows in all_events.items():
        (out / f"events-{metric}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "candidate-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    progress.stage_done("figures", 8)
    progress.finish()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
