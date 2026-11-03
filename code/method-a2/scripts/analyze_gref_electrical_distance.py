"""统一 IEEE37 电气尺度，比较固定候选数值与不同路径距离的关系。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

Q = np.array([[2, 0], [-1, np.sqrt(3)], [-1, -np.sqrt(3)]]) / np.sqrt(6)
MODE = np.exp(-2j * np.pi * np.arange(3) / 3) / np.sqrt(3)
METHOD = Path(__file__).resolve().parents[1]
NAMES = {"hop": "跳数", "length_m": "线路长度", "impedance_pu": "模态阻抗", "positive_pu": "正序阻抗"}
METRICS = {"energy": "候选能量", "reconstruction": "响应重构误差"}


def port_admittance(elements, pair):
    """按母线相别合并并联设备，接地导体不作为独立端口。"""
    total = np.zeros((6, 6), complex)
    for item in elements:
        buses = [b.split(".")[0] for b in item["buses"]]
        if len(buses) != 2 or set(buses) != set(pair):
            raise ValueError("端口元素必须连接给定的两个母线")
        data = np.asarray(item["yprim"])
        n = 2 * item["nconds"]
        y = (data[::2] + 1j * data[1::2]).reshape((n, n), order="F")
        mapping = [None if phase == 0 else 3 * pair.index(buses[i // item["nconds"]]) + phase - 1
                   for i, phase in enumerate(item["node_order"])]
        for i, u in enumerate(mapping):
            for j, v in enumerate(mapping):
                if u is not None and v is not None:
                    total[u, v] += y[i, j]
    return total


def differential_impedance(y_si, voltage_ln, s_base=1e6):
    """在共同三相容量及各端额定电压下提取可传输的无零序阻抗。"""
    d = np.diag(np.repeat(voltage_ln, 3))
    y = d @ y_si @ d / (s_base / 3)
    injection = np.vstack([Q, -Q])
    inverse = np.linalg.pinv(y, rcond=1e-10)
    residual = np.linalg.norm(y @ inverse @ injection - injection) / np.linalg.norm(injection)
    if residual > 1e-7:
        raise ValueError(f"所选模态不是该端口可传输的电流，残差 {residual}")
    return injection.T @ inverse @ injection, float(residual)


def weights(raw, official_file, data_dirs):
    """核验原始长度、候选顺序和图连接，返回具有明确量纲的边权。"""
    buses = [b["name"] for b in raw["buses"] if b["name"] != "sourcebus"]
    base = {b["name"]: b["kv_ln"] * 1000 for b in raw["buses"]}
    official = pd.read_excel(official_file, header=None).iloc[3:]
    lengths = {frozenset((str(int(a)), str(int(b)))): float(length)
               for a, b, length, _ in official.itertuples(index=False, name=None)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(buses)))
    for directory in data_dirs:
        edges = np.load(directory / "edge_index.npy")
        pairs = {tuple(sorted((int(u), int(v)))) for u, v in edges}
        if graph.number_of_edges() and pairs != set(graph.edges()):
            raise ValueError("两组数据拓扑不一致")
        graph.add_edges_from(pairs)
    if len(buses) != 38 or not nx.is_tree(graph):
        raise ValueError("候选轴和拓扑不符合 IEEE37 数据契约")
    elements = {}
    for item in raw["lines"] + raw["transformers"]:
        pair = tuple(b.split(".")[0] for b in item["buses"])
        if "sourcebus" not in pair:
            elements.setdefault(frozenset(pair), []).append(item)
    records = []
    for u, v in sorted(graph.edges()):
        pair = (buses[u], buses[v])
        items = elements[frozenset(pair)]
        regular = len(items) == 1 and "rmatrix" in items[0] and items[0]["nphases"] == 3
        residual, matrix_check = None, None
        if regular:
            line = items[0]
            source_pair = frozenset("799" if b == "799r" else b for b in pair)
            feet = lengths[source_pair]
            if not np.isclose(line["length"] * 1000, feet):
                raise ValueError("算例隐含长度单位与 IEEE 官方英尺长度不一致")
            z = (np.array(line["rmatrix"]).reshape(3, 3) + 1j * np.array(line["xmatrix"]).reshape(3, 3)) * line["length"]
            z_pu = z / (base[pair[0]] ** 2 / (1e6 / 3))
            z_modal = Q.T @ z_pu @ Q
            # 从原始导纳的互端块独立恢复串联矩阵，避免误用单位长度参数。
            y = port_admittance(items, pair)
            recovered = -np.linalg.inv(y[:3, 3:])
            matrix_check = float(np.max(np.abs(recovered - z)))
            if matrix_check > 1e-8:
                raise ValueError("线路矩阵与实际串联导纳不一致")
            length_m = feet * .3048
        else:
            z_modal, residual = differential_impedance(port_admittance(items, pair), [base[b] for b in pair])
            length_m = 0.
        q_pos = Q.T @ MODE
        record = dict(u=u, v=v, bus_u=pair[0], bus_v=pair[1],
                      elements=[item["name"] for item in items],
                      linecode=items[0].get("linecode", "device-bank"),
                      hop=1., length_m=length_m,
                      impedance_pu=float(np.linalg.norm(z_modal) / np.sqrt(2)),
                      positive_pu=float(abs(q_pos.conj() @ z_modal @ q_pos)),
                      current_projection_residual=residual, line_matrix_max_error_ohm=matrix_check)
        if record["impedance_pu"] <= 0 or record["positive_pu"] <= 0:
            raise ValueError("电气边权必须为正")
        records.append(record)
        graph.edges[u, v].update({k: record[k] for k in NAMES})
    # 复现旧提取器输出，独立验证实际母线索引与保存数据相符。
    old_params = {}
    for item in raw["lines"]:
        pair = frozenset(b.split(".")[0] for b in item["buses"])
        r, x = item["r1"] * item["length"], item["x1"] * item["length"]
        old_params[pair] = (r, x, float(np.hypot(r, x)))
    for item in raw["transformers"]:
        old_params[frozenset(b.split(".")[0] for b in item["buses"])] = (0., item["xhl_percent"], item["xhl_percent"])
    for directory in data_dirs:
        for (u, v), attr in zip(np.load(directory / "edge_index.npy"), np.load(directory / "edge_attr.npy")):
            if not np.allclose(old_params[frozenset((buses[u], buses[v]))], attr[:3], rtol=2e-6):
                raise ValueError("母线顺序或原始参数不能复现保存的边特征")
    return graph, records, buses


def relationship(distance, values, root, tolerance):
    """仅排除真实候选，保留零长度产生的等距错误候选。"""
    keep = np.arange(len(values)) != root
    x, y = np.asarray(distance)[keep], np.asarray(values)[keep]
    farther = x[None, :] > x[:, None] + 1e-10 * np.max(x)
    differences = y[None, :] - y[:, None]
    return dict(spearman=float(spearmanr(x, y).statistic), pearson=float(pearsonr(x, y).statistic),
                pair_violation=float(np.mean(differences[farther] < -tolerance)),
                ordered_pair_count=int(farther.sum()))


def savefig(fig, name, directory):
    """保留便于浏览及论文排版的两种图件格式。"""
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plots(events, graph, matrices, diameter, output):
    """对同一事件的三种距离作配对展示，并汇总完整测试集。"""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for i, metric in enumerate(METRICS):
        for j, dataset in enumerate(("a", "b")):
            ax = axes[i, j]
            rows = [r for r in events if r["metric"] == metric and r["dataset"] == dataset]
            data = [[r["relations"][k]["spearman"] for r in rows] for k in NAMES]
            ax.boxplot(data, labels=list(NAMES.values()), showfliers=False)
            for r in rows:
                ax.plot(range(1, 5), [r["relations"][k]["spearman"] for k in NAMES], alpha=.035, color="#347db2", lw=.6)
            for x, values in enumerate(data, 1):
                ax.text(x, np.median(values)+.06, f"{np.median(values):.3f}", ha="center", fontsize=9)
            ax.axhline(0, color="grey", ls="--", lw=.8)
            ax.set_title(f"{dataset.upper()}：{METRICS[metric]}，228 个事件")
            ax.set_ylabel("错误候选的 Spearman ρ")
            ax.set_ylim(-1.05, 1.1)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("固定候选数值，仅更换距离定义；连线为同一事件；箱体为四分位区间")
    savefig(fig, "correlation-comparison", output)
    selected = {}
    for metric in METRICS:
        rows = [r for r in events if r["metric"] == metric]
        delta = lambda r: r["relations"]["impedance_pu"]["spearman"] - r["relations"]["hop"]["spearman"]
        for category, eligible in (("improved", [r for r in rows if delta(r) > .05]),
                                   ("stable", [r for r in rows if abs(delta(r)) <= .05]),
                                   ("decreased", [r for r in rows if delta(r) < -.05])):
            pool = sorted(eligible, key=lambda r: (delta(r), r["dataset"], r["test_index"]))
            cases = []
            for dataset in ("a", "b"):
                subset = [r for r in pool if r["dataset"] == dataset]
                if subset:
                    cases.append(subset[len(subset)//2])
            rest = [r for r in pool if (r["dataset"], r["group_id"]) not in {(c["dataset"], c["group_id"]) for c in cases}]
            if rest:
                cases.append(rest[len(rest)//2])
            if not cases:
                continue
            selected[f"{metric}-{category}"] = [{k: c[k] for k in ("dataset", "test_index", "group_id", "true_location")} for c in cases]
            fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.8*len(cases)), squeeze=False)
            for row, case in enumerate(cases):
                root, values = case["true_location"], np.array(case["values"])
                for col, key in enumerate(("hop", "length_m", "impedance_pu")):
                    ax = axes[row, col]
                    distance = matrices[key][root] / diameter[key]
                    for u, v in graph.edges():
                        if matrices["hop"][root, u] > matrices["hop"][root, v]:
                            u, v = v, u
                        color = "#d94740" if values[v] < values[u]-case["tolerance"] else "#d1d7df"
                        ax.plot(distance[[u, v]], values[[u, v]], color=color, lw=.8, zorder=1)
                    ax.scatter(distance, values, s=25, c=matrices["hop"][root], cmap="viridis", zorder=3)
                    ax.scatter([0], [values[root]], marker="*", c="red", s=120, zorder=5)
                    for n in range(38):
                        ax.annotate(str(n), (distance[n], values[n]), xytext=(2, 3), textcoords="offset points", fontsize=5.5)
                    rho = case["relations"][key]["spearman"]
                    ax.set_title(f"{case['dataset'].upper()} 测试 {case['test_index']} / 故障 {root} / ρ={rho:.3f}", fontsize=10)
                    ax.set_xlabel(f"{NAMES[key]} / 全网对应直径")
                    ax.set_xlim(-.03, 1.03)
                    ax.set_ylabel(METRICS[metric])
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
                    ax.grid(alpha=.2)
            title = {"improved": "相关性提高超过 0.05", "stable": "相关性变化不超过 0.05", "decreased": "相关性降低超过 0.05"}[category]
            fig.suptitle(f"{METRICS[metric]}：{title}的 {len(cases)} 个案例；红线为向外下降，星号为真实故障点")
            savefig(fig, f"{metric}-{category}", output)
    return selected


def package_outputs(out, audit_dir, summary):
    """生成静态图件目录与包含参数来源的独立交付包。"""
    source = out / "source-data"
    source.mkdir(exist_ok=True)
    for name in ("ieee37.dss", "IEEELineCodes.DSS", "IEEE37_BusXY.csv", "Line Data.xls", "feeder37.zip"):
        shutil.copy2(audit_dir / name, source / name)
    heading = "<meta charset='utf-8'><title>GReF 电气加权距离审计</title>"
    body = ["<h1>GReF 电气加权距离审计</h1><p>每组 228 个测试事件；统一 1 MVA 标幺基准。图中距离按全网对应直径归一化。</p>",
            "<p>阻抗为额定变比下的无零序模态代理；逐边单调性不因正权重改变。</p>",
            "<table><tr><th>对象</th><th>跳数 ρ</th><th>长度 ρ</th><th>模态阻抗 ρ</th></tr>"]
    for metric in METRICS:
        for dataset in ("a", "b"):
            cells = "".join(f"<td>{summary[metric][dataset][key]['spearman_median']:.3f}</td>" for key in ("hop", "length_m", "impedance_pu"))
            body.append(f"<tr><td>{dataset.upper()} {METRICS[metric]}</td>{cells}</tr>")
    body.append("</table><p>表中为逐事件错误候选 Spearman 相关的中位数。</p>")
    for path in sorted((out / "figures").glob("*.png")):
        body.append(f"<h2>{path.stem}</h2><a href='figures/{path.stem}.svg'>SVG 图件</a><img src='figures/{path.name}' loading='lazy'>")
    body.append("<p><a href='candidate-values.csv'>全部候选数值</a> · <a href='analysis.json'>完整统计</a> · <a href='events.json'>逐事件结果</a> · <a href='opendss-raw.json'>原始电气参数</a></p>")
    style = "<style>body{max-width:1400px;margin:30px auto;font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.6}img{width:100%;display:block}td,th{padding:8px 20px;border:1px solid #ddd}table{border-collapse:collapse}</style>"
    (out / "gallery.html").write_text(heading+style+"".join(body), encoding="utf-8")
    with zipfile.ZipFile(out / "figures-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix not in (".zip", ".log"):
                archive.write(path, path.relative_to(out))
        archive.write(source / "feeder37.zip", "source-data/feeder37.zip")


def main():
    """离线分析既有全部候选数值，不更改模型和历史结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=METHOD / "output/gref/electrical-distance-20261103")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "analysis.json").exists():
        raise FileExistsError("拒绝覆盖已完成分析")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit_dir = METHOD / "data/electrical-audit-ieee37-20261103"
    source = METHOD / "output/gref/topology-analysis-20261103-final"
    data_dirs = [METHOD / f"data/{name}" for name in ("s0-ieee37-r0p01-20-loguniform", "s0-ieee37-r10-100-loguniform")]
    raw = json.loads((out / "opendss-raw.json").read_text(encoding="utf-8"))
    graph, edge_records, buses = weights(raw, audit_dir / "Line Data.xls", data_dirs)
    matrices = {k: np.array([[nx.shortest_path_length(graph, u, v, weight=k) for v in graph] for u in graph]) for k in NAMES}
    diameter = {k: float(m.max()) for k, m in matrices.items()}
    np.savez(out / "distances.npz", **matrices, buses=np.array(buses))
    events, candidates, invariance = [], [], 0
    for metric in METRICS:
        for row in json.loads((source / f"events-{metric}.json").read_text(encoding="utf-8")):
            root, values = row["true_location"], row["values"]
            relations = {k: relationship(m[root], values, root, row["tolerance"]) for k, m in matrices.items()}
            assert np.isclose(relations["hop"]["spearman"], row["wrong_candidate_spearman"])
            events.append({**row, "metric": metric, "relations": relations})
            for u, v in row["outward_edges"]:
                assert matrices["impedance_pu"][root, v] > matrices["impedance_pu"][root, u]
                invariance += 1
            for k, value in enumerate(values):
                candidates.append(dict(dataset=row["dataset"], test_index=row["test_index"], group_id=row["group_id"],
                                       metric=metric, true_location=root, candidate=k, bus=buses[k], value=value,
                                       **{name: float(m[root, k]) for name, m in matrices.items()},
                                       **{name+"_normalized": float(m[root, k]/diameter[name]) for name, m in matrices.items()}))
    summary = {}
    for metric in METRICS:
        summary[metric] = {}
        for dataset in ("a", "b"):
            rows = [r for r in events if r["metric"] == metric and r["dataset"] == dataset]
            report = {}
            for key in NAMES:
                rho = np.array([r["relations"][key]["spearman"] for r in rows])
                delta = rho - [r["relations"]["hop"]["spearman"] for r in rows]
                report[key] = dict(n=len(rows), spearman_median=float(np.median(rho)),
                                   spearman_q1=float(np.quantile(rho, .25)), spearman_q3=float(np.quantile(rho, .75)),
                                   pearson_median=float(np.median([r["relations"][key]["pearson"] for r in rows])),
                                   positive_count=int((rho > 0).sum()), delta_median=float(np.median(delta)),
                                   improved_count=int((delta > .05).sum()), stable_count=int((np.abs(delta) <= .05).sum()),
                                   decreased_count=int((delta < -.05).sum()),
                                   pair_violation_median=float(np.median([r["relations"][key]["pair_violation"] for r in rows])),
                                   group_medians={str(g): float(np.median([r["relations"][key]["spearman"] for r in rows if r["group_id"] == g])) for g in sorted({r["group_id"] for r in rows})})
            summary[metric][dataset] = report
    # 剔除低压叶节点的敏感性对照，不重新选取事件或调整主分析权重。
    low_voltage = buses.index("775")
    sensitivity = {}
    for metric in METRICS:
        sensitivity[metric] = {}
        for dataset in ("a", "b"):
            rows = [r for r in events if r["metric"] == metric and r["dataset"] == dataset and r["true_location"] != low_voltage]
            retained = np.arange(38) != low_voltage
            remap = {int(k): i for i, k in enumerate(np.flatnonzero(retained))}
            sensitivity[metric][dataset] = {"n_events": len(rows), "n_wrong_candidates": 36}
            for key, matrix in matrices.items():
                rho = [relationship(matrix[r["true_location"]][retained], np.array(r["values"])[retained],
                                    remap[r["true_location"]], r["tolerance"])["spearman"] for r in rows]
                sensitivity[metric][dataset][key] = float(np.median(rho))
    selected = plots(events, graph, matrices, diameter, figures)
    inputs = [out / "opendss-raw.json", audit_dir / "Line Data.xls", *audit_dir.glob("*.dss"), audit_dir / "IEEELineCodes.DSS",
              source / "events-energy.json", source / "events-reconstruction.json", Path(__file__)]
    result = dict(summary=summary, diameter=diameter, s_base_va=1e6, buses=buses, edge_parameters=edge_records,
                  exclude_low_voltage_leaf_sensitivity=sensitivity,
                  verified_outward_edges=invariance, selected=selected,
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  source_url="https://ewh.ieee.org/soc/pes/dsacom/testfeeders/feeder37.zip",
                  distance_definition="sum edge Frobenius norm of zero-sequence-free nominal modal impedance / sqrt(2)")
    for name, data in (("analysis.json", result), ("events.json", events)):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "candidate-values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    package_outputs(out, audit_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
