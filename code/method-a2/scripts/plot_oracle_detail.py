"""放大真实响应差异主体，避免低压末端极值遮蔽其他候选的变化。"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_oracle_distance import DISTANCES, CATEGORIES, METHOD
from scripts.analyze_gref_electrical_distance import savefig


def main():
    """仅改变展示范围，统计和候选集合仍使用完整分析。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=METHOD / "output/oracle-distance/ieee37-20261103/final")
    args = parser.parse_args()
    out = args.output.resolve()
    meta = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "events.json").read_text(encoding="utf-8"))
    with np.load(Path(meta["distance_source"]) / "distances.npz") as saved:
        matrices = {k: saved[k] for k in DISTANCES}
        buses = saved["buses"].tolist()
    leaf = buses.index("775")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    additions = []
    for category, selected in meta["selected"].items():
        if not selected:
            continue
        cases = [next(r for r in rows if r["dataset"] == s["dataset"] and r["test_index"] == s["test_index"] and r["coordinate"] == "pu") for s in selected]
        fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.9*len(cases)), squeeze=False)
        for i, row in enumerate(cases):
            values, root = np.array(row["values"]), row["true_location"]
            retained = np.arange(38) != leaf
            limit = float(values[retained].max()) * 1.12
            for j, key in enumerate(DISTANCES):
                ax = axes[i, j]
                distance = matrices[key][root] / matrices[key].max()
                for (u, v), delta in zip(row["outward_edges"], row["edge_deltas"]):
                    if leaf in (u, v):
                        continue
                    color = "#d73935" if delta < -row["tolerance"] else ("#e59b26" if abs(delta) <= row["tolerance"] else "#b9c4d0")
                    ax.plot(distance[[u, v]], values[[u, v]], color=color, lw=1, zorder=1)
                ax.scatter(distance[retained], values[retained], c=values[retained], cmap="viridis", s=30, zorder=3)
                ax.scatter([0], [0], c="red", marker="*", s=100, zorder=4)
                for n in np.flatnonzero(retained):
                    ax.annotate(str(n), (distance[n], values[n]), xytext=(2, 3), textcoords="offset points", fontsize=6)
                ax.set_xlim(-.03, 1.03)
                ax.set_ylim(-limit*.04, limit)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
                ax.set_xlabel(DISTANCES[key]+" / 全网对应直径")
                ax.set_ylabel("真实响应均方差（pu²）")
                ax.set_title(f"{row['dataset'].upper()} 族 {row['group_id']} / 故障 {root} | 全候选 ρ={row['relations'][key]['spearman']:.3f}", fontsize=10)
                ax.grid(alpha=.2)
        fig.suptitle(f"{CATEGORIES[category]}：主体候选纵轴放大图\n775 候选极值不在本图展示范围；分类与相关统计仍包含完整候选，参见对应全范围图", fontsize=12)
        name = "oracle-"+category+"-detail"
        savefig(fig, name, out / "figures")
        additions.append(f"<h2>{CATEGORIES[category]}：主体放大</h2><a href='figures/{name}.svg'>SVG</a><img src='figures/{name}.png'>")
    gallery = out / "gallery.html"
    gallery.write_text(gallery.read_text(encoding="utf-8")+"".join(additions), encoding="utf-8")
    (out / "detail-plot-provenance.json").write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256((out / "analysis.json").read_bytes()).hexdigest(), displayed_leaf_exclusion="775 (plot only)"), indent=2), encoding="utf-8")
    with zipfile.ZipFile(out / "figures-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.rglob("*")):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.relative_to(out))


if __name__ == "__main__":
    main()
