"""用一个真实事件的一条路径解释候选响应差异图的读法。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def main():
    """保留真实数据，仅简化为一条路径和一种距离。"""
    method = Path(__file__).resolve().parents[1]
    out = method / "output/oracle-distance/ieee37-20261103/final"
    rows = json.loads((out / "events.json").read_text(encoding="utf-8"))
    event = next(r for r in rows if r["dataset"] == "a" and r["coordinate"] == "pu" and r["test_index"] == 647)
    with np.load(method / "output/gref/electrical-distance-20261103/final/distances.npz") as saved:
        buses, distances = saved["buses"].tolist(), saved["impedance_pu"]
    graph = nx.Graph(event["outward_edges"])
    root, near, far = event["true_location"], buses.index("799r"), buses.index("799")
    path = nx.shortest_path(graph, root, far)
    x, y = distances[root, path], np.array(event["values"])[path]
    assert path[-2:] == [near, far] and x[-1] > x[-2] and y[-1] < y[-2]
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 3.2]})
    ax = axes[0]
    colors = ["#d9473f" if n == root else "#e6a02a" if n == near else "#287cb1" if n == far else "#dfe5ec" for n in path]
    for i in range(len(path)-1):
        ax.annotate("", (i+1-.18, .5), (i+.18, .5), arrowprops=dict(arrowstyle="->", color="#7a8795", lw=1.5))
    ax.scatter(range(len(path)), [.5]*len(path), c=colors, s=1050, zorder=3)
    for i, n in enumerate(path):
        ax.text(i, .5, buses[n], ha="center", va="center", fontsize=10)
    for n, text in ((root, "真实故障点"), (near, "较近候选"), (far, "更远候选")):
        ax.text(path.index(n), .83, text, ha="center", fontsize=10)
    ax.set_xlim(-.5, len(path)-.5)
    ax.set_ylim(.05, 1.15)
    ax.axis("off")
    ax = axes[1]
    ax.plot(x, y, "o-", color="#aab6c3", lw=2, markersize=6)
    ax.plot(x[-2:], y[-2:], color="#d9473f", lw=3, zorder=3)
    ax.scatter(x[-2:], y[-2:], c=colors[-2:], s=110, zorder=4)
    ax.scatter([0], [0], marker="*", c="#d9473f", s=180, zorder=5)
    ax.annotate(f"799r：较近\n距离 {x[-2]:.5f}\n差异 {y[-2]:.5f}", (x[-2], y[-2]), xytext=(x[-2]-.006, y[-2]+.0008), fontsize=11,
                bbox=dict(boxstyle="round,pad=.4", fc="#fff6e4", ec="#e6a02a"))
    ax.annotate(f"799：更远\n距离 {x[-1]:.5f}\n差异 {y[-1]:.6f}", (x[-1], y[-1]), xytext=(x[-1]-.003, y[-1]-.005), fontsize=11,
                bbox=dict(boxstyle="round,pad=.4", fc="#eef6fc", ec="#287cb1"))
    drop = 100*(1-y[-1]/y[-2])
    ax.text(.025, .013, f"距离变大\n差异反而下降\n{drop:.1f}%", color="#bf3831", fontsize=12, ha="left")
    for n, xx, yy in zip(path[1:-2], x[1:-2], y[1:-2]):
        ax.annotate(buses[n], (xx, yy), xytext=(3, 6), textcoords="offset points", fontsize=9)
    ax.annotate("真实故障 709\n与自身差异为 0", (0, 0), xytext=(.0008, .0014), fontsize=10)
    ax.set_xlim(-.0006, .030)
    ax.set_ylim(-.0006, .0205)
    ax.set_xlabel("离真实故障点 709 的电气距离（pu）——越往右越远", fontsize=11)
    ax.set_ylabel("真实响应差异（pu²）\n越往上差异越大", fontsize=11)
    ax.grid(alpha=.2)
    fig.suptitle("只看一条路径：候选更远，真实响应差异却更小\nA 组第 17 族；节点使用实际母线名；未使用模型预测", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .93])
    for extension in ("png", "svg"):
        fig.savefig(out / f"reading-guide.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
