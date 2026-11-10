"""按 IEEE37 母线坐标重绘拓扑，并把母线编号替换为候选索引。

数据来源为项目已有的 IEEE37 官方母线坐标 `IEEE37_BusXY.csv`、真实拓扑边表与
候选顺序（`distances.npz` 的 `buses`）。图中每个节点旁的数字为候选索引 0–37；
索引 38 为正常候选，不对应母线，因此不出现在拓扑中。原 10.1109/TPWRD.2011.2165972
图件仅作为版式参考，本脚本不读取该位图，也不修改任何既有文件。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

METHOD = Path(__file__).resolve().parents[1]
if str(METHOD) not in sys.path:
    sys.path.insert(0, str(METHOD))

# 节点标签偏移（横轴已按 x-scale 缩放），用于避免拥挤区域的索引重叠。
LABEL_OFFSETS = {
    "799": (0.11, 0.03), "799r": (-0.11, -0.02), "701": (0.09, 0.00),
    "702": (0.10, -0.11), "705": (0.00, -0.13), "713": (0.00, 0.13),
    "703": (0.10, 0.00), "727": (0.00, 0.13), "744": (0.00, 0.13),
    "729": (-0.09, 0.00), "728": (0.00, -0.13), "730": (0.09, 0.00),
    "704": (0.00, 0.13), "714": (0.09, -0.11), "718": (0.00, -0.13),
    "720": (0.09, 0.00), "706": (0.09, 0.00), "725": (0.09, 0.00),
    "707": (0.09, 0.00), "724": (0.09, 0.00), "722": (-0.09, 0.00),
    "708": (-0.09, 0.11), "732": (-0.09, 0.00), "733": (0.09, -0.04),
    "734": (0.09, 0.00), "710": (-0.09, 0.00), "735": (0.00, -0.13),
    "736": (-0.09, 0.10), "737": (0.00, -0.13), "738": (0.00, -0.13),
    "711": (0.00, -0.13), "740": (0.00, 0.13), "741": (0.09, 0.00),
    "742": (-0.09, 0.00), "712": (0.00, 0.13), "731": (0.09, 0.00),
    "709": (0.11, -0.09), "775": (0.00, -0.12),
}


def parse_args(argv=None) -> argparse.Namespace:
    """解析输出目录、坐标与边表路径。"""
    parser = argparse.ArgumentParser(description="重绘 IEEE37 拓扑并把母线编号替换为候选索引")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--bus-xy", default=str(METHOD / "data/electrical-audit-ieee37-20261103/IEEE37_BusXY.csv"))
    parser.add_argument("--data-dir", default=str(METHOD / "data/s0-ieee37-r0p01-20-loguniform"))
    parser.add_argument("--buses-npz", default=str(METHOD / "output/gref/electrical-distance-20261103/final/distances.npz"))
    parser.add_argument("--x-scale", type=float, default=2.7, help="横轴放大倍数，使版式接近参考图")
    return parser.parse_args(argv)


def _resolve(value: str) -> Path:
    """把相对路径解析为项目内路径。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), METHOD.parents[1], METHOD):
        if (base / path).exists():
            return (base / path).resolve()
    return (Path.cwd() / path).resolve()


def _read_coordinates(path: Path) -> dict[str, tuple[float, float]]:
    """读取母线与画布坐标。"""
    coordinates = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            coordinates[row[0].strip()] = (float(row[1]), float(row[2]))
    return coordinates


def _transformer(ax, start, end, radius: float, color: str = "black") -> None:
    """在边的中点画双圆变压器符号。"""
    middle = (start + end) / 2.0
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    for sign in (-1.0, 1.0):
        center = middle + sign * radius * 1.05 * direction
        ax.add_patch(Circle(center, radius, facecolor="white", edgecolor=color, lw=1.0, zorder=4))


def _regulator(ax, start, end, size: float) -> None:
    """在边的中点画调压器方框。"""
    middle = (start + end) / 2.0
    ax.add_patch(Rectangle(middle - size / 2.0, size, size, facecolor="white",
                           edgecolor="black", lw=1.0, zorder=4))
    ax.text(middle[0], middle[1], "R", fontsize=7, ha="center", va="center", zorder=5)


def main(argv=None) -> int:
    """绘制拓扑并写出 PNG/SVG、索引对照表与说明。"""
    args = parse_args(argv)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    coordinates = _read_coordinates(_resolve(args.bus_xy))
    buses = [str(name) for name in np.load(_resolve(args.buses_npz))["buses"].tolist()]
    index_of = {bus: index for index, bus in enumerate(buses)}
    edges = np.load(_resolve(args.data_dir) / "edge_index.npy")
    pairs = {tuple(sorted(pair)) for pair in map(tuple, edges.tolist())}
    scale = float(args.x_scale)

    points: dict[str, np.ndarray] = {}
    for bus in buses:
        x, y = coordinates[bus]
        points[bus] = np.array([x * scale, y], dtype=float)
    # 顶部三节点与 775 在原始坐标中几乎重合，作图时略微拉开以显示索引。
    points["SourceBus"] = np.array([0.0, 0.92])
    points["799"] = np.array([0.0, 0.40])
    points["799r"] = np.array([0.0, 0.15])
    points["775"] = np.array([0.0, -5.52])

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8.8, 9.2))
    for u, v in sorted(pairs):
        a, b = points[buses[u]], points[buses[v]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.25, zorder=1,
                solid_capstyle="round")
    # 电源变压器（sourcebus–799）、709–775 变压器、799–799r 调压器。
    _transformer(ax, points["SourceBus"], points["799"], radius=0.09)
    _transformer(ax, points["709"], points["775"], radius=0.09)
    _regulator(ax, points["799"], points["799r"], size=0.16)
    # 电源符号。
    ax.plot([-0.42, 0.42], [points["SourceBus"][1] + 0.14] * 2, color="black", lw=3.2,
            solid_capstyle="butt", zorder=3)
    ax.text(0.50, points["SourceBus"][1] + 0.14, "sourcebus（非候选）", fontsize=8,
            color="#555555", va="center")

    for bus in buses:
        x, y = points[bus]
        ax.scatter([x], [y], s=52, color="black", zorder=5)
        dx, dy = LABEL_OFFSETS.get(bus, (0.09, 0.0))
        horizontal = "left" if dx >= 0 else "right"
        ax.text(x + dx, y + dy, str(index_of[bus]), fontsize=10, fontweight="bold",
                ha=horizontal, va="center", zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("IEEE 37 节点测试馈线：候选索引拓扑", fontsize=14, pad=14)
    ax.text(0.5, -0.035,
            "节点旁数字为候选索引（0–37）；索引 38 为正常候选，不对应母线，不在拓扑中。\n"
            "双圆：变压器；R：调压器；灰色横条为电源侧。",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333333")
    fig.tight_layout()
    fig.savefig(output / "ieee37_topology_index.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "ieee37_topology_index.svg", bbox_inches="tight", facecolor="white")
    with (output / "index_bus_mapping.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_index", "bus", "x", "y"])
        for bus in buses:
            writer.writerow([index_of[bus], bus, f"{coordinates[bus][0]:.4f}", f"{coordinates[bus][1]:.4f}"])
    (output / "README.md").write_text(
        "# IEEE37 候选索引拓扑图\n\n"
        "- `ieee37_topology_index.png` / `.svg`：按官方母线坐标重绘，节点旁数字为候选索引。\n"
        "- `index_bus_mapping.csv`：候选索引与母线编号的完整对照。\n"
        "- 索引 38 为正常候选，不对应母线。\n"
        "- 顶部 799/799r 与电源侧在原坐标中几乎重合，作图时略微拉开；变压器与调压器以双圆与 R 符号表示。\n",
        encoding="utf-8")
    print(f"已写出 {output / 'ieee37_topology_index.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
