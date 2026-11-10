"""绘制 S1 拓扑边屏蔽鲁棒性图（两张独立图）。

图 1：屏蔽 0/1/2/3 条无向边时测试集 Top-1 的均值、最小值与最大值。
图 2：单条边屏蔽时 15 条无向边各自的 Top-1 变化量（相对 S0），按变化量升序排列。

数据全部从既有运行产物现算，不使用任何硬编码数值：
- 逐组合结果与分档均值：output/candidate-score-full-lg-run3-seed342-ckpt/s1_topk_report_1_2_3_edges.json
- S0 基线：output/candidate-score-full-lg-run3-seed342-ckpt/s1_edge_mask_sweep.json

用法：
    python code/method-a2/scripts/plot_s1_robustness.py
    python code/method-a2/scripts/plot_s1_robustness.py --run-dir <另一个运行目录>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图

import matplotlib.pyplot as plt

# 中文字体与负号显示：按 Windows、macOS、Linux 的常见字体依次回退
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 配色：与项目其他图件保持同一套克制色板
C_MAIN = "#2f5f87"
C_BAD = "#a4534a"
C_ZERO = "#8a8a80"
C_GRID = "#e6e6e6"
C_MUTED = "#555555"


def load_runs(run_dir: Path) -> tuple[float, list[dict]]:
    """读取 S0 基线（百分数）与逐组合记录。"""
    report_path = run_dir / "s1_topk_report_1_2_3_edges.json"
    base_path = run_dir / "s1_edge_mask_sweep.json"

    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    with base_path.open("r", encoding="utf-8") as fh:
        sweep = json.load(fh)

    baseline = float(sweep["baseline"]["top1"]) * 100.0
    records = []
    for rec in report["records"]:
        edges = rec["屏蔽边"]
        records.append(
            {
                "edges": list(edges),
                "k": len(edges),
                "top1": float(rec["测试集"]["top1"]) * 100.0,
                "top3": float(rec["测试集"]["top3"]) * 100.0,
            }
        )
    return baseline, records


def summarize(baseline: float, records: list[dict]) -> dict[int, dict]:
    """按屏蔽边数聚合：均值、最小值、最大值与组合数。"""
    groups: dict[int, list[float]] = {}
    for rec in records:
        groups.setdefault(rec["k"], []).append(rec["top1"])

    summary = {}
    for k in sorted(groups):
        vals = groups[k]
        summary[k] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }
    summary[0] = {"n": 1, "mean": baseline, "min": baseline, "max": baseline}
    return dict(sorted(summary.items()))


def plot_edge_count(summary: dict[int, dict], baseline: float, out_dir: Path) -> Path:
    """图 1：屏蔽边数与 Top-1 的范围、均值。"""
    ks = list(summary.keys())
    means = [summary[k]["mean"] for k in ks]
    lows = [summary[k]["min"] for k in ks]
    highs = [summary[k]["max"] for k in ks]
    counts = [summary[k]["n"] for k in ks]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.fill_between(
        ks,
        lows,
        highs,
        color=C_MAIN,
        alpha=0.10,
        linewidth=0,
        label="全部组合的最小—最大范围",
    )
    ax.plot(ks, means, color=C_MAIN, linewidth=1.1, zorder=3)
    ax.plot(
        ks,
        means,
        linestyle="none",
        marker="D",
        markersize=3.4,
        color=C_MAIN,
        zorder=4,
        label="均值",
    )
    ax.axhline(
        baseline,
        color="#777777",
        linewidth=0.6,
        linestyle=(0, (4, 2.5)),
        zorder=2,
        label=f"S0 = {baseline:.1f}",
    )

    # 标注各档的均值与极值
    lowest = min(lows)
    for k, mean, lo, hi in zip(ks, means, lows, highs):
        if k == 0:
            continue
        ax.annotate(
            f"{mean:.1f}",
            xy=(k, mean),
            xytext=(0, -9),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6,
            color=C_MAIN,
        )
        ax.annotate(
            f"{hi:.1f}",
            xy=(k, hi),
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=5.5,
            color=C_MUTED,
        )
        # 最低点若放在下方会被坐标轴截断，改放到顶点上方、范围带内侧
        is_lowest = lo <= lowest + 1e-9
        ax.annotate(
            f"{lo:.1f}",
            xy=(k, lo),
            xytext=(0, 8) if is_lowest else (-3, -9),
            textcoords="offset points",
            ha="center" if is_lowest else "right",
            va="bottom" if is_lowest else "top",
            fontsize=5.5,
            color=C_MUTED,
        )

    ax.set_xlim(-0.35, 3.45)
    ax.set_ylim(40, 71)
    ax.set_xticks(ks)
    # 用“组”代替 n，直接说明每档是枚举出来的组合数
    ax.set_xticklabels([f"{k}\n({c} 组)" for k, c in zip(ks, counts)], fontsize=7)
    ax.set_xlabel("屏蔽的无向边数量", fontsize=8)
    ax.set_ylabel("测试集 Top-1（%）", fontsize=8)
    ax.set_title("拓扑边屏蔽下的定位精度", fontsize=9.5, pad=8)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        loc="lower left",
        fontsize=5.6,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.6,
    )

    out = out_dir / "fig1-s1-edge-count.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_edge(baseline: float, records: list[dict], out_dir: Path) -> Path:
    """图 2：单条边屏蔽时各边的 Top-1 变化量。"""
    single = [r for r in records if r["k"] == 1]
    single.sort(key=lambda r: r["top1"])
    deltas = [r["top1"] - baseline for r in single]
    ranks = list(range(1, len(single) + 1))

    colors = [C_BAD if d < -1e-9 else (C_MAIN if d > 1e-9 else C_ZERO) for d in deltas]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.barh(ranks, deltas, height=0.62, color=colors, linewidth=0)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.invert_yaxis()

    # 每根条末端标注数值
    for bar, delta in zip(bars, deltas):
        y = bar.get_y() + bar.get_height() / 2
        if delta < 0:
            ax.annotate(
                f"{delta:.1f}",
                xy=(delta, y),
                xytext=(-2, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=5.5,
                color=C_MUTED,
            )
        elif delta > 0:
            ax.annotate(
                f"+{delta:.1f}",
                xy=(delta, y),
                xytext=(2, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=5.5,
                color=C_MUTED,
            )
        else:
            ax.annotate(
                "0.0",
                xy=(0.0, y),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=5.5,
                color=C_MUTED,
            )

    # 零变化量没有柱子可画，单独用圆点标出，避免整行空白
    zero_ranks = [r for r, d in zip(ranks, deltas) if abs(d) < 1e-9]
    if zero_ranks:
        ax.plot(
            [0.0] * len(zero_ranks),
            zero_ranks,
            linestyle="none",
            marker="o",
            markersize=2.8,
            color="#666666",
            zorder=5,
        )

    ax.set_yticks(ranks)
    ax.set_yticklabels([str(r) for r in ranks], fontsize=5.5)
    ax.set_ylim(len(ranks) + 0.6, 0.4)
    ax.set_xlim(-8.4, 1.9)
    ax.set_xlabel("ΔTop-1（百分点，相对 S0）", fontsize=8)
    ax.set_ylabel("单边组合（按变化量升序）", fontsize=8)
    ax.set_title(f"单条边屏蔽的定位影响（{len(single)} 条无向边）", fontsize=9.5, pad=8)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out = out_dir / "fig2-s1-single-edge.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """入口：解析参数、读数据、出两张图并打印核对用摘要。"""
    method_dir = Path(__file__).resolve().parents[1]
    default_run = method_dir / "output" / "candidate-score-full-lg-run3-seed342-ckpt"

    parser = argparse.ArgumentParser(description="绘制 S1 边屏蔽鲁棒性图")
    parser.add_argument("--run-dir", type=Path, default=default_run, help="运行产物目录")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=method_dir / "output" / "visualizations",
        help="输出目录",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline, records = load_runs(args.run_dir)
    summary = summarize(baseline, records)

    print(f"S0 基线：{baseline:.2f}%")
    for k, s in summary.items():
        print(
            f"  {k} 条边：{s['n']:>3} 组，均值 {s['mean']:.2f}，"
            f"最小 {s['min']:.2f}，最大 {s['max']:.2f}"
        )

    single = sorted((r for r in records if r["k"] == 1), key=lambda r: r["top1"])
    print("单边屏蔽（按 Top-1 升序）：")
    for rec in single:
        print(f"  {rec['top1'] - baseline:+6.2f}  {' + '.join(rec['edges'])}")

    fig1 = plot_edge_count(summary, baseline, args.out_dir)
    fig2 = plot_single_edge(baseline, records, args.out_dir)
    print(f"已写出：{fig1}")
    print(f"已写出：{fig2}")


if __name__ == "__main__":
    main()
