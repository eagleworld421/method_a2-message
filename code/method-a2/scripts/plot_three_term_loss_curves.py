"""绘制GRReF 模型的逐轮损失曲线：能量、排序、平滑与总损失。

只使用各运行目录已保存的 `train_log.csv`，不改动训练代码或既有产物。图中
各分量按配置权重折算为对总损失的贡献，使四条曲线满足
``总损失 = 能量 + 0.1 × 排序 + 0.1 × 平滑``；实线为训练侧、虚线为验证侧。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METHOD = Path(__file__).resolve().parents[1]
if str(METHOD) not in sys.path:
    sys.path.insert(0, str(METHOD))

RUNS = (
    ("A", "0.01–20 Ω", METHOD / "output/gref/three-term-ieee37-a-seed342"),
    ("B", "10–100 Ω", METHOD / "output/gref/three-term-ieee37-b-seed342"),
)
COLORS = {"energy": "#1f77b4", "rank": "#ff7f0e", "smooth": "#2ca02c", "total": "#111111"}
LABELS = {
    "energy": "能量项",
    "rank": "排序项",
    "smooth": "平滑项",
    "total": "总损失",
}


def parse_args(argv=None) -> argparse.Namespace:
    """解析输出目录。"""
    parser = argparse.ArgumentParser(description="绘制GRReF 的训练与验证曲线")
    parser.add_argument("--output", required=True, help="输出目录，必须尚不存在")
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


def _read_log(run_dir: Path) -> dict[str, np.ndarray]:
    """读取逐轮损失并折算为加权分量。"""
    with (run_dir / "train_log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epoch = np.array([int(row["epoch"]) for row in rows])
    energy = np.array([float(row["train_energy_mean"]) for row in rows])
    rank = np.array([float(row["train_rank_mean"]) for row in rows])
    smooth = np.array([float(row["train_smooth_mean"]) for row in rows])
    total = np.array([float(row["train_batch_total_mean"]) for row in rows])
    val_energy = np.array([float(row["val_energy"]) for row in rows])
    val_rank = np.array([float(row["val_rank"]) for row in rows])
    val_smooth = np.array([float(row["val_smooth"]) for row in rows])
    val_total = np.array([float(row["val_total"]) for row in rows])
    values = {
        "epoch": epoch,
        "energy": energy, "rank": 0.1 * rank, "smooth": 0.1 * smooth,
        "total": total,
        "val_energy": val_energy, "val_rank": 0.1 * val_rank,
        "val_smooth": 0.1 * val_smooth, "val_total": val_total,
    }
    for prefix in ("", "val_"):
        reconstructed = values[prefix + "energy"] + values[prefix + "rank"] + values[prefix + "smooth"]
        if not np.allclose(reconstructed, values[prefix + "total"], rtol=1e-5, atol=1e-7):
            raise ValueError("总损失与三项加权贡献不一致，请核对权重口径")
    return values


def main(argv=None) -> int:
    """绘制 1×2 面板并写出 PNG/SVG。"""
    args = parse_args(argv)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=False)
    for axis, (label, resistance, run_dir) in zip(axes, RUNS):
        values = _read_log(run_dir)
        epoch = values["epoch"]
        for key in ("energy", "rank", "smooth", "total"):
            axis.plot(epoch, values[key], color=COLORS[key], lw=1.7, label=LABELS[key])
            axis.plot(epoch, values["val_" + key], color=COLORS[key], lw=1.1,
                      ls="--", alpha=.85)
        axis.set_yscale("log")
        axis.set_xlabel("训练轮次", fontsize=10)
        axis.set_ylabel("加权损失（对数坐标）", fontsize=10)
        axis.set_title(f"{label} 组（{resistance}）", fontsize=12)
        axis.grid(alpha=.25, which="both")
        axis.set_xlim(1, int(epoch.max()))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        best = int(config["best_epoch"])
        axis.axvline(best, color="#888888", ls=":", lw=1)
        axis.text(best, axis.get_ylim()[1], f" 最佳轮次 {best}", fontsize=8,
                  color="#555555", va="top")
    handles = [
        Line2D([0], [0], color=COLORS[key], lw=1.8, label=LABELS[key])
        for key in ("energy", "rank", "smooth", "total")
    ]
    handles += [
        Line2D([0], [0], color="#777777", lw=1.8, label="实线：训练"),
        Line2D([0], [0], color="#777777", lw=1.2, ls="--", label="虚线：验证"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
               columnspacing=1.6)
    fig.suptitle("三项损失曲线；各分量已按权重 (1, 0.1, 0.1) 折算",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output / "three-term-loss-curves.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "three-term-loss-curves.svg", bbox_inches="tight", facecolor="white")
    (output / "README.md").write_text(
        "# 三项损失曲线\n\n"
        "- `three-term-loss-curves.png` / `.svg`：A、B 两组GRReF 模型的逐轮曲线。\n"
        "- 实线为训练侧、虚线为验证侧；能量项权重 1，排序项与平滑项权重 0.1，"
        "所以四条曲线满足总损失 = 三项加权贡献之和。\n"
        "- 纵轴为对数坐标；竖直虚线为按验证 Top-1 选出的最佳轮次（A 193、B 160）。\n"
        "- 数据来源：`three-term-ieee37-a-seed342/train_log.csv` 与 `.../b-...`。\n",
        encoding="utf-8")
    print(f"已写出 {output / 'three-term-loss-curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
