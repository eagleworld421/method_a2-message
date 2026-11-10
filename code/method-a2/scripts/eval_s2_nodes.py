"""在冻结模型上评估 S2 部分节点观测，不修改数据、模型参数与真实拓扑。

执行顺序固定为：先核对输入数据、模型配置、标准化、测试划分和节点掩码，任一检查
不通过即停止；全部通过后才计算完整观测基线，并按每个观测规模逐组掩码推理。输出
`s2_node_metrics.json`、`s2_node_metrics.md`、`node_masks.npz` 与检查记录。

两个冻结模型必须使用同一份 `node_masks.npz`：首个运行按固定种子生成并落盘，后续
运行通过 `--node-masks` 读取该文件，并核对文件内容与同一种子重新生成的结果一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.data import load_arrays, make_batch
from src.gref.model import GrefPredictor
from src.training.evaluation import rank_metrics

# 计划要求记录的名次与间隔指标
METRIC_KEYS = (
    "top1",
    "top3",
    "top5",
    "mean_rank",
    "median_rank",
    "signed_margin_median",
    "signed_margin_negative_ratio",
)
# 计划要求报告相对完整观测基线变化的指标
DELTA_KEYS = ("top1", "top3", "top5", "mean_rank")
MODEL_LABELS = {"gref": "标准 GReF", "three-term": "三项损失"}


def parse_args() -> argparse.Namespace:
    """解析 S2 节点观测评估参数。"""
    parser = argparse.ArgumentParser(description="评估冻结 GReF 在 S2 部分节点观测下的定位表现")
    parser.add_argument("--run-dir", type=Path, required=True, help="冻结模型运行目录")
    parser.add_argument("--data-dir", type=Path, default=None, help="数据目录；缺省读取运行配置")
    parser.add_argument("--output-dir", type=Path, default=None, help="评估输出目录")
    parser.add_argument("--observed-nodes", type=int, nargs="+", required=True,
                        help="每个 S2 条件保留的观测节点数，按给定顺序编号掩码种子")
    parser.add_argument("--masks-per-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--node-masks", type=Path, default=None,
                        help="复用既有 node_masks.npz，保证多个模型使用完全相同的掩码")
    parser.add_argument("--reference-run-dir", type=Path, default=None,
                        help="对照运行目录，用于核对测试划分与模型配置一致")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-events", type=int, default=None,
                        help="仅评估前若干测试事件；缺省时评估完整测试集")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true",
                        help="只执行执行前一致性检查，不创建输出目录、不计算指标")
    parser.add_argument("--checks-output", type=Path, default=None,
                        help="仅检查模式下检查记录的写出路径")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算文件内容哈希，用于核对原始数据未被改写。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """按连续内存字节计算数组哈希；不依赖 npz 压缩时间戳。"""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def data_fingerprint(directory: Path) -> dict:
    """记录数据目录内数组与元数据的尺寸、修改时间和内容哈希。"""
    names = sorted({p.name for p in directory.glob("*.npy")}
                   | {p.name for p in directory.glob("*.npz")}
                   | {"meta.json"})
    fingerprint = {}
    for name in names:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"数据目录缺少 {name}")
        stat = path.stat()
        fingerprint[name] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        }
    return fingerprint


def add_check(checks: list[dict], name: str, passed: bool, detail: str = "") -> None:
    """登记一条检查结果，detail 只写可复核的简短说明。"""
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def build_node_masks(
    n_nodes: int, observed_nodes: int, count: int, seed: int,
) -> np.ndarray:
    """按固定种子生成去重的节点观测掩码，返回 `[count,n_nodes]`。

    每个掩码从 `n_nodes` 个节点中无放回抽取 `observed_nodes` 个，节点索引升序记录；
    同一观测规模内不重复。相同参数必然得到相同数组，便于跨模型复用。
    """
    if not 1 <= observed_nodes <= n_nodes:
        raise ValueError("observed_nodes 必须位于 [1,n_nodes]")
    if count < 1:
        raise ValueError("masks_per_level 必须为正整数")
    rng = np.random.default_rng(seed)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    attempts = 0
    while len(unique) < count and attempts < count * 100:
        attempts += 1
        selected = tuple(sorted(rng.choice(n_nodes, size=observed_nodes, replace=False).tolist()))
        if selected in unique:
            continue
        mask = np.zeros(n_nodes, dtype=np.float32)
        mask[list(selected)] = 1.0
        unique[selected] = mask
    if len(unique) != count:
        raise RuntimeError("无法生成足够的不重复节点掩码")
    return np.stack(list(unique.values())).astype(np.float32)


def validate_mask_level(
    masks: np.ndarray, observed_nodes: int, n_nodes: int,
) -> list[str]:
    """核对单级掩码的形状、二值性、观测数、索引顺序与级内去重，返回问题列表。"""
    problems = []
    if masks.ndim != 2 or masks.shape[1] != n_nodes:
        return [f"形状为 {getattr(masks, 'shape', None)}，要求 [*,{n_nodes}]"]
    if not np.isin(masks, [0.0, 1.0]).all():
        problems.append("存在非零一取值")
    counts = masks.sum(axis=1)
    if not np.all(counts == observed_nodes):
        problems.append(f"观测节点数不全为 {observed_nodes}")
    indices = [np.flatnonzero(row) for row in masks]
    if any(not np.all(np.diff(index) > 0) for index in indices):
        problems.append("节点索引未严格升序")
    if len({tuple(index.tolist()) for index in indices}) != len(indices):
        problems.append("级内存在重复掩码")
    return problems


def load_mask_file(
    path: Path, levels: list[int], n_nodes: int,
) -> dict[int, np.ndarray]:
    """读取共享掩码文件，缺失的观测规模直接报错。"""
    archive = np.load(path)
    masks: dict[int, np.ndarray] = {}
    for level in levels:
        key = str(level)
        if key not in archive.files:
            raise KeyError(f"掩码文件缺少观测规模 {key}，现有键为 {archive.files}")
        array = np.asarray(archive[key], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != n_nodes:
            raise ValueError(f"观测规模 {level} 的掩码形状 {array.shape} 与节点数 {n_nodes} 不一致")
        masks[level] = array
    return masks


def ensure_finite_metrics(metrics: dict, where: str) -> None:
    """拒绝非有限指标，避免把 NaN 或 inf 写入正式结果。"""
    for key in METRIC_KEYS:
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"{where} 的指标 {key} 非有限：{value}")


@torch.no_grad()
def evaluate_mask(
    model: GrefPredictor, data: dict, indices: np.ndarray, normalization: dict,
    device: torch.device, node_mask: np.ndarray, batch_size: int, n_fault: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """在固定节点观测掩码下计算定位指标，并返回逐事件名次与 signed margin。"""
    scores, locations = [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        batch = make_batch(data, chunk, normalization, device, graph_view="true")
        batch["node_mask"] = torch.as_tensor(
            np.repeat(node_mask[None, :], len(chunk), axis=0),
            dtype=torch.float32, device=device,
        )
        output = model(
            batch["observation"], batch["edge_index"], batch["edge_attr"],
            edge_mask=batch["edge_mask"], node_mask=batch["node_mask"],
            candidate_mask=batch["candidate_mask"],
        )
        valid = batch["candidate_mask"][:, :n_fault].bool()
        value = torch.where(valid, output["candidate_score"][:, :n_fault], torch.inf)
        scores.append(value.cpu().numpy())
        locations.append(batch["location"].cpu().numpy())
    scores_array = np.concatenate(scores)
    locations_array = np.concatenate(locations)
    order = np.argsort(scores_array, axis=1, kind="stable")
    ranks = (order == locations_array[:, None]).nonzero()[1] + 1
    sorted_scores = np.take_along_axis(scores_array, order, axis=1)
    true_scores = scores_array[np.arange(scores_array.shape[0]), locations_array]
    hardest = np.where(order[:, 0] == locations_array, sorted_scores[:, 1], sorted_scores[:, 0])
    # 与 rank_metrics 的间隔口径一致：最佳错误候选能量减正确候选能量
    signed_margins = hardest - true_scores
    metrics = rank_metrics(scores_array, locations_array)
    if (abs(float(np.median(signed_margins)) - metrics["signed_margin_median"]) > 1e-6
            or abs(float((signed_margins < 0).mean())
                   - metrics["signed_margin_negative_ratio"]) > 1e-9):
        raise RuntimeError("逐事件间隔与 rank_metrics 的间隔口径不一致")
    return metrics, ranks, signed_margins


def preflight(
    run_dir: Path, data_dir: Path, levels: list[int], masks_per_level: int,
    seed: int, node_masks: Path | None, reference_run_dir: Path | None,
    device: torch.device,
) -> dict:
    """执行输入、模型、标准化、测试划分与掩码一致性检查。

    返回检查记录、数据与模型对象；调用方只在全部检查通过后继续推理。
    """
    checks: list[dict] = []
    started = time.time()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    # 数据读取与元数据核对
    data = load_arrays(data_dir)
    meta = data["meta"]
    n_nodes = int(data["X_obs"].shape[1])
    n_fault = int(data["X_obs"].shape[1])
    n_candidates = int(data["signature_bank"].shape[1])
    n_samples = int(data["X_obs"].shape[0])
    add_check(checks, "data_dir", data_dir.is_dir(), str(data_dir))
    add_check(
        checks, "dataset_meta",
        meta.get("case") == "ieee37" and meta.get("scenario") == "S0"
        and int(meta.get("n_nodes", -1)) == n_nodes
        and int(meta.get("n_candidates", -1)) == n_candidates,
        f"case={meta.get('case')}, scenario={meta.get('scenario')}, "
        f"n_nodes={meta.get('n_nodes')}, n_candidates={meta.get('n_candidates')}, "
        f"电阻={meta.get('res_min')}–{meta.get('res_max')} Ω（{meta.get('impedance_sampling')}）",
    )
    add_check(
        checks, "array_shapes",
        data["X_obs"].shape[1:] == (n_nodes, int(meta.get("window_len", data["X_obs"].shape[2])),
                                    int(meta.get("feature_dim", data["X_obs"].shape[3])))
        and data["mask"].shape == (n_samples, n_nodes)
        and data["candidate_mask"].shape == (n_samples, n_candidates)
        and n_candidates == n_fault + 1,
        f"X_obs={tuple(data['X_obs'].shape)}, signature_bank={tuple(data['signature_bank'].shape)}, "
        f"mask={tuple(data['mask'].shape)}, candidate_mask={tuple(data['candidate_mask'].shape)}",
    )

    # 标准化与模型配置核对
    normalization_file = np.load(run_dir / "normalization.npz")
    normalization = {key: normalization_file[key] for key in ("mean", "std")}
    add_check(
        checks, "normalization",
        normalization["mean"].shape == (n_nodes, int(meta.get("feature_dim", 0)))
        and normalization["std"].shape == normalization["mean"].shape
        and np.isfinite(normalization["mean"]).all()
        and np.isfinite(normalization["std"]).all()
        and (normalization["std"] > 0).all(),
        f"mean/std 形状 {normalization['mean'].shape}",
    )
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model_config = dict(checkpoint["model_config"])
    add_check(
        checks, "model_config",
        int(model_config.get("n_nodes", -1)) == n_nodes
        and int(model_config.get("time_steps", -1)) == int(data["X_obs"].shape[2])
        and int(model_config.get("feature_dim", -1)) == int(data["X_obs"].shape[3]),
        f"model_config={model_config}",
    )
    model = GrefPredictor(**model_config).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    add_check(
        checks, "model_state_dict",
        not load_result.missing_keys and not load_result.unexpected_keys,
        f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}, "
        f"参数数={sum(p.numel() for p in model.parameters())}",
    )
    model.eval()

    # 测试划分核对
    test_idx = np.load(run_dir / "split.npz")["test_idx"]
    labels = np.asarray(data["y_loc"][test_idx])
    groups = np.asarray(data["group_id"][test_idx])
    if int(meta.get("n_test", -1)) != len(test_idx):
        add_check(
            checks, "test_split_source", True,
            f"运行划分 {len(test_idx)} 个测试事件，数据目录内置 test_idx 为 "
            f"{int(np.load(data_dir / 'test_idx.npy').size)}；本次沿用运行 split.npz",
        )
    problems = []
    if test_idx.size == 0:
        problems.append("测试索引为空")
    if test_idx.min() < 0 or test_idx.max() >= n_samples:
        problems.append("测试索引越界")
    if np.unique(test_idx).size != test_idx.size:
        problems.append("测试索引重复")
    if (labels < 0).any():
        problems.append("测试集含无故障样本")
    if (labels >= n_fault).any():
        problems.append("故障标签超出故障候选轴")
    add_check(
        checks, "test_split", not problems,
        f"n_test={test_idx.size}, 故障族数={np.unique(groups).size}, 问题={problems or '无'}",
    )
    add_check(
        checks, "candidate_axis",
        bool(np.asarray(data["candidate_mask"][test_idx][:, :n_fault]).all())
        and int(data["signature_bank"].shape[1]) == n_fault + 1,
        f"故障候选 {n_fault} 个，末尾正常候选 1 个，测试事件故障候选全部有效",
    )

    if reference_run_dir is not None:
        reference_split = np.load(reference_run_dir / "split.npz")
        reference_config = json.loads((reference_run_dir / "config.json").read_text(encoding="utf-8"))
        same_split = all(
            np.array_equal(np.load(run_dir / "split.npz")[key], reference_split[key])
            for key in reference_split.files if key in np.load(run_dir / "split.npz").files
        )
        reference_checkpoint = torch.load(reference_run_dir / "model.pt", map_location="cpu",
                                         weights_only=True)
        add_check(
            checks, "reference_run",
            same_split and dict(reference_checkpoint["model_config"]) == model_config,
            f"对照运行 {reference_run_dir.name}：划分一致={same_split}，"
            f"模型配置一致={dict(reference_checkpoint['model_config']) == model_config}，"
            f"对照损失模式={reference_config.get('loss_mode', 'gref')}",
        )

    # 掩码生成或读取，并核对同一种子可复现
    masks_by_level: dict[int, np.ndarray] = {}
    mask_source = "generated"
    if node_masks is not None:
        masks_by_level = load_mask_file(node_masks, levels, n_nodes)
        mask_source = str(node_masks)
    for level_index, level in enumerate(levels):
        generated = build_node_masks(n_nodes, int(level), masks_per_level,
                                     seed + 1009 * level_index)
        masks = masks_by_level.get(level, generated)
        masks_by_level[level] = masks
        problems = validate_mask_level(masks, int(level), n_nodes)
        if masks.shape[0] != masks_per_level:
            problems.append(f"掩码组数为 {masks.shape[0]}，要求 {masks_per_level}")
        add_check(
            checks, f"node_masks_{level}", not problems,
            f"{masks.shape[0]} 组掩码，每组保留 {level} 个节点，问题={problems or '无'}",
        )
        if node_masks is not None:
            same = masks.shape == generated.shape and np.array_equal(masks, generated)
            add_check(
                checks, f"node_masks_{level}_reproducible", same,
                "读取文件与同一种子重新生成的结果逐元素一致"
                if same else "读取文件与同一种子重新生成的结果不一致",
            )

    passed = all(item["passed"] for item in checks)
    return {
        "config": config,
        "data": data,
        "meta": meta,
        "normalization": normalization,
        "checkpoint": checkpoint,
        "model": model,
        "test_idx": test_idx,
        "masks_by_level": masks_by_level,
        "mask_source": mask_source,
        "checks": checks,
        "passed": passed,
        "preflight_seconds": time.time() - started,
        "n_nodes": n_nodes,
        "n_fault": n_fault,
        "n_candidates": n_candidates,
    }


def _stats(values: list) -> dict | None:
    """对一组掩码级读数给出均值、最小值与最大值；空集或含缺失时忽略缺失值。"""
    finite = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {"mean": float(np.mean(finite)), "min": float(np.min(finite)),
            "max": float(np.max(finite))}


def _subset_metrics(
    ranks: np.ndarray, margins: np.ndarray, select: np.ndarray,
) -> dict | None:
    """在给定事件子集上按 `rank_metrics` 口径计算完整定位指标；子集为空时返回 None。"""
    if not select.any():
        return None
    ranks = ranks[select]
    margins = margins[select]
    return {
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "top5": float((ranks <= 5).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "signed_margin_median": float(np.median(margins)),
        "signed_margin_negative_ratio": float((margins < 0).mean()),
    }


def summarize(
    records: list[dict], clean: dict, levels: list[int], locations: np.ndarray,
) -> dict:
    """汇总每个观测规模的指标、相对基线变化与真实节点观测条件的完整分解。"""
    summary: dict[str, dict] = {}
    for level in levels:
        subset = [row for row in records if row["observed_nodes"] == level]
        metrics = [row["metrics"] for row in subset]
        conditional_rows: list[dict] = []
        for row in subset:
            observed = np.isin(locations, row["observed_node_indices"])
            ranks = np.asarray(row["ranks"], dtype=np.int64)
            margins = np.asarray(row["signed_margins"], dtype=np.float32)
            conditional_rows.append({
                "n_true_node_observed": int(observed.sum()),
                "observed": _subset_metrics(ranks, margins, observed),
                "missing": _subset_metrics(ranks, margins, ~observed),
            })
        conditional = {
            "n_true_node_observed": _stats([item["n_true_node_observed"] for item in conditional_rows])
        }
        for group in ("observed", "missing"):
            conditional[group] = {
                key: _stats([item[group][key] if item[group] is not None else None
                             for item in conditional_rows])
                for key in METRIC_KEYS
            }
        summary[str(level)] = {
            "metrics": {
                key: {
                    "mean": float(np.mean([item[key] for item in metrics])),
                    "min": float(np.min([item[key] for item in metrics])),
                    "max": float(np.max([item[key] for item in metrics])),
                }
                for key in METRIC_KEYS
            },
            "delta": {
                key: {
                    "mean": float(np.mean([item[key] - clean[key] for item in metrics])),
                    "min": float(np.min([item[key] - clean[key] for item in metrics])),
                    "max": float(np.max([item[key] - clean[key] for item in metrics])),
                }
                for key in DELTA_KEYS
            },
            "conditional": conditional,
            "n_masks": len(subset),
        }
    return summary


def write_markdown(
    path: Path, payload: dict, model_label: str, levels: list[int],
) -> None:
    """写出面向复核的指标摘要，数值保留四位小数，变化以百分点表示。"""
    clean = payload["clean"]
    lines = [
        f"# {model_label} S2 部分节点观测评估",
        "",
        f"- 运行目录：`{payload['run_dir']}`",
        f"- 数据目录：`{payload['data_dir']}`",
        f"- 损失模式：`{payload['loss_mode']}`",
        f"- 测试事件数：{payload['n_test']}",
        f"- 节点数／故障候选数：{payload['n_nodes']}／{payload['n_fault_candidates']}",
        f"- 掩码来源：`{payload['mask_source']}`",
        f"- 掩码种子：{payload['seed']}，每级 {payload['masks_per_level']} 组",
        f"- 执行前检查：{len(payload['checks'])} 项全部通过",
        "",
        "## 完整观测基线",
        "",
        f"- Top-1／Top-3／Top-5：{clean['top1']:.4f}／{clean['top3']:.4f}／{clean['top5']:.4f}",
        f"- 平均名次／名次中位数：{clean['mean_rank']:.4f}／{clean['median_rank']:.4f}",
        f"- signed margin 中位数／负间隔比例："
        f"{clean['signed_margin_median']:.6f}／{clean['signed_margin_negative_ratio']:.4f}",
        "",
        "## 各观测规模",
        "",
        "每级按十组掩码给出均值与变化；条件分解按真实故障节点本身是否落在观测节点集合内划分，"
        "并在两个子集上分别给出完整定位指标。",
        "",
    ]
    for level in levels:
        item = payload["summary"][str(level)]
        metrics, delta = item["metrics"], item["delta"]
        condition = item["conditional"]
        lines.append(
            f"- 保留 {level} 个节点（{item['n_masks']} 组掩码）："
            f"Top-1={metrics['top1']['mean']:.4f}"
            f"（{delta['top1']['mean'] * 100:+.2f} pp，范围 "
            f"{metrics['top1']['min']:.4f}–{metrics['top1']['max']:.4f}），"
            f"Top-3={metrics['top3']['mean']:.4f}（{delta['top3']['mean'] * 100:+.2f} pp），"
            f"Top-5={metrics['top5']['mean']:.4f}（{delta['top5']['mean'] * 100:+.2f} pp），"
            f"平均名次={metrics['mean_rank']['mean']:.4f}"
            f"（{delta['mean_rank']['mean']:+.4f}），"
            f"名次中位数={metrics['median_rank']['mean']:.4f}，"
            f"signed margin 中位数={metrics['signed_margin_median']['mean']:.6f}，"
            f"负间隔比例={metrics['signed_margin_negative_ratio']['mean']:.4f}"
        )
        lines.append(
            f"  - 条件分解（真实故障节点被观测，"
            f"{condition['n_true_node_observed']['mean']:.0f} 个事件）："
            f"Top-1={condition['observed']['top1']['mean']:.4f}、"
            f"Top-3={condition['observed']['top3']['mean']:.4f}、"
            f"Top-5={condition['observed']['top5']['mean']:.4f}、"
            f"平均名次={condition['observed']['mean_rank']['mean']:.4f}、"
            f"名次中位数={condition['observed']['median_rank']['mean']:.4f}、"
            f"signed margin 中位数={condition['observed']['signed_margin_median']['mean']:.6f}、"
            f"负间隔比例={condition['observed']['signed_margin_negative_ratio']['mean']:.4f}"
        )
        if condition["missing"]["top1"] is not None:
            lines.append(
                f"  - 条件分解（真实故障节点未被观测，"
                f"{payload['n_test'] - condition['n_true_node_observed']['mean']:.0f} 个事件）："
                f"Top-1={condition['missing']['top1']['mean']:.4f}、"
                f"Top-3={condition['missing']['top3']['mean']:.4f}、"
                f"Top-5={condition['missing']['top5']['mean']:.4f}、"
                f"平均名次={condition['missing']['mean_rank']['mean']:.4f}、"
                f"名次中位数={condition['missing']['median_rank']['mean']:.4f}、"
                f"signed margin 中位数={condition['missing']['signed_margin_median']['mean']:.6f}、"
                f"负间隔比例={condition['missing']['signed_margin_negative_ratio']['mean']:.4f}"
            )
    lines.extend([
        "",
        "## 产物",
        "",
        "- 完整指标与逐掩码结果：`s2_node_metrics.json`；",
        "- 执行前检查记录：`s2_input_checks.json`；",
        "- 节点掩码数组：`node_masks.npz`（键为观测节点数，行内节点索引升序）。",
        "",
        "## 边界",
        "",
        "- 只改变部署输入的 `node_mask`，不重训练、不修改真实拓扑、不重新生成物理响应；",
        "- 单训练种子、单测试划分、每级 10 组掩码，结果不含置信区间；",
        "- 排序指标使用同一候选有效性掩码，正常候选不进入定位评价；",
        "- CUDA 归约次序非确定，同参数重跑的候选分数差异约 1e-7，近并列事件的名次可能变化一个事件；"
        "本文件数值对应当前归档的运行产物。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """执行检查与推理，并写出指标 JSON、Markdown、掩码与检查记录。"""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = (args.data_dir or Path(config["data_dir"])).resolve()
    output_dir = (args.output_dir or (run_dir / "s2-node-eval")).resolve()
    levels = [int(level) for level in args.observed_nodes]
    if len(set(levels)) != len(levels):
        raise ValueError("observed_nodes 不允许重复")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    fingerprint_before = None if args.check_only else data_fingerprint(data_dir)
    state = preflight(run_dir, data_dir, levels, args.masks_per_level, args.seed,
                      args.node_masks, args.reference_run_dir, device)
    checks = state["checks"]
    if not state["passed"]:
        failures = [item for item in checks if not item["passed"]]
        print(json.dumps({"checks_passed": False, "failures": failures}, ensure_ascii=False, indent=2))
        return 1
    model: GrefPredictor = state["model"]
    data: dict = state["data"]
    normalization = state["normalization"]
    test_idx = state["test_idx"]
    n_nodes = state["n_nodes"]
    n_fault = state["n_fault"]

    if args.check_only:
        payload = {
            "run_dir": str(run_dir), "data_dir": str(data_dir),
            "loss_mode": config.get("loss_mode", "gref"),
            "observed_nodes": levels, "masks_per_level": args.masks_per_level,
            "seed": args.seed, "device": str(device),
            "n_test": int(len(test_idx)), "n_nodes": n_nodes,
            "n_fault_candidates": n_fault,
            "checks_passed": True, "checks": checks,
        }
        if args.checks_output is not None:
            args.checks_output.parent.mkdir(parents=True, exist_ok=True)
            args.checks_output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    # 完整观测基线，并与运行目录已保存的测试指标核对可复现性
    started = time.time()
    clean, _, _ = evaluate_mask(model, data, test_idx, normalization, device,
                                np.ones(n_nodes, dtype=np.float32), args.batch_size, n_fault)
    ensure_finite_metrics(clean, "完整观测基线")
    baseline_seconds = time.time() - started
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["score"]
    tolerance = {"signed_margin_median": 1e-4}
    differences = {
        key: float(abs(clean[key] - stored[key]))
        for key in METRIC_KEYS if key in stored
    }
    reproducible = all(
        difference <= tolerance.get(key, 1e-6) for key, difference in differences.items()
    )
    add_check(
        checks, "baseline_reproducible", reproducible,
        "重算基线与运行目录 metrics.json 的测试指标一致，最大差="
        f"{max(differences.values()):.3e}；重算 Top-1={clean['top1']:.4f}",
    )

    records: list[dict] = []
    mask_checksums = {}
    started = time.time()
    for level in levels:
        masks = state["masks_by_level"][level]
        mask_checksums[str(level)] = array_sha256(masks)
        for mask_index, mask in enumerate(masks):
            metrics, ranks, signed_margins = evaluate_mask(
                model, data, test_idx, normalization, device, mask,
                args.batch_size, n_fault)
            ensure_finite_metrics(metrics, f"观测规模 {level} 第 {mask_index} 组掩码")
            records.append({
                "observed_nodes": int(level),
                "mask_index": int(mask_index),
                "mask_sha256": array_sha256(mask),
                "observed_node_indices": np.flatnonzero(mask).tolist(),
                "ranks": ranks.tolist(),
                "signed_margins": signed_margins.tolist(),
                "metrics": metrics,
                "delta": {key: float(metrics[key] - clean[key]) for key in DELTA_KEYS},
            })
    masks_seconds = time.time() - started

    locations = np.asarray(data["y_loc"][test_idx], dtype=np.int64)
    summary = summarize(records, clean, levels, locations)
    fingerprint_after = data_fingerprint(data_dir)
    add_check(
        checks, "data_unchanged", fingerprint_after == fingerprint_before,
        f"核对 {len(fingerprint_before)} 个数据文件的尺寸、修改时间与 SHA256",
    )
    add_check(
        checks, "metrics_finite", True,
        f"完整观测基线、{len(records)} 组掩码指标与分层汇总均为有限数值",
    )
    np.savez_compressed(output_dir / "node_masks.npz",
                        **{str(level): state["masks_by_level"][level] for level in levels})
    payload = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "loss_mode": config.get("loss_mode", "gref"),
        "model_config": dict(state["checkpoint"]["model_config"]),
        "loss_config": {
            key: config[key] for key in ("temperature", "lambda_family", "lambda_energy",
                                        "lambda_rank", "lambda_smooth", "rank_temperature")
            if key in config
        },
        "device": {"type": str(device),
                   "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "n_nodes": n_nodes,
        "n_fault_candidates": n_fault,
        "n_candidates": state["n_candidates"],
        "n_test": int(len(test_idx)),
        "seed": args.seed,
        "masks_per_level": args.masks_per_level,
        "observed_nodes": levels,
        "mask_rule": "第 i 个观测规模使用 seed=seed+1009*i，从 n_nodes 个节点无放回抽取，按节点索引升序记录",
        "mask_source": state["mask_source"],
        "mask_sha256": mask_checksums,
        "clean": clean,
        "clean_reference": {key: float(stored[key]) for key in METRIC_KEYS if key in stored},
        "summary": summary,
        "test_locations": locations.tolist(),
        "records": records,
        "checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "timing_seconds": {
            "preflight": state["preflight_seconds"],
            "baseline": baseline_seconds,
            "masks": masks_seconds,
        },
        "data_fingerprint": fingerprint_after,
    }
    if not payload["checks_passed"]:
        failures = [item for item in checks if not item["passed"]]
        (output_dir / "s2_input_checks.json").write_text(
            json.dumps({"checks_passed": False, "failures": failures},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"执行中检查失败：{failures}")
    (output_dir / "s2_node_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "s2_input_checks.json").write_text(
        json.dumps({key: payload[key] for key in
                    ("run_dir", "data_dir", "loss_mode", "observed_nodes", "masks_per_level",
                     "seed", "device", "n_test", "n_nodes", "n_fault_candidates",
                     "checks_passed", "checks", "data_fingerprint")},
                   ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(output_dir / "s2_node_metrics.md", payload,
                   MODEL_LABELS.get(str(payload["loss_mode"]), str(payload["loss_mode"])), levels)
    print(json.dumps({
        "output_dir": str(output_dir), "loss_mode": payload["loss_mode"],
        "checks_passed": payload["checks_passed"], "n_records": len(records),
        "summary": summary,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
