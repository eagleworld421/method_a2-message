"""Method-A2 判据一：共享状态可恢复性 Probe 执行脚本。

本脚本实现
`docs/project/plans/Method-A2-判据一-共享状态可恢复性Probe设计.md`
规定的判据一实验：冻结既有部署 predictor 的共享编码器，在其表示上另接
岭回归与多类逻辑回归，测量共享状态 (H_m, C_m) 是否进入表示。

执行约束（设计文档第 6.1 节）：

- 不训练、不修改既有模型；编码器全程 eval() 且不接收梯度。
- 不修改 src/ 下任何文件，不新增模型结构，不引入新的训练损失。
- 全部新增内容限于本脚本、一个测试文件与一个产物目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.model.privileged_response_predictor import PrivilegedResponseSystem  # noqa: E402


# 15 维工况向量的键顺序，按设计文档第 6.3 节固定为字典序。
LOAD_MULTIPLIER_KEYS = (
    "611", "634a", "634b", "634c", "645", "646", "652",
    "670a", "670b", "670c", "671", "675a", "675b", "675c", "692",
)

# 编码器构造超参数，按设计文档第 6.2 节逐项固定。
MODEL_KWARGS = {
    "n_nodes": 16,
    "time_steps": 12,
    "feature_dim": 6,
    "candidate_feature_dim": 10,
    "temporal_hidden": 64,
    "temporal_out": 64,
    "gnn_hidden": 64,
    "candidate_hidden": 64,
    "hidden_dim": 64,
    "impedance_hidden": 16,
}

# 正则强度候选集，按设计文档第 6.4 节固定。
C_GRID = (0.01, 0.1, 1.0, 10.0)
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

# 响应距离的节点—特征权重中的数值稳定项，按设计文档第 4 节。
DISTANCE_EPS = 1e-8

# 本机 sklearn 1.0.2 与 scipy 1.13.1 不兼容：Ridge 的默认 solver（auto/cholesky）
# 会调用已被移除的 scipy.linalg.solve(sym_pos=...) 而抛 TypeError。改用迭代式
# lsqr 求解器，实测与 svd 求解器的系数最大差为 1.1e-8。
RIDGE_SOLVER = "lsqr"
RIDGE_TOL = 1e-8

# 多类逻辑回归的求解器。规格只要求 multinomial 与 max_iter ≥ 2000，未指定 solver。
# 本机实测：同一 16 类问题上 lbfgs 需 166.4 秒（196/312/466/206 次迭代），
# newton-cg 只需 1.4 秒（19～22 次迭代），相差约 119 倍；二者均为 multinomial
# 口径。lbfgs 的耗时来自其拟牛顿近似在近奇异 Hessian 上反复试探步长。
CLASSIFICATION_SOLVER = "newton-cg"

# 浅层非线性 probe 的网格与预算。刻意只用一个隐藏层、宽度不超过 512，以维持
# 设计文档第 3 节"另接一个很小的模型"的定位；正则强度由验证划分选择。
MLP_WIDTHS = (32, 128, 512)
MLP_WEIGHT_DECAYS = (1e-4, 1e-2)
MLP_EPOCHS = 400
MLP_LEARNING_RATE = 1e-3
MLP_SELECTION_SEED = 0
MLP_REPORT_SEEDS = (0, 1, 2)

DATASET_FILES = (
    "X_obs.npy", "paired_response.npy", "candidate_features.npy",
    "node_features.npy", "edge_index.npy", "edge_attr.npy",
    "edge_mask.npy", "candidate_mask.npy", "node_mask.npy",
    "feature_scaler.npz", "event_metadata.jsonl",
    "train_idx.npy", "val_idx.npy", "test_idx.npy",
)


def utc_now() -> str:
    """返回秒级 UTC 时间戳字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    """计算文件 sha256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_probe(repo: Path) -> dict:
    """记录仓库 HEAD 与工作区状态，用于固化代码版本。"""
    info = {"repo": str(repo)}
    for key, command in (
        ("head", ["rev-parse", "HEAD"]),
        ("status_porcelain", ["status", "--porcelain"]),
    ):
        try:
            done = subprocess.run(
                ["git", "-C", str(repo), *command],
                capture_output=True, text=True, timeout=30,
            )
            info[key] = done.stdout.strip() if done.returncode == 0 else f"<error {done.returncode}>"
        except Exception as exc:  # noqa: BLE001 - 版本信息缺失不应中断实验
            info[key] = f"<unavailable: {type(exc).__name__}>"
    return info


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Method-A2 判据一共享状态可恢复性 Probe")
    parser.add_argument("--data-dir", type=Path, required=True, help="paired-v1 数据集目录")
    parser.add_argument("--checkpoint", type=Path, required=True, help="teacher.pt 路径")
    parser.add_argument("--output-root", type=Path, default=METHOD_DIR / "output" / "error-source-v2")
    parser.add_argument("--log-root", type=Path, default=METHOD_DIR / "logs" / "error-source-v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument(
        "--source-repo", default="auto",
        help="数据与权重的来源仓库标识；auto 表示按路径推断",
    )
    parser.add_argument(
        "--block-limit", type=int, default=None,
        help="每个划分最多使用的 block 数，用于快速冒烟；默认使用全部 block。"
             "按 block 截断可保持划分不重叠与块内共享工况两条性质。",
    )
    parser.add_argument(
        "--probe-kinds", choices=("linear", "mlp"), default="linear",
        help="probe 拟合器：linear 为岭回归与多类逻辑回归；mlp 为单隐藏层浅层非线性。"
             "一次运行只使用一种，以保证产物的口径单一、可独立复核。",
    )
    return parser.parse_args()


def low_precision(matrix: np.ndarray) -> np.ndarray:
    """把整数标签矩阵转为浮点，避免整型运算溢出。"""
    return matrix.astype(np.float32)


# --------------------------------------------------------------------------
# 步骤一：表示提取
# --------------------------------------------------------------------------

def load_encoder(checkpoint: Path, device: str):
    """载入 teacher checkpoint 的共享编码器并断言权重完全匹配。"""
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise KeyError(f"checkpoint 缺少 state_dict：{checkpoint}")
    model = PrivilegedResponseSystem(**MODEL_KWARGS)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if list(missing) or list(unexpected):
        raise RuntimeError(
            f"checkpoint 与模型结构不匹配：missing={list(missing)}, "
            f"unexpected={list(unexpected)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model, payload


def extract_representations(model, dataset: dict, device: str, batch_size: int) -> dict:
    """按 batch 前向全部事件，只调用候选条件编码器，不调用任何解码头。"""
    x_obs = dataset["X_obs"]
    n_events = x_obs.shape[0]
    edge_index = torch.as_tensor(dataset["edge_index"], device=device)
    edge_attr = torch.as_tensor(dataset["edge_attr"], device=device)
    candidate_features = torch.as_tensor(dataset["candidate_features"], device=device)
    node_features = torch.as_tensor(dataset["node_features"], device=device)

    collected = {name: [] for name in (
        "node_repr", "global_repr", "candidate_embed", "attention",
    )}
    forward_seconds = 0.0
    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            stop = min(start + batch_size, n_events)
            batch_x = torch.as_tensor(x_obs[start:stop], device=device)
            batch_edge_mask = torch.as_tensor(dataset["edge_mask"][start:stop], device=device)
            batch_candidate_mask = torch.as_tensor(
                dataset["candidate_mask"][start:stop], device=device
            )
            started = time.perf_counter()
            encoded = model.encoder(
                batch_x, edge_index, edge_attr, batch_edge_mask,
                candidate_features, node_features,
                candidate_mask=batch_candidate_mask,
            )
            forward_seconds += time.perf_counter() - started
            for name in collected:
                collected[name].append(encoded[name].detach().cpu().numpy())

    node_repr = np.concatenate(collected["node_repr"], axis=0)
    global_repr = np.concatenate(collected["global_repr"], axis=0)
    candidate_embed = np.concatenate(collected["candidate_embed"], axis=0)
    attention = np.concatenate(collected["attention"], axis=0)

    features = {
        "rep_node_mean": node_repr.mean(axis=1),
        "rep_node_flat": node_repr.reshape(node_repr.shape[0], -1),
        "rep_global": global_repr,
        "rep_candidate_mean": candidate_embed.mean(axis=1),
        "rep_attention_mean": attention.mean(axis=1),
    }
    observations = {
        "obs_node_time_mean": x_obs.mean(axis=(1, 2)),
        "obs_node_flat": x_obs.mean(axis=2).reshape(n_events, -1),
        "obs_full": x_obs.reshape(n_events, -1),
    }
    arrays = {
        "node_repr": node_repr,
        "global_repr": global_repr,
        "candidate_embed": candidate_embed,
        "attention": attention,
        **features,
        **observations,
    }
    return {
        "arrays": arrays,
        "features": features,
        "observations": observations,
        "forward_seconds": forward_seconds,
    }


def candidate_embed_variation(candidate_embed: np.ndarray) -> dict:
    """检验 candidate_embed 是否随事件变化（设计文档预期其为常量）。"""
    per_event = candidate_embed.reshape(candidate_embed.shape[0], -1)
    spread = per_event.max(axis=0) - per_event.min(axis=0)
    return {
        "per_event_element_max_spread": float(spread.max()),
        "per_event_element_mean_spread": float(spread.mean()),
        "is_constant_across_events": bool(spread.max() <= 1e-6),
    }


# --------------------------------------------------------------------------
# 步骤二：条件标签构造
# --------------------------------------------------------------------------

def load_condition_metadata(data_dir: Path) -> list:
    """逐行读取 event_metadata.jsonl。"""
    path = data_dir / "event_metadata.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"事件元数据为空：{path}")
    return rows


def build_targets(rows: list) -> dict:
    """构造 H、故障类型、log1p 阻抗、位置与故障时刻目标。"""
    load_multipliers = np.asarray(
        [[float(row["load_multipliers"][key]) for key in LOAD_MULTIPLIER_KEYS] for row in rows],
        dtype=np.float64,
    )
    impedance = np.asarray([float(row["true_impedance"]) for row in rows], dtype=np.float64)
    return {
        "H": load_multipliers,
        # 统一保持 [N,D] 二维形状，使逐维报告与块级聚合对回归目标一致。
        "log1p_impedance": np.log1p(impedance)[:, None],
        "impedance_raw": impedance,
        "fault_class": np.asarray([int(row["fault_class"]) for row in rows], dtype=np.int64),
        "true_location": np.asarray([int(row["true_location"]) for row in rows], dtype=np.int64),
        "fault_delay_steps": np.asarray(
            [int(row["fault_delay_steps"]) for row in rows], dtype=np.int64
        ),
        "block_id": np.asarray([int(row["block_id"]) for row in rows], dtype=np.int64),
        "event_index": np.asarray([int(row["event_index"]) for row in rows], dtype=np.int64),
        "event_id": [str(row["event_id"]) for row in rows],
    }


# --------------------------------------------------------------------------
# 度量
# --------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 R²、RMSE 与 MAE；R² 以真实标签方差为分母。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    variance = float(((y_true - y_true.mean(axis=0)) ** 2).sum())
    residual_sum = float((residual ** 2).sum())
    r2 = 1.0 - residual_sum / variance if variance > 0 else float("nan")
    return {
        "r2": r2,
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "mae": float(np.abs(residual).mean()),
        "n_samples": int(y_true.shape[0]),
        "n_dims": int(y_true.shape[1]) if y_true.ndim > 1 else 1,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """计算准确率、宏平均 F1、混淆矩阵与基线。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.tolist(), y_pred.tolist()):
        confusion[truth, pred] += 1
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else float("nan")
    f1_scores = []
    for label in range(n_classes):
        true_positive = float(confusion[label, label])
        false_positive = float(confusion[:, label].sum() - true_positive)
        false_negative = float(confusion[label, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator > 0 else 0.0)
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "majority_baseline": float(counts.max() / counts.sum()) if counts.sum() else float("nan"),
        "random_baseline": 1.0 / float(n_classes),
        "confusion_matrix": confusion.tolist(),
        "n_samples": int(y_true.size),
    }


def block_ids_for(indices: np.ndarray, block_id: np.ndarray) -> np.ndarray:
    """返回给定事件索引对应的 block 编号。"""
    return block_id[indices]


def limit_blocks(splits: dict, block_id: np.ndarray, block_limit: int) -> dict:
    """按 block 截断每个划分，用于快速冒烟。

    以 block 为单位截断可保持划分不重叠与块内共享工况两条性质；按事件截断
    会破坏这两条性质并使 probe 因泄漏而虚高。
    """
    if block_limit < 1:
        raise ValueError("block_limit 必须为正整数")
    limited = {}
    for key in ("train", "val", "test"):
        blocks = np.unique(block_id[splits[key]])
        keep = blocks[:block_limit]
        limited[key] = splits[key][np.isin(block_id[splits[key]], keep)]
    return limited


def aggregate_by_block(features: np.ndarray, blocks: np.ndarray) -> tuple:
    """按 block 对特征取均值，返回块级特征与块标签。"""
    unique = np.unique(blocks)
    aggregated = np.stack([features[blocks == value].mean(axis=0) for value in unique], axis=0)
    return aggregated, unique


# --------------------------------------------------------------------------
# 步骤三：probe 拟合
# --------------------------------------------------------------------------

def standardize(train: np.ndarray, others: list) -> tuple:
    """按训练侧统计量标准化，返回训练侧标准化结果与其余矩阵。"""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled_train = (train - mean) / std
    return scaled_train, [((matrix - mean) / std) for matrix in others]


def fit_regression(features: dict, target: np.ndarray, splits: dict, seed: int,
                   permute_train_labels: bool = False) -> dict:
    """在验证划分上选择岭回归正则强度，并在测试划分上报告一次。

    `permute_train_labels=True` 时打乱训练侧目标而保持特征不变，测试侧仍用
    真实目标评估，用于标签置换负对照。
    """
    from sklearn.linear_model import Ridge

    x_train = features["train"]
    x_val = features["val"]
    x_test = features["test"]
    (x_train_s, x_val_s, x_test_s), target_scaled, target_stats = _scale_regression(
        x_train, [x_val, x_test], target, splits
    )
    y_train = target_scaled[splits["train"]].copy()
    if permute_train_labels:
        rng = np.random.default_rng(seed)
        y_train = y_train[rng.permutation(y_train.shape[0])]
    y_val = target_scaled[splits["val"]]
    y_test = target_scaled[splits["test"]]

    selection = []
    best_alpha, best_score, best_model = None, -np.inf, None
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=float(alpha), solver=RIDGE_SOLVER, tol=RIDGE_TOL,
                      random_state=seed)
        model.fit(x_train_s, y_train)
        score = regression_metrics(y_val, model.predict(x_val_s))["r2"]
        selection.append({"alpha": float(alpha), "val_r2": score})
        if np.isfinite(score) and score > best_score:
            best_alpha, best_score, best_model = float(alpha), score, model
    if best_model is None:
        raise RuntimeError("岭回归未产出可用模型")

    # 报告口径统一换算回原始物理单位：R² 与尺度无关，RMSE/MAE 必须换算。
    target_mean = np.asarray(target_stats["mean"], dtype=np.float64)
    target_std = np.asarray(target_stats["std"], dtype=np.float64)
    predicted_raw = best_model.predict(x_test_s) * target_std + target_mean

    return {
        "selection": selection,
        "selected_alpha": best_alpha,
        "target_stats": target_stats,
        "event": regression_metrics(target[splits["test"]], predicted_raw),
        "model": best_model,
        "predict": best_model.predict,
        "test_features": x_test_s,
        "test_blocks": splits["test_blocks"],
        "target_test": target[splits["test"]],
    }


def _scale_regression(x_train, x_others, target, splits):
    """标准化特征与回归目标，目标统计量仅取自训练侧。"""
    x_train_s, scaled = standardize(x_train, list(x_others))
    target_mean = target[splits["train"]].mean(axis=0)
    target_std = target[splits["train"]].std(axis=0)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)
    target_scaled = (target - target_mean) / target_std
    return (x_train_s, *scaled), target_scaled, {
        "mean": target_mean.tolist(),
        "std": target_std.tolist(),
    }


def fit_classification(features: dict, labels: np.ndarray, splits: dict, n_classes: int,
                       seed: int, permute_train_labels: bool = False) -> dict:
    """在验证划分上选择逻辑回归正则强度，并在测试划分上报告一次。

    `permute_train_labels=True` 时打乱训练侧标签而保持特征不变，用于标签
    置换负对照；测试侧始终使用真实标签评估。
    """
    from sklearn.linear_model import LogisticRegression

    x_train = features["train"]
    x_val = features["val"]
    x_test = features["test"]
    x_train_s, (x_val_s, x_test_s) = standardize(x_train, [x_val, x_test])

    y_train = labels[splits["train"]].copy()
    if permute_train_labels:
        rng = np.random.default_rng(seed)
        y_train = y_train[rng.permutation(y_train.size)]
    y_val = labels[splits["val"]]
    y_test = labels[splits["test"]]

    selection = []
    best_c, best_score, best_model = None, -np.inf, None
    for c_value in C_GRID:
        model = LogisticRegression(
            C=float(c_value), max_iter=2000, multi_class="multinomial",
            solver=CLASSIFICATION_SOLVER, random_state=seed,
        )
        model.fit(x_train_s, y_train)
        score = classification_metrics(y_val, model.predict(x_val_s), n_classes)["accuracy"]
        iterations = int(np.ravel(model.n_iter_)[0]) if hasattr(model, "n_iter_") else None
        selection.append({
            "C": float(c_value), "val_accuracy": score,
            "n_iter": iterations,
            "reached_iteration_limit": bool(iterations is not None and iterations >= 2000),
        })
        if np.isfinite(score) and score > best_score:
            best_c, best_score, best_model = float(c_value), score, model
    if best_model is None:
        raise RuntimeError("逻辑回归未产出可用模型")

    return {
        "selection": selection,
        "selected_C": best_c,
        "event": classification_metrics(y_test, best_model.predict(x_test_s), n_classes),
        "model": best_model,
        "predict": best_model.predict,
        "test_features": x_test_s,
        "test_blocks": splits["test_blocks"],
        "labels_test": labels[splits["test"]],
    }


def _train_shallow_mlp(x_train, y_train, out_dim, task, width, weight_decay,
                       epochs, learning_rate, seed):
    """全批量训练一个单隐藏层 MLP，刻意保持浅层以维持探针的小模型定位。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    module = nn.Sequential(
        nn.Linear(int(x_train.shape[1]), int(width)),
        nn.ReLU(),
        nn.Linear(int(width), int(out_dim)),
    )
    optimizer = torch.optim.Adam(
        module.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    if task == "regression":
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        loss_function = nn.functional.mse_loss
    else:
        y_tensor = torch.as_tensor(y_train, dtype=torch.long)
        loss_function = nn.functional.cross_entropy
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = loss_function(module(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    module.eval()
    return module


def _seeded_predictors(module, task, seed):
    """把单个网络包装为“接受标准化特征、返回标准化输出或类别”的预测函数。"""
    import torch

    def predict(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            outputs = module(torch.as_tensor(np.asarray(features, dtype=np.float32))).numpy()
        if task == "classification":
            return outputs.argmax(axis=1).astype(np.int64)
        return outputs.astype(np.float64)

    return predict


def _average_predictors(predictors: list):
    """多随机种子预测平均；分类按多数投票，回归按均值。"""
    def predict(features: np.ndarray) -> np.ndarray:
        stacked = np.stack([fn(features) for fn in predictors], axis=0)
        if stacked.dtype.kind in "iu":
            votes = np.stack(
                [(stacked == label).sum(axis=0) for label in range(int(stacked.max()) + 1)],
                axis=1,
            )
            return votes.argmax(axis=1).astype(np.int64)
        return stacked.mean(axis=0)

    return predict


def fit_nonlinear_regression(features: dict, target: np.ndarray, splits: dict, seed: int,
                             permute_train_labels: bool = False) -> dict:
    """浅层非线性回归 probe：验证划分选宽度与权重衰减，测试划分只报告一次。"""
    x_train = features["train"]
    x_val = features["val"]
    x_test = features["test"]
    (x_train_s, x_val_s, x_test_s), target_scaled, target_stats = _scale_regression(
        x_train, [x_val, x_test], target, splits
    )
    y_train = target_scaled[splits["train"]].copy()
    if permute_train_labels:
        rng = np.random.default_rng(seed)
        y_train = y_train[rng.permutation(y_train.shape[0])]
    y_val = target_scaled[splits["val"]]

    selection = []
    best_key, best_score = None, -np.inf
    for width in MLP_WIDTHS:
        for weight_decay in MLP_WEIGHT_DECAYS:
            module = _train_shallow_mlp(
                x_train_s, y_train, y_train.shape[1], "regression", width,
                weight_decay, MLP_EPOCHS, MLP_LEARNING_RATE, MLP_SELECTION_SEED,
            )
            score = regression_metrics(
                y_val, _seeded_predictors(module, "regression", MLP_SELECTION_SEED)(x_val_s)
            )["r2"]
            selection.append({
                "width": int(width), "weight_decay": float(weight_decay),
                "epochs": int(MLP_EPOCHS), "val_r2": score,
            })
            if np.isfinite(score) and score > best_score:
                best_key, best_score = (int(width), float(weight_decay)), score
    if best_key is None:
        raise RuntimeError("浅层非线性回归未产出可用模型")

    predictors = []
    for report_seed in MLP_REPORT_SEEDS:
        module = _train_shallow_mlp(
            x_train_s, y_train, y_train.shape[1], "regression", best_key[0],
            best_key[1], MLP_EPOCHS, MLP_LEARNING_RATE, report_seed,
        )
        predictors.append(_seeded_predictors(module, "regression", report_seed))
    predict = _average_predictors(predictors)

    target_mean = np.asarray(target_stats["mean"], dtype=np.float64)
    target_std = np.asarray(target_stats["std"], dtype=np.float64)
    predicted_raw = predict(x_test_s) * target_std + target_mean
    return {
        "selection": selection,
        "selected_alpha": None,
        "selected_mlp": {"width": best_key[0], "weight_decay": best_key[1],
                         "seeds": list(MLP_REPORT_SEEDS), "epochs": int(MLP_EPOCHS)},
        "target_stats": target_stats,
        "event": regression_metrics(target[splits["test"]], predicted_raw),
        "model": None,
        "predict": predict,
        "test_features": x_test_s,
        "test_blocks": splits["test_blocks"],
        "target_test": target[splits["test"]],
    }


def fit_nonlinear_classification(features: dict, labels: np.ndarray, splits: dict,
                                 n_classes: int, seed: int,
                                 permute_train_labels: bool = False) -> dict:
    """浅层非线性分类 probe：验证划分选宽度与权重衰减，测试划分只报告一次。"""
    x_train = features["train"]
    x_val = features["val"]
    x_test = features["test"]
    x_train_s, (x_val_s, x_test_s) = standardize(x_train, [x_val, x_test])

    y_train = labels[splits["train"]].copy()
    if permute_train_labels:
        rng = np.random.default_rng(seed)
        y_train = y_train[rng.permutation(y_train.size)]
    y_val = labels[splits["val"]]
    y_test = labels[splits["test"]]

    selection = []
    best_key, best_score = None, -np.inf
    for width in MLP_WIDTHS:
        for weight_decay in MLP_WEIGHT_DECAYS:
            module = _train_shallow_mlp(
                x_train_s, y_train, n_classes, "classification", width,
                weight_decay, MLP_EPOCHS, MLP_LEARNING_RATE, MLP_SELECTION_SEED,
            )
            score = classification_metrics(
                y_val, _seeded_predictors(module, "classification", MLP_SELECTION_SEED)(x_val_s),
                n_classes,
            )["accuracy"]
            selection.append({
                "width": int(width), "weight_decay": float(weight_decay),
                "epochs": int(MLP_EPOCHS), "val_accuracy": score,
            })
            if np.isfinite(score) and score > best_score:
                best_key, best_score = (int(width), float(weight_decay)), score
    if best_key is None:
        raise RuntimeError("浅层非线性分类未产出可用模型")

    predictors = []
    for report_seed in MLP_REPORT_SEEDS:
        module = _train_shallow_mlp(
            x_train_s, y_train, n_classes, "classification", best_key[0],
            best_key[1], MLP_EPOCHS, MLP_LEARNING_RATE, report_seed,
        )
        predictors.append(_seeded_predictors(module, "classification", report_seed))
    predict = _average_predictors(predictors)

    return {
        "selection": selection,
        "selected_C": None,
        "selected_mlp": {"width": best_key[0], "weight_decay": best_key[1],
                         "seeds": list(MLP_REPORT_SEEDS), "epochs": int(MLP_EPOCHS)},
        "event": classification_metrics(y_test, predict(x_test_s), n_classes),
        "model": None,
        "predict": predict,
        "test_features": x_test_s,
        "test_blocks": splits["test_blocks"],
        "labels_test": labels[splits["test"]],
    }


def fit_probe(features: dict, target: np.ndarray, splits: dict, seed: int,
              kind: str, task: str, n_classes: int = None,
              permute_train_labels: bool = False) -> dict:
    """按 probe 类型分派到线性或浅层非线性拟合器。"""
    if kind == "linear":
        if task == "regression":
            return fit_regression(features, target, splits, seed, permute_train_labels)
        return fit_classification(features, target, splits, n_classes, seed,
                                  permute_train_labels)
    if kind == "mlp":
        if task == "regression":
            return fit_nonlinear_regression(features, target, splits, seed,
                                            permute_train_labels)
        return fit_nonlinear_classification(features, target, splits, n_classes, seed,
                                            permute_train_labels)
    raise ValueError(f"未知 probe 类型：{kind}")


def block_level_regression(result: dict) -> dict:
    """块级回归口径：块内特征先取均值，再预测一次，与部署时一次一个工况一致。"""
    aggregated, blocks = aggregate_by_block(result["test_features"], result["test_blocks"])
    predictions = result["predict"](aggregated)
    stats = result["target_stats"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    predicted_raw = predictions * std + mean
    truth_per_block = np.stack(
        [result["target_test"][result["test_blocks"] == value][0] for value in blocks], axis=0
    )
    return regression_metrics(truth_per_block, predicted_raw)


def block_level_classification(result: dict, n_classes: int) -> dict:
    """块级分类口径：块内特征先取均值，再预测一次。"""
    aggregated, blocks = aggregate_by_block(result["test_features"], result["test_blocks"])
    predictions = result["predict"](aggregated)
    truth_per_block = np.asarray(
        [result["labels_test"][result["test_blocks"] == value][0] for value in blocks],
        dtype=np.int64,
    )
    return classification_metrics(truth_per_block, predictions, n_classes)


def per_dimension_regression(result: dict, dim_names: list) -> list:
    """逐维报告回归目标的事件级与块级指标。"""
    stats = result["target_stats"]
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float64).reshape(-1)

    def two_dimensional(values: np.ndarray) -> np.ndarray:
        """把一维回归目标提升为 [N,1]，避免逐维切片越界。"""
        array = np.asarray(values, dtype=np.float64)
        return array[:, None] if array.ndim == 1 else array

    predicted_event = two_dimensional(result["predict"](result["test_features"])) * std + mean
    truth_event = two_dimensional(result["target_test"])
    aggregated, blocks = aggregate_by_block(result["test_features"], result["test_blocks"])
    predicted_block = two_dimensional(result["predict"](aggregated)) * std + mean
    truth_block = np.stack(
        [truth_event[result["test_blocks"] == value][0] for value in blocks], axis=0
    )
    rows = []
    for position, name in enumerate(dim_names):
        event_metric = regression_metrics(
            truth_event[:, position:position + 1], predicted_event[:, position:position + 1]
        )
        block_metric = regression_metrics(
            truth_block[:, position:position + 1], predicted_block[:, position:position + 1]
        )
        rows.append({
            "dimension": name,
            "event_r2": event_metric["r2"],
            "event_rmse": event_metric["rmse"],
            "event_mae": event_metric["mae"],
            "block_r2": block_metric["r2"],
            "block_rmse": block_metric["rmse"],
            "block_mae": block_metric["mae"],
        })
    return rows


# --------------------------------------------------------------------------
# 步骤四：负对照
# --------------------------------------------------------------------------

def build_negative_controls(arrays: dict, splits: dict, targets: dict, seed: int,
                            kind: str = "linear") -> dict:
    """执行设计文档第 6.5 节规定的四类负对照。"""
    rng = np.random.default_rng(seed)
    primary = arrays["rep_all"]
    controls = {}

    noise = rng.normal(size=primary.shape).astype(primary.dtype)
    zeros = np.zeros_like(primary)
    for name, replacement, description in (
        ("random_representation", noise, "与表示同形状的高斯噪声替换表示"),
        ("zero_representation", zeros, "全零特征替换表示"),
    ):
        features = {key: replacement[splits[key]] for key in ("train", "val", "test")}
        result = fit_probe(features, targets["H"], splits, seed, kind=kind, task="regression")
        controls[name] = {
            "description": description,
            "target": "H",
            "event_r2": result["event"]["r2"],
            "block_r2": block_level_regression(result)["r2"],
            "selected_alpha": result["selected_alpha"],
            "selected_mlp": result.get("selected_mlp"),
        }

    features = {key: primary[splits[key]] for key in ("train", "val", "test")}
    permuted = fit_probe(features, targets["H"], splits, seed, kind=kind,
                         task="regression", permute_train_labels=True)
    controls["label_permutation"] = {
        "description": "训练侧打乱 H 标签，测试侧使用真实标签",
        "target": "H",
        "event_r2": permuted["event"]["r2"],
        "block_r2": block_level_regression(permuted)["r2"],
        "selected_alpha": permuted["selected_alpha"],
        "selected_mlp": permuted.get("selected_mlp"),
    }

    location = fit_probe(features, targets["true_location"], splits, seed, kind=kind,
                         task="classification", n_classes=16)
    controls["position_positive_control"] = {
        "description": "真实位置 16 类分类，随机基线 1/16=0.0625",
        "target": "true_location",
        "event_accuracy": location["event"]["accuracy"],
        "event_macro_f1": location["event"]["macro_f1"],
        "block_accuracy": block_level_classification(location, 16)["accuracy"],
        "random_baseline": 0.0625,
    }
    return controls


# --------------------------------------------------------------------------
# 步骤五：块内一致性
# --------------------------------------------------------------------------

def block_consistency(features: dict, block_id: np.ndarray, splits: dict) -> dict:
    """计算块内与块间离散度比值，该检验不依赖标签。"""
    result = {}
    for name, matrix in features.items():
        subset = matrix[splits["train"]]
        blocks = block_id[splits["train"]]
        global_mean = subset.mean(axis=0)
        within, between = [], []
        for value in np.unique(blocks):
            members = subset[blocks == value]
            block_mean = members.mean(axis=0)
            within.append(np.linalg.norm(members - block_mean, axis=1).mean())
            between.append(np.linalg.norm(block_mean - global_mean))
        ratio = float(np.median(within) / max(np.median(between), 1e-12))
        within_median = float(np.median(within))
        if within_median <= 1e-9:
            interpretation = (
                "表示跨事件近乎恒定，比值趋零，但该结果不携带任何工况信息，"
                "不得据此判为稳定"
            )
        elif ratio < 1.0:
            interpretation = "比值显著小于 1 表示表示在工况意义上稳定"
        else:
            interpretation = "比值接近或大于 1 表示表示主要由事件特异信息主导"
        result[name] = {
            "within_block_dispersion_median": within_median,
            "between_block_dispersion_median": float(np.median(between)),
            "ratio_median": ratio,
            "interpretation": interpretation,
        }
    return result


# --------------------------------------------------------------------------
# 位置可恢复性独立核验（设计文档第 1.3 节推论）
# --------------------------------------------------------------------------

def verify_position_recoverability(data_dir: Path, dataset: dict, targets: dict) -> dict:
    """按正式 d_S 口径复算单事件族内的位置取回命中率与间隔。"""
    scaler = np.load(data_dir / "feature_scaler.npz")
    if "node_scale" not in scaler:
        return {"status": "skipped", "reason": "feature_scaler 缺少 node_scale"}
    node_scale = np.asarray(scaler["node_scale"], dtype=np.float64)
    weights = 1.0 / (node_scale ** 2 + DISTANCE_EPS)
    x_obs = np.asarray(dataset["X_obs"], dtype=np.float64)
    n_events, n_candidates, n_nodes, n_steps, n_features = dataset["paired_response"].shape
    denominator = float(n_nodes * n_steps * n_features)

    # 分块累加，避免把完整 [1600,17,16,12,6] 的差分张量一次性展开到内存。
    distance = np.empty((n_events, n_candidates), dtype=np.float64)
    paired = dataset["paired_response"]
    for start in range(0, n_events, 200):
        stop = min(start + 200, n_events)
        chunk = np.asarray(paired[start:stop], dtype=np.float64)
        block_x = x_obs[start:stop]
        paired_term = np.einsum("mkntf,mkntf,nf->mk", chunk, chunk, weights)
        cross_term = np.einsum("mkntf,mntf,nf->mk", chunk, block_x, weights)
        obs_term = np.einsum("mntf,mntf,nf->m", block_x, block_x, weights)
        distance[start:stop] = (
            paired_term - 2.0 * cross_term + obs_term[:, None]
        ) / denominator
        del chunk

    order = np.argsort(distance, axis=1)
    minimum = np.take_along_axis(distance, order[:, :1], axis=1)[:, 0]
    second = np.take_along_axis(distance, order[:, 1:2], axis=1)[:, 0]
    argmin = order[:, 0]
    true_location = targets["true_location"]
    margin = second - minimum
    return {
        "status": "ok",
        "formula": "d_S = sum_{n,t,f} (A-B)^2 / (s_{n,f}^2+1e-8) / (N*T*F)",
        "denominator": denominator,
        "n_candidates": int(n_candidates),
        "argmin_hit_rate": float(np.mean(argmin == true_location)),
        "unique_minimum_rate": float(
            np.mean((minimum < second) & (second > 0)) if n_candidates > 1 else 1.0
        ),
        "margin_min": float(margin.min()),
        "margin_p05": float(np.percentile(margin, 5)),
        "margin_median": float(np.median(margin)),
        "rho_min": float(np.sqrt(max(minimum.min(), 0.0))),
        "rho_median": float(np.sqrt(max(np.median(minimum), 0.0))),
        "rho_note": (
            "X_m 与真实位置候选的 d_S 精确为 0（实测逐元素差为 0），"
            "故 rho 的最小值与中位数均为 0；该字段只作口径声明，不用于跨量比较"
        ),
    }


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------

def summarize_dimensions(rows: list, target: str, thresholds: dict) -> dict:
    """按阈值把逐维块级结果分为可恢复／未定／不可恢复三组。

    设计文档第 8 节规定 S1／S2 记为未定时必须指明哪些维度可恢复，本函数承担
    该要求：平均值会掩盖个别维度不可恢复的情形。
    """
    selected = [row for row in rows if row["target"] == target]

    def group(predicate) -> list:
        return [row["dimension"] for row in selected if predicate(float(row["block_r2"]))]

    return {
        "target": target,
        "criterion": (
            f"块级 R² >= {thresholds['s1_s2_pass']} 记为可恢复；"
            f"<= {thresholds['s1_s2_fail']} 记为不可恢复；其间记为未定"
        ),
        "n_dimensions": len(selected),
        "recoverable": group(lambda value: value >= thresholds["s1_s2_pass"]),
        "undetermined": group(
            lambda value: thresholds["s1_s2_fail"] < value < thresholds["s1_s2_pass"]
        ),
        "not_recoverable": group(lambda value: value <= thresholds["s1_s2_fail"]),
    }


def decide(s1_block_r2: float, s2_block_r2: float, s2_event_r2: float,
           controls: dict, thresholds: dict) -> dict:
    """按设计文档第 8 节给出 S1／S2 判定、三分支结论与分支 D 状态。"""
    def label(value: float) -> str:
        if not np.isfinite(value):
            return "未定"
        if value >= thresholds["s1_s2_pass"]:
            return "成立"
        if value <= thresholds["s1_s2_fail"]:
            return "不成立"
        return "未定"

    s1_label = label(s1_block_r2)
    s2_label = label(s2_block_r2)
    random_r2 = controls["random_representation"]["block_r2"]
    permuted_r2 = controls["label_permutation"]["block_r2"]
    beats_random = bool(np.isfinite(s2_block_r2) and s2_block_r2 > random_r2)
    beats_permutation = bool(np.isfinite(s2_block_r2) and s2_block_r2 > permuted_r2)
    if s2_label == "成立" and not (beats_random and beats_permutation):
        s2_label = "不成立"

    generalization_gap = s2_event_r2 - s2_block_r2
    branch_d = bool(np.isfinite(generalization_gap)
                    and generalization_gap > thresholds["branch_d_gap"])
    if branch_d and s2_label == "成立":
        s2_label = "未定"

    if s1_label == "成立" and s2_label == "不成立":
        conclusion = "提取失败"
        action = "设计共享状态表示 U，并把表示是否被解码器使用交给判据二验证"
    elif s1_label == "不成立":
        conclusion = "输入信息缺失"
        action = "讨论补充输入（例如显式提供工作点或缩短观测口径），不设计 U"
    elif s1_label == "成立" and s2_label == "成立":
        conclusion = "瓶颈在表示之后"
        action = "转入判据二与输出侧检查，不设计 U"
    elif branch_d:
        conclusion = "证据不足"
        action = "按分支 D 处置，以块内一致性检验与判据二为主，并声明表示不具备工况级稳定性"
    elif s1_label == "成立" and s2_label == "未定":
        conclusion = "S2 未定：表示携带部分状态信息但未达到判定阈值"
        action = "报告逐维结果与负对照，补齐判据二后再决定是否设计 U；不单凭本判据设计 U"
    else:
        conclusion = "证据不足"
        action = "补齐负对照或增加样本后再判定，本判据暂不给出方向性结论"

    return {
        "thresholds": thresholds,
        "s1": {
            "label": s1_label,
            "block_r2_on_H": s1_block_r2,
            "source": "观测 X_m 池化特征（obs_full）",
        },
        "s2": {
            "label": s2_label,
            "block_r2_on_H": s2_block_r2,
            "event_r2_on_H": s2_event_r2,
            "source": "编码器表示（rep_all）",
            "beats_random_control": beats_random,
            "beats_label_permutation_control": beats_permutation,
            "random_control_block_r2": random_r2,
            "label_permutation_block_r2": permuted_r2,
        },
        "generalization_gap_event_minus_block": generalization_gap,
        "branch_d": {
            "active": branch_d,
            "gap_threshold": thresholds["branch_d_gap"],
            "statement": (
                "当前表示不具备工况级稳定性"
                if branch_d
                else "事件级与块级差距未超过阈值，表示不受本分支约束"
            ),
        },
        "conclusion": conclusion,
        "next_action": action,
        "cannot_conclude": [
            "probe 成功不等于模型在使用该信息，S2 成立只排除信息没进表示",
            "probe 在同一批数据上拟合，不能证明因果关系",
            "本判据不能替代判据二对条件已知时映射可学性的判定",
        ],
    }


# --------------------------------------------------------------------------
# 产物写出
# --------------------------------------------------------------------------

def write_json(path: Path, payload) -> None:
    """写出 UTF-8 JSON。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def write_conditions_csv(path: Path, targets: dict) -> None:
    """写出逐事件未标准化条件表。"""
    header = (["event_id", "event_index", "true_location", "fault_class",
               "true_impedance", "fault_delay_steps", "block_id"]
              + [f"H_{key}" for key in LOAD_MULTIPLIER_KEYS])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for position, event_id in enumerate(targets["event_id"]):
            writer.writerow(
                [event_id, int(targets["event_index"][position]),
                 int(targets["true_location"][position]), int(targets["fault_class"][position]),
                 float(targets["impedance_raw"][position]),
                 int(targets["fault_delay_steps"][position]), int(targets["block_id"][position])]
                + [float(value) for value in targets["H"][position]]
            )


def write_per_dimension_csv(path: Path, rows: list) -> None:
    """写出逐维回归结果。"""
    fields = ["target", "dimension", "event_r2", "event_rmse", "event_mae",
              "block_r2", "block_rmse", "block_mae"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_heartbeat(path: Path, payload: dict) -> None:
    """追加一条心跳记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _fmt(value) -> str:
    """把数值格式化为便于阅读的字符串。"""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.6e}"
        return f"{value:+.4f}" if value < 0 else f"{value:.4f}"
    return str(value)


def _selection_text(entry: dict) -> str:
    """把选中的正则强度或浅层非线性配置渲染成一行文本。"""
    if entry.get("selected_mlp"):
        info = entry["selected_mlp"]
        return (f"选中 MLP 配置：宽度 {info['width']}、"
                f"权重衰减 {_fmt(info['weight_decay'])}、种子 {info['seeds']}")
    if entry.get("selected_C") is not None:
        return f"选中 C = {_fmt(entry['selected_C'])}"
    return f"选中 α = {_fmt(entry['selected_alpha'])}"


def write_report(path: Path, config: dict, probe_metrics: dict, controls: dict,
                 consistency: dict, position: dict, decision: dict,
                 per_dimension: list) -> None:
    """写出中文执行报告；结果以列表呈现，不使用表格。"""
    lines: list[str] = []
    add = lines.append
    run_id = config["run_id"]
    s1 = decision["s1"]
    s2 = decision["s2"]

    add(f"<!-- 摘要：Method-A2 判据一共享状态可恢复性 Probe 的执行报告（运行标识 {run_id}）。"
        f"冻结既有 teacher checkpoint 的共享编码器，在表示与观测池化特征上"
        f"分别拟合岭回归与多类逻辑回归，测量共享状态 (H,C) 是否进入表示；结论为 "
        f"S1 {s1['label']}、S2 {s2['label']}，判定为「{decision['conclusion']}」。"
        f"含四类负对照、块内一致性、位置可恢复性核验与全部口径声明。"
        + ("**本文为冒烟子采样运行，不是正式实验结论。**" if decision.get("smoke_run") else "")
        + " -->")
    add("")
    add("# Method-A2 判据一执行报告：共享状态可恢复性 Probe")
    add("")
    add("## 1. 运行标识与来源")
    add("")
    add(f"- 运行标识：`{run_id}`")
    add(f"- 数据集：`{config['data_dir']}`")
    add(f"- 权重：`{config['checkpoint']}`")
    add(f"- 权重 sha256：`{config['checkpoint_sha256']}`")
    add(f"- 来源仓库标识：`{config['source_repo']}`")
    add(f"- 设备：`{config['device']}`；批大小 {config['batch_size']}；随机种子 {config['seed']}")
    add(f"- 表示提取前向耗时：{_fmt(config['forward_seconds'])} 秒；"
        f"全流程耗时：{_fmt(config['total_seconds'])} 秒")
    checkpoint_meta = config.get("checkpoint_meta") or {}
    add(f"- 权重训练信息：epoch {checkpoint_meta.get('epoch')}、"
        f"best_epoch {checkpoint_meta.get('best_epoch')}、"
        f"best_val_loss {_fmt(checkpoint_meta.get('best_val_loss'))}、"
        f"variant `{checkpoint_meta.get('variant')}`")
    git_info = config.get("git", {}).get("method_a2", {})
    add(f"- 代码版本：HEAD `{git_info.get('head')}`")
    add("")
    add("本运行未迁移数据集与权重，按设计文档第 0 节与第 10 节的要求，"
        "实际读取路径与来源仓库已在 `config.json` 中固化，未静默拼接另一仓库的数据。")
    add("")
    add("## 2. 判定结论")
    add("")
    if decision.get("smoke_run"):
        add(f"> **本文件为冒烟子采样运行的结果，不是正式实验结论。** "
            f"{decision.get('invalid_reason', '')}")
        add("")
    add(f"- S1（观测可恢复）：**{s1['label']}**，口径为 15 维工况的块级 R² = "
        f"{_fmt(s1['block_r2_on_H'])}，特征源为 {s1['source']}")
    add(f"- S2（表示可恢复）：**{s2['label']}**，口径为同目标同划分下的块级 R² = "
        f"{_fmt(s2['block_r2_on_H'])}，事件级 R² = {_fmt(s2['event_r2_on_H'])}，"
        f"特征源为 {s2['source']}")
    add(f"- S2 是否高于随机表示对照：{_fmt(s2['beats_random_control'])}"
        f"（对照块级 R² = {_fmt(s2['random_control_block_r2'])}）")
    add(f"- S2 是否高于标签置换对照：{_fmt(s2['beats_label_permutation_control'])}"
        f"（对照块级 R² = {_fmt(s2['label_permutation_block_r2'])}）")
    add(f"- 块级泛化差距（事件级 R² − 块级 R²）："
        f"{_fmt(decision['generalization_gap_event_minus_block'])}，"
        f"分支 D 阈值 {_fmt(decision['branch_d']['gap_threshold'])}")
    add(f"- 分支 D：{'触发' if decision['branch_d']['active'] else '未触发'}"
        f"；{decision['branch_d']['statement']}")
    add(f"- **三分支结论：{decision['conclusion']}**")
    add(f"- 后续动作：{decision['next_action']}")
    add("")
    add("## 3. 运行配置与口径声明")
    add("")
    split_rule = config["split_rule"]
    add(f"- 划分来源：{split_rule['source']}")
    add(f"- 事件数：训练 {split_rule['counts']['train']}、验证 {split_rule['counts']['val']}、"
        f"测试 {split_rule['counts']['test']}")
    add(f"- 块数：训练 {split_rule['block_counts']['train']}、"
        f"验证 {split_rule['block_counts']['val']}、测试 {split_rule['block_counts']['test']}")
    add(f"- 泄漏防护：{split_rule['leakage_protection']}；"
        f"实测 block 跨划分交集 train-val {split_rule['block_overlap']['train_val']}、"
        f"train-test {split_rule['block_overlap']['train_test']}、"
        f"val-test {split_rule['block_overlap']['val_test']}")
    add(f"- 标准化：{config['standardization']}")
    add(f"- 正则候选集：分类 C ∈ {config['regularization_grid']['classification_C']}；"
        f"回归 α ∈ {config['regularization_grid']['regression_alpha']}")
    add(f"- probe 拟合器：`{config['probe_kind']}`"
        + (
            f"；{config['mlp_grid']['architecture']}，网格为 宽度 {config['mlp_grid']['widths']} ×"
            f" 权重衰减 {config['mlp_grid']['weight_decays']}，epochs"
            f" {config['mlp_grid']['epochs']}；{config['mlp_grid']['selection_rule']}"
            if config.get("mlp_grid")
            else "（岭回归与多类逻辑回归，均为线性口径）"
        ))
    add(f"- 回归求解器：`{config['regression_solver']['solver']}`，"
        f"tol = {config['regression_solver']['tol']}；理由：{config['regression_solver']['reason']}")
    add(f"- 分类求解器：`{config['classification_solver']['solver']}`，"
        f"max_iter = {config['classification_solver']['max_iter']}，"
        f"multi_class = {config['classification_solver']['multi_class']}；"
        f"理由：{config['classification_solver']['reason']}")
    add(f"- 确定性说明：{config['determinism_note']}")
    add(f"- 距离口径：{config['distance_convention']}")
    add("- 事件级口径为 240 个测试事件；块级口径为 60 个测试块，块内特征先取均值再预测一次，"
        "与部署时一次一个工况一致。")
    add(f"- 数据来源标识：`{config['source_repo']}`；"
        f"数据目录 `{config['data_dir']}`")
    add("")
    add("## 4. 表示清单与池化方式")
    add("")
    for name, info in config["representation_inventory"].items():
        if name == "observation_poolings":
            continue
        if "raw_shape" in info:
            add(f"- `{name}`：原始形状 {info['raw_shape']}；池化形式 {info['poolings']}")
        else:
            add(f"- `{name}`：形状 {info['shape']}；由 {info['composition']} 拼接")
    add("- 观测池化（用于 S1 对照）："
        + "；".join(f"`{key}` {value}"
                    for key, value in config["representation_inventory"]["observation_poolings"].items()))
    variation = config["candidate_embed_variation"]
    add(f"- 候选嵌入跨事件变化实测：逐元素最大极差 {_fmt(variation['per_event_element_max_spread'])}，"
        f"平均极差 {_fmt(variation['per_event_element_mean_spread'])}，"
        f"是否跨事件恒定：{_fmt(variation['is_constant_across_events'])}")
    add("")
    add("## 5. 主结果：S1 与 S2 在同一目标、同一划分、同一正则选择下")
    add("")
    add("目标为 15 维工况 H 时，各特征源的回归结果如下。")
    add("")
    groups = [
        ("编码器表示（S2 口径）", ["rep_node_mean", "rep_node_flat", "rep_global",
                                  "rep_candidate_mean", "rep_attention_mean", "rep_all"]),
        ("观测池化（S1 口径）", ["obs_node_time_mean", "obs_node_flat", "obs_full"]),
        ("表示与观测拼接（辅助）", ["concat_rep_all_obs_full"]),
    ]
    for title, names in groups:
        add(f"### {title}")
        add("")
        for source in names:
            entry = probe_metrics.get(f"H|{source}")
            if not entry:
                continue
            add(f"- `{source}`：{_selection_text(entry)}；"
                f"事件级 R² = {_fmt(entry['event']['r2'])}、"
                f"RMSE = {_fmt(entry['event']['rmse'])}、MAE = {_fmt(entry['event']['mae'])}；"
                f"块级 R² = {_fmt(entry['block']['r2'])}、"
                f"RMSE = {_fmt(entry['block']['rmse'])}")
        add("")
    add("目标为 log1p 故障阻抗、故障类型与真实位置时，结果如下。")
    add("")
    for target, label in (("log1p_impedance", "log1p 故障阻抗（回归）"),
                          ("fault_class", "故障类型（5 类分类）"),
                          ("true_location", "真实位置（16 类分类，正对照）")):
        add(f"### {label}")
        add("")
        for source in ("rep_all", "obs_full", "rep_node_flat"):
            entry = probe_metrics.get(f"{target}|{source}")
            if not entry:
                continue
            if entry["task"] == "regression":
                add(f"- `{source}`：{_selection_text(entry)}；"
                    f"事件级 R² = {_fmt(entry['event']['r2'])}；"
                    f"块级 R² = {_fmt(entry['block']['r2'])}")
            else:
                add(f"- `{source}`：{_selection_text(entry)}；"
                    f"事件级准确率 = {_fmt(entry['event']['accuracy'])}、"
                    f"宏平均 F1 = {_fmt(entry['event']['macro_f1'])}；"
                    f"块级准确率 = {_fmt(entry['block']['accuracy'])}")
        add("")
    add("### 正则强度选择过程")
    add("")
    if config.get("mlp_grid"):
        add("每个（目标，特征源）组合都在验证划分上独立选择隐藏层宽度与权重衰减，"
            "选定配置再用多个随机种子重训并聚合预测；测试划分只报告一次。"
            "完整网格结果见 `probe_metrics.json` 的 `alpha_selection` 与 `C_selection` 字段。")
    else:
        add("每个（目标，特征源）组合都在验证划分上独立选参，测试划分只报告一次，"
            "完整选参曲线见 `probe_metrics.json` 的 `alpha_selection` 与 `C_selection` 字段。")
    add("")
    add("## 6. 逐维工况结果")
    add("")
    add("15 维负荷乘子逐维报告，避免平均掩盖个别维度不可恢复的情形；完整数值见 "
        "`probe_per_dimension.csv`。")
    add("")
    summary = decision.get("per_dimension") or {}
    for key, label in (("observation_obs_full", "观测特征 obs_full（S1 口径）"),
                       ("representation_rep_all", "编码器表示 rep_all（S2 口径）")):
        entry = summary.get(key)
        if not entry:
            continue
        add(f"### {label}的分维判定")
        add("")
        add(f"- 判定口径：{entry['criterion']}")
        add(f"- 可恢复（{len(entry['recoverable'])}/{entry['n_dimensions']} 维）："
            + ("、".join(f"`{name}`" for name in entry["recoverable"]) or "无"))
        add(f"- 未定（{len(entry['undetermined'])}/{entry['n_dimensions']} 维）："
            + ("、".join(f"`{name}`" for name in entry["undetermined"]) or "无"))
        add(f"- 不可恢复（{len(entry['not_recoverable'])}/{entry['n_dimensions']} 维）："
            + ("、".join(f"`{name}`" for name in entry["not_recoverable"]) or "无"))
        add("")
    add("逐维原始数值如下（目标为 H、特征源为 rep_all）。")
    add("")
    for row in per_dimension:
        if row["target"] != "H|rep_all":
            continue
        add(f"- `{row['dimension']}`：事件级 R² = {_fmt(row['event_r2'])}、"
            f"RMSE = {_fmt(row['event_rmse'])}；块级 R² = {_fmt(row['block_r2'])}、"
            f"RMSE = {_fmt(row['block_rmse'])}")
    add("")
    add("## 7. 负对照")
    add("")
    add("四类负对照与主结果并列如下，均以 15 维工况 H 为目标、使用表示特征 `rep_all`。")
    add("")
    for name, value in controls.items():
        add(f"- `{name}`：{value.get('description', '')}")
        if "event_r2" in value:
            add(f"  - 事件级 R² = {_fmt(value['event_r2'])}；"
                f"块级 R² = {_fmt(value['block_r2'])}"
                + (f"；选中 α = {_fmt(value['selected_alpha'])}"
                   if "selected_alpha" in value else ""))
        else:
            add(f"  - 事件级准确率 = {_fmt(value['event_accuracy'])}、"
                f"宏平均 F1 = {_fmt(value['event_macro_f1'])}；"
                f"块级准确率 = {_fmt(value['block_accuracy'])}；"
                f"随机基线 = {_fmt(value['random_baseline'])}")
    add("")
    add("## 8. 块内一致性（标签无关）")
    add("")
    add("利用同一 block 的 4 个事件共享 (H,C) 这一数据事实，比较块内与块间离散度比值。")
    add("")
    for name, value in consistency.items():
        add(f"- `{name}`：块内离散度中位数 {_fmt(value['within_block_dispersion_median'])}、"
            f"块间离散度中位数 {_fmt(value['between_block_dispersion_median'])}、"
            f"比值 {_fmt(value['ratio_median'])}；{value['interpretation']}")
    add("")
    add("该检验只作为 S2 的辅助证据，不能替代 probe。")
    add("")
    add("## 9. 位置可恢复性独立核验")
    add("")
    if position.get("status") == "ok":
        add(f"- 口径：{position['formula']}，分母 {_fmt(position['denominator'])}，"
            f"候选数 {position['n_candidates']}")
        add(f"- 单事件族内 argmin 命中真实位置的比率：{_fmt(position['argmin_hit_rate'])}")
        add(f"- 唯一最小值比率：{_fmt(position['unique_minimum_rate'])}")
        add(f"- 最小值与次小值间隔：最小 {_fmt(position['margin_min'])}、"
            f"p05 {_fmt(position['margin_p05'])}、中位数 {_fmt(position['margin_median'])}")
        add(f"- 开方口径 ρ：最小值 {_fmt(position['rho_min'])}、"
            f"中位数 {_fmt(position['rho_median'])}")
        if position.get("rho_note"):
            add(f"- ρ 口径说明：{position['rho_note']}")
    else:
        add(f"- 未执行：{position.get('reason')}")
    add("")
    add("## 10. 本判据不能证明的边界")
    add("")
    for item in decision["cannot_conclude"]:
        add(f"- {item}")
    add("")
    add("## 11. 产物清单")
    add("")
    for name, description in (
        ("config.json", "运行配置、实际读取路径、权重 sha256、表示清单、划分规则、正则网格与选参结果"),
        ("representations.npz", "各表示与池化形式的数组，含 event_indices 与 block_ids"),
        ("conditions_raw.csv", "逐事件未标准化条件表"),
        ("probe_metrics.json", "每个（目标，特征源）的事件级与块级指标、选参曲线"),
        ("probe_per_dimension.csv", "15 维工况与阻抗的逐维结果"),
        ("negative_controls.json", "四类负对照结果"),
        ("block_consistency.json", "块内与块间离散度及其比值"),
        ("position_recoverability.json", "单事件族内位置取回与间隔的独立核验"),
        ("decision.json", "S1 与 S2 判定标签、阈值、三分支结论与分支 D 状态"),
    ):
        add(f"- `{name}`：{description}")
    add("")
    add("本文为实验执行后的结果文档，全部数值由 "
        "`scripts/run_error_source_criterion1_probe.py` 单次运行产生。")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """执行判据一实验并写出全部产物。"""
    args = parse_args()
    started = time.perf_counter()
    run_id = args.run_id or f"c1run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed{args.seed}"
    output_dir = args.output_root / run_id / "criterion-1"
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = args.log_root / run_id / "heartbeat.jsonl"

    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    for required in (data_dir, checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"路径不存在：{required}")
    missing_files = [name for name in DATASET_FILES if not (data_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"数据集缺少文件：{missing_files}")

    print(f"[1/6] 载入数据与权重，source_repo={args.source_repo}", flush=True)
    dataset = {
        "X_obs": np.load(data_dir / "X_obs.npy", allow_pickle=False),
        "paired_response": np.load(data_dir / "paired_response.npy", allow_pickle=False),
        "candidate_features": np.load(data_dir / "candidate_features.npy", allow_pickle=False),
        "node_features": np.load(data_dir / "node_features.npy", allow_pickle=False),
        "edge_index": np.load(data_dir / "edge_index.npy", allow_pickle=False),
        "edge_attr": np.load(data_dir / "edge_attr.npy", allow_pickle=False),
        "edge_mask": np.load(data_dir / "edge_mask.npy", allow_pickle=False),
        "candidate_mask": np.load(data_dir / "candidate_mask.npy", allow_pickle=False),
        "node_mask": np.load(data_dir / "node_mask.npy", allow_pickle=False),
    }
    rows = load_condition_metadata(data_dir)
    targets = build_targets(rows)
    n_events = dataset["X_obs"].shape[0]
    if len(rows) != n_events:
        raise ValueError(f"元数据事件数 {len(rows)} 与数组事件数 {n_events} 不一致")
    print(f"      事件 {n_events}，节点 {dataset['X_obs'].shape[1]}，"
          f"窗口 {dataset['X_obs'].shape[2]}，特征 {dataset['X_obs'].shape[3]}", flush=True)

    splits = {
        key: np.load(data_dir / f"{key}.npy", allow_pickle=False)
        for key in ("train_idx", "val_idx", "test_idx")
    }
    splits = {
        "train": splits["train_idx"], "val": splits["val_idx"], "test": splits["test_idx"],
    }
    block_id = targets["block_id"]
    if args.block_limit is not None:
        splits = limit_blocks(splits, block_id, args.block_limit)
        print(f"      冒烟子采样：每划分最多 {args.block_limit} 个 block，"
              f"实际事件数 训练 {splits['train'].size}、"
              f"验证 {splits['val'].size}、测试 {splits['test'].size}", flush=True)
    splits["test_blocks"] = block_ids_for(splits["test"], block_id)
    split_blocks = {key: set(block_ids_for(splits[key], block_id).tolist())
                    for key in ("train", "val", "test")}
    overlap = {
        "train_val": len(split_blocks["train"] & split_blocks["val"]),
        "train_test": len(split_blocks["train"] & split_blocks["test"]),
        "val_test": len(split_blocks["val"] & split_blocks["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"划分存在 block 泄漏：{overlap}")

    model, payload = load_encoder(checkpoint, args.device)
    checkpoint_sha256 = sha256_file(checkpoint)
    print(f"[2/6] 表示提取（device={args.device}, batch={args.batch_size}）", flush=True)
    extraction = extract_representations(model, dataset, args.device, args.batch_size)
    arrays = extraction["arrays"]
    arrays["rep_all"] = np.concatenate(
        [arrays[name] for name in ("rep_node_mean", "rep_node_flat", "rep_global",
                                   "rep_candidate_mean", "rep_attention_mean")],
        axis=1,
    )
    arrays["concat_rep_all_obs_full"] = np.concatenate(
        [arrays["rep_all"], arrays["obs_full"]], axis=1
    )
    write_heartbeat(heartbeat_path, {
        "stage": "representations_done", "run_id": run_id,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "forward_seconds": round(extraction["forward_seconds"], 4),
    })
    print(f"      前向耗时 {extraction['forward_seconds']:.2f}s；"
          f"rep_all 维度 {arrays['rep_all'].shape[1]}", flush=True)

    feature_names = ["rep_node_mean", "rep_node_flat", "rep_global",
                     "rep_candidate_mean", "rep_attention_mean", "rep_all"]
    observation_names = ["obs_node_time_mean", "obs_node_flat", "obs_full"]

    def sliced(name: str) -> dict:
        matrix = arrays[name]
        return {key: matrix[splits[key]] for key in ("train", "val", "test")}

    print("[3/6] 主 probe 拟合", flush=True)
    probe_metrics = {}
    regression_results = {}
    # 逐维报告只用到这三个组合；其余结果不再持有，避免长期驻留大量神经网络模块。
    needed_for_per_dimension = {("H", "rep_all"), ("H", "obs_full"),
                                ("log1p_impedance", "rep_all")}
    partial_path = output_dir / "probe_metrics.partial.json"

    def persist_partial() -> None:
        """每次拟合后立即落盘，避免末段失败导致整轮结果丢失。"""
        write_json(partial_path, {
            "status": "partial",
            "completed_fits": len(probe_metrics),
            "results": probe_metrics,
        })

    for target_name, dim_names in (
        ("H", list(LOAD_MULTIPLIER_KEYS)),
        ("log1p_impedance", ["log1p_impedance"]),
    ):
        target = targets[target_name]
        for source_name in feature_names + observation_names + ["concat_rep_all_obs_full"]:
            result = fit_probe(sliced(source_name), target, splits, args.seed,
                               kind=args.probe_kinds, task="regression")
            if (target_name, source_name) in needed_for_per_dimension:
                regression_results[(target_name, source_name)] = result
            probe_metrics[f"{target_name}|{source_name}"] = {
                "target": target_name, "feature_source": source_name,
                "probe_kind": args.probe_kinds, "task": "regression",
                "selected_alpha": result["selected_alpha"],
                "selected_mlp": result.get("selected_mlp"),
                "alpha_selection": result["selection"],
                "event": result["event"], "block": block_level_regression(result),
            }
            block_r2 = probe_metrics[f"{target_name}|{source_name}"]["block"]["r2"]
            event_r2 = probe_metrics[f"{target_name}|{source_name}"]["event"]["r2"]
            del result
            persist_partial()
            print(f"      {target_name:<16} {source_name:<24} "
                  f"event R²={event_r2:+.4f} block R²={block_r2:+.4f}", flush=True)

    for target_name, n_classes in (("fault_class", 5), ("true_location", 16)):
        labels = targets[target_name]
        for source_name in feature_names + observation_names + ["concat_rep_all_obs_full"]:
            result = fit_probe(sliced(source_name), labels, splits, args.seed,
                               kind=args.probe_kinds, task="classification",
                               n_classes=n_classes)
            probe_metrics[f"{target_name}|{source_name}"] = {
                "target": target_name, "feature_source": source_name,
                "probe_kind": args.probe_kinds, "task": "classification",
                "selected_C": result["selected_C"],
                "selected_mlp": result.get("selected_mlp"),
                "C_selection": result["selection"],
                "event": result["event"],
                "block": block_level_classification(result, n_classes),
            }
            persist_partial()
            print(f"      {target_name:<16} {source_name:<24} "
                  f"event acc={result['event']['accuracy']:.4f} "
                  f"block acc={block_level_classification(result, n_classes)['accuracy']:.4f}",
                  flush=True)

    write_heartbeat(heartbeat_path, {
        "stage": "probes_done", "run_id": run_id,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    })

    print("[4/6] 负对照", flush=True)
    controls = build_negative_controls(arrays, splits, targets, args.seed,
                                       kind=args.probe_kinds)
    for name, value in controls.items():
        print(f"      {name}: {json.dumps(value, ensure_ascii=False)[:150]}", flush=True)

    print("[5/6] 块内一致性与位置可恢复性核验", flush=True)
    consistency = block_consistency({**extraction["features"], "rep_all": arrays["rep_all"]},
                                    block_id, splits)
    position = verify_position_recoverability(data_dir, dataset, targets)
    print(f"      位置取回命中率 = {position.get('argmin_hit_rate')}", flush=True)

    per_dimension = []
    for row in per_dimension_regression(regression_results[("H", "rep_all")],
                                        list(LOAD_MULTIPLIER_KEYS)):
        per_dimension.append({**row, "target": "H|rep_all"})
    for row in per_dimension_regression(regression_results[("H", "obs_full")],
                                        list(LOAD_MULTIPLIER_KEYS)):
        per_dimension.append({**row, "target": "H|obs_full"})
    for row in per_dimension_regression(regression_results[("log1p_impedance", "rep_all")],
                                        ["log1p_impedance"]):
        per_dimension.append({**row, "target": "log1p_impedance|rep_all"})

    print("[6/6] 判定与产物写出", flush=True)
    thresholds = {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3}
    decision = decide(
        s1_block_r2=probe_metrics["H|obs_full"]["block"]["r2"],
        s2_block_r2=probe_metrics["H|rep_all"]["block"]["r2"],
        s2_event_r2=probe_metrics["H|rep_all"]["event"]["r2"],
        controls=controls, thresholds=thresholds,
    )
    decision["smoke_run"] = args.block_limit is not None
    # 设计文档第 8 节：S1／S2 记为未定时必须指明哪些维度可恢复。
    decision["per_dimension"] = {
        "observation_obs_full": summarize_dimensions(per_dimension, "H|obs_full", thresholds),
        "representation_rep_all": summarize_dimensions(per_dimension, "H|rep_all", thresholds),
    }
    if decision["smoke_run"]:
        decision["invalid_reason"] = (
            f"冒烟子采样运行（每划分 {args.block_limit} 个 block），"
            "样本量不足以支撑 S1／S2 判定，以下标签仅供链路验证，不得作为实验结论引用"
        )

    config = {
        "run_id": run_id,
        "started_utc": utc_now(),
        "source_repo": args.source_repo,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_meta": {key: payload.get(key) for key in
                            ("epoch", "best_epoch", "best_val_loss", "variant")},
        "dataset_meta": json.loads((data_dir / "meta.json").read_text(encoding="utf-8")),
        "representation_inventory": {
            "node_repr": {"raw_shape": list(arrays["node_repr"].shape),
                          "poolings": ["rep_node_mean", "rep_node_flat"]},
            "global_repr": {"raw_shape": list(arrays["global_repr"].shape),
                            "poolings": ["rep_global"]},
            "candidate_embed": {"raw_shape": list(arrays["candidate_embed"].shape),
                                "poolings": ["rep_candidate_mean"]},
            "attention": {"raw_shape": list(arrays["attention"].shape),
                          "poolings": ["rep_attention_mean"]},
            "rep_all": {"shape": list(arrays["rep_all"].shape),
                        "composition": ["rep_node_mean", "rep_node_flat", "rep_global",
                                        "rep_candidate_mean", "rep_attention_mean"]},
            "observation_poolings": {name: list(arrays[name].shape) for name in observation_names},
        },
        "candidate_embed_variation": candidate_embed_variation(arrays["candidate_embed"]),
        "determinism_note": (
            "既有编码器不含 dropout 层，eval() 下前向确定，故未做多次前向取平均"
        ),
        "split_rule": {
            "source": "数据集自带 train_idx/val_idx/test_idx",
            "counts": {key: int(splits[key].size) for key in ("train", "val", "test")},
            "block_counts": {key: int(len(split_blocks[key])) for key in ("train", "val", "test")},
            "block_overlap": overlap,
            "leakage_protection": "按 block 切分，禁止按事件随机切分",
        },
        "standardization": "probe 输入用训练侧统计量；回归目标按训练侧均值与标准差标准化后拟合",
        "probe_kind": args.probe_kinds,
        "regularization_grid": {"classification_C": list(C_GRID), "regression_alpha": list(ALPHA_GRID)},
        "mlp_grid": {
            "widths": list(MLP_WIDTHS), "weight_decays": list(MLP_WEIGHT_DECAYS),
            "epochs": int(MLP_EPOCHS), "learning_rate": float(MLP_LEARNING_RATE),
            "selection_seed": int(MLP_SELECTION_SEED),
            "report_seeds": list(MLP_REPORT_SEEDS),
            "architecture": "单隐藏层 MLP（Linear-ReLU-Linear），全批量 Adam",
            "selection_rule": "宽度与权重衰减在验证划分上选择；选定配置用多个随机种子重训并聚合预测",
        } if args.probe_kinds == "mlp" else None,
        "regression_solver": {
            "solver": RIDGE_SOLVER, "tol": RIDGE_TOL,
            "reason": (
                "sklearn 1.0.2 与 scipy 1.13.1 不兼容，Ridge 默认 auto/cholesky 调用"
                "已移除的 scipy.linalg.solve(sym_pos=...) 抛 TypeError；lsqr 与 svd 的"
                "系数最大差实测 1.1e-8"
            ),
        },
        "classification_solver": {
            "solver": CLASSIFICATION_SOLVER, "max_iter": 2000, "multi_class": "multinomial",
            "reason": (
                "规格未指定 solver；同一 16 类问题上 lbfgs 需 166.4 秒、"
                "newton-cg 需 1.4 秒，相差约 119 倍，二者均为 multinomial 口径"
            ),
        },
        "selected_regularization": {
            f"{target}|{source}": (
                probe_metrics[f"{target}|{source}"].get("selected_mlp")
                if args.probe_kinds == "mlp"
                else probe_metrics[f"{target}|{source}"].get("selected_C",
                     probe_metrics[f"{target}|{source}"].get("selected_alpha"))
            )
            for target in ("H", "log1p_impedance", "fault_class", "true_location")
            for source in ("rep_all", "obs_full")
        },
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "subsampling": {
            "block_limit": args.block_limit,
            "is_smoke": args.block_limit is not None,
            "note": (
                "冒烟子采样运行：样本量不足以支撑 S1／S2 判定，结论不得引用"
                if args.block_limit is not None else "未子采样，使用全部 block"
            ),
        },
        "forward_seconds": extraction["forward_seconds"],
        "total_seconds": round(time.perf_counter() - started, 4),
        "git": {"method_a2": git_probe(METHOD_DIR.parents[1])},
        "distance_convention": (
            "d_S 为平方量，rho=sqrt(d_S) 为开方口径；状态向量 (H,C) 与响应张量不同空间，"
            "本判据对状态误差只报原始单位，rho 仅用于响应空间的位置可恢复性核验"
        ),
    }
    write_json(output_dir / "config.json", config)

    np.savez_compressed(
        output_dir / "representations.npz",
        event_indices=targets["event_index"], block_ids=block_id,
        **{name: arrays[name] for name in arrays},
    )
    write_conditions_csv(output_dir / "conditions_raw.csv", targets)
    write_json(output_dir / "probe_metrics.json", {"thresholds": thresholds, "results": probe_metrics})
    write_per_dimension_csv(output_dir / "probe_per_dimension.csv", per_dimension)
    write_json(output_dir / "negative_controls.json", controls)
    write_json(output_dir / "block_consistency.json", consistency)
    write_json(output_dir / "position_recoverability.json", position)
    write_json(output_dir / "decision.json", decision)
    write_report(output_dir / "report.md", config, probe_metrics, controls,
                 consistency, position, decision, per_dimension)
    if partial_path.exists():
        partial_path.unlink()
    write_heartbeat(heartbeat_path, {
        "stage": "run_done", "run_id": run_id,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "output_dir": str(output_dir),
    })

    print(json.dumps({
        "run_id": run_id, "output_dir": str(output_dir),
        "S1": decision["s1"], "S2": decision["s2"],
        "conclusion": decision["conclusion"], "branch_d": decision["branch_d"]["active"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
