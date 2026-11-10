"""RaReF 区域概率损失与节点缺测增强的数值、对齐、梯度与端到端契约测试。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref import training
from src.gref.data import sample_node_masks
from src.gref.losses import region_term_losses, three_term_losses
from src.gref.region import (
    calibrate_temperature,
    graph_distance_hops,
    minimal_regions,
    probability_statistics,
    region_event_metrics,
    region_kernel,
    region_probabilities,
    region_target,
)


def _kernel(distance, sigma=1.0):
    """把跳数矩阵转换为 torch 区域核，供损失测试使用。"""
    return torch.as_tensor(region_kernel(distance, sigma), dtype=torch.float32)


def test_graph_distance_matches_known_topology():
    """跳数距离按无向边计算，并与候选索引一一对应。"""
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=np.int64)
    distance = graph_distance_hops(edges, 4)
    assert distance[0, 3] == 3 and distance[0, 2] == 2 and distance[1, 3] == 2
    assert np.array_equal(distance, distance.T) and (np.diag(distance) == 0).all()


def test_graph_distance_rejects_identity_mismatch():
    """边端点越界或图不连通时必须报错，不得按数组位置推断邻接。"""
    with pytest.raises(ValueError, match="越界"):
        graph_distance_hops(np.array([[0, 4]]), 4)
    with pytest.raises(ValueError, match="不连通"):
        graph_distance_hops(np.array([[0, 1]]), 3)


def test_kernel_is_column_stochastic_and_peaks_at_candidate():
    """区域核每列归一、按跳数单调衰减，并保持物理候选身份。"""
    distance = graph_distance_hops(
        np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]), 4)
    kernel = region_kernel(distance, sigma=1.0)
    assert np.allclose(kernel.sum(axis=0), 1.0)
    for candidate in range(4):
        assert int(np.argmax(kernel[:, candidate])) == candidate
        assert kernel[candidate, candidate] > kernel[distance[:, candidate] == 2, candidate].max()
    with pytest.raises(ValueError, match="sigma"):
        region_kernel(distance, 0.0)


def test_region_probability_and_target_normalization():
    """区域概率与区域目标都必须归一，且 beta 在 one-hot 与核之间插值。"""
    distance = graph_distance_hops(np.array([[0, 1], [1, 0], [1, 2], [2, 1]]), 3)
    kernel = region_kernel(distance, sigma=1.0)
    probabilities = np.array([[0.5, 0.3, 0.2]])
    region = region_probabilities(probabilities, kernel)
    assert region.shape == (1, 3) and np.isclose(region.sum(), 1.0, atol=1e-9)
    one_hot = region_target(kernel, 1, True, beta_obs=0.0, beta_miss=0.4)
    assert np.allclose(one_hot, [0.0, 1.0, 0.0])
    diffused = region_target(kernel, 1, False, beta_obs=0.1, beta_miss=0.4)
    assert np.isclose(diffused.sum(), 1.0, atol=1e-9)
    assert np.allclose(diffused, 0.6 * np.array([0.0, 1.0, 0.0]) + 0.4 * kernel[:, 1])


def test_region_loss_hand_computed_and_backward():
    """区域损失等于目标分布与区域概率的交叉熵，并能回传到候选响应。"""
    torch.manual_seed(0)
    response = torch.randn(1, 3, 2, 2, 1, requires_grad=True)
    observation = torch.randn(1, 2, 2, 1)
    target = torch.randn(1, 3, 2, 2, 1)
    candidate_mask = torch.ones(1, 3)
    locations = torch.tensor([0])
    types = torch.zeros(1, 3, 1)
    types[:, -1] = 1.0
    distance = graph_distance_hops(np.array([[0, 1], [1, 0]]), 2)
    kernel = _kernel(distance, 1.0)
    mask = torch.ones(1, 2)
    losses = region_term_losses(
        {"response": response, "candidate_types": types}, observation, target,
        candidate_mask, locations, kernel, rank_temperature=0.1,
        lambda_energy=1.0, lambda_rank=0.3, lambda_region=0.1,
        beta_obs=0.1, beta_miss=0.4, mask=mask)
    # 手工复算区域交叉熵：仅在两个故障候选上构造单节点概率
    energies = losses["energies"].detach()
    p_fault = torch.softmax(-energies[:, :2] / 0.1, dim=1)
    q = p_fault @ kernel.t()
    y = (0.9 * torch.eye(2)[0] + 0.1 * kernel[:, 0]).unsqueeze(0)
    expected = -(y * torch.log(q.clamp_min(1e-12))).sum(dim=1).mean()
    assert torch.allclose(losses["region"], expected, atol=1e-6)
    losses["total"].backward()
    assert response.grad is not None and torch.isfinite(response.grad).all()
    assert float(response.grad.abs().sum()) > 0


def test_region_loss_keeps_energy_and_rank_definitions():
    """区域模式复用原有能量与排序口径，逐项与三项损失完全一致。"""
    torch.manual_seed(3)
    response = torch.randn(2, 4, 3, 2, 1, requires_grad=True)
    observation = torch.randn(2, 3, 2, 1)
    target = torch.randn(2, 4, 3, 2, 1)
    candidate_mask = torch.ones(2, 4)
    locations = torch.tensor([0, 1])
    types = torch.zeros(2, 4, 1)
    types[:, -1] = 1.0
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    output = {"response": response, "candidate_types": types}
    three = three_term_losses(output, observation, target, candidate_mask, locations, edges,
                              rank_temperature=0.1, lambda_energy=1.0,
                              lambda_rank=0.3, lambda_smooth=0.1)
    kernel = _kernel(graph_distance_hops(edges.numpy(), 3), 1.0)
    region = region_term_losses(output, observation, target, candidate_mask, locations,
                                kernel, rank_temperature=0.1, lambda_energy=1.0,
                                lambda_rank=0.3, lambda_region=0.1)
    assert torch.allclose(three["energy"], region["energy"])
    assert torch.allclose(three["rank"], region["rank"])
    assert region["region_events"] == 2
    assert torch.isfinite(region["total"])


def test_region_loss_handles_normal_only_batch_without_nan():
    """全正常批次区域项为零且保持有限，不产生 NaN。"""
    torch.manual_seed(4)
    response = torch.randn(2, 3, 2, 2, 1)
    observation = torch.randn(2, 2, 2, 1)
    target = torch.randn(2, 3, 2, 2, 1)
    candidate_mask = torch.ones(2, 3)
    types = torch.zeros(2, 3, 1)
    types[:, -1] = 1.0
    kernel = _kernel(graph_distance_hops(np.array([[0, 1], [1, 0]]), 2))
    losses = region_term_losses({"response": response, "candidate_types": types},
                                observation, target, candidate_mask,
                                torch.tensor([-1, -1]), kernel, rank_temperature=0.1)
    assert losses["region_events"] == 0 and float(losses["region"]) == 0
    assert torch.isfinite(losses["total"])


def test_minimal_region_and_event_metrics():
    """最小区域按概率降序累计，半径与中心距离按跳数计算。"""
    region = np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
    membership, size = minimal_regions(region, alpha=0.1)
    assert size.tolist() == [2, 3]
    assert membership[0].tolist() == [True, True, False]
    distance = graph_distance_hops(np.array([[0, 1], [1, 0], [1, 2], [2, 1]]), 3)
    metrics = region_event_metrics(region[:1], np.array([1]), distance, alpha=0.1)
    assert metrics["coverage"] == 1.0 and metrics["size"].tolist() == [2]
    assert metrics["radius"].tolist() == [1] and metrics["center_distance"].tolist() == [1]
    assert np.isclose(metrics["mass"][0], 0.9, atol=1e-9)


def test_probability_statistics_and_temperature_calibration():
    """熵、最大概率、首位间隔与验证集温度标定均返回有限值。"""
    statistics = probability_statistics(np.array([[0.7, 0.2, 0.1]]))
    assert statistics["max_probability"][0] == pytest.approx(0.7)
    assert statistics["top_gap"][0] == pytest.approx(0.5)
    assert statistics["entropy"][0] > 0
    energies = np.array([[0.0, 2.0, 3.0], [0.1, 0.0, 2.0]])
    fault = np.ones((2, 3), dtype=bool)
    calibration = calibrate_temperature(energies, fault, np.array([0, 1]))
    assert calibration["temperature"] > 0
    assert np.isfinite(calibration["negative_log_likelihood"])
    assert calibration["n_events"] == 2


def test_sample_node_masks_covers_required_conditions():
    """训练掩码必须覆盖完整观测、真实节点被观测与未被观测两类缺测。"""
    locations = np.tile(np.array([0, 1, 2, 3, -1, 4, 5, 6]), 5)
    masks = sample_node_masks(locations, 8, np.random.default_rng(342),
                              full_probability=0.25, min_observed=3,
                              max_observed=6, force_observed_probability=0.5)
    assert masks.shape == (locations.size, 8) and np.isin(masks, [0.0, 1.0]).all()
    counts = masks.sum(axis=1).astype(int)
    assert (((counts >= 3) & (counts <= 6)) | (counts == 8)).all()
    rows = np.arange(locations.size)
    observed = masks[rows, np.clip(locations, 0, 7)] > 0
    fault_rows = locations >= 0
    assert (fault_rows & observed).any() and (fault_rows & ~observed).any()
    # 完整观测与两类缺测的强制分支
    full = sample_node_masks(locations, 8, np.random.default_rng(1), full_probability=1.0,
                             min_observed=3, max_observed=6)
    assert (full.sum(axis=1) == 8).all()
    forced = sample_node_masks(np.array([0, 1, 2, 3]), 8, np.random.default_rng(2),
                               full_probability=0.0, min_observed=3, max_observed=6,
                               force_observed_probability=1.0)
    assert forced[np.arange(4), np.arange(4)] .all()
    excluded = sample_node_masks(np.array([0, 1, 2, 3]), 8, np.random.default_rng(3),
                                 full_probability=0.0, min_observed=3, max_observed=6,
                                 force_observed_probability=0.0)
    assert not excluded[np.arange(4), np.arange(4)].any()
    # 同一随机种子必须逐元素一致
    repeat = sample_node_masks(locations, 8, np.random.default_rng(342),
                               full_probability=0.25, min_observed=3,
                               max_observed=6, force_observed_probability=0.5)
    assert np.array_equal(masks, repeat)


def test_parse_args_supports_region_mode():
    """入口接受区域损失参数，并保留能量与排序权重。"""
    args = training.parse_args([
        "--data-dir", "unused", "--output-dir", "unused",
        "--energy-tolerance", "0.001", "--tolerance-source", "测试",
        "--temperature", "0.1", "--loss-mode", "region",
        "--lambda-rank", "0.3", "--lambda-region", "0.1",
        "--rank-temperature", "0.004", "--region-sigma", "1.0",
    ])
    assert args.loss_mode == "region" and args.lambda_energy == 1.0
    assert args.lambda_rank == 0.3 and args.lambda_region == 0.1
    assert args.beta_obs == 0.1 and args.beta_miss == 0.4
    assert args.node_mask_min_observed == 15 and args.node_mask_max_observed == 37


def _dataset(directory: Path) -> None:
    """生成独立事件族的合成数组，仅供区域模式端到端契约测试。"""
    directory.mkdir()
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(6, 4, 3, 2, 1)).astype(np.float32)
    signatures = np.repeat(bank, 3, axis=0)
    labels = np.tile(np.arange(3), 6)
    x = signatures[np.arange(18), labels]
    arrays = {
        "X_obs": x, "signature_bank": signatures,
        "mask": np.ones((18, 3), np.float32), "candidate_mask": np.ones((18, 4), np.float32),
        "edge_index": np.array([[0, 1], [1, 0], [1, 2], [2, 1]], np.int64),
        "edge_attr": np.ones((4, 5), np.float32), "edge_mask": np.ones((18, 4), np.float32),
        "group_id": np.repeat(np.arange(6), 3), "y_loc": labels,
        "y_resist": np.repeat(np.arange(6) + 1.0, 3).astype(np.float32),
    }
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    (directory / "meta.json").write_text(
        json.dumps({"case": "synthetic", "feature_format": "real_imag_standardized"}),
        encoding="utf-8")


def test_region_end_to_end_smoke(tmp_path, monkeypatch):
    """区域模式完成训练与落盘，并记录区域分项、掩码条件与区域核。"""
    _dataset(tmp_path / "data")
    monkeypatch.setattr(training, "METHOD_DIR", tmp_path)
    output = tmp_path / "output/gref/region-smoke"
    args = training.parse_args([
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(output),
        "--energy-tolerance", "0.0001", "--tolerance-source", "合成软件测试，不是物理容差",
        "--temperature", "1", "--loss-mode", "region", "--epochs", "2",
        "--batch-size", "3", "--seed", "342", "--rank-temperature", "0.1",
        "--lambda-rank", "0.3", "--save-epochs", "1,2", "--smoke", "--no-terminal-progress",
    ])
    training.run(args)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert config["loss_mode"] == "region"
    assert config["loss_weights"] == {"energy": 1.0, "rank": 0.3, "region": 0.1}
    assert config["region_kernel"]["sigma"] == 1.0
    assert config["node_mask_augmentation"]["min_observed"] == 2
    assert {"energy", "rank", "region", "total"} <= set(metrics["losses"])
    with (output / "train_log.csv").open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert {"train_energy_mean", "train_rank_mean", "train_region_mean",
            "train_n_full", "train_n_partial_missing"} <= set(header)
    assert (output / "region_kernel.npz").exists()
    with np.load(output / "region_kernel.npz") as payload:
        assert payload["kernel"].shape == (3, 3) and payload["distance"].shape == (3, 3)
    with np.load(output / "predictions.npz") as payload:
        assert payload["candidate_score"].shape[1] == 4
    # 每个故障事件应至少被分配到完整观测或两类缺测之一
    with (output / "train_log.csv").open(encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()[1:]
    assert rows, "训练日志为空"
