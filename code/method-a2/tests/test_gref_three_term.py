"""GRReF（能量、排序、图平滑）的数值、掩码与端到端契约测试。"""

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
from src.gref.losses import (
    graph_smoothness_terms,
    list_energy_loss,
    reconstruction_energy,
    three_term_losses,
    unique_undirected_edges,
)
from src.gref.model import GrefPredictor


def _types(candidates: int, batch: int = 1) -> torch.Tensor:
    """构造末位为正常候选、其余为故障候选的类型位。"""
    types = torch.zeros(batch, candidates, 1)
    types[:, -1] = 1.0
    return types


def test_reconstruction_energy_matches_hand_computation():
    """逐候选均方重构能量按有效观测数归一，掩码节点不参与。"""
    response = torch.tensor([[[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]])
    target = torch.zeros_like(response)
    assert torch.allclose(reconstruction_energy(response, target), torch.tensor([[2.5, 12.5]]))
    masked = reconstruction_energy(response, target, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(masked, torch.tensor([[1.0, 9.0]]))


def test_list_loss_zero_at_true_energies_and_grows_under_reversal():
    """预测能量与真实能量一致时 KL 为零，顺序反转时增大。"""
    true = torch.tensor([[0.1, 0.2, 0.3]])
    fault = torch.tensor([[True, True, False]])
    locations = torch.tensor([0])
    perfect, count = list_energy_loss(true, true, fault, locations, 0.1)
    assert count == 1 and abs(float(perfect)) < 1e-12
    reversed_energies = torch.tensor([[0.3, 0.2, 0.1]])
    violated, _ = list_energy_loss(reversed_energies, true, fault, locations, 0.1)
    partial, _ = list_energy_loss(torch.tensor([[0.1, 0.3, 0.2]]), true, fault, locations, 0.1)
    assert float(violated) > float(partial) > 0


def test_list_loss_ignores_normal_and_invalid_candidates():
    """正常候选或被掩码候选的能量变化不改变列表损失。"""
    true = torch.tensor([[0.1, 0.2, 0.3]])
    fault = torch.tensor([[True, True, False]])
    locations = torch.tensor([0])
    changed = torch.tensor([[0.1, 0.2, 1e6]])
    first, _ = list_energy_loss(torch.tensor([[0.15, 0.25, 0.0]]), true, fault, locations, 0.1)
    second, _ = list_energy_loss(torch.tensor([[0.15, 0.25, 1e6]]), true, fault, locations, 0.1)
    assert torch.allclose(first, second)
    masked_fault = torch.tensor([[True, True, False]])
    masked_true = torch.tensor([[0.1, 0.2, 9.0]])
    third, _ = list_energy_loss(torch.tensor([[0.2, 0.1, 0.0]]), masked_true, masked_fault, locations, 0.1)
    assert float(third) > 0


def test_list_loss_skips_normal_events_and_rejects_bad_temperature():
    """全正常批次返回零且无有效事件；温度必须为正有限数。"""
    energies = torch.ones(2, 3)
    fault = torch.tensor([[True, True, False], [True, True, False]])
    value, count = list_energy_loss(energies, energies, fault, torch.tensor([-1, -1]), 0.1)
    assert count == 0 and float(value) == 0
    for temperature in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="温度"):
            list_energy_loss(energies, energies, fault, torch.tensor([0, 0]), temperature)


def test_smoothness_zero_for_constant_error_and_positive_for_rough_error():
    """常数误差场位于拉普拉斯零空间，粗糙误差产生正平滑值。"""
    target = torch.zeros(1, 2, 3, 1, 1)
    constant = target + 2.0
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    numerator, denominator = graph_smoothness_terms(constant, target, torch.ones(1, 2), edges)
    assert abs(float(numerator)) < 1e-6
    # 四条有向边去重为两条无向边，分母按 2*T*F*有效边权和*候选数计算
    assert denominator == pytest.approx(2.0 * 1 * 1 * 2 * 2)
    rough = target.clone()
    rough[0, :, 0] = 1.0
    rough[0, :, 2] = -1.0
    value, _ = graph_smoothness_terms(rough, target, torch.ones(1, 2), edges)
    assert float(value) > 0


def test_smoothness_invariant_to_node_permutation():
    """节点、边端点与响应同步置换时平滑值不变。"""
    torch.manual_seed(3)
    response = torch.randn(2, 3, 3, 2, 1)
    target = torch.randn_like(response)
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]])
    mask = torch.ones(2, 3)
    perm = torch.tensor([2, 0, 1])
    first, first_denominator = graph_smoothness_terms(response, target, torch.ones(2, 3), edges, mask=mask)
    second, second_denominator = graph_smoothness_terms(
        response[:, :, perm], target[:, :, perm], torch.ones(2, 3), perm[edges], mask=mask)
    assert torch.allclose(first, second, atol=1e-5)
    assert first_denominator == pytest.approx(second_denominator)


def test_smoothness_drops_edges_touching_masked_nodes():
    """节点观测掩码关闭后，与其相连的边不再计入平滑项。"""
    response = torch.randn(1, 2, 3, 1, 1)
    target = torch.zeros_like(response)
    edges = torch.tensor([[0, 1], [1, 2]])
    visible, denominator = graph_smoothness_terms(
        response, target, torch.ones(1, 2), edges, mask=torch.tensor([[1.0, 0.0, 1.0]]))
    assert float(visible) == 0
    assert denominator == 0


def test_unique_undirected_edges_average_duplicate_weights():
    """有向边去重为无向边，重复边权取平均。"""
    edges = torch.tensor([[0, 1], [1, 0], [1, 2]])
    unique, weights = unique_undirected_edges(edges, torch.tensor([2.0, 4.0, 3.0]))
    assert unique.shape == (2, 2)
    assert torch.allclose(weights, torch.tensor([3.0, 3.0]))


def _three_term_inputs():
    """构造三项损失所需的随机批次，用于数值与梯度契约。"""
    torch.manual_seed(11)
    response = torch.randn(2, 4, 3, 2, 1) * 0.3
    response.requires_grad_(True)
    target = 0.3 * torch.randn(2, 4, 3, 2, 1)
    observation = target[:, 0].clone()
    candidate_mask = torch.ones(2, 4)
    locations = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    output = {"response": response, "candidate_types": _types(4, 2)}
    return output, observation, target, candidate_mask, locations, edges


def test_three_term_loss_composition_and_rank_isolation():
    """总损失等于三项加权和，且正常候选变化不影响排序项。"""
    output, observation, target, mask, locations, edges = _three_term_inputs()
    losses = three_term_losses(output, observation, target, mask, locations, edges,
                               rank_temperature=0.1, lambda_energy=1.0,
                               lambda_rank=0.1, lambda_smooth=0.1)
    expected = losses["energy"] + 0.1 * losses["rank"] + 0.1 * losses["smooth"]
    assert torch.allclose(losses["total"], expected, atol=1e-6)
    changed = output["response"].detach().clone()
    changed[:, -1] = 1e4
    second = three_term_losses(
        {"response": changed, "candidate_types": output["candidate_types"]},
        observation, target, mask, locations, edges, rank_temperature=0.1)
    assert torch.allclose(losses["rank"], second["rank"])


def test_three_term_loss_normal_batch_has_zero_rank_but_keeps_energy():
    """全正常批次排序项为零，能量与平滑项仍可训练。"""
    output, observation, target, mask, _, edges = _three_term_inputs()
    losses = three_term_losses(output, observation, target, mask, torch.tensor([-1, -1]),
                               edges, rank_temperature=0.1)
    assert float(losses["rank"]) == 0 and losses["rank_events"] == 0
    losses["total"].backward()
    assert output["response"].grad is not None
    assert torch.isfinite(output["response"].grad).all()


def test_three_term_gradients_reach_encoder_and_decoder():
    """三项总损失对编码器与响应解码器均产生非零梯度，且不存在可训练评分头。"""
    torch.manual_seed(5)
    model = GrefPredictor(3, 2, 1, temporal_hidden=8, shared_dim=8,
                          decoder_hidden=8, dropout=0.0, temporal_layers=1)
    x = torch.zeros(2, 3, 2, 1)
    target = torch.randn(2, 4, 3, 2, 1)
    mask = torch.ones(2, 4)
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    attr = torch.ones(4, 5)
    output = model(x, edges, attr, candidate_mask=mask)
    losses = three_term_losses(output, x, target, mask, torch.tensor([0, 1]), edges,
                               rank_temperature=0.1)
    losses["total"].backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())
    assert not any("score_head" in name for name, _ in model.named_parameters())


def test_parse_args_supports_three_term_mode():
    """入口接受三项损失参数并保留首轮参数默认值。"""
    args = training.parse_args([
        "--data-dir", "unused", "--output-dir", "unused",
        "--energy-tolerance", "0.001", "--tolerance-source", "测试",
        "--temperature", "0.1", "--loss-mode", "three-term",
        "--lambda-rank", "0.2", "--save-epochs", "1,50",
    ])
    assert args.loss_mode == "three-term" and args.lambda_energy == 1.0
    assert args.lambda_smooth == 0.1 and args.rank_temperature is None
    assert args.save_epochs == "1,50"
    default = training.parse_args([
        "--data-dir", "unused", "--output-dir", "unused",
        "--energy-tolerance", "0.001", "--tolerance-source", "测试", "--temperature", "0.1",
    ])
    assert default.loss_mode == "gref"


def _dataset(directory: Path) -> None:
    """生成独立事件族的合成数组，仅供软件契约与端到端测试。"""
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


def test_three_term_end_to_end_smoke(tmp_path, monkeypatch):
    """三项损失模式完成训练与落盘，并保存排序温度、分段检查点与优化器状态。"""
    _dataset(tmp_path / "data")
    monkeypatch.setattr(training, "METHOD_DIR", tmp_path)
    output = tmp_path / "output/gref/three-term-smoke"
    args = training.parse_args([
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(output),
        "--energy-tolerance", "0.0001", "--tolerance-source", "合成软件测试，不是物理容差",
        "--temperature", "1", "--loss-mode", "three-term", "--epochs", "2",
        "--batch-size", "3", "--seed", "342", "--save-epochs", "1,2",
        "--smoke", "--no-terminal-progress",
    ])
    training.run(args)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert config["loss_mode"] == "three-term"
    assert config["rank_temperature"] > 0 and config["rank_calibration"]["n_events"] > 0
    assert {"energy", "rank", "smooth", "total"} <= set(metrics["losses"])
    with (output / "train_log.csv").open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert {"train_energy_mean", "train_rank_mean", "train_smooth_mean"} <= set(header)
    assert (output / "optimizer.pt").exists()
    assert (output / "checkpoint-epoch001.pt").exists()
    assert (output / "checkpoint-epoch002.pt").exists()
    with np.load(output / "predictions.npz") as payload:
        assert payload["candidate_score"].shape[1] == 4
    snapshot = json.loads(
        (tmp_path / "logs/three-term-smoke/progress.latest.json").read_text(encoding="utf-8"))
    assert snapshot["kind"] == "end"
