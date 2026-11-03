"""GReF 的物理评分、门控梯度、训练隔离和端到端产物验证。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.data import fit_normalization, load_arrays
from src.gref.losses import gref_losses, response_energy, teacher_gate
from src.gref.model import GrefPredictor
from src.gref import training
from src.training.evaluation import rank_metrics


def _tiny():
    """构造包含正常候选且教师能量可手算的两节点批次。"""
    x = torch.zeros(2, 2, 4, 1)
    target = torch.stack([torch.stack([x[0], x[0] + 1, x[0] + 2]),
                          torch.stack([x[1] + 1, x[1], x[1] + 2])])
    return x, target, torch.tensor([0, 1]), torch.ones(2, 3)


def _model_inputs():
    """提供固定图和无随机失活的小模型以核验梯度及置换。"""
    torch.manual_seed(9)
    model = GrefPredictor(2, 4, 1, temporal_hidden=8, shared_dim=8,
                          decoder_hidden=8, dropout=0.0, temporal_layers=1)
    x, target, labels, mask = _tiny()
    edges, attr = torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 5)
    return model, x, target, labels, mask, edges, attr


def test_energy_mask_denominator_and_missing_values():
    """掩码排除的通道不参与分子和分母，包括其缺失值。"""
    x = torch.zeros(1, 2, 1, 1)
    response = torch.tensor([[[[[2.0]], [[float('nan')]]], [[[3.0]], [[99.0]]]]])
    assert torch.equal(response_energy(x, response, torch.tensor([[1., 0.]])), torch.tensor([[4., 9.]]))
    with pytest.raises(ValueError, match="有效观测"):
        response_energy(x, response, torch.zeros(1, 2))


def test_teacher_gate_boundary_conflict_normal_and_single_candidate():
    """边界间隔、教师冲突、正常行和无负候选行均不产生定位权重。"""
    energies = torch.tensor([[0., .2, 0.], [.3, .1, 0.], [0., 1., 0.], [0., 8., 0.]])
    labels = torch.tensor([0, 0, -1, 0])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
    weights, margins = teacher_gate(energies, labels, mask, .2)
    assert torch.equal(weights, torch.zeros(4))
    assert margins[1] < 0
    weights, _ = teacher_gate(energies, labels, mask, torch.tensor([.1, .1, .1, .1]))
    assert weights[0].item() == pytest.approx(.5)
    assert not weights.requires_grad


@pytest.mark.parametrize("tolerance", [0., -1., float("nan"), float("inf")])
def test_gate_rejects_unmeasured_or_invalid_tolerance(tolerance):
    """禁止隐式零容差和非有限容差。"""
    with pytest.raises(ValueError, match="容差"):
        teacher_gate(torch.tensor([[0., 1.]]), torch.tensor([0]), torch.ones(1, 2), tolerance)


def test_masked_or_out_of_bounds_true_candidate_is_rejected():
    """真实位置不能被有效候选掩码排除或越过候选轴。"""
    with pytest.raises(ValueError, match="有效故障候选"):
        teacher_gate(torch.zeros(1, 3), torch.tensor([1]), torch.tensor([[1, 0, 0]]), .1)
    with pytest.raises(ValueError, match="越界"):
        teacher_gate(torch.zeros(1, 3), torch.tensor([3]), torch.ones(1, 3), .1)


def test_loc_gradient_reaches_response_decoder_and_encoder():
    """仅定位项反传时编码器与响应输出层均收到非零梯度，旧评分头无参数。"""
    model, x, target, labels, mask, edges, attr = _model_inputs()
    output = model(x, edges, attr, candidate_mask=mask)
    losses = gref_losses(output, x, target, mask, labels, tolerance=.1, temperature=.5)
    losses["loc"].backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())
    assert not any("score_head" in name for name, _ in model.named_parameters())
    assert torch.allclose(output["candidate_score"], response_energy(x, output["response"]))


@pytest.mark.parametrize("normal", [False, True])
def test_closed_gate_or_normal_batch_keeps_family_training(normal):
    """全门控关闭或全正常时定位返回可微零值，重构仍有梯度。"""
    x, target, labels, mask = _tiny()
    prediction = torch.randn_like(target, requires_grad=True)
    types = torch.tensor([[[0.], [0.], [1.]]]).expand(2, -1, -1)
    losses = gref_losses({"response": prediction, "candidate_types": types}, x, target,
                         mask, torch.full_like(labels, -1) if normal else labels,
                         tolerance=2., temperature=1.)
    assert losses["loc"].item() == 0
    assert losses["active_events"].item() == 0
    losses["total"].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


@pytest.mark.parametrize("normal", [True, False])
def test_normal_and_invalid_candidates_do_not_enter_loc(normal):
    """正常和无效候选的任意响应修改不改变定位项。"""
    x, target, labels, mask = _tiny()
    types = torch.tensor([[[0.], [0.], [1.]]]).expand(2, -1, -1)
    if not normal:
        types = torch.zeros_like(types)
        mask[:, -1] = 0
    first = torch.randn_like(target)
    second = first.clone()
    second[:, -1] = 1e6
    options = dict(tolerance=.1, temperature=1.)
    a = gref_losses({"response": first, "candidate_types": types}, x, target, mask, labels, **options)
    b = gref_losses({"response": second, "candidate_types": types}, x, target, mask, labels, **options)
    assert torch.equal(a["loc"], b["loc"])
    if normal:
        assert b["family"] > a["family"]
    else:
        assert torch.equal(a["family"], b["family"])


def test_unobserved_nodes_cannot_change_score():
    """未观测节点的输入值不会通过编码器或能量计算泄入评分。"""
    model, x, _, _, mask, edges, attr = _model_inputs()
    model.eval()
    node_mask = torch.tensor([[1., 0.], [1., 0.]])
    changed = x.clone()
    changed[:, 1] = 1e6
    with torch.no_grad():
        first = model(x, edges, attr, candidate_mask=mask, node_mask=node_mask)
        second = model(changed, edges, attr, candidate_mask=mask, node_mask=node_mask)
    assert torch.allclose(first["candidate_score"], second["candidate_score"])


def test_node_permutation_preserves_energies():
    """节点、边端点与候选标记同步置换时每个候选能量不变。"""
    model, x, _, _, mask, edges, attr = _model_inputs()
    model.eval()
    with torch.no_grad():
        original = model(x, edges, attr, candidate_mask=mask)
        perm = torch.tensor([1, 0])
        changed = model(x[:, perm], perm[edges], attr,
                        candidate_markers=original["candidate_markers"][:, :, perm],
                        candidate_types=original["candidate_types"], candidate_mask=mask)
    assert torch.allclose(changed["candidate_score"], original["candidate_score"], atol=1e-6)


def test_candidate_permutation_and_response_ablation():
    """候选标记与类型同步置换时评分等变，候选无关响应使评分完全并列。"""
    model, x, _, _, mask, edges, attr = _model_inputs()
    model.eval()
    with torch.no_grad():
        output = model(x, edges, attr, candidate_mask=mask)
        perm = torch.tensor([2, 0, 1])
        changed = model(x, edges, attr, candidate_markers=output["candidate_markers"][:, perm],
                        candidate_types=output["candidate_types"][:, perm], candidate_mask=mask[:, perm])
    assert torch.allclose(changed["candidate_score"], output["candidate_score"][:, perm], atol=1e-6)
    shuffled_energy = response_energy(x, output["response"][:, perm])
    assert torch.allclose(shuffled_energy, output["candidate_score"][:, perm])
    shared = output["response"].mean(1, keepdim=True).expand_as(output["response"])
    energy = response_energy(x, shared)
    assert torch.equal(energy[:, 0], energy[:, 1])


def test_masked_missing_targets_do_not_poison_family_loss():
    """未观测节点的缺失目标不会污染重构或定位损失。"""
    x, target, labels, mask = _tiny()
    target[:, :, 1] = torch.nan
    prediction = torch.randn_like(target, requires_grad=True)
    types = torch.tensor([[[0.], [0.], [1.]]]).expand(2, -1, -1)
    losses = gref_losses({"response": prediction, "candidate_types": types}, x, target,
                         mask, labels, tolerance=.1, temperature=1., mask=torch.tensor([[1., 0.], [1., 0.]]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :, 1].abs().sum() == 0


def _dataset(directory):
    """生成独立事件族的合成数组，仅供软件契约与端到端测试。"""
    directory.mkdir()
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(6, 3, 2, 4, 1)).astype(np.float32)
    signatures = np.repeat(bank, 2, axis=0)
    labels = np.tile(np.arange(2), 6)
    x = signatures[np.arange(12), labels]
    arrays = {"X_obs": x, "signature_bank": signatures,
              "mask": np.ones((12, 2), np.float32), "candidate_mask": np.ones((12, 3), np.float32),
              "edge_index": np.array([[0, 1], [1, 0]], np.int64),
              "edge_attr": np.ones((2, 5), np.float32), "edge_mask": np.ones((12, 2), np.float32),
              "group_id": np.repeat(np.arange(6), 2), "y_loc": labels,
              "y_resist": np.repeat(np.arange(6) + 1., 2).astype(np.float32)}
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    (directory / "meta.json").write_text(json.dumps({"case": "synthetic", "feature_format": "real_imag_standardized"}), encoding="utf-8")
    return arrays


def test_normalization_excludes_validation_and_test(tmp_path):
    """改变未纳入训练的响应不改变训练侧统计量，统计值与直接计算一致。"""
    arrays = _dataset(tmp_path / "data")
    indices = np.arange(4)
    norm = fit_normalization(arrays, indices)
    expected = arrays["signature_bank"][indices]
    assert np.allclose(norm["mean"], expected.mean(axis=(0, 1, 3)), atol=1e-6)
    assert np.allclose(norm["std"], expected.std(axis=(0, 1, 3)), atol=1e-6)
    arrays["signature_bank"][4:] = 1e9
    after = fit_normalization(arrays, indices)
    assert np.array_equal(norm["mean"], after["mean"])
    assert np.array_equal(norm["std"], after["std"])


def test_end_to_end_smoke_recomputes_saved_scores(tmp_path, monkeypatch):
    """在临时目录完成训练与落盘，复算评分、门控、最佳轮次及进度终态。"""
    _dataset(tmp_path / "data")
    monkeypatch.setattr(training, "METHOD_DIR", tmp_path)
    output = tmp_path / "output/gref/synthetic-smoke"
    args = training.parse_args(["--data-dir", str(tmp_path / "data"), "--output-dir", str(output),
                               "--energy-tolerance", "0.0001", "--tolerance-source", "合成软件测试，不是物理容差",
                               "--temperature", "1", "--epochs", "2", "--batch-size", "4",
                               "--smoke", "--no-terminal-progress"])
    training.run(args)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["smoke"] is True and config["best_epoch"] in (1, 2)
    with np.load(output / "predictions.npz") as pred:
        energies = response_energy(torch.from_numpy(pred["observation"]), torch.from_numpy(pred["response"]),
                                   torch.from_numpy(pred["node_mask"])).numpy()
        assert np.allclose(energies, pred["candidate_score"], atol=1e-6)
        assert metrics["score"]["top1"] == rank_metrics(energies[:, :-1], pred["location"])["top1"]
        assert len(pred["location"]) == metrics["n_fault_events"]
    snapshot = json.loads((tmp_path / "logs/synthetic-smoke/progress.latest.json").read_text(encoding="utf-8"))
    assert snapshot["kind"] == "end"
    assert (output / "gate_audit.json").exists()
    with pytest.raises(FileExistsError):
        training.run(args)


def test_cli_requires_explicit_tolerance_source_and_temperature():
    """正式入口不提供未标定的容差和温度默认值。"""
    with pytest.raises(SystemExit):
        training.parse_args(["--data-dir", "unused", "--output-dir", "unused"])
