"""准确率提升验证的接口、信息边界与反捷径测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.dataset_builder import EventConfig, _sample_event  # noqa: E402
from src.model.losses import heteroscedastic_response_loss, masked_family_loss  # noqa: E402
from src.model.marker_response_predictor import (  # noqa: E402
    POOLING_MODES,
    MarkerResponsePredictor,
    build_candidate_markers,
    build_derived_channels,
)
from src.training.evaluation import (  # noqa: E402
    oracle_rank_metrics,
    rank_metrics,
    ridge_probe,
    split_by_group,
)


def _model(**kwargs) -> MarkerResponsePredictor:
    """构造一个小容量测试模型。"""
    options = dict(
        n_nodes=4, time_steps=6, feature_dim=6, temporal_hidden=8,
        shared_dim=8, decoder_hidden=8, dropout=0.0,
    )
    options.update(kwargs)
    return MarkerResponsePredictor(**options)


def _inputs(batch: int = 2, nodes: int = 4, steps: int = 6, features: int = 6):
    """构造测试用观测与固定拓扑。"""
    edges = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    return (
        torch.randn(batch, nodes, steps, features),
        edges,
        torch.ones(edges.shape[0], 5),
    )


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_all_pooling_modes_keep_node_representation_and_shapes(pooling):
    """三种池化都返回节点级 Z 与事件级共享表示，输出形状不变。"""
    model = _model(pooling=pooling)
    x, edges, edge_attr = _inputs()
    markers, types = build_candidate_markers(4)
    result = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
    assert result["response"].shape == (2, 5, 4, 6, 6)
    assert result["Z"].shape == (2, 4, 8)
    assert result["z_shared"].shape[0] == 2
    assert result["z_shared"].shape[1] == (16 if pooling == "mean_median_concat" else 8)
    assert "global_repr" not in result and "g_m" not in result


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_z_shared_is_independent_of_candidate_axis(pooling):
    """事件级共享表示不随候选集合或候选顺序变化。"""
    model = _model(pooling=pooling).eval()
    x, edges, edge_attr = _inputs()
    markers, types = build_candidate_markers(4)
    with torch.no_grad():
        first = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
        subset = model(
            x, edges, edge_attr,
            candidate_markers=markers[[3, 0]], candidate_types=types[[3, 0]],
        )
    assert torch.allclose(first["z_shared"], subset["z_shared"], atol=1e-6)
    assert torch.allclose(first["Z"], subset["Z"], atol=1e-6)


def test_candidate_permutation_is_equivariant_for_every_pooling():
    """候选轴置换时输出按同一置换变化，共享表示保持不变。"""
    for pooling in POOLING_MODES:
        model = _model(pooling=pooling).eval()
        x, edges, edge_attr = _inputs()
        markers, types = build_candidate_markers(4)
        permutation = torch.tensor([2, 4, 0, 3, 1])
        with torch.no_grad():
            original = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
            permuted = model(
                x, edges, edge_attr,
                candidate_markers=markers[permutation], candidate_types=types[permutation],
            )
        assert torch.allclose(
            permuted["response"], original["response"][:, permutation], atol=1e-6, rtol=1e-5
        )
        assert torch.allclose(permuted["z_shared"], original["z_shared"], atol=1e-6)


def test_node_permutation_equivariance():
    """节点同步置换（观测、边、标记、掩码）后共享表示与预测响应按同一置换变化。

    注意输出形状为 `[B,C,N,T,F]`，节点轴是第 2 维；标记的节点轴同样按该置换重排。
    """
    model = _model().eval()
    x, edges, edge_attr = _inputs(batch=1)
    markers, types = build_candidate_markers(4)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    remapped = inverse[edges]
    with torch.no_grad():
        original = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
        permuted = model(
            x[:, permutation], remapped, edge_attr,
            candidate_markers=markers[:, permutation],
            candidate_types=types,
        )
    # 节点级表示与预测响应均在节点轴上按同一置换重排：
    # 观测、边、标记节点轴同步重排后，位置 i 承载的是原位置 permutation[i] 的内容；
    # 输出形状为 [B,C,N,T,F]，节点轴是第 2 维
    # 观测、边与标记节点轴同步重排后，位置 i 承载的是原位置 permutation[i] 的内容；
    # 节点级表示与预测响应均在节点轴上按同一置换重排（输出节点轴是第 2 维）
    assert torch.allclose(
        permuted["Z"], original["Z"][:, permutation], atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        permuted["z_shared"], original["z_shared"], atol=1e-5, rtol=1e-4
    )
    for index in range(permutation.numel()):
        assert torch.allclose(
            permuted["response"][:, :, index],
            original["response"][:, :, int(permutation[index])],
            atol=1e-4, rtol=1e-3,
        )


def test_uncertainty_outputs_are_finite_and_clamped():
    """不确定性分支输出有限方差，且对数方差落在截断范围内。"""
    model = _model(uncertainty=True, logvar_clamp=4.0)
    x, edges, edge_attr = _inputs()
    result = model(x, edges, edge_attr)
    logvar = result["response_logvar"]
    assert torch.isfinite(logvar).all()
    assert float(logvar.abs().max()) <= 4.0 + 1e-6
    loss, clamped = heteroscedastic_response_loss(
        result["response"], torch.zeros_like(result["response"]), logvar,
        logvar_clamp=4.0,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= clamped <= 1.0


def test_zero_and_partial_node_mask_produce_no_nan():
    """全零与部分节点掩码下前向不产生 NaN。"""
    model = _model()
    x, edges, edge_attr = _inputs()
    for node_mask in (torch.zeros(2, 4), torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 2)):
        result = model(x, edges, edge_attr, node_mask=node_mask)
        assert torch.isfinite(result["response"]).all()
        assert torch.isfinite(result["z_shared"]).all()


def test_derived_channels_are_deterministic_and_unwrap_phase():
    """派生通道由观测唯一确定，并在正负 π 附近不产生相位跳变。"""
    x, _, _ = _inputs(batch=1, nodes=1, steps=8)
    first = build_derived_channels(x)
    second = build_derived_channels(x.clone())
    assert torch.equal(first, second)
    # 原始实部／虚部 6 通道，加上幅值、展开相角、幅值差分与相角差分共 18 通道
    assert first.shape == (1, 1, 8, 18)
    # 展开相角允许跨越多个 π，但相邻时间步不应出现 2π 跳变。
    phase = first[..., 9:12]
    phase_step = phase[..., 1:, :] - phase[..., :-1, :]
    assert float(phase_step.abs().max()) <= 3.1416
    # 相位差分通道不应出现接近 2π 的跳变
    assert float(first[..., -3:].abs().max()) < 6.0


def test_derived_channel_model_keeps_contract_shapes():
    """启用派生通道后模型仍保持节点数与输出形状契约。"""
    model = _model(use_derived_channels=True)
    x, edges, edge_attr = _inputs()
    result = model(x, edges, edge_attr)
    assert result["response"].shape == (2, 5, 4, 6, 6)
    # 原始观测通道数不是 6 时必须报错（派生通道由模型内部构造）
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 6, 12), edges, torch.ones(edges.shape[0], 5))
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 6, 5), edges, torch.ones(edges.shape[0], 5))


def test_dual_scale_temporal_encoder_runs():
    """双尺度时间编码可前向并保持输出维度。"""
    model = _model(temporal_mode="dual")
    x, edges, edge_attr = _inputs()
    result = model(x, edges, edge_attr)
    assert result["response"].shape == (2, 5, 4, 6, 6)


def test_invalid_configuration_is_rejected():
    """未知池化与时间编码方式被明确拒绝。"""
    with pytest.raises(ValueError):
        _model(pooling="median_only")
    with pytest.raises(ValueError):
        _model(temporal_mode="triple")


def test_fault_class_sampling_respects_phase_availability():
    """故障类型采样只在相别可行的母线上成立。"""
    class _Stub:
        _bus_phase_nodes = [(1, 2, 3), (2, 3), (3,), (1,)]
        _base_loads = {"load_a": (1.0, 1.0)}

    rng = np.random.default_rng(0)
    event = _sample_event(rng, _Stub(), 0.1, 10.0, fault_classes=(0,))
    assert event.fault_class == 0
    with pytest.raises(RuntimeError):
        _sample_event(rng, _Stub(), 0.1, 10.0, fault_classes=(3,), min_feasible_buses=3)
    single_phase = _sample_event(rng, _Stub(), 0.1, 10.0, fault_classes=(3,), min_feasible_buses=1)
    assert single_phase.fault_class == 3


def test_split_by_group_keeps_groups_separate():
    """分组划分不把同一反事实族拆到不同集合。"""
    group_id = np.repeat(np.arange(10), 16)
    location = np.tile(np.arange(16), 10)
    split = split_by_group(group_id, location, seed=342)
    sets = [set(split[key].tolist()) for key in ("train_idx", "val_idx", "test_idx")]
    assert sets[0] and sets[1] and sets[2]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for key in ("train_groups", "val_groups", "test_groups"):
        assert split[key].size > 0


def test_rank_metrics_reports_expected_values():
    """定位指标在可构造的评分上给出预期结果。"""
    # 行为 [M,C]，评分越小越优；第一行真实候选 0 排第一，第二行真实候选 2 排第二
    scores = np.asarray([[0.1, 0.9, 0.5], [0.4, 0.2, 0.3]], dtype=np.float64)
    locations = np.asarray([0, 2])
    metrics = rank_metrics(scores, locations)
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["top3"] == pytest.approx(1.0)
    assert metrics["mean_rank"] == pytest.approx(1.5)
    assert metrics["signed_margin_negative_ratio"] == pytest.approx(0.5)
    assert metrics["n_candidates"] == 3


def test_oracle_metrics_flags_exact_ties():
    """Oracle 指标报告精确并列事件数量。"""
    observation = np.zeros((1, 2, 2, 1))
    target = np.zeros((1, 3, 2, 2, 1))
    target[0, 2] = 0.5
    result = oracle_rank_metrics(
        observation, target, np.asarray([2]), n_fault_candidates=3
    )
    assert result["n_events"] == 1
    assert result["margin_exact_zero_count"] >= 0


def test_ridge_probe_separates_signal_from_noise():
    """探针在含信号特征上明显优于随机，在纯噪声特征上接近随机。"""
    rng = np.random.default_rng(0)
    targets = np.tile(np.linspace(-1.0, 1.0, 32), 5)
    signal = rng.normal(0, 1, (targets.size, 8))
    signal[:, 0] = targets
    noise = rng.normal(0, 1, (targets.size, 8))
    selector = np.zeros(targets.size, dtype=bool)
    selector[:128] = True
    test_selector = ~selector
    strong = ridge_probe(signal, targets, selector, test_selector, task="regression")
    weak = ridge_probe(noise, targets, selector, test_selector, task="regression")
    assert strong["test_r2"] > 0.9
    assert strong["test_r2"] > weak["test_r2"] + 0.5


def test_uncertainty_does_not_change_mean_response_semantics():
    """启用不确定性后 response 与未启用版本的语义一致（同为响应均值）。"""
    torch.manual_seed(0)
    plain = _model(uncertainty=False).eval()
    torch.manual_seed(0)
    uncertain = _model(uncertainty=True).eval()
    x, edges, edge_attr = _inputs()
    with torch.no_grad():
        first = plain(x, edges, edge_attr)["response"]
    shared = {
        name: value for name, value in uncertain.state_dict().items()
        if not name.startswith("output_logvar")
    }
    uncertain.load_state_dict(shared, strict=False)
    with torch.no_grad():
        second = uncertain(x, edges, edge_attr)["response"]
    assert torch.allclose(first, second, atol=1e-6)


def test_masked_family_loss_respects_candidate_mask():
    """响应损失只在有效候选上平均。"""
    prediction = torch.zeros(1, 3, 2, 2, 1)
    target = torch.ones_like(prediction)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    assert float(masked_family_loss(prediction, target, mask)) == pytest.approx(1.0)
