"""判据一共享状态可恢复性 Probe 的纯函数测试。

测试只覆盖不依赖真实数据集与 checkpoint 的纯函数：度量定义、按 block 聚合、
训练侧标准化、正则选择口径与判定分支。需要真实数据与权重的端到端路径由
`scripts/run_error_source_criterion1_probe.py` 的运行记录与产物承担。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

METHOD_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = METHOD_DIR / "scripts"
for path in (str(METHOD_DIR), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

MODULE = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "run_error_source_criterion1_probe",
        SCRIPTS_DIR / "run_error_source_criterion1_probe.py",
    )
)
MODULE.__loader__.exec_module(MODULE)


def test_load_multiplier_keys_are_sorted_and_complete():
    """工况键顺序必须是设计文档固定的 15 键字典序。"""
    keys = MODULE.LOAD_MULTIPLIER_KEYS
    assert len(keys) == 15
    assert list(keys) == sorted(keys)
    assert set(keys) == {
        "611", "634a", "634b", "634c", "645", "646", "652",
        "670a", "670b", "670c", "671", "675a", "675b", "675c", "692",
    }


def test_regression_metrics_perfect_and_mean_predictions():
    """完美预测的 R² 为 1，均值预测的 R² 为 0。"""
    truth = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    perfect = MODULE.regression_metrics(truth, truth)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["rmse"] == pytest.approx(0.0)
    assert perfect["mae"] == pytest.approx(0.0)
    assert perfect["n_samples"] == 4

    mean_only = MODULE.regression_metrics(truth, np.repeat(truth.mean(axis=0)[None, :], 4, axis=0))
    assert mean_only["r2"] == pytest.approx(0.0, abs=1e-12)
    assert mean_only["rmse"] > 0.0


def test_regression_metrics_single_dimension_and_denominator():
    """R² 的分母取真实标签方差，负值预测必须给出负 R²。"""
    truth = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    assert MODULE.regression_metrics(truth, truth)["r2"] == pytest.approx(1.0)
    worse = MODULE.regression_metrics(truth, -truth)
    assert worse["r2"] < 0.0


def test_classification_metrics_perfect_and_baselines():
    """完美分类的准确率与宏平均 F1 均为 1，且混淆矩阵按真实类别累加。"""
    labels = np.asarray([0, 1, 2, 1, 0, 2], dtype=np.int64)
    metric = MODULE.classification_metrics(labels, labels, 3)
    assert metric["accuracy"] == pytest.approx(1.0)
    assert metric["macro_f1"] == pytest.approx(1.0)
    assert metric["random_baseline"] == pytest.approx(1.0 / 3.0)
    assert metric["majority_baseline"] == pytest.approx(2.0 / 6.0)
    confusion = np.asarray(metric["confusion_matrix"])
    assert confusion.shape == (3, 3)
    assert confusion.trace() == 6


def test_classification_metrics_majority_class_prediction():
    """全预测多数类时准确率等于多数类基线。"""
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    metric = MODULE.classification_metrics(labels, np.zeros_like(labels), 2)
    assert metric["accuracy"] == pytest.approx(metric["majority_baseline"])


def test_standardize_uses_train_statistics_only():
    """标准化必须只用训练侧统计量，验证与测试侧不得参与。"""
    train = np.asarray([[0.0, 0.0], [2.0, 4.0]])
    val = np.asarray([[100.0, 100.0]])
    test = np.asarray([[-100.0, -100.0]])
    scaled_train, (scaled_val, scaled_test) = MODULE.standardize(train, [val, test])
    assert scaled_train.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-12)
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    assert scaled_val[0] == pytest.approx((val[0] - train_mean) / train_std)
    assert scaled_test[0] == pytest.approx((test[0] - train_mean) / train_std)


def test_regression_target_statistics_come_from_train_split():
    """回归目标标准化统计量只能取自训练划分。"""
    target = np.asarray([[1.0], [2.0], [3.0], [1000.0]])
    splits = {"train": np.asarray([0, 1, 2]), "val": np.asarray([3]), "test": np.asarray([3])}
    features = {
        "train": np.zeros((3, 2)), "val": np.zeros((1, 2)), "test": np.zeros((1, 2)),
    }
    _, _, stats = MODULE._scale_regression(
        features["train"], [features["val"], features["test"]], target, splits
    )
    assert stats["mean"][0] == pytest.approx(float(target[:3].mean()))
    assert stats["std"][0] == pytest.approx(float(target[:3].std()))


def test_block_ids_and_aggregation_follow_dataset_block_structure():
    """block 聚合按 4 事件一块取均值，块标签唯一。"""
    block_id = np.repeat(np.arange(4), 4)
    features = np.arange(32, dtype=np.float64).reshape(16, 2)
    aggregated, blocks = MODULE.aggregate_by_block(features, block_id)
    assert blocks.tolist() == [0, 1, 2, 3]
    assert aggregated.shape == (4, 2)
    assert aggregated[0].tolist() == pytest.approx(features[:4].mean(axis=0).tolist())
    assert aggregated[3].tolist() == pytest.approx(features[12:].mean(axis=0).tolist())


def test_block_ids_for_maps_event_indices():
    """事件索引到 block 编号的映射必须逐项一致。"""
    block_id = np.repeat(np.arange(5), 4)
    indices = np.asarray([0, 3, 4, 19])
    assert MODULE.block_ids_for(indices, block_id).tolist() == [0, 0, 1, 4]


def test_limit_blocks_truncates_whole_blocks_and_keeps_splits_disjoint():
    """冒烟截断以 block 为单位，不得按事件截断而破坏划分或块内共享工况。"""
    block_id = np.repeat(np.arange(12), 4)
    splits = {
        "train": np.arange(0, 20), "val": np.arange(20, 28), "test": np.arange(28, 48),
    }
    limited = MODULE.limit_blocks(splits, block_id, 3)
    for key in ("train", "val", "test"):
        blocks = np.unique(block_id[limited[key]])
        assert blocks.size <= 3
        counts = np.bincount(block_id[limited[key]])
        assert set(counts[counts > 0].tolist()) == {4}, "每个保留块必须仍是完整 4 事件"
    kept = {key: set(block_id[limited[key]].tolist()) for key in limited}
    assert not (kept["train"] & kept["val"])
    assert not (kept["train"] & kept["test"])
    assert not (kept["val"] & kept["test"])


def test_limit_blocks_rejects_non_positive_limit():
    """非法截断上限必须报错而不是静默产出空划分。"""
    with pytest.raises(ValueError):
        MODULE.limit_blocks({"train": np.arange(4), "val": np.arange(4),
                             "test": np.arange(4)}, np.zeros(4, dtype=np.int64), 0)


def test_build_targets_extracts_conditions_in_fixed_order():
    """条件标签按固定键序取出，并保留原始阻抗与故障时刻。"""
    rows = [
        {
            "load_multipliers": {key: float(index) for index, key in
                                 enumerate(MODULE.LOAD_MULTIPLIER_KEYS)},
            "true_impedance": 0.1015, "fault_class": 3, "true_location": 7,
            "fault_delay_steps": 0, "block_id": 1, "event_index": 4, "event_id": "e0004",
        }
    ]
    targets = MODULE.build_targets(rows)
    assert targets["H"].shape == (1, 15)
    assert targets["H"][0].tolist() == pytest.approx(
        [float(index) for index in range(15)]
    )
    # 回归目标统一保持 [N,D] 二维形状，逐维报告才不会切片越界。
    assert targets["log1p_impedance"].shape == (1, 1)
    assert targets["log1p_impedance"][0, 0] == pytest.approx(float(np.log1p(0.1015)))
    assert targets["impedance_raw"][0] == pytest.approx(0.1015)
    assert targets["fault_class"][0] == 3
    assert targets["true_location"][0] == 7
    assert targets["event_id"] == ["e0004"]


def test_candidate_embed_variation_detects_constancy():
    """常量候选嵌入必须被识别为跨事件不变。"""
    constant = np.ones((6, 4, 3), dtype=np.float32)
    verdict = MODULE.candidate_embed_variation(constant)
    assert verdict["is_constant_across_events"] is True
    assert verdict["per_event_element_max_spread"] == pytest.approx(0.0)

    varying = constant.copy()
    varying[3, 0, 0] = 2.0
    assert MODULE.candidate_embed_variation(varying)["is_constant_across_events"] is False


def _controls(random_r2: float, permutation_r2: float) -> dict:
    """构造判定函数所需的最小负对照结构。"""
    return {
        "random_representation": {"block_r2": random_r2},
        "label_permutation": {"block_r2": permutation_r2},
    }


def test_decide_both_criteria_hold_points_to_decoder():
    """S1 与 S2 均成立时结论为瓶颈在表示之后。"""
    decision = MODULE.decide(0.8, 0.7, 0.72, _controls(0.0, 0.0),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert decision["s1"]["label"] == "成立"
    assert decision["s2"]["label"] == "成立"
    assert decision["conclusion"] == "瓶颈在表示之后"
    assert decision["branch_d"]["active"] is False


def test_decide_observation_informative_but_representation_not_is_extraction_failure():
    """S1 成立而 S2 不成立时结论为提取失败。"""
    decision = MODULE.decide(0.8, 0.05, 0.06, _controls(0.0, 0.0),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert decision["conclusion"] == "提取失败"
    assert "U" in decision["next_action"]


def test_decide_uninformative_observation_points_to_input_definition():
    """S1 不成立时结论为输入信息缺失。"""
    decision = MODULE.decide(0.05, 0.7, 0.72, _controls(0.0, 0.0),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert decision["s1"]["label"] == "不成立"
    assert decision["conclusion"] == "输入信息缺失"


def test_decide_s2_requires_beating_both_negative_controls():
    """S2 高于阈值但未超过负对照时必须记为不成立。"""
    thresholds = {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3}
    beaten_by_random = MODULE.decide(0.9, 0.6, 0.6, _controls(0.8, 0.0), thresholds)
    assert beaten_by_random["s2"]["label"] == "不成立"
    assert beaten_by_random["s2"]["beats_random_control"] is False

    beaten_by_permutation = MODULE.decide(0.9, 0.6, 0.6, _controls(0.0, 0.9), thresholds)
    assert beaten_by_permutation["s2"]["label"] == "不成立"
    assert beaten_by_permutation["s2"]["beats_label_permutation_control"] is False


def test_decide_branch_d_triggers_on_event_block_gap():
    """事件级明显高于块级时触发分支 D，S2 记为未定。"""
    decision = MODULE.decide(0.9, 0.6, 0.95, _controls(0.0, 0.0),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert decision["branch_d"]["active"] is True
    assert decision["s2"]["label"] == "未定"
    assert decision["generalization_gap_event_minus_block"] == pytest.approx(0.35)
    assert decision["conclusion"] == "证据不足"


def test_decide_s2_undetermined_without_branch_d():
    """S2 落在阈值带内且未触发分支 D 时，结论必须区别于分支 D 情形。"""
    decision = MODULE.decide(0.8, 0.12, -1.5, _controls(-0.3, -0.1),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert decision["s1"]["label"] == "成立"
    assert decision["s2"]["label"] == "未定"
    assert decision["branch_d"]["active"] is False
    assert decision["conclusion"].startswith("S2 未定")
    assert "设计 U" in decision["next_action"]


def test_summarize_dimensions_splits_by_thresholds():
    """逐维判定必须按阈值分成可恢复／未定／不可恢复三组。"""
    rows = [
        {"target": "H|rep_all", "dimension": "a", "block_r2": 0.93},
        {"target": "H|rep_all", "dimension": "b", "block_r2": 0.50},
        {"target": "H|rep_all", "dimension": "c", "block_r2": 0.30},
        {"target": "H|rep_all", "dimension": "d", "block_r2": 0.10},
        {"target": "H|rep_all", "dimension": "e", "block_r2": -0.40},
        {"target": "H|obs_full", "dimension": "a", "block_r2": 0.99},
    ]
    summary = MODULE.summarize_dimensions(
        rows, "H|rep_all", {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1}
    )
    assert summary["n_dimensions"] == 5
    assert summary["recoverable"] == ["a", "b"]
    assert summary["undetermined"] == ["c"]
    assert summary["not_recoverable"] == ["d", "e"]


def test_block_consistency_flags_constant_representation(tmp_path):
    """跨事件恒定的表示不得被判为工况意义上稳定。"""
    block_id = np.repeat(np.arange(4), 4)
    constant = np.ones((16, 3), dtype=np.float64)
    result = MODULE.block_consistency({"rep": constant}, block_id, {"train": np.arange(16)})
    assert result["rep"]["ratio_median"] == pytest.approx(0.0, abs=1e-12)
    assert "不携带任何工况信息" in result["rep"]["interpretation"]


def test_decide_records_cannot_conclude_boundaries():
    """判定必须附带本判据不能证明的三条边界。"""
    decision = MODULE.decide(0.8, 0.7, 0.7, _controls(0.0, 0.0),
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    assert len(decision["cannot_conclude"]) == 3
    joined = " ".join(decision["cannot_conclude"])
    assert "不能证明因果关系" in joined
    assert "判据二" in joined


def test_sha256_file_matches_hashlib(tmp_path):
    """文件摘要函数与标准库结果一致。"""
    import hashlib

    target = tmp_path / "payload.bin"
    target.write_bytes(b"method-a2-criterion1")
    expected = hashlib.sha256(b"method-a2-criterion1").hexdigest()
    assert MODULE.sha256_file(target) == expected


def test_block_consistency_reports_ratio_and_invariance_to_labels():
    """块内一致性检验不依赖标签，比值由离散度决定。"""
    rng = np.random.default_rng(0)
    block_id = np.repeat(np.arange(4), 4)
    stable = np.repeat(rng.normal(size=(4, 2)), 4, axis=0)
    splits = {"train": np.arange(16)}
    stable_result = MODULE.block_consistency({"rep": stable}, block_id, splits)
    assert stable_result["rep"]["ratio_median"] == pytest.approx(0.0, abs=1e-9)

    noisy = rng.normal(size=(16, 2))
    noisy_result = MODULE.block_consistency({"rep": noisy}, block_id, splits)
    assert noisy_result["rep"]["ratio_median"] > stable_result["rep"]["ratio_median"]


def _synthetic_artifacts():
    """构造 write_report 所需的最小产物集合。"""
    metric = {
        "target": "H", "feature_source": "rep_all", "task": "regression",
        "selected_alpha": 1.0,
        "event": {"r2": 0.11, "rmse": 0.2, "mae": 0.15, "n_samples": 240, "n_dims": 15},
        "block": {"r2": 0.62, "rmse": 0.1, "mae": 0.08, "n_samples": 60, "n_dims": 15},
    }
    config = {
        "run_id": "c1run-test", "data_dir": "/data", "checkpoint": "/ckpt/teacher.pt",
        "checkpoint_sha256": "abc", "source_repo": "method_a", "device": "cpu",
        "batch_size": 16, "seed": 342, "forward_seconds": 2.0, "total_seconds": 600.0,
        "checkpoint_meta": {"epoch": 100, "best_epoch": 95, "best_val_loss": 0.006,
                            "variant": "teacher"},
        "git": {"method_a2": {"head": "deadbeef"}},
        "split_rule": {
            "source": "数据集自带", "counts": {"train": 1120, "val": 240, "test": 240},
            "block_counts": {"train": 280, "val": 60, "test": 60},
            "block_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "leakage_protection": "按 block 切分",
        },
        "standardization": "训练侧统计量",
        "regularization_grid": {"classification_C": [0.01, 1.0], "regression_alpha": [1e-3, 1.0]},
        "regression_solver": {"solver": "lsqr", "tol": 1e-8, "reason": "scipy 不兼容"},
        "classification_solver": {"solver": "newton-cg", "max_iter": 2000,
                                  "multi_class": "multinomial", "reason": "lbfgs 慢 119 倍"},
        "determinism_note": "无 dropout",
        "distance_convention": "d_S 为平方量",
        "representation_inventory": {
            "node_repr": {"raw_shape": [1600, 16, 64], "poolings": ["rep_node_mean"]},
            "rep_all": {"shape": [1600, 1232], "composition": ["rep_node_mean"]},
            "observation_poolings": {"obs_full": [1600, 3456]},
        },
        "candidate_embed_variation": {
            "per_event_element_max_spread": 0.0, "per_event_element_mean_spread": 0.0,
            "is_constant_across_events": True,
        },
    }
    controls = {
        "random_representation": {"description": "噪声", "event_r2": -0.01, "block_r2": 0.0,
                                  "selected_alpha": 1.0},
        "zero_representation": {"description": "全零", "event_r2": 0.0, "block_r2": 0.0,
                                "selected_alpha": 1.0},
        "label_permutation": {"description": "训练侧打乱标签", "event_r2": -0.02,
                              "block_r2": 0.01, "selected_alpha": 1.0},
        "position_positive_control": {"description": "位置", "event_accuracy": 0.9,
                                      "event_macro_f1": 0.89, "block_accuracy": 0.95,
                                      "random_baseline": 0.0625},
    }
    consistency = {"rep_all": {"within_block_dispersion_median": 1.0,
                               "between_block_dispersion_median": 2.0,
                               "ratio_median": 0.5, "interpretation": "稳定"}}
    position = {"status": "ok", "formula": "d_S", "denominator": 1152.0,
                "n_candidates": 17, "argmin_hit_rate": 1.0, "unique_minimum_rate": 1.0,
                "margin_min": 5e-4, "margin_p05": 8e-4, "margin_median": 1.1e-3,
                "rho_min": 0.02, "rho_median": 0.03}
    decision = MODULE.decide(0.62, 0.62, 0.11, controls,
                             {"s1_s2_pass": 0.5, "s1_s2_fail": 0.1, "branch_d_gap": 0.3})
    per_dimension = [{"target": "H|rep_all", "dimension": "611", "event_r2": 0.1,
                      "event_rmse": 0.2, "event_mae": 0.15, "block_r2": 0.6,
                      "block_rmse": 0.1, "block_mae": 0.08}]
    return config, {"H|rep_all": metric}, controls, consistency, position, decision, per_dimension


def test_write_report_renders_complete_document(tmp_path):
    """报告生成必须覆盖全部规定章节，且不使用表格。"""
    artifacts = _synthetic_artifacts()
    target = tmp_path / "report.md"
    MODULE.write_report(target, *artifacts)
    text = target.read_text(encoding="utf-8")
    for heading in ("# Method-A2 判据一执行报告", "## 1. 运行标识与来源", "## 2. 判定结论",
                    "## 3. 运行配置与口径声明", "## 4. 表示清单与池化方式",
                    "## 5. 主结果", "## 6. 逐维工况结果", "## 7. 负对照",
                    "## 8. 块内一致性", "## 9. 位置可恢复性独立核验",
                    "## 10. 本判据不能证明的边界", "## 11. 产物清单"):
        assert heading in text, heading
    assert "<!-- 摘要：" in text
    assert "|" not in text.split("## 1.")[1].split("## 2.")[0], "运行标识段不得使用表格"


def test_write_report_handles_skipped_position_check(tmp_path):
    """位置核验被跳过时报告必须如实说明而不是崩溃。"""
    artifacts = list(_synthetic_artifacts())
    artifacts[4] = {"status": "skipped", "reason": "feature_scaler 缺少 node_scale"}
    target = tmp_path / "report.md"
    MODULE.write_report(target, *artifacts)
    assert "未执行：feature_scaler 缺少 node_scale" in target.read_text(encoding="utf-8")


def test_write_conditions_and_per_dimension_csv_round_trip(tmp_path):
    """两个 CSV 产物必须可写出且字段与行数正确。"""
    rows = [{
        "load_multipliers": {key: 1.0 for key in MODULE.LOAD_MULTIPLIER_KEYS},
        "true_impedance": 4.27, "fault_class": 3, "true_location": 2,
        "fault_delay_steps": 0, "block_id": 0, "event_index": 0, "event_id": "e0000",
    }]
    targets = MODULE.build_targets(rows)

    conditions = tmp_path / "conditions_raw.csv"
    MODULE.write_conditions_csv(conditions, targets)
    header = conditions.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:7] == ["event_id", "event_index", "true_location", "fault_class",
                          "true_impedance", "fault_delay_steps", "block_id"]
    assert header[7:] == [f"H_{key}" for key in MODULE.LOAD_MULTIPLIER_KEYS]
    assert len(conditions.read_text(encoding="utf-8").strip().splitlines()) == 2

    per_dimension = tmp_path / "probe_per_dimension.csv"
    MODULE.write_per_dimension_csv(per_dimension, [{
        "target": "H|rep_all", "dimension": "611", "event_r2": 0.1, "event_rmse": 0.2,
        "event_mae": 0.15, "block_r2": 0.6, "block_rmse": 0.1, "block_mae": 0.08,
    }])
    written = per_dimension.read_text(encoding="utf-8").strip().splitlines()
    assert written[0].split(",") == ["target", "dimension", "event_r2", "event_rmse",
                                     "event_mae", "block_r2", "block_rmse", "block_mae"]
    assert written[1].startswith("H|rep_all,611,")


def test_write_json_is_utf8_and_round_trips(tmp_path):
    """产物 JSON 必须为 UTF-8 且中文不被转义为 ASCII。"""
    target = tmp_path / "decision.json"
    MODULE.write_json(target, {"结论": "提取失败", "值": 0.12})
    text = target.read_text(encoding="utf-8")
    assert "提取失败" in text
    assert np.isclose(json.loads(text)["值"], 0.12)
