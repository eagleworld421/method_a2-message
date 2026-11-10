"""反事实评分与预测响应距离的独立评价。"""

from __future__ import annotations

import numpy as np


def distance_gap_metrics(scores: np.ndarray, true_distances: np.ndarray,
                         predicted_distances: np.ndarray, n_fault: int) -> dict:
    """报告评分、真实距离和预测距离的候选间距关系。"""
    score = scores[:, :n_fault]
    true = true_distances[:, :n_fault]
    pred = predicted_distances[:, :n_fault]
    true_order = np.argsort(true, axis=1)
    score_order = np.argsort(score, axis=1)
    pred_order = np.argsort(pred, axis=1)
    return {
        "score_true_distance_spearman": float(np.mean([np.corrcoef(true[i], score[i])[0, 1] for i in range(len(true)) if np.std(true[i]) > 0 and np.std(score[i]) > 0])),
        "pred_true_distance_spearman": float(np.mean([np.corrcoef(true[i], pred[i])[0, 1] for i in range(len(true)) if np.std(true[i]) > 0 and np.std(pred[i]) > 0])),
        "score_true_order_agreement": float(np.mean(np.all(score_order == true_order, axis=1))),
        "pred_true_order_agreement": float(np.mean(np.all(pred_order == true_order, axis=1))),
        "true_gap_median": float(np.median(np.diff(np.take_along_axis(true, true_order, axis=1), axis=1))),
    }
