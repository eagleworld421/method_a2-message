"""交叉核对报告中的关键数字与产物文件，避免人工誊写错误。"""

import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1] / "output" / "node-marker-mvp"


def load(run: str, name: str) -> dict:
    """读取指定运行的 JSON 产物。"""
    return json.loads((ROOT / run / name).read_text(encoding="utf-8"))


def main() -> int:
    """打印需要核对的数字。"""
    checks = {
        "stage0 测试集 Oracle": (
            ROOT / "stage0-oracle-lg-run3" / "oracle_metrics.json",
            ["normalized", "raw", "normalized_margin_median", "normalized_margin_min",
             "error_budget", "normalized_margin_below_budget_ratio"],
        ),
        "stage0 验证集 Oracle": (
            ROOT / "stage0-oracle-lg-run3-val" / "oracle_metrics.json",
            ["normalized", "raw"],
        ),
    }
    for label, (path, keys) in checks.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"--- {label} ---")
        for key in keys:
            print(f"  {key} = {payload[key]}")

    print("--- stage1 mean 种子 342 ---")
    validation = load("stage1-sweep/stage1-mean__single-seed342", "validation_metrics.json")
    metrics = load("stage1-sweep/stage1-mean__single-seed342", "metrics.json")
    print("  val", validation["plain"]["top1"], validation["plain"]["top3"])
    print("  test", metrics["plain"]["top1"], metrics["plain"]["top3"])
    print("--- stage1 mean 种子 343 ---")
    validation = load("stage1-sweep/stage1-mean__single-seed343", "validation_metrics.json")
    metrics = load("stage1-sweep/stage1-mean__single-seed343", "metrics.json")
    print("  val", validation["plain"]["top1"], validation["plain"]["top3"])
    print("  test", metrics["plain"]["top1"], metrics["plain"]["top3"])
    print("--- stage3 种子 342 ---")
    validation = load("stage3-sweep/stage3-uncertainty__mean-seed342", "validation_metrics.json")
    metrics = load("stage3-sweep/stage3-uncertainty__mean-seed342", "metrics.json")
    print("  val plain", validation["plain"]["top1"], "unc", validation["uncertainty"]["top1"])
    print("  test plain", metrics["plain"]["top1"], "unc", metrics["uncertainty"]["top1"],
          "unc top3", metrics["uncertainty"]["top3"])
    print("  family loss", metrics["losses"]["family"], "clamped", metrics["losses"]["logvar_clamped_ratio"])
    print("  var_err_corr", metrics["variance_error_correlation"])
    print("  val unc error/margin", validation["uncertainty"]["prediction_error_to_margin_ratio_median"])
    print("  test unc error/margin", metrics["uncertainty"]["prediction_error_to_margin_ratio_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
