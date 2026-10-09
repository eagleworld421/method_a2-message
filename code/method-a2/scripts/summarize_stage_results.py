"""汇总全部已完成的阶段运行，输出分组统计与探针读数。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1] / "output" / "node-marker-mvp"


def main() -> int:
    """按阶段与配置分组打印统计量。"""
    for stage in ("stage1-sweep", "stage2-sweep"):
        base = ROOT / stage
        if not base.exists():
            continue
        print(f"=== {stage} ===")
        buckets: dict[str, list[dict]] = {}
        for directory in sorted(base.glob("*")):
            validation_path = directory / "validation_metrics.json"
            metrics_path = directory / "metrics.json"
            if not validation_path.exists() or not metrics_path.exists():
                print(f"  跳过未完成：{directory.name}")
                continue
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            probes_path = directory / "probes.json"
            probes = (
                json.loads(probes_path.read_text(encoding="utf-8"))
                if probes_path.exists() else {}
            )
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            arm = directory.name.rsplit("-seed", 1)[0]
            bucket = buckets.setdefault(arm, [])
            row = {
                "seed": directory.name.rsplit("-seed", 1)[-1],
                "val_top1": validation["plain"]["top1"],
                "val_top3": validation["plain"]["top3"],
                "test_top1": metrics["plain"]["top1"],
                "test_top3": metrics["plain"]["top3"],
                "test_rank": metrics["plain"]["mean_rank"],
                "val_family": validation["losses"]["family"],
                "probe_z": probes.get("z_shared_fault_location", {}).get("test_accuracy"),
                "probe_node": probes.get("node_repr_fault_location", {}).get("test_accuracy"),
                "probe_chance": probes.get("z_shared_fault_location", {}).get("chance_accuracy"),
                "parameters": config.get("trainable_parameters"),
                "best_epoch": config.get("best_epoch"),
                "val_unc_top1": validation.get("uncertainty", {}).get("top1"),
                "test_unc_top1": metrics.get("uncertainty", {}).get("top1"),
                "test_unc_top3": metrics.get("uncertainty", {}).get("top3"),
                "var_err_corr": metrics.get("variance_error_correlation"),
            }
            bucket.append(row)
            print(
                "  {name:44s} val={vt:.4f}/{v3:.4f} test={tt:.4f}/{t3:.4f} rank={tr:.2f} "
                "probe z={pz} node={pn} unc={un} corr={corr}".format(
                    name=directory.name, vt=row["val_top1"], v3=row["val_top3"],
                    tt=row["test_top1"], t3=row["test_top3"], tr=row["test_rank"],
                    pz=_fmt(row["probe_z"]), pn=_fmt(row["probe_node"]),
                    un=_fmt(row["test_unc_top1"]), corr=_fmt(row["var_err_corr"], 3),
                )
            )
        for arm, rows in buckets.items():
            if len(rows) < 2:
                continue
            for key in ("val_top1", "val_top3", "test_top1", "test_top3", "test_rank"):
                values = [row[key] for row in rows if row[key] is not None]
                if len(values) > 1:
                    print(
                        f"  [汇总] {arm:44s} {key:10s} "
                        f"mean={statistics.mean(values):.4f} sd={statistics.stdev(values):.4f} n={len(values)}"
                    )
    return 0


def _fmt(value, digits: int = 4) -> str:
    """格式化可能为空的数值。"""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
