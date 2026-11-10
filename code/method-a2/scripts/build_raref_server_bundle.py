"""从当前工作区快照构建可独立迁移的 RaReF 服务器运行包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = METHOD_DIR.parents[1]
DATASET = "s0-ieee37-r0p01-20-loguniform"
DATASETS = {DATASET: (0.01, 20.0), "s0-ieee37-r10-100-loguniform": (10.0, 100.0)}
REFERENCE = "region-exp2-b40-s10-seed342"
DOC_NAME = "Method-A2-RaReF服务器运行包说明.md"
DATA_FILES = ("X_obs.npy", "signature_bank.npy", "mask.npy", "candidate_mask.npy",
              "edge_index.npy", "edge_attr.npy", "edge_mask.npy", "group_id.npy",
              "y_loc.npy", "y_resist.npy", "meta.json", "feature_scaler.npz")
SOURCE_FILES = (
    "src/__init__.py", "src/progress.py", "src/gref/__init__.py", "src/gref/model.py",
    "src/gref/losses.py", "src/gref/data.py", "src/gref/region.py", "src/gref/training.py",
    "src/model/__init__.py", "src/model/marker_response_predictor.py", "src/model/gnn.py",
    "src/model/temporal.py", "src/model/losses.py", "src/training/__init__.py",
    "src/training/marker_training.py", "src/training/evaluation.py",
    "src/data_generation/__init__.py", "src/data_generation/topology.py",
    "scripts/train_gref.py", "scripts/evaluate_gref.py", "scripts/eval_region_term.py",
)
CONFIG_KEYS = (
    "energy_tolerance", "tolerance_source", "temperature", "loss_mode", "lambda_energy",
    "lambda_rank", "lambda_region", "region_sigma", "beta_obs", "beta_miss",
    "node_mask_full_probability", "node_mask_min_observed", "node_mask_max_observed",
    "node_mask_force_observed_probability", "rank_temperature", "save_epochs", "edge_drop_prob",
    "epochs", "batch_size", "learning_rate", "val_ratio", "test_ratio", "seed",
)


def sha256(path):
    """流式计算文件摘要，避免载入完整响应库。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path, value):
    """统一使用 UTF-8 与 Linux 换行写出生成文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def git_value(source, *arguments):
    """记录来源仓库状态；无 Git 时保留明确说明。"""
    result = subprocess.run(["git", "-C", str(source), *arguments], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build(source, destination):
    """复制最小依赖闭包、登记数据及协议，生成完整性清单和传输归档。"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    output_root = source / "code/method-a2/output"
    if output_root not in destination.parents:
        raise ValueError("运行包必须交付到来源方法的 output/ 下")
    archive = destination.with_name(destination.name + ".tar.gz")
    if destination.exists() or archive.exists():
        raise FileExistsError("目标目录或归档已存在，禁止覆盖")
    method = source / "code/method-a2"
    registry = source / "docs/project/Method-A2-主要实验数据集登记.md"
    registry_text = registry.read_text(encoding="utf-8")
    contracts = {}
    for dataset, (res_min, res_max) in DATASETS.items():
        if f"code/method-a2/data/{dataset}/" not in registry_text:
            raise ValueError(f"数据集缺少登记：{dataset}")
        meta = json.loads((method / "data" / dataset / "meta.json").read_text(encoding="utf-8"))
        expected = {"case": "ieee37", "scenario": "S0", "n_nodes": 38, "n_candidates": 39,
                    "n_samples": 2736, "res_min": res_min, "res_max": res_max,
                    "impedance_sampling": "log_uniform"}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"登记数据元信息不一致：{dataset} {key}={meta.get(key)!r}")
        contracts[dataset] = expected
    reference_path = method / "output/gref" / REFERENCE / "config.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    config = {key: reference[key] for key in CONFIG_KEYS}
    if (config["loss_mode"], config["lambda_energy"], config["lambda_rank"], config["lambda_region"],
        config["beta_obs"], config["beta_miss"], config["region_sigma"]) != ("region", 1., .141, .194, .05, .4, 1.):
        raise ValueError("参考运行与最新报告选定的 RaReF 配置不一致")
    sources = [(method / name, Path("code/method-a2") / name) for name in SOURCE_FILES]
    sources += [(method / "data" / dataset / name, Path("code/method-a2/data") / dataset / name)
                for dataset in DATASETS for name in DATA_FILES]
    sources += [(Path(__file__).with_name("run_raref_server.py"), Path("code/method-a2/scripts/run_raref_server.py")),
                (REPO_DIR / "docs/project/plans" / DOC_NAME, Path("docs/README.md")),
                (registry, Path("docs/dataset_registry.md")),
                (reference_path, Path("docs/reference_config.json"))]
    for report in ("Method-A2-区域概率损失多种子配对复核报告.md",
                   "Method-A2-区域概率损失未观测定向与置信度校准实验报告.md"):
        sources.append((source / "docs/project/reports" / report, Path("docs/project/reports") / report))
    missing = [str(path) for path, _ in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"打包依赖缺失：{missing}")
    destination.mkdir(parents=True)
    for original, relative in sources:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    for name in ("output/gref", "logs", "checkpoint"):
        (destination / "code/method-a2" / name).mkdir(parents=True, exist_ok=True)
    write_text(destination / "code/method-a2/raref_config.json",
               json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    write_text(destination / "requirements.txt", "numpy==1.26.4\n")
    write_text(destination / "environment.yml", """name: raref
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - numpy==1.26.4
      - --extra-index-url https://download.pytorch.org/whl/cu121
      - torch==2.2.1+cu121
""")
    write_text(destination / "setup_env.sh", """#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python3.10}"
flavor="${TORCH_FLAVOR:-cu121}"
case "$flavor" in cu121|cpu) ;; *) echo 'TORCH_FLAVOR must be cu121 or cpu' >&2; exit 2 ;; esac
"$python_bin" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 11), "Use Python 3.9-3.11; recommended 3.10"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "torch==2.2.1+$flavor" --index-url "https://download.pytorch.org/whl/$flavor"
.venv/bin/python -m pip check
echo 'Activate: source .venv/bin/activate'
""")
    write_text(destination / "run.sh", """#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python -u code/method-a2/scripts/run_raref_server.py "$@"
""")
    files = {path.relative_to(destination).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest = {"format_version": 2, "method": "RaReF", "dataset": DATASET, "datasets": contracts,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": git_value(source, "rev-parse", "HEAD"),
                "source_branch": git_value(source, "branch", "--show-current"),
                "source_status": git_value(source, "status", "--short"),
                "reference_run": REFERENCE, "dataset_contract": contracts[DATASET],
                "weights_included": False, "files": files}
    write_text(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(archive, "w:gz", compresslevel=3) as handle:
        handle.add(destination, arcname=destination.name)
    write_text(archive.with_name(archive.name + ".sha256"), f"{sha256(archive)}  {archive.name}\n")
    return {"directory": str(destination), "archive": str(archive), "files": len(files),
            "archive_bytes": archive.stat().st_size}


def main():
    """解析来源工作区与新建交付目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_DIR)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.destination), ensure_ascii=False))


if __name__ == "__main__":
    main()
