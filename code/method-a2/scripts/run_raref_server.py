"""服务器包的环境检查、训练、验证、冻结测试与合成冒烟入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1]
BUNDLE_DIR = METHOD_DIR.parents[1]


def read_json(path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """保存严格 JSON。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    """分块计算大文件摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle():
    """校验包内清单中的代码、配置与数据，拒绝路径逃逸。"""
    manifest = read_json(BUNDLE_DIR / "manifest.json")
    for name, record in manifest["files"].items():
        path = (BUNDLE_DIR / name).resolve()
        if BUNDLE_DIR.resolve() not in path.parents:
            raise ValueError(f"清单路径越界：{name}")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"文件缺失或校验失败：{name}")
    return manifest


def check_environment(device):
    """验证实际计算设备，并保存可复核的软硬件信息。"""
    import numpy as np
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 PyTorch 无法访问 GPU；请检查驱动和安装环境")
    target = torch.device(device)
    probe = torch.ones(4, device=target)
    assert float((probe * probe).sum().cpu()) == 4.0
    info = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None}
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


def run_child(run_id, label, arguments):
    """执行原有阶段入口，将完整控制台输出写入独立日志。"""
    log_dir = METHOD_DIR / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    command = [sys.executable, "-u", *map(str, arguments)]
    print(f"阶段 {label}；日志：{log_path}", flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        subprocess.run(command, cwd=METHOD_DIR, env=environment, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def synthetic_data(directory):
    """生成仅供软件冒烟验证的临时合成数据，不构成实验数据集。"""
    import numpy as np

    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(6, 5, 4, 12, 6)).astype(np.float32)
    signatures = np.repeat(bank, 4, axis=0)
    labels = np.tile(np.arange(4), 6)
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=np.int64)
    arrays = {"X_obs": signatures[np.arange(24), labels], "signature_bank": signatures,
              "mask": np.ones((24, 4), np.float32), "candidate_mask": np.ones((24, 5), np.float32),
              "edge_index": edges, "edge_attr": np.ones((6, 5), np.float32),
              "edge_mask": np.ones((24, 6), np.float32), "group_id": np.repeat(np.arange(6), 4),
              "y_loc": labels, "y_resist": np.repeat(np.arange(6) + 1., 4).astype(np.float32)}
    for name, value in arrays.items():
        np.save(directory / f"{name}.npy", value)
    write_json(directory / "meta.json", {"case": "synthetic", "feature_format": "real_imag_standardized"})


def parse_args(argv=None):
    """解析阶段与有限运行覆盖项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train", "validate", "test", "all", "smoke"))
    parser.add_argument("--run-id", default="raref-ieee37-a-seed342")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=("s0-ieee37-r0p01-20-loguniform",
                                             "s0-ieee37-r10-100-loguniform"), default=None,
                        help="训练时选择包内数据集；验证和测试默认复用训练记录")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.run_id):
        parser.error("run-id 只允许字母、数字、点、下划线与连字符")
    for name in ("epochs", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name} 必须为正整数")
    if args.stage in ("validate", "test") and (args.epochs is not None or args.seed is not None):
        parser.error("验证与测试必须复用训练记录，不可覆盖 epochs 或 seed")
    if args.stage == "smoke" and args.dataset is not None:
        parser.error("合成冒烟不使用正式数据集，请勿指定 dataset")
    return args


def execute(args):
    """串联既有实现；所有测试均在验证完成的冻结检查点上执行。"""
    manifest = verify_bundle()
    environment = check_environment(args.device)
    if args.stage == "check":
        print(f"文件校验通过：{len(manifest['files'])} 个文件", flush=True)
        return
    smoke = args.stage == "smoke"
    dataset = args.dataset or manifest["dataset"]
    if dataset not in manifest.get("datasets", {manifest["dataset"]: {}}):
        raise ValueError(f"运行包中不存在所选数据集：{dataset}")
    config = read_json(METHOD_DIR / "raref_config.json")
    run_id = f"smoke-{uuid.uuid4().hex[:10]}" if smoke else args.run_id
    run_dir = METHOD_DIR / "output" / "gref" / run_id
    data_dir = METHOD_DIR / "data" / (run_id if smoke else dataset)
    stages = ("train", "validate", "test") if args.stage in ("all", "smoke") else (args.stage,)
    if "train" in stages:
        if run_dir.exists() or (METHOD_DIR / "logs" / run_id).exists():
            raise FileExistsError("运行标识已存在；请使用新的 --run-id")
        if smoke:
            synthetic_data(data_dir)
        for name in ("epochs", "batch_size", "seed"):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        if smoke:
            config.update(epochs=1, batch_size=8, save_epochs="", node_mask_min_observed=2,
                          node_mask_max_observed=4, energy_tolerance=1e-12)
        command = [METHOD_DIR / "scripts/train_gref.py", "--data-dir", data_dir,
                   "--output-dir", run_dir, "--device", args.device,
                   "--skip-test-evaluation", "--no-terminal-progress"]
        for name, value in config.items():
            command += ["--" + name.replace("_", "-"), str(value)]
        run_child(run_id, "train", command)
        write_json(run_dir / "server_run.json", {
            "smoke": smoke, "data_dir": data_dir.relative_to(METHOD_DIR).as_posix(),
            "manifest_sha256": sha256(BUNDLE_DIR / "manifest.json"), "environment": environment})
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=True)
        (run_dir / "environment.freeze.txt").write_text(frozen.stdout, encoding="utf-8")
    state = read_json(run_dir / "server_run.json")
    if state["manifest_sha256"] != sha256(BUNDLE_DIR / "manifest.json"):
        raise ValueError("训练与评价的运行包清单不一致")
    data_dir = METHOD_DIR / state["data_dir"]
    if args.dataset is not None and data_dir.name != args.dataset:
        raise ValueError("验证与测试的数据集必须与训练记录一致")
    trained = read_json(run_dir / "config.json")
    if trained["loss_mode"] != "region":
        raise ValueError("只支持 RaReF 区域损失运行")
    common = ["--run-dir", run_dir, "--data-dir", data_dir, "--device", args.device,
              "--batch-size", args.batch_size or trained["batch_size"]]
    region_args = ["--seed", trained["seed"], "--alphas", 0.05, 0.1, 0.2]
    if state["smoke"]:
        region_args += ["--observed-nodes", 2, "--masks-per-level", 2]
    frozen_files = ("model.pt", "normalization.npz", "split.npz", "region_kernel.npz", "config.json")
    fingerprint = {name: sha256(run_dir / name) for name in frozen_files}
    if "validate" in stages:
        if (run_dir / "server_test.started.json").exists():
            raise FileExistsError("测试已启动，禁止在同一运行上重新执行验证流程")
        run_child(run_id, "validate-region", [METHOD_DIR / "scripts/eval_region_term.py", *common,
                  *region_args, "--split", "validation", "--mask-seed-offset", 1000])
        write_json(run_dir / "server_validation.json", {"frozen_files": fingerprint})
    if "test" in stages:
        validation = read_json(run_dir / "server_validation.json")
        if fingerprint != validation["frozen_files"]:
            raise ValueError("检查点、划分或标准化在验证后发生变化")
        # 独占标记阻止重复评价及并发测试；失败时保留标记供人工核验。
        with (run_dir / "server_test.started.json").open("x", encoding="utf-8") as handle:
            json.dump({"frozen_files": fingerprint}, handle)
        run_child(run_id, "test-standard", [METHOD_DIR / "scripts/evaluate_gref.py", *common])
        run_child(run_id, "test-region", [METHOD_DIR / "scripts/eval_region_term.py", *common,
                  *region_args, "--split", "test"])
        write_json(run_dir / "server_complete.json", {"status": "complete", "smoke": state["smoke"],
                   "outputs": ["metrics.json", "region_metrics.json", "predictions.npz"],
                   "frozen_files": fingerprint})
    print(f"阶段完成；产物：{run_dir}", flush=True)


if __name__ == "__main__":
    execute(parse_args())
