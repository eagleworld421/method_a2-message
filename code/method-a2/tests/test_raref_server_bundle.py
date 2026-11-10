"""验证服务器包迁移完整性、拒绝覆盖和运行入口参数约束。"""

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    """按文件路径载入脚本，避免依赖完整开发工作区。"""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_raref_server_bundle")
runner = load_script("run_raref_server")


class BundleTests(unittest.TestCase):
    """以隔离的来源目录检查构建与文件校验契约。"""

    def setUp(self):
        """建立具有登记元信息的最小来源快照。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.method = self.source / "code/method-a2"
        self.destination = self.method / "output/server.v1"
        for name in builder.SOURCE_FILES:
            self.put(self.method / name, "\"\"\"测试来源。\"\"\"\n")
        data = self.method / "data" / builder.DATASET
        for name in builder.DATA_FILES:
            self.put(data / name, "fixture")
        self.meta = {"case": "ieee37", "scenario": "S0", "n_nodes": 38, "n_candidates": 39,
                     "n_samples": 2736, "res_min": 0.01, "res_max": 20.0,
                     "impedance_sampling": "log_uniform"}
        self.put(data / "meta.json", json.dumps(self.meta))
        for dataset, (res_min, res_max) in builder.DATASETS.items():
            if dataset == builder.DATASET:
                continue
            other_data = self.method / "data" / dataset
            for name in builder.DATA_FILES:
                self.put(other_data / name, "fixture")
            self.put(other_data / "meta.json", json.dumps(dict(self.meta, res_min=res_min, res_max=res_max)))
        self.put(self.source / "docs/project/Method-A2-主要实验数据集登记.md",
                 "\n".join(f"登记：code/method-a2/data/{dataset}/" for dataset in builder.DATASETS))
        for name in ("Method-A2-区域概率损失多种子配对复核报告.md",
                     "Method-A2-区域概率损失未观测定向与置信度校准实验报告.md"):
            self.put(self.source / "docs/project/reports" / name, "来源报告")
        config = dict.fromkeys(builder.CONFIG_KEYS, 1)
        config.update(loss_mode="region", lambda_energy=1., lambda_rank=.141,
                      lambda_region=.194, beta_obs=.05, beta_miss=.4, region_sigma=1.)
        self.put(self.method / "output/gref" / builder.REFERENCE / "config.json", json.dumps(config))

    def put(self, path, value):
        """写出测试夹具文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def test_archive_and_manifest_survive_relocation(self):
        """归档内路径与逐文件摘要在独立目录中保持一致。"""
        result = builder.build(self.source, self.destination)
        self.assertTrue(result["archive"].endswith("server.v1.tar.gz"))
        with tarfile.open(result["archive"]) as archive:
            names = archive.getnames()
            self.assertIn("server.v1/code/method-a2/src/gref/region.py", names)
            self.assertIn("server.v1/code/method-a2/data/" + builder.DATASET + "/signature_bank.npy", names)
            self.assertIn("server.v1/code/method-a2/data/s0-ieee37-r10-100-loguniform/signature_bank.npy", names)
        moved = self.source / "relocated"
        self.destination.rename(moved)
        with patch.object(runner, "BUNDLE_DIR", moved):
            manifest = runner.verify_bundle()
            self.assertEqual(manifest["dataset"], builder.DATASET)
            self.assertEqual(set(manifest["datasets"]), set(builder.DATASETS))
            self.assertFalse(manifest["weights_included"])
            (moved / "code/method-a2/src/gref/region.py").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                runner.verify_bundle()

    def test_rejects_output_outside_method_and_overwrite(self):
        """交付必须遵守 output 边界，且已有文件不得被覆盖。"""
        with self.assertRaisesRegex(ValueError, "output"):
            builder.build(self.source, self.source / "other")
        self.destination.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            builder.build(self.source, self.destination)

    def test_metadata_mismatch_stops_before_writing(self):
        """电阻范围与登记不一致时不得生成包。"""
        self.meta["res_max"] = 100
        self.put(self.method / "data" / builder.DATASET / "meta.json", json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "res_max"):
            builder.build(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_manifest_path_cannot_escape_package(self):
        """不信任清单中的绝对路径和父目录跳转。"""
        self.destination.mkdir(parents=True)
        self.put(self.destination / "manifest.json", json.dumps({"files": {"../outside": {}}}))
        with patch.object(runner, "BUNDLE_DIR", self.destination):
            with self.assertRaisesRegex(ValueError, "越界"):
                runner.verify_bundle()

    def test_high_resistance_metadata_mismatch_stops_before_writing(self):
        """第二套数据的电阻下限错误时同样拒绝打包。"""
        self.put(self.method / "data/s0-ieee37-r10-100-loguniform/meta.json",
                 json.dumps(dict(self.meta, res_min=20., res_max=100.)))
        with self.assertRaisesRegex(ValueError, "res_min"):
            builder.build(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_high_resistance_selection_reaches_training(self):
        """高阻数据选择必须实际传给训练入口并写入运行状态。"""
        high = "s0-ieee37-r10-100-loguniform"
        self.put(self.method / "raref_config.json", "{}")
        self.put(self.source / "manifest.json", "{}")
        run = self.method / "output/gref/high-run"

        def fake_train(run_id, label, command):
            """模拟训练落盘，核对实际命令中的数据路径。"""
            self.assertEqual(label, "train")
            self.assertEqual(Path(command[command.index("--data-dir") + 1]).name, high)
            self.assertIn("--no-cuda-cache", command)
            self.assertNotIn("--no-terminal-progress", command)
            for name in ("model.pt", "normalization.npz", "split.npz", "region_kernel.npz"):
                self.put(run / name, "frozen")
            self.put(run / "config.json", json.dumps({"loss_mode": "region", "batch_size": 64, "seed": 342}))

        with patch.object(runner, "METHOD_DIR", self.method), patch.object(runner, "BUNDLE_DIR", self.source), \
                patch.object(runner, "verify_bundle", return_value={"dataset": builder.DATASET, "datasets": builder.DATASETS}), \
                patch.object(runner, "check_environment", return_value={}), \
                patch.object(runner, "run_child", side_effect=fake_train):
            runner.execute(runner.parse_args(["train", "--run-id", "high-run", "--dataset", high,
                                              "--device", "cpu", "--no-cuda-cache"]))
        state = json.loads((run / "server_run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["data_dir"], "data/" + high)

    def test_evaluation_rejects_switching_dataset(self):
        """冻结评价不能切换到另一套数据。"""
        self.frozen_run()
        high = "s0-ieee37-r10-100-loguniform"
        with patch.object(runner, "METHOD_DIR", self.method), patch.object(runner, "BUNDLE_DIR", self.source), \
                patch.object(runner, "verify_bundle", return_value={"dataset": builder.DATASET, "datasets": builder.DATASETS}), \
                patch.object(runner, "check_environment", return_value={}), patch.object(runner, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "数据集必须"):
                runner.execute(runner.parse_args(["test", "--run-id", "test-run", "--dataset", high, "--device", "cpu"]))
            child.assert_not_called()

    def test_run_identifier_and_frozen_stage_constraints(self):
        """拒绝运行目录逃逸和测试时更换划分种子。"""
        for command in (["train", "--run-id", "../bad"], ["train", "--epochs", "0"],
                        ["test", "--seed", "343"], ["validate", "--epochs", "2"]):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                runner.parse_args(command)
        args = runner.parse_args(["all", "--run-id", "server-342", "--device", "cuda:0"])
        self.assertEqual(args.device, "cuda:0")

    def frozen_run(self):
        """建立已验证的冻结运行，仅用于阶段保护测试。"""
        run = self.method / "output/gref/test-run"
        self.put(self.method / "raref_config.json", "{}")
        self.put(self.source / "manifest.json", "{}")
        for name in ("model.pt", "normalization.npz", "split.npz", "region_kernel.npz"):
            self.put(run / name, "frozen")
        self.put(run / "config.json", json.dumps({"loss_mode": "region", "batch_size": 64, "seed": 342}))
        fingerprint = {name: runner.sha256(run / name) for name in
                       ("model.pt", "normalization.npz", "split.npz", "region_kernel.npz", "config.json")}
        self.put(run / "server_validation.json", json.dumps({"frozen_files": fingerprint}))
        self.put(run / "server_run.json", json.dumps({"smoke": False, "data_dir": "data/example",
                  "manifest_sha256": runner.sha256(self.source / "manifest.json")}))
        return run

    def test_frozen_model_change_prevents_test(self):
        """验证后模型变化时，不得启动任何测试子进程。"""
        run = self.frozen_run()
        self.put(run / "model.pt", "changed")
        with patch.object(runner, "METHOD_DIR", self.method), patch.object(runner, "BUNDLE_DIR", self.source), \
                patch.object(runner, "verify_bundle", return_value={"dataset": "example"}), \
                patch.object(runner, "check_environment", return_value={}), patch.object(runner, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "验证后"):
                runner.execute(runner.parse_args(["test", "--run-id", "test-run", "--device", "cpu"]))
            child.assert_not_called()

    def test_test_stage_cannot_run_twice(self):
        """测试阶段成功后再次调用必须拒绝，防止重复使用测试集。"""
        run = self.frozen_run()
        with patch.object(runner, "METHOD_DIR", self.method), patch.object(runner, "BUNDLE_DIR", self.source), \
                patch.object(runner, "verify_bundle", return_value={"dataset": "example"}), \
                patch.object(runner, "check_environment", return_value={}), patch.object(runner, "run_child") as child:
            args = runner.parse_args(["test", "--run-id", "test-run", "--device", "cpu"])
            runner.execute(args)
            self.assertEqual(child.call_count, 2)
            self.assertTrue((run / "server_complete.json").is_file())
            with self.assertRaises(FileExistsError):
                runner.execute(args)
            self.assertEqual(child.call_count, 2)


if __name__ == "__main__":
    unittest.main()
