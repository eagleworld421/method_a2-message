<!-- 摘要：提供 RaReF 双 IEEE37 数据服务器包的 README，列出 CUDA 12.8 环境安装、完整运行命令参数与可选项，说明 CUDA 训练加速、进度条、后台日志、产物与验证边界。 -->

# RaReF 服务器运行包说明

## 1. 版本与数据范围

本包以 2026-11-10 最新报告选定的 `region-exp2-b40-s10-seed342` 配置为默认值，使用 RaReF 区域损失模式。能量、排序、区域权重分别为 1、0.141、0.194，β_obs=0.05、β_miss=0.4、σ=1.0，排序温度固定为 0.0040640985。默认训练 200 轮、批次 64、学习率 0.001、随机种子 342。配置存于 `code/method-a2/raref_config.json`，原始参考配置保存在 `docs/reference_config.json`。

双数据集包 `raref-server-20261111.tar.gz` 包含以下两套已登记的 IEEE37 完整数据，分别训练与评价，不自动混合：

- A：`s0-ieee37-r0p01-20-loguniform`，电阻 0.01–20 Ω，对数均匀抽样。
- B：`s0-ieee37-r10-100-loguniform`，电阻 10–100 Ω，对数均匀抽样。

每套均为 38 个节点、39 个候选、2736 条样本和 40 个故障事件族，约 1.2 GB。原始 S0 数据保持完整；S2 在训练及评价时施加节点掩码。包内未包含预训练权重，默认从头训练；无需 OpenDSS、Windows COM 或原始仓库。B 使用相同 RaReF 默认超参数，排序温度及诊断能量容差沿用 A 的固定标定值；不将其解释为已针对 B 优化或验证的配置。

`manifest.json` 记录来源提交、分支、工作区状态以及每个交付文件的 SHA-256。数据登记快照见 `docs/dataset_registry.md`。默认配置主要在验证集上确定，报告未证明其未观测单点定位已恢复，也未证明其跨种子测试性能稳定。

## 2. 环境安装

目标平台为 Linux x86_64 与单张 NVIDIA GPU。以下主流程针对本次 CUDA 12.8 服务器，采用 Python 3.10、NumPy 1.26.4 和 PyTorch 2.7.1 CUDA 12.8 wheel。命令假定服务器已安装 Conda，安装过程需要访问软件源。建议预留至少 8 GB 主存和 8 GB 可用磁盘；显存需求以设备冒烟检查和实际批次为准。安装方式依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

### 2.1 上传、校验与解压

将 `.tar.gz` 与对应 `.sha256` 文件上传到服务器的同一目录，在该目录执行：

```bash
sha256sum -c raref-server-20261111.tar.gz.sha256
tar -xzf raref-server-20261111.tar.gz
cd raref-server-20261110-ab-perf
nvidia-smi
```

后续全部命令均从解压后的包根目录 `raref-server-20261110-ab-perf/` 执行。归档文件名按交付日期命名为 `raref-server-20261111.tar.gz`。

### 2.2 首次安装 CUDA 12.8 环境

```bash
conda create -n raref-cu128 python=3.10 pip -y
conda activate raref-cu128
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip check
```

若已按上述步骤建立 `raref-cu128`，直接激活即可，不必重新创建。该安装方式与包内 `setup_env.sh`、`environment.yml` 二选一；后两者固定为历史 PyTorch 2.2.1／CUDA 12.1 环境，不应在 CUDA 12.8 环境安装完成后再次执行。

### 2.3 环境与全链路检查

```bash
conda activate raref-cu128
bash run.sh check --device cuda
bash run.sh smoke --device cuda
```

`check` 校验文件完整性并执行设备张量运算，显示实际 PyTorch、CUDA 和 GPU 信息。正式 GPU 命令在 CUDA 不可用时直接报错。CUDA 12.8／PyTorch 2.7.1 环境尚未在目标服务器完成验收，须在两条命令均成功后启动正式实验。

`smoke` 使用独立生成的合成数组完成一轮训练、验证、标准测试和 S2 区域评价，不访问正式数据的测试划分。合成数据位于方法的 `data/smoke-*/`，输出位于 `output/gref/smoke-*/`，只证明软件链路可运行，不代表模型质量。

每次重新登录服务器，先执行 `conda activate raref-cu128`，再进入实际运行包目录后调用训练、验证或测试入口。

### 2.4 历史环境与 CPU 可选方式

复现来源软件版本时，可运行 `bash setup_env.sh`、`source .venv/bin/activate`，或者 `conda env create -f environment.yml`、`conda activate raref`。这些方式使用 PyTorch 2.2.1／CUDA 12.1，需先确认 GPU 架构支持。虚拟环境脚本默认调用 `python3.10`，可通过 `PYTHON_BIN=/path/to/python3.10 bash setup_env.sh` 指定 Python；支持 Python 3.9–3.11。CPU 环境安装采用 `TORCH_FLAVOR=cpu bash setup_env.sh`，后续命令使用 `--device cpu`。本包不含离线 wheel。

## 3. 训练、验证与测试

### 3.1 运行命令语法与参数

服务器入口的统一语法如下。`stage` 必须取 `check`、`train`、`validate`、`test`、`all` 或 `smoke` 之一。

```bash
bash run.sh <stage> [--run-id RUN_ID] [--device DEVICE] [--dataset DATASET]
             [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--seed SEED]
             [--no-cuda-cache]
```

入口参数的含义与默认值如下。

- `--run-id RUN_ID` 指定运行目录名，默认值为 `raref-ieee37-a-seed342`。它只是运行标识，不会从名称中解析随机种子。实际随机种子由 `--seed` 或配置文件决定。
- `--device DEVICE` 指定计算设备，默认值为 `cuda`。可填写 `cuda`、`cuda:0` 或 `cpu`。
- `--dataset DATASET` 选择训练数据，默认值为 `s0-ieee37-r0p01-20-loguniform`。可选值为 `s0-ieee37-r0p01-20-loguniform`（0.01–20 Ω）和 `s0-ieee37-r10-100-loguniform`（10–100 Ω）。验证和测试阶段从 `server_run.json` 读取训练时的数据集。
- `--epochs EPOCHS` 覆盖训练轮数，默认值为配置中的 200。只适用于 `train`、`all` 和 `smoke`，`smoke` 内部固定为 1 轮。
- `--batch-size BATCH_SIZE` 覆盖训练和评价批大小，默认值为配置中的 64。只适用于 `train`、`all` 和 `smoke`，`smoke` 内部固定为 8。
- `--seed SEED` 覆盖训练随机种子，默认值为配置中的 342。若希望使用随机种子 42，必须显式写 `--seed 42`，并建议同时把运行标识写成 `--run-id raref-ieee37-a-seed42`。
- `--no-cuda-cache` 关闭已标准化训练与验证张量的显存缓存，改为逐批从主存加载。默认不启用该选项。它只适用于包含训练的阶段。

各阶段的实际行为如下。

- `check` 只校验包完整性和计算设备，不创建训练运行目录。
- `train` 只训练并执行每轮验证选模，训练入口内部自动加入 `--skip-test-evaluation`，不会评价测试集。
- `validate` 在已训练模型上执行验证集区域评价和概率温度标定，并写入冻结文件指纹。该阶段不能覆盖 `--seed`、`--epochs` 或 `--batch-size`。
- `test` 读取验证阶段生成的冻结指纹，执行一次标准测试和一次区域测试。该阶段不能覆盖 `--seed`、`--epochs` 或 `--batch-size`。
- `all` 按顺序执行 `train`、`validate` 和 `test`。
- `smoke` 使用临时合成数据执行一轮训练、验证和测试，不接受 `--dataset`，运行标识自动生成。

运行标识、随机种子和数据集必须分别记录。例如，下面的命令实际使用随机种子 42：

```bash
bash run.sh train \
  --dataset s0-ieee37-r0p01-20-loguniform \
  --run-id raref-ieee37-a-seed42 \
  --seed 42 \
  --device cuda:0 \
  --epochs 200 \
  --batch-size 64
```

### 3.2 训练命令

以下两条命令分别训练 A（0.01–20 Ω）和 B（10–100 Ω），各默认 200 轮。按需要执行其中一条或依次执行两条；运行标识与数据集一一对应。训练阶段自动执行每轮验证选模，但不评价测试集。

```bash
conda activate raref-cu128
bash run.sh train --dataset s0-ieee37-r0p01-20-loguniform --run-id raref-ieee37-a-seed342 --device cuda
bash run.sh train --dataset s0-ieee37-r10-100-loguniform --run-id raref-ieee37-b-seed342 --device cuda
```

### 3.3 训练完成后的验证命令

对已完成训练的运行执行区域概率温度标定与完整／部分节点观测验证。该步骤是冻结测试的前置条件。

```bash
conda activate raref-cu128
bash run.sh validate --run-id raref-ieee37-a-seed342 --device cuda
bash run.sh validate --run-id raref-ieee37-b-seed342 --device cuda
```

### 3.4 测试命令

对应运行的训练和验证完成后，再执行以下测试。测试加载已选定模型，输出标准定位和 S2 区域评价结果，不更新模型参数。只训练了其中一套数据时，仅执行对应运行的验证和测试命令。

```bash
conda activate raref-cu128
bash run.sh test --run-id raref-ieee37-a-seed342 --device cuda
bash run.sh test --run-id raref-ieee37-b-seed342 --device cuda
```

验证与测试从 `server_run.json` 读取训练时的数据路径；无需重复提供 `--dataset`，若显式指定另一套数据则拒绝运行。运行标识必须与训练命令一致。

### 3.5 一次执行完整流程

也可用 `all` 依次完成训练、验证和测试，替代上述分阶段操作。以下使用不同运行标识，避免与分阶段示例覆盖；未提供 `--dataset` 时默认使用 A。

```bash
conda activate raref-cu128
bash run.sh all --dataset s0-ieee37-r0p01-20-loguniform --run-id raref-ieee37-a-all-seed342 --device cuda
bash run.sh all --dataset s0-ieee37-r10-100-loguniform --run-id raref-ieee37-b-all-seed342 --device cuda
```

### 3.6 训练参数与高级覆盖项

服务器入口只暴露运行阶段和少量常用覆盖项。需要改变损失、节点掩码或划分时，应修改运行包根目录的 `raref_config.json` 后使用新的 `--run-id`，不得复用已有运行目录。训练入口 `scripts/train_gref.py` 支持以下完整参数。

- 数据与输出：`--data-dir`、`--output-dir`、`--device`。
- 数值约束：`--energy-tolerance`（必填）、`--tolerance-source`（必填）、`--temperature`（必填）。
- 损失模式：`--loss-mode` 可选 `gref`、`three-term`、`region`，默认 `gref`；`--lambda-family` 默认 0.1；`--lambda-energy` 默认 1.0；`--lambda-rank` 默认 0.1；`--lambda-smooth` 默认 0.1；`--lambda-region` 默认 0.1。
- RaReF 区域参数：`--region-sigma` 默认 1.0；`--beta-obs` 默认 0.1；`--beta-miss` 默认 0.4；`--rank-temperature` 默认自动标定；`--smooth-edge-weights` 当前只能取 `unit`。
- 训练期节点观测：`--node-mask-full-probability` 默认 0.25；`--node-mask-min-observed` 默认 15；`--node-mask-max-observed` 默认 37；`--node-mask-force-observed-probability` 默认 0.5。
- 训练期拓扑消融：`--edge-drop-prob` 默认 0；`--edge-drop-max` 默认 3。正式主实验保持 `--edge-drop-prob 0`。
- 优化与划分：`--epochs` 默认 200；`--batch-size` 默认 64；`--learning-rate` 默认 0.001；`--val-ratio` 默认 0.15；`--test-ratio` 默认 0.15；`--seed` 默认 342。
- 产物和显示：`--save-epochs` 默认空字符串，可填写 `1,50,100,200`；`--skip-test-evaluation` 用于扫描阶段跳过测试；`--no-terminal-progress` 关闭终端进度条；`--no-cuda-cache` 关闭显存缓存。

直接调用训练入口的示例如下。该方式用于高级实验，服务器推荐入口仍为 `bash run.sh train`。

```bash
python -u code/method-a2/scripts/train_gref.py \
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform \
  --output-dir code/method-a2/output/gref/raref-ieee37-a-seed42 \
  --device cuda:0 \
  --energy-tolerance 2.231209336901537e-06 \
  --tolerance-source "首轮 A 标定 calibration-a.json，三项损失下仅用于门控诊断" \
  --temperature 0.012815295718610287 \
  --loss-mode region \
  --lambda-energy 1.0 --lambda-rank 0.141 --lambda-region 0.194 \
  --region-sigma 1.0 --beta-obs 0.05 --beta-miss 0.4 \
  --rank-temperature 0.0040640985 \
  --epochs 200 --batch-size 64 --learning-rate 0.001 --seed 42 \
  --val-ratio 0.15 --test-ratio 0.15 --save-epochs 1,50,100,200
```

### 3.7 协议与运行约束

训练阶段按事件族重新划分；默认训练含正常样本，验证与测试仅含故障样本。种子 342 时训练、验证、测试分别为 2280、228、228 条，原始数据内置索引不参与本流程。只在最终训练集拟合标准化参数。每轮使用完整观测验证集评价，按验证 Top-1 优先、重构能量次优选取最佳模型；训练阶段跳过测试。

训练增强的完整观测概率为 0.25，部分观测数区间为 15–37，故障样本强制包含或排除真实故障节点各半。验证区域评价仅在验证集完整观测上拟合概率温度，并采用独立掩码种子偏移 1000。测试冻结模型、划分、标准化和区域核；标准测试输出定位指标与反捷径审计，区域测试输出完整观测及 35、30、25、20、15 个节点各 10 组掩码评价。测试掩码使用训练种子，无偏移；α 为 0.05、0.1、0.2。

短程正式数据检查可以使用新的运行标识与 `--epochs 2`；显存不足时降低 `--batch-size 16`。改变批次会改变优化轨迹，须按新实验记录。随机种子可在训练或全流程入口通过 `--seed 343` 覆盖；该操作按种子重划事件族，固定排序温度不重新标定。不可把短程检查或新种子结果解释为默认 200 轮参考实验的复现。

### 3.8 CUDA 性能与训练进度

CUDA 训练使用融合 Adam。显存容量允许时，程序缓存由训练侧统计量标准化的训练和验证张量。默认 IEEE37 划分的缓存约为 1.0 GiB，是否启用依据启动时可用显存；显存共享或紧张时可用 `--no-cuda-cache` 保持逐批加载，实际状态记录于 `config.json` 的 `training_acceleration`。解码器复用各轴上的线性投影，损失复用已计算的候选能量。拆分运算及融合优化器会改变浮点舍入，训练轨迹可能与历史运行略有差异。

`train`、`all` 和 `smoke` 会在终端及 `train.log` 显示进度条、全部训练批次完成比例、当前轮次、已用时间及 ETA。已用时间从训练脚本开始记录数据准备进度时起算。ETA 根据实测批次速率估算区间，计入每轮验证和保存，随运行更新。`all` 在训练结束后独立报告后续阶段。

运行标识必须唯一；已有运行不覆盖。若只完成训练，可继续执行验证、测试。验证与测试不可重写训练种子；测试入口要求验证完成且模型、划分、标准化、区域核哈希保持一致。测试启动后写入独占标记，禁止再次调用测试或重新验证。若测试因外部故障中断，保留现场与日志，需人工核验后处理，不自动删除标记或重跑。当前不提供训练中断续训、多 GPU 或作业调度器脚本。

后台运行示例：

```bash
mkdir -p code/method-a2/logs
nohup bash run.sh all --run-id raref-a-server-342 --device cuda > code/method-a2/logs/launch-342.log 2>&1 &
tail -f code/method-a2/logs/launch-342.log
```

## 4. 输出与检查

所有路径相对运行包根目录。

- `code/method-a2/output/gref/<run-id>/`：最终交付目录。
- `config.json`、`server_run.json`、`environment.freeze.txt`：实际参数、包清单指纹、软硬件信息及已安装依赖版本。
- `train_log.csv`、`validation_metrics.json`：逐轮训练记录和最佳模型的完整验证指标。
- `validation_region_metrics.json`、`validation_region_predictions.npz`：验证分层、概率温度、覆盖率与区域规模。
- `model.pt`、`optimizer.pt`、`checkpoint-epoch*.pt`：复用既有实现保存的最佳模型、最佳轮优化器状态和指定轮快照；不承诺它们支持精确断点续训。
- `split.npz`、`normalization.npz`、`region_kernel.npz`：可复算的划分、标准化和区域核。
- `metrics.json`、`region_metrics.json`、`predictions.npz`、`shortcut_audit.json`、`dataset_audit.json`：标准与 S2 测试结果、逐事件预测和审计。
- `server_complete.json`：全部测试阶段成功完成的标记。存在 `server_test.started.json` 但没有完成标记表示测试尚未完整成功。
- `code/method-a2/logs/<run-id>/`：`train.log`、`validate-region.log`、`test-standard.log`、`test-region.log`、`progress.jsonl` 和 `progress.latest.json`。

脚本运行时控制台显示阶段及日志位置；子进程完整输出保存在上述日志。测试阶段的概率温度由同一冻结模型在验证集确定，不依据测试结果调整。区域覆盖率应与区域大小一起报告；验证集上的样本内校准不等同于独立测试保证。S1 拓扑错误评价不在本包范围内。

## 5. 重新构建与本地验收

以下命令只适用于包含完整来源数据和最新实现的开发仓库。构建工具独立于服务器运行入口，归档生成到对应方法的 `output/`。

```text
python code/method-a2/scripts/build_raref_server_bundle.py --source-root . --destination code/method-a2/output/raref-server-20261110-ab-perf
python -m unittest discover -s code/method-a2/tests -p test_raref_server_bundle.py -v
```

交付验证包括清单完整性、依赖导入、命令行契约、合成数据的完整端到端执行，以及重复测试拒绝。当前开发机为 Windows，未在目标 Linux GPU 上完成 200 轮正式训练；服务器需先运行 `check` 和 GPU `smoke` 后再开展正式实验。跨系统与 CUDA 算子可能产生数值差异，不承诺与历史结果逐位一致。
