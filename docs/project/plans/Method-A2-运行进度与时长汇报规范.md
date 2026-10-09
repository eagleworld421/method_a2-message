<!-- 摘要：规定 Method-A2 长时运行任务的进度落盘与时长汇报规范。针对“脚本在后台运行、用户看不到终端、无人能回答当前进度与预计剩余时间”的问题，规定进度文件的位置与推荐路径、逐行 JSON 记录格式与最小字段集、各层循环的埋点方式与写盘节流规则、读取入口脚本、后台启动与汇报协议、停滞判定阈值、无参照时的诚实表述要求，以及验收条件；本文为代码生成前的计划与设计文档，含代码示例，不含运行结果。 -->

# Method-A2：运行进度与时长汇报规范

> 日期：2026-09-24
> 状态：已实现并通过小规模可行性验证；长任务正式验收待执行
> 适用范围：`code/method-a2/` 下全部脚本与入口中单次运行超过一分钟的任务
> 优先级：若本文与代码、Git 规则或正式项目文档冲突，以代码/Git 与正式项目文档为准。

---

## 1. 问题与目标

### 1.1 现象

当前长时任务以两种方式运行：一种是 Agent 在后台启动脚本，另一种是用户在终端自行启动。在这两种情形下，用户都看不到运行中的终端画面，Agent 也无法读取终端画面。可读取的信息只有两类：一是程序已写入磁盘的文件，二是后台作业的输出缓冲区。

既有的进度记录只覆盖阶段边界。以判据一探针脚本的线性全量运行为例，其进度文件仅有五条记录，分别是表示提取完成、探针拟合完成、运行结束，其中探针拟合阶段占用总时长的约百分之八十八，在该阶段内部没有任何记录。因此，一次运行的耗时可以被事后算出，但运行过程中的位置不可知。

### 1.2 目标

用一个磁盘文件回答三个问题：现在做到哪一步；已经过去多久；预计还要多久。该文件必须在运行过程中持续更新，使得任何人在任意时刻读取它都能得到当前状态，且不要求读取者接触终端、不要求读取者与进程通信、不因进程中断而失效。

### 1.3 不在本文范围内

本文不改变实验设计、判定阈值与产物口径；不引入新的运行框架、调度器或外部服务；不要求把训练流程改造为可中断或可恢复。

## 2. 设计原则

- 进度写入磁盘文件，以追加方式逐行记录。不依赖终端、不依赖标准输出是否被重定向、不依赖是否有读取者在线。
- 位置与时间同时记录。只记录完成度无法判断速率，只记录耗时无法判断位置，两者必须成对出现。
- 记录必须立即落盘。每次写完即关闭文件，不使用长期持有的写入句柄，避免缓冲区未刷新导致读取者看到空文件。
- 记录位置覆盖到最内层循环。仅在最外层循环埋点，会使单个长单元内部重新变成空白，这正是当前问题的成因。
- 人的观感与机器的读取分开。面向终端的进度条可以保留用于本地调试；面向进度文件的是逐行文本记录，二者互不替代。
- 数字必须有依据。已经过去的时间是实数，直接报告；剩余时间是估计，必须给出依据与区间；没有同类参照时明确说明无法估计，不以单一数字冒充准确估计。
- 长时间无更新等同于异常信号，必须如实报告停滞，不得沿用失效的估计继续安抚。

## 3. 进度文件规范

### 3.1 路径

进度文件放在 `logs/` 下，与该次运行的日志同目录，不新建目录层级。相对方法目录的路径为 `logs/<run-id>/progress.jsonl`，与既有的 `logs/<run-id>/heartbeat.jsonl` 并列。既有 `heartbeat.jsonl` 的阶段级记录继续保留，用于确认阶段迁移，两者不互相替代。

同时维护一个只保留最新一条记录的快照文件 `logs/<run-id>/progress.latest.json`。其内容是最近一次写入的那条记录，读取者只需读取一个很小的文件即可获得当前状态，不必扫描整个追加文件。

运行标识的确定方式沿用既有做法：命令行显式给出时使用给定值，未给出时按 `c1run-<UTC 时间戳>-seed<种子>` 形式生成，并把该标识写入本次运行的全部产物与进度记录，使目录名、报告与进度文件可互相追溯。

### 3.2 记录格式

采用逐行 JSON，每行一条完整记录，行内不换行，编码为 UTF-8。字段命名使用简短英文，值中的说明性文字使用中文，便于阅读者直接理解。字段如下。

- `utc`：写入时刻的协调世界时，形如 `2026-09-24T10:31:07`。用于判断记录新鲜度。
- `run_id`：运行标识，用于确认读到的是目标运行。
- `stage`：当前阶段标识，取自既有阶段划分，例如 `representations`、`probes`、`controls`、`writeout`。
- `unit`：当前工作单元的类别，例如 `ridge`（岭回归拟合）、`logistic`（多类逻辑回归拟合）、`mlp`（浅层非线性拟合）、`epoch`（单个拟合内部的训练轮批次）。当同一阶段内不同单元的耗时相差一个数量级以上时，该字段是估计可用的前提。
- `done`：已完成的单元数。
- `total`：本阶段计划完成的单元总数；无法预先确定时写 `null`，并在 `note` 中说明原因。循环次数在运行前即可确定时不得写 `null`，否则百分比与剩余时间都不可用；判据一探针脚本的各阶段均属此类，其总数可由目标数与特征源数相乘直接得到。
- `percent`：`done / total` 的百分比，保留两位小数；`total` 为 `null` 时写 `null`。
- `elapsed_s`：自本次运行开始到当前时刻的秒数。
- `rate_s`：该类单元的平均单次耗时，单位秒；无已完成单元时为 `null`。
- `eta_s`：按当前速率估计的剩余秒数；无参照时写 `null`。
- `note`：自由文本，说明当前正在处理的对象，例如目标名与特征源名。
- `pid`：进程标识，用于区分同名的重新启动进程。
- `kind`：记录类型。每次运行开始时写一条 `start`，正常推进写 `unit`，长单元内部按时间间隔写 `tick`，阶段完成写 `stage_done`，全部完成写 `end`，未捕获异常写 `error`。

### 3.3 最小字段集

`utc`、`run_id`、`kind`、`stage`、`done`、`total`、`elapsed_s` 为必需字段，缺失任一字段视为实现不合格。其中 `total` 在无法预先确定时写 `null`，但该字段本身必须存在，以便读取方区分「不可估计」与「实现漏写」。其余字段可依实际情况省略，省略的字段读取方按不可用处理。

## 4. 埋点方式与代码示例

以下示例使用建议的实现位置与中文注释，可直接作为代码生成的依据。实现文件建议放在 `code/method-a2/src/progress.py`，脚本与入口从该模块导入。

### 4.1 进度记录器

```python
"""运行进度落盘工具：把循环进度写成逐行 JSON，供运行中途随时读取。"""

import json
import os
import time
from pathlib import Path


class ProgressWriter:
    """按阶段记录完成数、耗时与剩余时间，每次写入立即落盘。"""

    def __init__(self, path: Path, run_id: str, min_interval_s: float = 5.0) -> None:
        """
        path：进度文件路径，形如 logs/<run-id>/progress.jsonl。
        run_id：运行标识，写入每条记录用于核对。
        min_interval_s：两次写盘之间的最小间隔，用于节流高频更新。
        """
        self.path = Path(path)
        self.latest_path = self.path.with_name("progress.latest.json")
        self.run_id = run_id
        self.min_interval_s = float(min_interval_s)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._last_write = 0.0
        self._unit_totals: dict[str, float] = {}

    def _elapsed(self) -> float:
        """返回自本次运行开始以来经过的秒数。"""
        return time.perf_counter() - self.started

    def _write(self, record: dict) -> None:
        """追加一条记录并刷新快照文件；写完立即关闭文件句柄。"""
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.latest_path.open("w", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stage_start(self, stage: str, total: int | None) -> None:
        """标记阶段开始，写入一条 start 记录并重置该阶段的累计耗时。"""
        self._unit_totals = {}
        self._write({
            "kind": "start", "utc": self._utc(), "run_id": self.run_id,
            "stage": stage, "unit": None, "done": 0, "total": total,
            "percent": 0.0 if total else None, "elapsed_s": round(self._elapsed(), 1),
            "rate_s": None, "eta_s": None, "note": "", "pid": os.getpid(),
        })

    def stage_done(self, stage: str, total: int | None, note: str = "") -> None:
        """标记阶段结束。"""
        self.update(stage, unit=None, done=total or 0, total=total,
                    note=note or "阶段完成", kind="stage_done", force=True)

    def update(self, stage: str, unit: str | None, done: int, total: int | None,
               note: str = "", kind: str = "unit", force: bool = False,
               unit_seconds: float | None = None) -> None:
        """
        记录一次进度推进。
        done：已完成的单元数；total 为 None 时按不可估计处理。
        unit_seconds：本次单元的实测耗时；给出时用于更新该类单元的速率。
        kind：unit 表示正常推进，tick 表示长单元内部的定时记录。
        force：为真时忽略节流间隔，强制写入。
        """
        now = time.perf_counter()
        if not force and kind == "unit" and now - self._last_write < self.min_interval_s:
            return
        if unit_seconds is not None and unit is not None:
            history = self._unit_totals.setdefault(unit, [0.0, 0])  # [累计秒数, 次数]
            history[0] += float(unit_seconds)
            history[1] += 1
        rate = None
        if unit is not None and unit in self._unit_totals and self._unit_totals[unit][1] > 0:
            total_s, count = self._unit_totals[unit]
            rate = total_s / count
        remaining = None if total is None else max(int(total) - int(done), 0)
        eta = round(rate * remaining, 1) if (rate and remaining is not None) else None
        self._last_write = now
        self._write({
            "kind": kind, "utc": self._utc(), "run_id": self.run_id,
            "stage": stage, "unit": unit, "done": int(done), "total": total,
            "percent": round(100.0 * done / total, 2) if total else None,
            "elapsed_s": round(self._elapsed(), 1),
            "rate_s": round(rate, 3) if rate else None,
            "eta_s": eta, "note": note, "pid": os.getpid(),
        })

    def error(self, stage: str, note: str) -> None:
        """记录未捕获异常，保留出错位置供事后定位。"""
        self._write({
            "kind": "error", "utc": self._utc(), "run_id": self.run_id,
            "stage": stage, "unit": None, "done": None, "total": None,
            "percent": None, "elapsed_s": round(self._elapsed(), 1),
            "rate_s": None, "eta_s": None, "note": note, "pid": os.getpid(),
        })

    @staticmethod
    def _utc() -> str:
        """返回当前协调世界时字符串。"""
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
```

### 4.2 外层循环埋点

对每个阶段先调用 `stage_start` 给出总数，循环内每次推进调用一次 `update`。以判据一探针脚本的主探针阶段为例，其结构为两重循环，总数为目标数与特征源数的乘积：

```python
# 阶段开始：给出本阶段计划完成的总单元数
feature_sources = feature_names + observation_names + ["concat_rep_all_obs_full"]
total_units = len(("H", "log1p_impedance")) * len(feature_sources)
progress.stage_start("probes", total_units)

index = 0
for target_name, dim_names in (("H", list(LOAD_MULTIPLIER_KEYS)), ("log1p_impedance", ["log1p_impedance"])):
    target = targets[target_name]
    for source_name in feature_sources:
        unit_started = time.perf_counter()
        result = fit_probe(sliced(source_name), target, splits, args.seed,
                           kind=args.probe_kinds, task="regression")
        index += 1
        # 每次拟合结束记录一次，note 写明当前对象，便于直接判断跑到哪一个组合
        progress.update("probes", unit="ridge", done=index, total=total_units,
                        note=f"{target_name} | {source_name}",
                        unit_seconds=time.perf_counter() - unit_started)
```

各类探针应使用各自的 `unit` 值：岭回归写 `ridge`，多类逻辑回归写 `logistic`，浅层非线性写 `mlp`。这样剩余时间按类别分别估计，不会被慢单元与快单元的混合平均拉偏。

### 4.3 长单元内部埋点

单个单元本身耗时较长时，仅在该单元结束时记录仍会造成空白。在单元内部的轮次循环中按间隔记录 `tick`。以浅层非线性拟合的训练循环为例：

```python
def _train_shallow_mlp(x_train, y_train, out_dim, task, width, weight_decay,
                       epochs, learning_rate, seed, progress=None, tick_interval_s=30.0):
    """全批量训练一个单隐藏层 MLP，并按时间间隔记录训练进度。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    module = nn.Sequential(
        nn.Linear(int(x_train.shape[1]), int(width)),
        nn.ReLU(),
        nn.Linear(int(width), int(out_dim)),
    )
    optimizer = torch.optim.Adam(
        module.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    if task == "regression":
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        loss_function = nn.functional.mse_loss
    else:
        y_tensor = torch.as_tensor(y_train, dtype=torch.long)
        loss_function = nn.functional.cross_entropy

    unit_started = time.perf_counter()
    last_tick = unit_started
    for step in range(int(epochs)):
        optimizer.zero_grad()
        loss = loss_function(module(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        # 单轮计算很快，不做每轮写盘；仅当距上次记录超过阈值时记录一次
        if progress is not None and time.perf_counter() - last_tick >= tick_interval_s:
            last_tick = time.perf_counter()
            progress.update("probes", unit="epoch", done=step + 1, total=int(epochs),
                            note=f"width={width} wd={weight_decay}",
                            kind="tick", force=True,
                            unit_seconds=time.perf_counter() - unit_started)
    module.eval()
    return module
```

`tick` 记录中的 `done` 与 `total` 表示当前单元内部的轮次，读取者可据此判断该单元是否仍在推进。不要求每一轮都写盘，也不必为该检查使用额外线程；单次时间比较的开销可以忽略。

### 4.4 异常与中断

在入口函数的主体外层包一次异常捕获，捕获后写入 `error` 记录并保留原始异常抛出，使退出码与既有行为一致：

```python
try:
    run_experiment(...)
except Exception as exc:  # 记录出错位置后再抛出，不改变既有失败语义
    progress.error("unknown", f"{type(exc).__name__}: {exc}")
    raise
```

### 4.5 面向终端的进度条

本地调试时可以使用进度条工具，写法为 `for i in tqdm(range(n), file=sys.stdout, mininterval=5)`。该输出仅用于本地观察，不作为进度汇报的数据来源，原因是重定向到文件后回车符不再覆盖行首，历史进度会被压缩成难以按行切分的长行，读取成本高且信息不可靠。

## 5. 读取入口

新增 `code/method-a2/scripts/report_progress.py`，接受运行标识或进度文件路径，输出一段固定结构的文本，含四部分：记录新鲜度与存活判断、阶段与完成度、按类别给出的剩余时间估计、以及该估计的依据与可靠程度。核心读取逻辑如下：

```python
def load_records(path: Path) -> list[dict]:
    """读取进度文件中的全部记录；文件不存在或含残缺行时跳过该行。"""
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 进程被强制结束时会留下半行，跳过即可
    return records


def summarize(records: list[dict], now: float | None = None) -> dict:
    """汇总最近状态：新鲜度、完成度、分类别剩余时间与估计依据。"""
    if not records:
        return {"status": "no_records"}
    latest = records[-1]
    stamped = datetime.strptime(latest["utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    age = (now or datetime.now(timezone.utc).timestamp()) - stamped.timestamp()

    # 按 unit 聚合已完成单元的实测耗时，得到分类别的单次平均耗时
    per_unit: dict[str, list[float]] = {}
    for record in records:
        if record.get("unit") and record.get("rate_s"):
            per_unit.setdefault(record["unit"], []).append(float(record["rate_s"]))
    rates = {name: sum(values) / len(values) for name, values in per_unit.items()}

    remaining_units = None
    if latest.get("total"):
        remaining_units = max(int(latest["total"]) - int(latest.get("done") or 0), 0)
    rate = rates.get(latest.get("unit")) if latest.get("unit") else None
    eta = round(rate * remaining_units, 1) if (rate is not None and remaining_units is not None) else None
    if remaining_units == 0:
        basis = "本阶段单元已全部完成"
    elif eta is not None:
        basis = "按同类单元历史平均耗时外推"
    else:
        basis = "尚无同类单元完成，无法估计"

    return {
        "status": "running", "run_id": latest.get("run_id"), "stage": latest.get("stage"),
        "age_s": round(age, 1), "elapsed_s": latest.get("elapsed_s"),
        "done": latest.get("done"), "total": latest.get("total"),
        "percent": latest.get("percent"), "eta_s": eta,
        "eta_range_s": [round(eta * 0.8, 1), round(eta * 1.3, 1)] if eta is not None else None,
        "basis": basis,
        "sample_sizes": {name: len(values) for name, values in per_unit.items()},
        "note": latest.get("note"),
    }
```

三个必须遵守的实现细节，均由可行性验证暴露，忽略任一条都会使读取结果失真。

- 残缺行必须被跳过，不得让异常向上传播。进程被强制结束或断电时，最后一行往往只有半条 JSON。上述 `load_records` 之所以使用 `path.read_text()` 之外的逐行解析，正是因为直接解析整段文本会在遇到半行时整体失败；一旦整体失败，读取者会误判为「没有记录」，而实际情况是「记录到某一步为止」。规范要求读取方区分这两种情形。
- 速率与剩余量都必须以 `is not None` 判断，不得用真值判断。单元耗时有可能恰好为整数零，此时真值判断会把可用的速率误判为缺失。
- 剩余量为零时，依据文案不得写成「无法估计」。全部单元已完成与没有参照是两种不同情形，前者应报告为阶段完成。

命令行输出示例（仅为格式示例，数值不是运行结果）：

```text
运行 c1nl-20260923-seed342　状态：仍在推进
最后更新：12 秒前（写入于 2026-09-24T10:31:07Z）
阶段 probes：已完成 12/33（36.36%），已运行 8 分 20 秒
分类别速率：ridge 平均 3.1 秒/次（样本 9），mlp 平均 84.6 秒/次（样本 3）
预计剩余：约 14 分 06 秒（区间 11 分 17 秒 至 18 分 20 秒）
依据：剩余 mlp 单元按历史平均耗时外推，样本量 3，可靠程度中等
当前对象：H | rep_node_flat
```

## 6. 后台运行与汇报协议

### 6.1 启动方式

单次运行预计超过一分钟的任务，必须以后台方式启动，不得在前台等待。Agent 使用带后台参数的方式启动脚本，启动调用会立即返回一个作业标识，形如 `pwsh-1`，该标识需被记录，用于后续读取或终止。脚本的标准输出应同时写入 `logs/<run-id>/run.log`，使输出在作业结束后仍然可查。

### 6.2 用户询问时的回答方式

用户询问进度时，按以下顺序执行，不中断运行中的程序。

1. 读取进度快照文件 `logs/<run-id>/progress.latest.json`；若需看趋势与分类别速率，读取 `logs/<run-id>/progress.jsonl` 的最后若干行。
2. 计算最后一条记录的写入时刻与当前时刻之差，据此给出新鲜度。
3. 读取后台作业的增量输出，确认作业状态为运行中还是已结束。该读取默认不阻塞，返回的是上次读取之后的新增内容；仅在确实需要等待结束时才使用其等待参数，且等待有上限，不得用作长时间阻塞。
4. 输出固定四段：是否仍在推进；已完成数与百分比；预计剩余时间与区间；该估计的依据与样本量。

### 6.3 注意力占用

后台运行的目的正是让等待不占用注意力。作业结束时系统会主动投递一条完成通知，因此无需周期性地反复查看，也无需为等待而空转。仅在用户询问进度、或需要判断作业是否已结束以继续后续步骤时读取。禁止为等待结束而反复轮询或休眠。

### 6.4 停滞与失败判定

以时间阈值判定停滞：默认阈值为六十秒与该类别单元历史平均单次耗时的三倍中的较大者。超过阈值没有任何新记录时，必须报告为疑似停滞，并给出最后一条记录的写入时刻、阶段、完成数与当前对象。

判定停滞之后不得继续报告旧的估计值，原因是旧估计所依据的速率已经失效。此时可行的动作是读取后台作业输出查看是否已退出，并检查 `logs/<run-id>/run.log` 尾部与是否存在 `error` 记录。

进程已退出而进度文件停在中间时，报告应改述为「运行已中断于某阶段某对象」，并给出该位置，而不是继续给出剩余时间。

### 6.5 诚实表述要求

- 已用时长直接报告实测值，不换算、不修饰。
- 剩余时间必须同时给出区间与依据。依据不足时只报告已完成数、已完成类别速率与剩余单元数，由用户自行判断。
- 本轮尚无任何同类单元完成时，明确表述为无法估计，不得以其他类别速率代替，也不得给出单一数字。
- 完成度分母无法预先确定时，报告为「进度不可估比例」，并给出已完成的绝对单元数。
- 估计与事实必须可区分，任何估计都应能被后续记录证伪，例如估计剩余十分钟而实际用时显著不同时，应在下一次汇报中说明偏差。

## 7. 可行性验证与验收条件

### 7.1 已完成的可行性验证

本文规定的做法已用 `code/method-a2/scripts/progress_spec_feasibility_check.py` 在小规模上实测，该脚本不修改任何既有实验代码与产物，其运行产物写入 `logs/progress-selftest/`，可据 `run_id` 复核。验证结论如下。

- 运行中途可读取：在模拟循环中，每个单元结束后立即从磁盘重新读取，六个单元依次读到完成数一至六，百分比由百分之十六点六七推进至百分之百，最后一条记录的写入时刻与读取时刻之差不超过零点九秒。
- 剩余时间可自校准：即使单个单元的真实耗时不足一秒，读数仍随实测耗时调整。三个快单元之后按类别估计的剩余时间为零点二三秒、零点一一秒与零秒，进入慢单元后立刻上调至约二点零一秒与一点零一秒，说明估计依据的是实测速率而不是预设值。写入文件的 `rate_s` 与实测耗时一致，快单元为零点一零九七至零点一一四三秒，慢单元为一点零零六九至一点零一一七秒。
- 按类别估计是必要的：本用例中快单元平均约零点一一秒、慢单元平均约一点零一秒，若把两类混在一起平均得到约零点五六秒，与慢单元真实速率相差百分之四十四点五。这正是必须以 `unit` 分类估计的量化依据。
- 停滞可判定：最后一条记录写于一小时前时，按六十秒阈值判定为疑似停滞，并能给出阶段、完成数与当前对象。
- 中断可定位：在文件中人为写入半条 JSON 行后用读取逻辑解析，残缺行被跳过，其余两条记录正常读出，判定为运行未正常结束且无 `end` 记录，中断位置为 `probes` 阶段第五个单元，此时不报告剩余时间。

本次验证同时暴露了三处必须先修正的实现缺陷，已写回第 5 节：残缺行必须逐行容错解析，否则读取者会把「记录到某一步」误判为「没有记录」；速率与剩余量的判断必须使用 `is not None`，否则耗时恰为零的单元会让可用速率被判为缺失；剩余量为零时不得沿用「无法估计」文案。

本节验证覆盖的是规范可离线检验的部分，不替代第 7.2 节在长任务真实运行中的验收。

### 7.2 验收条件

代码生成完成后，按以下条件验证，全部满足方可视为实现合格。

- 在冒烟子采样规模下运行三次长任务，运行过程中任意时刻读取进度文件，最后一条记录的写入时刻与当前时刻之差不超过六倍写盘间隔。
- 单次运行结束后，进度文件中的 `end` 记录存在，其中的总耗时与运行结束时打印的总耗时一致，误差不超过一秒。
- 记录的阶段耗时之和与总耗时一致，误差不超过百分之一，可用于回答各模块耗时占比。
- 最长的单个单元内部存在至少一条 `tick` 记录；若某单元耗时超过写盘间隔却没有任何中间记录，视为埋点不合格。
- 用写入 `error` 记录后的产物不存在 `end` 记录，且 `report_progress.py` 能据此判断运行未正常结束。
- 人为在探针阶段中止进程后，`report_progress.py` 能指出中断所处的阶段与对象，不报告剩余时间。
- 冒烟运行不改变既有判定结论与既有产物内容；进度记录的加入不引入随机性来源，同一随机种子下判定结果与加入前一致。

## 8. 交叉引用

- 埋点对象与阶段划分依据：`docs/project/reports/Method-A2-判据一-共享状态可恢复性Probe执行结论.md`。
- 该实验的设计与产物清单：`docs/project/plans/Method-A2-判据一-共享状态可恢复性Probe设计.md`。
- 文件分类与输出位置约束：`code/CODE_CONVENTIONS.md`；代码范围执行要求：`code/AGENTS.md`。
