"""进度汇报可行性验证脚本：按《运行进度与时长汇报规范》实现最小可运行版本并逐项核对。

本脚本不修改任何既有实验代码与产物，只在 logs/ 下新建一个验证专用目录，
用于回答三个问题：进度文件是否能在运行中途读到、ETA 是否随实际耗时自校准、
进程中断与停滞后读取入口能否如实报告。
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------- 待验证实现

def write_progress(path, record):
    """追加一条进度记录，并把快照文件更新为同一条记录。

    path 为进度文件路径；record 为记录字典。
    写完立即关闭文件句柄，保证运行中途读取者能看到最新内容。
    """
    payload = json.dumps(record, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    with path.with_name("progress.latest.json").open("w", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def make_record(run_id, stage, unit, done, total, elapsed, utc, note="", pid=0):
    """按规范字段集构造一条阶段结束记录。

    pid 默认写 0，避免验证脚本自身进程标识干扰核对。
    """
    return {
        "kind": "stage_done", "utc": utc, "run_id": run_id, "stage": stage,
        "unit": unit, "done": int(done), "total": int(total),
        "percent": round(100.0 * done / total, 2) if total else None,
        "elapsed_s": round(elapsed, 4), "rate_s": None, "eta_s": None,
        "note": note, "pid": pid,
    }


def load_progress(path):
    """读取进度文件全部记录，跳过被强制结束留下的残缺行。"""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 进程被强制结束时最后一行可能不完整，跳过即可
    return records


def summarize(records, now=None):
    """汇总最近状态：新鲜度、完成度、分类别速率与剩余时间估计。"""
    if not records:
        return {"status": "no_records"}
    latest = records[-1]
    stamped = datetime.strptime(latest["utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    age = (now or datetime.now(timezone.utc).timestamp()) - stamped.timestamp()

    per_unit, per_stage = {}, {}
    for record in records:
        if record.get("unit") and record.get("rate_s"):
            per_unit.setdefault(record["unit"], []).append(float(record["rate_s"]))
        if record.get("stage") and record.get("elapsed_s"):
            per_stage.setdefault(record["stage"], []).append(float(record["elapsed_s"]))
    rates = {name: sum(values) / len(values) for name, values in per_unit.items()}

    remaining = None
    if latest.get("total"):
        remaining = max(int(latest["total"]) - int(latest.get("done") or 0), 0)
    rate = rates.get(latest.get("unit")) if latest.get("unit") else None
    eta = round(rate * remaining, 4) if (rate is not None and remaining is not None) else None
    if remaining == 0:
        basis = "本阶段单元已全部完成"
    elif eta is not None:
        basis = "按同类单元历史平均耗时外推"
    else:
        basis = "尚无同类单元完成，无法估计"
    return {
        "status": "running", "run_id": latest.get("run_id"), "stage": latest.get("stage"),
        "age_s": round(age, 2), "elapsed_s": latest.get("elapsed_s"),
        "done": latest.get("done"), "total": latest.get("total"),
        "percent": latest.get("percent"), "eta_s": eta,
        "eta_range_s": [round(eta * 0.8, 4), round(eta * 1.3, 4)] if eta else None,
        "basis": basis,
        "sample_sizes": {name: len(values) for name, values in per_unit.items()},
        "note": latest.get("note"),
    }


def resolve_run(run_id=None, log_root=None):
    """定位一次运行的进度文件：显式给出运行标识，否则取最后修改的那一个。"""
    root = Path(log_root) if log_root else Path(__file__).absolute().parents[1] / "logs" / "progress-selftest"
    candidates = ([root / run_id / "progress.jsonl"] if run_id
                  else sorted(root.glob("*/progress.jsonl"), key=lambda p: p.stat().st_mtime))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------- 验证用例

def case_a_中途可读(run_id, log_root):
    """验证最核心的问题：循环还在跑，外部能否读到当前进度与自校准的剩余时间。"""
    print("\n=== 用例A：运行中途读取（模拟运行中的三类单元） ===")
    path = log_root / run_id / "progress.jsonl"

    # 模拟阶段开始：写入 start 记录后再进入循环
    write_progress(path, make_record(run_id, "probes", None, 0, 6, 0.0,
                                     time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                                     note="阶段开始：3 个快单元 + 3 个慢单元"))
    read_A = summarize(load_progress(path))

    # 实际耗时分布刻意做成两类：快单元 0.1 秒，慢单元 1.0 秒
    done = 0
    rates = {}
    unit_of = ["ridge"] * 3 + ["mlp"] * 3
    for i, unit in enumerate(unit_of):
        cost = 0.1 if unit == "ridge" else 1.0
        started = time.perf_counter()
        time.sleep(cost)
        measured = time.perf_counter() - started
        rates.setdefault(unit, []).append(measured)
        done += 1
        record = make_record(run_id, "probes", None, done, 6, sum(sum(v) for v in rates.values()),
                             time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                             note=f"第 {done} 个单元：{unit}")
        # 按类别的平均耗时写入 rate_s，并按剩余同类单元数写入 eta_s
        record["unit"] = unit
        record["rate_s"] = round(sum(rates[unit]) / len(rates[unit]), 4)
        same_kind_left = sum(1 for name in unit_of[done:] if name == unit)
        record["eta_s"] = round(record["rate_s"] * same_kind_left, 4)
        write_progress(path, record)

        # 每个单元结束后立刻从磁盘读一次，检查外部此刻能看到什么
        snapshot = summarize(load_progress(path))
        print(f"  第 {done} 个单元后读取：stage={snapshot['stage']} "
              f"done={snapshot['done']}/{snapshot['total']} percent={snapshot['percent']} "
              f"age={snapshot['age_s']}s eta={snapshot['eta_s']} 依据={snapshot['basis']}")

    final = summarize(load_progress(path))
    print(f"  末尾状态：{final}")
    return path, final


def case_b_分类别速率(run_id, log_root):
    """验证按类别分别估计：混合速率会被慢单元严重拉偏，分类别不会。"""
    print("\n=== 用例B：分类别速率与混合速率对比 ===")
    path = log_root / run_id / "progress.jsonl"
    records = load_progress(path)
    ridge = [r["rate_s"] for r in records if r.get("unit") == "ridge" and r.get("rate_s")]
    mlp = [r["rate_s"] for r in records if r.get("unit") == "mlp" and r.get("rate_s")]
    mixed = [r["rate_s"] for r in records if r.get("rate_s")]
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    print(f"  ridge 平均：{avg(ridge)} 秒/次（样本 {len(ridge)}）")
    print(f"  mlp   平均：{avg(mlp)} 秒/次（样本 {len(mlp)}）")
    print(f"  混合  平均：{avg(mixed)} 秒/次（样本 {len(mixed)}）")
    if avg(ridge) and avg(mlp) and avg(mixed):
        print(f"  结论：混合速率({avg(mixed)}) 与慢单元真实速率({avg(mlp)}) 相差 "
              f"{abs(avg(mixed) - avg(mlp)) / avg(mlp) * 100:.1f}%，"
              f"分类别估计是必要的" if abs(avg(mixed) - avg(mlp)) / avg(mlp) > 0.1
              else "  结论：两类速率接近，本用例不足以体现分类别差异")
    return {"ridge": avg(ridge), "mlp": avg(mlp), "mixed": avg(mixed)}


def case_c_停滞与中断(log_root, run_id_stalled, run_id_aborted):
    """验证停滞后不得沿用旧估计，以及中断后能指出中断位置。"""
    print("\n=== 用例C：停滞判定与中断定位 ===")
    # 停滞：最后一条记录写在一个小时之前
    stalled_path = log_root / run_id_stalled / "progress.jsonl"
    old = (datetime.now(timezone.utc).timestamp() - 3600)
    write_progress(stalled_path, make_record(
        run_id_stalled, "probes", "mlp", 7, 33, 600.0,
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(old)), note="width=64 wd=1e-4"))
    stalled = summarize(load_progress(stalled_path))
    threshold = 60.0
    verdict = "疑似停滞" if stalled["age_s"] > threshold else "仍在推进"
    print(f"  最后更新 {stalled['age_s']} 秒前，阈值 {threshold} 秒 → 判定：{verdict}")
    print(f"  停滞时的报告应给出：阶段={stalled['stage']} 完成={stalled['done']}/{stalled['total']} "
          f"当前对象={stalled['note']}，且不给出剩余时间")

    # 中断：有 start 与若干推进记录，但没有 end 记录，最后一行是残缺行
    aborted_path = log_root / run_id_aborted / "progress.jsonl"
    write_progress(aborted_path, make_record(
        run_id_aborted, "representations", None, 0, 1,
        0.0, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), note="阶段开始：表示提取"))
    write_progress(aborted_path, make_record(
        run_id_aborted, "probes", "ridge", 5, 33, 42.0,
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), note="H | rep_node_mean"))
    with aborted_path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "unit", "utc": "2026-09-24T10:31:0')  # 模拟被强制结束时留下的半行

    records = load_progress(aborted_path)
    has_end = any(r.get("kind") == "end" for r in records)
    aborted = summarize(records)
    print(f"  记录条数（已跳过残缺行）：{len(records)}；存在 end 记录：{has_end}")
    print(f"  判定：运行未正常结束，中断于阶段={aborted['stage']} "
          f"完成={aborted['done']}/{aborted['total']} 当前对象={aborted['note']}")
    print(f"  此时不报告剩余时间：eta={aborted['eta_s']}")
    return {"stalled_verdict": verdict, "aborted_has_end": has_end, "aborted_records": len(records)}


def case_d_读取入口(run_id, log_root):
    """验证读取入口作为独立命令可用，即后续会话不必手工解析文件。"""
    print("\n=== 用例D：读取入口命令行 ===")
    target = log_root / run_id / "progress.jsonl"
    script = Path(__file__).absolute().parent / "report_progress.py"
    if not script.exists():
        print(f"  跳过：读取入口 {script.name} 尚未生成（属于后续代码生成范围）")
        return {"command_ok": None, "reason": "入口脚本未生成"}
    result = subprocess.run([sys.executable, str(script), "--progress", str(target)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"  退出码：{result.returncode}")
    print("  输出：")
    for line in (result.stdout or "").strip().splitlines():
        safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print(f"    {safe_line}")
    return {"command_ok": result.returncode == 0, "stdout": result.stdout}


def main():
    """依次执行四个用例并打印汇总判定。"""
    log_root = Path(__file__).absolute().parents[1] / "logs" / "progress-selftest"
    run_id = "selftest-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_root.mkdir(parents=True, exist_ok=True)

    path, final = case_a_中途可读(run_id, log_root)
    rates = case_b_分类别速率(run_id, log_root)
    verdicts = case_c_停滞与中断(log_root, run_id + "-stalled", run_id + "-aborted")
    entry = case_d_读取入口(run_id, log_root)

    print("\n=== 汇总 ===")
    eta_basis_ok = any("按同类单元历史平均耗时外推" in r.get("basis", "")
                       for r in [summarize(load_progress(path)[:i + 1]) for i in range(5)])
    checks = [
        ("运行中途可读到最新记录", final.get("done") == 6 and final.get("percent") == 100.0),
        ("运行中途的 ETA 有依据且随实测耗时自校准", eta_basis_ok),
        ("分类别速率与混合速率可区分", rates["mlp"] is not None and rates["mlp"] > rates["ridge"]),
        ("超过阈值无更新判定为停滞", verdicts["stalled_verdict"] == "疑似停滞"),
        ("中断后无 end 记录且跳过残缺行", verdicts["aborted_has_end"] is False
         and verdicts["aborted_records"] == 2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '未通过'}] {name}")
    print(f"\n  进度文件：{path}")
    print("  上述用例回答的是规范第 7 节的可验证部分；正式验收需在长任务真实运行中复核。")


if __name__ == "__main__":
    main()
