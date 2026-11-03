"""从独立 IEEE37 副本导出线路矩阵及额定状态设备原始导纳。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def export(master, destination):
    """仅在审计副本内关闭调压控制并提取串联参数。"""
    import win32com.client

    engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
    engine.AllowForms = False
    engine.DataPath = str(master.parent)
    engine.Text.Command = f'Compile "{master}"'
    circuit = engine.ActiveCircuit
    operational_taps = {}
    i = circuit.Transformers.First
    while i:
        operational_taps[circuit.Transformers.Name] = []
        for w in range(1, circuit.Transformers.NumWindings + 1):
            circuit.Transformers.Wdg = w
            operational_taps[circuit.Transformers.Name].append(float(circuit.Transformers.Tap))
        i = circuit.Transformers.Next
    engine.Text.Command = "set controlmode=off"
    for name in circuit.Transformers.AllNames:
        engine.Text.Command = f"edit transformer.{name} %noloadloss=0 %imag=0 ppm_antifloat=0"
        for w in (1, 2):
            engine.Text.Command = f"edit transformer.{name} wdg={w} tap=1"
    circuit.Solution.Solve()

    def element():
        """保存端子、相别及以西门子为单位的复导纳交错数组。"""
        e = circuit.ActiveCktElement
        return dict(buses=list(e.BusNames), nphases=int(e.NumPhases),
                    nconds=int(e.NumConductors), nterminals=int(e.NumTerminals),
                    node_order=list(e.NodeOrder), yprim=list(e.Yprim))

    buses = []
    for name in circuit.AllBusNames:
        circuit.SetActiveBus(name)
        buses.append(dict(name=name, kv_ln=float(circuit.ActiveBus.kVBase),
                          nodes=list(circuit.ActiveBus.Nodes)))
    lines = []
    i = circuit.Lines.First
    while i:
        row = element()
        row.update(name=circuit.Lines.Name, length=float(circuit.Lines.Length),
                   units=int(circuit.Lines.Units), linecode=circuit.Lines.LineCode,
                   r1=float(circuit.Lines.R1), x1=float(circuit.Lines.X1),
                   rmatrix=list(circuit.Lines.Rmatrix), xmatrix=list(circuit.Lines.Xmatrix))
        lines.append(row)
        i = circuit.Lines.Next
    transformers = []
    i = circuit.Transformers.First
    while i:
        row = element()
        row.update(name=circuit.Transformers.Name, xhl_percent=float(circuit.Transformers.Xhl), windings=[])
        for w in range(1, circuit.Transformers.NumWindings + 1):
            circuit.Transformers.Wdg = w
            row["windings"].append(dict(kv=float(circuit.Transformers.kV),
                                        kva=float(circuit.Transformers.kVA),
                                        r_percent=float(circuit.Transformers.R),
                                        tap=float(circuit.Transformers.Tap),
                                        is_delta=bool(circuit.Transformers.IsDelta)))
        transformers.append(row)
        i = circuit.Transformers.Next
    result = dict(source_file=str(master), engine_version=engine.Version, buses=buses,
                  lines=lines, transformers=transformers, operational_taps=operational_taps,
                  converged=bool(circuit.Solution.Converged),
                  audit_edits="controlmode=off; transformer taps=1; no-load loss, magnetizing current and antifloat=0")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    """以独立子进程隔离本机 COM 退出挂起，完成文件为成功依据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    master, output = args.master.resolve(), args.output.resolve()
    if args.worker:
        export(master, output)
        return
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--master", str(master), "--output", str(output)]
    with output.with_suffix(".log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=0x08000000)
        deadline = time.monotonic() + 45
        try:
            while time.monotonic() < deadline:
                if output.exists():
                    json.loads(output.read_text(encoding="utf-8"))
                    print("Export complete:", output, flush=True)
                    return
                if process.poll() is not None:
                    raise RuntimeError(f"参数导出失败，查看 {output.with_suffix('.log')}")
                time.sleep(.2)
            raise TimeoutError("参数导出未在 45 秒内完成")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
