"""汇总故障相别有效性诊断结果，列出未生效的故障类型与母线组合。"""

import re
from pathlib import Path

path = Path(__file__).absolute().parents[1] / "logs" / "diagnose-fault-phase.txt"
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
current = ""
ineffective = []
effective_counts: dict[str, int] = {}
for line in lines:
    if line.startswith("==="):
        current = line.strip("= ").strip()
        effective_counts.setdefault(current, 0)
        continue
    match = re.search(r"bus\s+(\d+)\s+(\S+)\s+phases=\(([^)]*)\)\s+max\|dV\|=([0-9.eE+-]+)", line)
    if not match:
        continue
    bus, name, phases, delta = match.group(1), match.group(2), match.group(3), float(match.group(4))
    if delta < 1e-3:
        ineffective.append((current, bus, name, phases, delta))
    else:
        effective_counts[current] = effective_counts.get(current, 0) + 1
print("未生效的（故障类型, 母线）组合：")
for row in ineffective:
    print("  {:<10s} bus {:>2s} {:<9s} phases=({}) max|dV|={:.3e}".format(*row))
print()
print("各故障类型生效的母线数：", effective_counts)
