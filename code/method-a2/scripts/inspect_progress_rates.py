"""读取指定进度文件，核对实测速率随完成量的变化趋势。"""

import json
import sys
from pathlib import Path

path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).absolute().parents[1] / "logs" / "progress-smoke" / "progress.jsonl"
)
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
units = [record for record in records if record["kind"] == "unit"]
print("unit records:", len(units))
for index in (9, 49, 99, 149, 199, 249, 299, len(units) - 1):
    if index < len(units):
        record = units[index]
        print(
            "done={:4d} elapsed={:8.1f}s rate_s={} eta_s={} note={}".format(
                record["done"],
                record["elapsed_s"],
                record["rate_s"],
                record["eta_s"],
                record["note"],
            )
        )
print("stage_done elapsed:", [r["elapsed_s"] for r in records if r["kind"] == "stage_done"])
print("end elapsed:", [r["elapsed_s"] for r in records if r["kind"] == "end"])
