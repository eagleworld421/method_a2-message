"""GReF 的独立命令行入口，不改变既有 Node Marker 训练路径。"""

import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.training import parse_args, run


if __name__ == "__main__":
    run(parse_args())
