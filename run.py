#!/usr/bin/env python3
"""進入點。`python run.py` 直接開 UI，其他子命令見 `python run.py --help`。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mt.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
