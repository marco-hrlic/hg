import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
ROOT = EVAL.parent
for p in (str(EVAL), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
