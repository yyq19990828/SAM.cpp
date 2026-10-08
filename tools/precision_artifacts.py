#!/usr/bin/env python3
"""Compatibility entry point for the grouped tools.

The historical flat module path keeps working for imports and scripts; the
implementation lives in tools.maintenance.precision_artifacts.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(_ROOT / "tools/maintenance/precision_artifacts.py"), run_name="__main__")
else:
    from tools.maintenance import precision_artifacts as _implementation  # noqa: E402
    sys.modules[__name__] = _implementation
