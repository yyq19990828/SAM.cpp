#!/usr/bin/env python3
"""Compatibility entry point for the grouped tools.

The historical flat module path keeps working for imports and scripts; the
implementation lives in tools.quantize.cache_quantization.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(_ROOT / "tools/quantize/cache_quantization.py"), run_name="__main__")
else:
    from tools.quantize import cache_quantization as _implementation  # noqa: E402
    sys.modules[__name__] = _implementation
