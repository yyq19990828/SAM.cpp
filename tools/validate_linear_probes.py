#!/usr/bin/env python3
"""Compatibility entry point for the grouped tools.

The historical flat module path keeps working for imports and scripts; the
implementation lives in tools.validation.validate_linear_probes.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.validation import validate_linear_probes as _implementation  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Importing the historical module returns the real implementation module so
# module-level patching and attribute access keep their existing semantics.
sys.modules[__name__] = _implementation
