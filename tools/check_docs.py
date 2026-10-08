#!/usr/bin/env python3
"""Check bilingual measurement/artifact tables and local Markdown links."""

from pathlib import Path
import os
import re

# Historical plans keep their original path narrative. Links from those
# documents still resolve through this explicit compiled-library migration
# mapping; unmapped missing targets keep failing the check.
MOVED_SOURCE_PATHS = {
    "include/sam/internal/models/sam3/execution.hpp": "src/models/sam3/execution.hpp",
    "include/sam/internal/models/sam3/vision.hpp": "src/models/sam3/vision.hpp",
    "include/sam/internal/models/sam3/tracking/execution.hpp": "src/models/sam3/video/execution.hpp",
    "include/sam/internal/runtime/ggml/graph.hpp": "src/runtime/ggml/graph.hpp",
    "include/sam/internal/runtime/ggml/runtime.hpp": "src/runtime/ggml/runtime.hpp",
    "include/sam/internal/runtime/ggml/backend.hpp": "src/runtime/ggml/backend.hpp",
}


def resolve_doc_target(root, path, target):
    destination = target.split("#")[0]
    direct = path.parent / destination
    if direct.is_file():
        return direct
    try:
        relative = os.path.normpath(str(direct))
        relative = Path(relative).relative_to(root).as_posix()
    except ValueError:
        return None
    replacement = MOVED_SOURCE_PATHS.get(relative)
    if replacement is None:
        return None
    moved = root / replacement
    return moved if moved.is_file() else None


def table_facts(text):
    facts = []
    for line in text.splitlines():
        if line.startswith("|") and (re.search(r"\d+\.\d+|[0-9a-f]{64}", line)):
            facts.append(re.findall(r"[0-9a-f]{64}|(?<!\w)\d+(?:\.\d+)?", line))
    return facts


def check(root):
    root = Path(root)
    for name in ("BENCHMARK", "MODEL_ZOO", "docs/quantization", "docs/visual-examples"):
        english, chinese = root / (name + ".md"), root / (name + "_zh.md")
        if table_facts(english.read_text()) != table_facts(chinese.read_text()):
            raise ValueError(f"bilingual measurement/artifact table differs: {name}")
    files = [root / name for name in ("README.md", "BENCHMARK.md", "BENCHMARK_zh.md", "MODEL_ZOO.md", "MODEL_ZOO_zh.md")]
    files += list((root / "docs").rglob("*.md"))
    for path in files:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            if resolve_doc_target(root, path, target) is None:
                raise ValueError(f"missing local doc target: {path}: {target}")
    return len(files)


if __name__ == "__main__":
    print(f"PASS: bilingual tables and local links in {check(Path(__file__).resolve().parents[1])} documents")
