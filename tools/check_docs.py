#!/usr/bin/env python3
"""Check bilingual measurement/artifact tables and local Markdown links."""

from pathlib import Path
import re


def table_facts(text):
    facts = []
    for line in text.splitlines():
        if line.startswith("|") and (re.search(r"\d+\.\d+|[0-9a-f]{64}", line)):
            facts.append(re.findall(r"[0-9a-f]{64}|(?<!\w)\d+(?:\.\d+)?", line))
    return facts


def check(root):
    root = Path(root)
    for name in ("BENCHMARK", "MODEL_ZOO"):
        english, chinese = root / (name + ".md"), root / (name + "_zh.md")
        if table_facts(english.read_text()) != table_facts(chinese.read_text()):
            raise ValueError(f"bilingual measurement/artifact table differs: {name}")
    files = [root / name for name in ("README.md", "BENCHMARK.md", "BENCHMARK_zh.md", "MODEL_ZOO.md", "MODEL_ZOO_zh.md")]
    files += list((root / "docs").rglob("*.md"))
    for path in files:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            destination = target.split("#")[0]
            if not (path.parent / destination).is_file():
                raise ValueError(f"missing local doc target: {path}: {target}")
    return len(files)


if __name__ == "__main__":
    print(f"PASS: bilingual tables and local links in {check(Path(__file__).resolve().parents[1])} documents")
