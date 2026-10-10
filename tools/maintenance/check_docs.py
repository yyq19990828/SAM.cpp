#!/usr/bin/env python3
"""Check bilingual measurement/artifact tables and local Markdown links."""

from pathlib import Path
import os
import re
import subprocess
from urllib.parse import unquote, urlsplit

# Historical plans keep their original path narrative. Links from those
# documents still resolve through this explicit compiled-library migration
# mapping; unmapped missing targets keep failing the check.
MOVED_SOURCE_PATHS = {
    "examples/CMakeLists.txt": "apps/CMakeLists.txt",
    "examples/image.cpp": "apps/image/image.cpp",
    "examples/image_io.cpp": "support/image_io/image_io.cpp",
    "examples/image_io.hpp": "support/image_io/image_io.hpp",
    "examples/image_support.hpp": "support/image_io/image_support.hpp",
    "examples/stb/stb_image.h": "third_party/stb/stb_image.h",
    "examples/stb/stb_image_write.h": "third_party/stb/stb_image_write.h",
    "examples/video.cpp": "apps/video/video.cpp",
    "include/sam/internal/models/sam3/execution.hpp": "src/models/sam3/execution.hpp",
    "include/sam/internal/models/sam3/tracking/execution.hpp": "src/models/sam3/video/execution.hpp",
    "include/sam/internal/models/sam3/vision.hpp": "src/models/sam3/vision.hpp",
    "include/sam/internal/runtime/ggml/backend.hpp": "src/runtime/ggml/backend.hpp",
    "include/sam/internal/runtime/ggml/graph.hpp": "src/runtime/ggml/graph.hpp",
    "include/sam/internal/runtime/ggml/runtime.hpp": "src/runtime/ggml/runtime.hpp",
    "tests/backend_test_support.hpp": "tests/runtime/ggml/backend_test_support.hpp",
    "tests/test_backend.cpp": "tests/runtime/ggml/test_backend.cpp",
    "tests/test_contracts.cpp": "tests/api/test_contracts.cpp",
    "tests/test_cuda_concat.cpp": "tests/runtime/ggml/test_cuda_concat.cpp",
    "tests/test_cuda_copy.cpp": "tests/runtime/ggml/test_cuda_copy.cpp",
    "tests/test_cuda_numerics.cpp": "tests/runtime/ggml/test_cuda_numerics.cpp",
    "tests/test_graph.cpp": "tests/runtime/ggml/test_graph.cpp",
    "tests/test_graph_profile.cpp": "tests/runtime/ggml/test_graph_profile.cpp",
    "tests/test_graph_workspace.cpp": "tests/runtime/ggml/test_graph_workspace.cpp",
    "tests/test_host_tensor.cpp": "tests/runtime/ggml/test_host_tensor.cpp",
    "tests/test_image.cpp": "tests/models/test_image.cpp",
    "tests/test_image_cli.cpp": "tests/models/test_image_cli.cpp",
    "tests/test_precision.cpp": "tests/runtime/ggml/test_precision.cpp",
    "tests/test_precision_output.cpp": "tests/runtime/ggml/test_precision_output.cpp",
    "tests/test_prepare_ggml.cmake": "tests/integration/test_prepare_ggml.cmake",
    "tests/test_quantization.cpp": "tests/runtime/ggml/test_quantization.cpp",
    "tests/test_rope.cpp": "tests/runtime/ggml/test_rope.cpp",
    "tests/test_session.cpp": "tests/models/test_session.cpp",
    "tests/test_task_contract.cpp": "tests/api/test_task_contract.cpp",
    "tests/test_tracking.cpp": "tests/models/test_tracking.cpp",
    "tests/test_two_tu_main.cpp": "tests/api/test_two_tu_main.cpp",
    "tests/test_video.cpp": "tests/models/test_video.cpp",
    "tests/test_video_long_session.cpp": "tests/models/test_video_long_session.cpp",
    "tests/test_video_session.cpp": "tests/models/test_video_session.cpp",
    "tests/test_window.cpp": "tests/runtime/ggml/test_window.cpp",
    "tests/two_tu_other.cpp": "tests/api/two_tu_other.cpp",
    "tools/archive_validation.py": "tools/maintenance/archive_validation.py",
    "tools/benchmark_precision.py": "tools/archive/precision_v2_v3/benchmark_precision.py",
    "tools/benchmark_video.py": "tools/benchmark/benchmark_video.py",
    "tools/cache_codec_probe.cpp": "tools/quantize/cache_codec_probe.cpp",
    "tools/cache_image_probe.cpp": "tools/quantize/cache_image_probe.cpp",
    "tools/cache_quantization.py": "tools/quantize/cache_quantization.py",
    "tools/calibration.py": "tools/quantize/calibration.py",
    "tools/check_docs.py": "tools/maintenance/check_docs.py",
    "tools/coco_acceptance.py": "tools/archive/precision_v2_v3/coco_acceptance.py",
    "tools/convert_sam3.py": "tools/convert/convert_sam3.py",
    "tools/cuda_linear_probe.cu": "tools/benchmark/cuda_linear_probe.cu",
    "tools/evaluate_coco_screening.py": "tools/validation/evaluate_coco_screening.py",
    "tools/evaluate_precision.py": "tools/archive/precision_v2_v3/evaluate_precision.py",
    "tools/export_calibration.py": "tools/quantize/export_calibration.py",
    "tools/export_memory_selection_goldens.py": "tools/validation/export_memory_selection_goldens.py",
    "tools/export_precision_outputs.py": "tools/archive/precision_v2_v3/export_precision_outputs.py",
    "tools/export_quantized_reference.py": "tools/validation/export_quantized_reference.py",
    "tools/export_reference.py": "tools/validation/export_reference.py",
    "tools/export_video_reference.py": "tools/validation/export_video_reference.py",
    "tools/freeze_precision_campaign.py": "tools/archive/precision_v2_v3/freeze_precision_campaign.py",
    "tools/generate_video_benchmark.py": "tools/benchmark/generate_video_benchmark.py",
    "tools/generate_video_cases.py": "tools/benchmark/generate_video_cases.py",
    "tools/ggml_cuda_cache_bridge.cu": "tools/quantize/ggml_cuda_cache_bridge.cu",
    "tools/ggml_linear_probe.cpp": "tools/benchmark/ggml_linear_probe.cpp",
    "tools/graph_profile.hpp": "tools/benchmark/graph_profile.hpp",
    "tools/linear_probe.hpp": "tools/benchmark/linear_probe.hpp",
    "tools/precision_acceptance.py": "tools/archive/precision_v2_v3/precision_acceptance.py",
    "tools/precision_artifacts.py": "tools/archive/precision_v2_v3/precision_artifacts.py",
    "tools/precision_benchmark_probe.cpp": "tools/benchmark/precision_benchmark_probe.cpp",
    "tools/precision_image_probe.cpp": "tools/validation/precision_image_probe.cpp",
    "tools/precision_output.hpp": "tools/validation/precision_output.hpp",
    "tools/precision_performance.py": "tools/archive/precision_v2_v3/precision_performance.py",
    "tools/prepare_coco_acceptance.py": "tools/validation/prepare_coco_acceptance.py",
    "tools/prepare_coco_calibration.py": "tools/quantize/prepare_coco_calibration.py",
    "tools/prepare_linear_probe.py": "tools/benchmark/prepare_linear_probe.py",
    "tools/prepare_precision_inputs.py": "tools/archive/precision_v2_v3/prepare_precision_inputs.py",
    "tools/prepare_precision_performance.py": "tools/archive/precision_v2_v3/prepare_precision_performance.py",
    "tools/prepare_reference_source.py": "tools/validation/prepare_reference_source.py",
    "tools/profile_graph.cpp": "tools/benchmark/profile_graph.cpp",
    "tools/qualify_video_benchmark.py": "tools/benchmark/qualify_video_benchmark.py",
    "tools/quantize_rows.cpp": "tools/quantize/quantize_rows.cpp",
    "tools/render_image_comparison.py": "tools/visualization/render_image_comparison.py",
    "tools/runtime_quantization.py": "tools/quantize/runtime_quantization.py",
    "tools/sam3_artifacts.py": "tools/convert/sam3_artifacts.py",
    "tools/sam3_gguf.py": "tools/convert/sam3_gguf.py",
    "tools/sam3_tensor_schema.json": "tools/convert/sam3_tensor_schema.json",
    "tools/screen_runtime_quantization.py": "tools/quantize/screen_runtime_quantization.py",
    "tools/study_activation_quantization.py": "tools/quantize/study_activation_quantization.py",
    "tools/test_cache_quantization.py": "tests/tools/test_cache_quantization.py",
    "tools/test_calibration.py": "tests/tools/test_calibration.py",
    "tools/test_coco_acceptance.py": "tests/tools/test_coco_acceptance.py",
    "tools/test_coco_screening.py": "tests/tools/test_coco_screening.py",
    "tools/test_modular_quantization_tools.py": "tests/tools/test_modular_quantization_tools.py",
    "tools/test_precision_acceptance.py": "tests/tools/test_precision_acceptance.py",
    "tools/test_precision_artifacts.py": "tests/tools/test_precision_artifacts.py",
    "tools/test_precision_dataset.py": "tests/tools/test_precision_dataset.py",
    "tools/test_precision_performance.py": "tests/tools/test_precision_performance.py",
    "tools/test_precision_regression.py": "tests/tools/test_precision_regression.py",
    "tools/test_quantization_tools.py": "tests/tools/test_quantization_tools.py",
    "tools/test_runtime_quantization.py": "tests/tools/test_runtime_quantization.py",
    "tools/test_tools.py": "tests/tools/test_tools.py",
    "tools/validate_image.py": "tools/validation/validate_image.py",
    "tools/validate_linear_probes.py": "tools/validation/validate_linear_probes.py",
    "tools/validate_precision_regression.py": "tools/archive/precision_v2_v3/validate_precision_regression.py",
    "tools/validate_video.py": "tools/validation/validate_video.py",
    "tools/verify_linear_probe.py": "tools/validation/verify_linear_probe.py",
    "tools/verify_precision_f16.py": "tools/validation/verify_precision_f16.py",
}


def doc_target_path(path, target):
    # Check the linked path itself: an ignored symlink to a public source is
    # still absent in a clean checkout. Normalize '..' without following it.
    return Path(os.path.abspath(path.parent / unquote(target.split("#", 1)[0])))


def resolve_doc_target(root, path, target):
    direct = doc_target_path(path, target)
    try:
        relative = direct.relative_to(root).as_posix()
    except ValueError:
        return None
    if direct.is_file():
        return direct
    replacement = MOVED_SOURCE_PATHS.get(relative)
    if replacement is None:
        return None
    moved = root / replacement
    return moved if moved.is_file() else None


def local_doc_targets(text):
    # Inline links/images and reference definitions share the same destination
    # rules. Angle brackets permit spaces; optional link titles are not paths.
    patterns = (r"\]\(\s*(?:<([^>\n]+)>|([^\s)]+))",
                r"(?m)^[ \t]{0,3}\[[^]\n]+\]:[ \t]*(?:<([^>\n]+)>|([^\s]+))")
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            target = match.group(1) or match.group(2)
            if not target.startswith(("#", "//")) and not urlsplit(target).scheme:
                yield target


def ignored_doc_targets(root, destinations):
    if not destinations:
        return set()
    result = subprocess.run(["git", "check-ignore", "--stdin", "-z"], cwd=root,
                            input="\0".join(sorted(destinations)) + "\0",
                            capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"cannot check Git ignore rules: {result.stderr.strip()}")
    return set(result.stdout.split("\0")) - {""}


def table_facts(text):
    facts = []
    for line in text.splitlines():
        if line.startswith("|") and (re.search(r"\d+\.\d+|[0-9a-f]{64}", line)):
            facts.append(re.findall(r"[0-9a-f]{64}|(?<!\w)\d+(?:\.\d+)?", line))
    return facts


def check(root):
    root = Path(root).resolve()
    bilingual = ["BENCHMARK", "MODEL_ZOO", "docs/quantization", "docs/visual-examples"]
    bilingual.extend(path.relative_to(root).with_suffix("").as_posix()
                     for path in sorted((root / "benchmarks").glob("*.md"))
                     if not path.stem.endswith("_zh") and path.name != "README.md")
    for name in bilingual:
        english, chinese = root / (name + ".md"), root / (name + "_zh.md")
        if not chinese.is_file():
            raise ValueError(f"missing Chinese measurement document: {name}_zh.md")
        if table_facts(english.read_text()) != table_facts(chinese.read_text()):
            raise ValueError(f"bilingual measurement/artifact table differs: {name}")
    # Check source documents (including newly written ones) without walking
    # generated dependency docs, model downloads or virtual environments.
    inventory = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
        cwd=root, text=True)
    files = sorted({root / name for name in inventory.split("\0") if name and (root / name).is_file()})
    links = []
    for path in files:
        for target in local_doc_targets(path.read_text()):
            try:
                relative = doc_target_path(path, target).relative_to(root).as_posix()
            except ValueError:
                raise ValueError(f"missing local doc target: {path}: {target}")
            links.append((path, target, relative))
    ignored = ignored_doc_targets(root, {relative for _, _, relative in links})
    for path, target, relative in links:
        if relative in ignored:
            raise ValueError(f"ignored local doc target: {path}: {target}; use a plain archive path instead")
        if resolve_doc_target(root, path, target) is None:
            raise ValueError(f"missing local doc target: {path}: {target}")
    return len(files)


if __name__ == "__main__":
    print(f"PASS: bilingual tables and local links in {check(Path(__file__).resolve().parents[2])} documents")
