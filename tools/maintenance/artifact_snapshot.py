"""Hash-bound producer provenance, independent of quality policies."""

from importlib.metadata import version
from pathlib import Path
import csv
import io
import os
import platform
import re
import shutil
import subprocess

from tools.convert.sam3_artifacts import artifact_path, sha256_file


def runtime_environment(backend):
    """Bind the host and visible device inventory, without volatile utilization."""
    value = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
             "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        value["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                             if line.startswith("model name")), value["cpu"])
    if backend == "cuda":
        fields = ("index", "uuid", "name", "driver_version", "compute_cap", "memory.total")
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                check=True, text=True, capture_output=True)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if not rows or any(len(row) != len(fields) for row in rows):
            raise ValueError("cannot establish the CUDA device inventory")
        value["cuda_devices"] = [dict(zip(fields, [field.strip() for field in row])) for row in rows]
        value["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value

def repository_root():
    return Path(__file__).resolve().parents[2]

def source_snapshot():
    # Freeze shared contracts as well as the entry point. Future mutations are
    # allowed only before a new campaign, not while its batches are running.
    # Compiled src/ implementations, the grouped tools, apps/support and the
    # vendor sources the probes compile belong to the same identity.
    root = repository_root()
    paths = {root / "CMakeLists.txt", root / "tools/CMakeLists.txt",
             root / "apps/CMakeLists.txt", root / "support/image_io/CMakeLists.txt",
             root / "tests/CMakeLists.txt"}
    for pattern in ("tools/**/*.py", "tools/**/*.cpp", "tools/**/*.hpp", "tools/**/*.cu",
                    "tools/**/*.json", "tools/**/*.lock",
                    "apps/**/*.cpp", "apps/**/*.hpp", "apps/**/CMakeLists.txt",
                    "support/**/*.cpp", "support/**/*.hpp", "support/**/CMakeLists.txt",
                    "third_party/stb/*.h",
                    "include/**/*.hpp",
                    "src/**/*.cpp", "src/**/*.hpp", "src/**/*.cu", "src/**/CMakeLists.txt",
                    "tests/**/*.py", "tests/**/*.hpp", "tests/**/*.json", "tests/**/*.txt",
                    "tests/**/*.inc", "tests/**/*.cmake", "tests/**/CMakeLists.txt",
                    "cmake/**/*.cmake", "cmake/**/*.in", "cmake/**/*.patch"):
        paths.update(root.glob(pattern))
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}

def archive_sources(directory, identities):
    root = repository_root()
    archived = {}
    for name, expected in identities.items():
        path = Path(name)
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts[0] not in ("include", "src", "tools", "cmake", "tests", "apps",
                                     "support", "third_party") and relative.as_posix() != "CMakeLists.txt":
            continue
        if path.suffix not in (".py", ".hpp", ".h", ".cpp", ".cu", ".cmake", ".in", ".patch", ".json",
                               ".txt", ".inc", ".lock"):
            continue
        destination = directory / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256_file(destination) != expected:
            raise ValueError("producer source changed while archiving")
        archived[name] = destination.relative_to(directory).as_posix()
    return archived

def verify_export_artifacts(directory, manifest):
    for name, digest in manifest["artifact_sha256"].items():
        archived = manifest.get("archived_sources", {}).get(name)
        path = artifact_path(directory, archived) if archived is not None else Path(name)
        if sha256_file(path) != digest:
            raise ValueError(f"export artifact changed: {name}")

def native_snapshot(binary, model):
    manifest = model.with_suffix(model.suffix + ".manifest.json")
    paths = {binary.resolve(), model.resolve(), manifest.resolve()}
    linked = subprocess.run(["ldd", str(binary)], check=True, text=True, capture_output=True).stdout
    if "not found" in linked:
        raise ValueError("native export has unresolved linked libraries")
    for line in linked.splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if match:
            paths.add(Path(match.group(1)))
    return {str(path): sha256_file(path) for path in sorted(paths)}

def packages():
    return {name: version(name) for name in ("numpy", "pillow", "pycocotools", "torch")}
