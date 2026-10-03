#!/usr/bin/env python3
"""Copy local validation evidence into an exclusive, hash-verified private bundle."""

import argparse
import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile

from sam3_artifacts import artifact_path, read_json, sha256_file, write_json


def copy_file(source, target):
    if sys.platform == "darwin":
        clone = ctypes.CDLL(None, use_errno=True).clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(target), 0) == 0:
            shutil.copystat(source, target)
            return "apfs-clone"
    shutil.copy2(source, target)
    return "copy"


def create_bundle(sources, output):
    sources = [Path(source).resolve(strict=True) for source in sources]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("archive destination already exists")
    for source in sources:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("archive source/destination overlap")
        if not (source.is_file() or source.is_dir()):
            raise ValueError("archive source is not a regular file or directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".validation-archive-", dir=output.parent))
    staging.chmod(0o700)
    report = {"schema_version": 1, "complete": False, "kind": "private-validation-byte-archive",
              "scope": "Historical evidence preservation, not new model acceptance or automatic baseline eligibility.",
              "roots": [], "files": {}, "bytes": 0, "copies": {}}
    try:
        for index, source in enumerate(sources):
            prefix = Path("data") / f"{index:02d}-{source.name}"
            report["roots"].append({"source": str(source), "bundle": prefix.as_posix(), "directory": source.is_dir()})
            paths = [source] if source.is_file() else sorted(source.rglob("*"))
            for path in paths:
                if path.is_symlink() and (path.is_dir() or not path.exists()):
                    raise ValueError(f"archive rejects directory/broken symlink: {path}")
                if not path.is_file():
                    continue
                target_name = prefix if source.is_file() else prefix / path.relative_to(source)
                target = staging / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = sha256_file(path)
                method = copy_file(path.resolve(), target)
                if sha256_file(target) != digest:
                    raise ValueError(f"archive copy hash differs: {path}")
                size = target.stat().st_size
                report["files"][target_name.as_posix()] = {"sha256": digest, "bytes": size}
                report["bytes"] += size
                report["copies"][method] = report["copies"].get(method, 0) + 1
            write_json(staging / "archive.json", report)
            print(f"Archived {source.name}: {len(report['files'])} files, {report['bytes']} bytes", flush=True)
        if not report["files"]:
            raise ValueError("archive contains no files")
        report["complete"] = True
        write_json(staging / "archive.json", report)
        # mkdir is the exclusive publication lock; rename into it cannot replace user data.
        output.mkdir(mode=0o700)
        os.rename(staging, output / "bundle")
        return output / "bundle" / "archive.json"
    except BaseException:
        write_json(staging / "archive.json", report)
        print(f"Incomplete archive preserved: {staging}", file=sys.stderr)
        raise


def verify_bundle(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / "archive.json")
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("complete") is not True or not isinstance(report.get("files"), dict) or not report["files"]):
        raise ValueError("archive is incomplete or empty")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("archive contains a symlink")
    actual = {path.relative_to(directory).as_posix() for path in paths if path.is_file()}
    if actual != set(report["files"]) | {"archive.json"}:
        raise ValueError("archive file inventory changed")
    if type(report.get("bytes")) is not int or report["bytes"] < 0:
        raise ValueError("archive byte total is invalid")
    total = 0
    for name, metadata in report["files"].items():
        path = artifact_path(directory, name)
        if path.is_symlink() or path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"archive file changed: {name}")
        total += path.stat().st_size
    if total != report["bytes"]:
        raise ValueError("archive byte total differs from files")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source", action="append", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            print(create_bundle(args.source, args.output))
        else:
            report = verify_bundle(args.bundle)
            print(f"PASS: {len(report['files'])} archived files, {report['bytes']} bytes")
    except (OSError, ValueError) as error:
        parser.exit(1, f"archive failed: {error}\n")


if __name__ == "__main__":
    main()
