#!/usr/bin/env python3
"""Read-only discovery of Mac profiling tools. Never records, replays or launches an app."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


TOOLS = ("xctrace", "gpucapture", "gpudebug", "metalperftrace")


def run(argv):
    """Use argument vectors only; retain errors as discovery data."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:12000],
            "stderr": result.stderr.strip()[:3000],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "returncode": None, "error": str(error)}


def probe(include_templates=False, include_help=False):
    report = {
        "schema_version": 1,
        "kind": "read-only-mac-metal-tool-discovery",
        "platform": sys.platform,
        "architecture": platform.machine(),
        "recording_started": False,
        "replay_started": False,
        "tools": {},
        "diagnostic_environment_keys_present": sorted(
            key for key in os.environ
            if key.startswith("GGML_METAL_") or key in {
                "MTL_CAPTURE_ENABLED", "METAL_CAPTURE_ENABLED",
                "MTLCAPTURE_WAIT_FOR_SIGNAL", "MTL_DEBUG_LAYER",
                "MTL_SHADER_VALIDATION", "MTL_DEVICE_WRAPPER_TYPE",
            }
        ),
        "not_tested": ["recording permissions", "GPU counters", "trace replay", "shader source mapping"],
    }
    if sys.platform != "darwin":
        report["limitation"] = "This probe supports macOS; no tool commands were run."
        return report
    report["macos"] = run(["/usr/bin/sw_vers"])
    report["developer_directory"] = run(["/usr/bin/xcode-select", "-p"])
    report["xcode_version"] = run(["/usr/bin/xcodebuild", "-version"])
    xcrun = shutil.which("xcrun")
    if not xcrun:
        report["limitation"] = "xcrun is unavailable; no profiling tool was invoked."
        return report
    for name in TOOLS:
        discovery = run([xcrun, "--find", name])
        found = discovery.get("returncode") == 0 and bool(discovery.get("stdout"))
        item = {"available": found, "discovery": discovery}
        if found and include_help:
            argv = [xcrun, name, "help", "record"] if name == "xctrace" else [xcrun, name, "--help"]
            item["help"] = run(argv)
        report["tools"][name] = item
    if include_templates and report["tools"]["xctrace"]["available"]:
        report["templates"] = run([xcrun, "xctrace", "list", "templates"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", action="store_true", help="also list installed xctrace templates")
    parser.add_argument("--help-text", action="store_true", help="also read global tool help (xctrace record help)")
    args = parser.parse_args()
    print(json.dumps(probe(args.templates, args.help_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
