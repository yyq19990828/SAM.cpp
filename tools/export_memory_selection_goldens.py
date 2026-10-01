#!/usr/bin/env python3
"""Export weight-free selector goldens from the pinned Meta functions.

Only the hash-verified select_closest_cond_frames and frame_filter functions
are executed. No package imports, model construction, or checkpoint access.
"""

import argparse
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

from sam3_artifacts import sha256_file


SOURCES = {
    "sam3/model/sam3_tracker_utils.py": "dc5fdeba2d4416f273394a9bd4450dff050608db6009a4546ca68adbaa24a640",
    "sam3/model/sam3_tracker_base.py": "b2b52409c002e1590262375aa794f8ab67e7476f42f8fee41a76de0c14aa62e2",
}


def function_from_source(path, name, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"pinned selector source hash differs: {path}")
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f"expected one pinned function: {name}")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def export(source, output):
    closest = function_from_source(source / "sam3/model/sam3_tracker_utils.py", "select_closest_cond_frames",
                                   SOURCES["sam3/model/sam3_tracker_utils.py"])
    frame_filter = function_from_source(source / "sam3/model/sam3_tracker_base.py", "frame_filter",
                                        SOURCES["sam3/model/sam3_tracker_base.py"])
    parameters = SimpleNamespace(max_obj_ptrs_in_encoder=16, mf_threshold=0.01)
    lines = ["# Meta 2345a4ad109ac29c569da749c91d84f10dc08c40 unpruned forward selector goldens.",
             "# scenario frame spatial_count (frame temporal_position)* pointer_count (frame relative_position)*"]
    for scenario, (birth, count) in enumerate(((0, 1000), (0, 64), (37, 1000))):
        conditioning, non_conditioning = {}, {}
        for frame in range(birth, count):
            if frame - birth == 11:
                recent = [index for index in conditioning if frame - 7 <= index < frame]
                for index in recent:
                    non_conditioning[index] = conditioning.pop(index)
            if scenario == 2 and frame - birth == 12:
                for record in non_conditioning.values():
                    record["eff_iou_score"] = 0.0
            if frame > birth:
                selected, unselected = closest(frame, conditioning, 4, keep_first_cond_frame=False)
                valid = frame_filter(parameters, {"cond_frame_outputs": conditioning,
                                                  "non_cond_frame_outputs": non_conditioning}, False, frame, count, 1)
                spatial = [(index, 0) for index in selected]
                pointers = [(index, frame - index) for index in selected]
                available = non_conditioning.keys() | unselected.keys()
                for position in range(1, 7):
                    relative = 7 - position
                    if relative <= len(valid) and valid[-relative] in available:
                        spatial.append((valid[-relative], position))
                for relative in range(1, min(count, 16)):
                    if relative >= len(valid):
                        break
                    if valid[-relative] in available:
                        pointers.append((valid[-relative], relative))
                values = [scenario, frame, len(spatial)]
                values += [value for entry in spatial for value in entry]
                values += [len(pointers)] + [value for entry in pointers for value in entry]
                lines.append(" ".join(map(str, values)))
            record = {"eff_iou_score": 0.0 if scenario == 1 and frame > 1 else (frame * 37 % 31) / 100.0}
            if (frame - birth) % 16 == 0 or (frame - birth < 15 and (frame - birth) % 4 == 0):
                conditioning[frame] = record
            else:
                non_conditioning[frame] = record
    data = ("\n".join(lines) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(data)
    print(f"Wrote {len(lines) - 2} selector cases; SHA-256 {hashlib.sha256(data).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        export(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"selector export failed: {error}\n")


if __name__ == "__main__":
    main()
