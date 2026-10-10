"""Video benchmark source receipts and mutation rejection, with simulated inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.benchmark import benchmark_video
from tools.convert.sam3_artifacts import read_json, sha256_file, write_json


class VideoBenchmarkChecks(unittest.TestCase):
    def fixture(self, root):
        project = root / "project"
        # Minimal compiled repository, including the old benchmark's dependencies.
        sources = (
            "CMakeLists.txt", "src/CMakeLists.txt", "src/api/model.cpp",
            "src/models/sam3/model.hpp", "src/runtime/ggml/kernel.cu", "include/sam/sam.hpp",
            "apps/CMakeLists.txt", "apps/video/main.cpp", "support/image_io/CMakeLists.txt",
            "support/image_io/image_io.cpp", "tools/CMakeLists.txt", "tests/CMakeLists.txt",
            "cmake/install.cmake", "cmake/ggml.cmake", "cmake/prepare_ggml.cmake",
            "cmake/patches/ggml-precise-metal.patch", "cmake/patches/ggml-precise-cuda.patch",
            "cmake/patches/ggml-short-dot-cuda.patch",
            "tools/convert/sam3_tensor_schema.json", "tools/convert/sam3_artifacts.py",
            "tools/convert/sam3_gguf.py", "tools/benchmark/benchmark_video.py",
            "tools/validation/validate_video.py", "tools/validation/validate_image.py",
        )
        for name in sources:
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original source\n")
        args = SimpleNamespace(
            build_dir=root / "build", model=root / "model.gguf", reference=root / "reference",
            validation=root / "validation.json", fixture_manifest=root / "fixture.json",
            qualification=root / "qualification.json", recipe_script=root / "recipe.py",
            output=root / "output", backend="cpu", threads=4, workload="one-object",
            timeout_seconds=10,
        )
        executable = args.build_dir / "examples/sam_video"
        executable.parent.mkdir(parents=True)
        executable.write_text("simulated executable\n")
        args.model.write_text("simulated weights\n")
        write_json(args.model.with_suffix(".gguf.manifest.json"),
                   {"output": {"sha256": sha256_file(args.model)}})
        args.reference.mkdir()
        reference = {"cases": [{"id": str(i)} for i in range(5)], "checkpoint": {"sha256": "a" * 64}}
        write_json(args.reference / "manifest.json", reference)
        artifacts = {str(path): sha256_file(path) for path in (executable, args.model)}
        write_json(args.validation, {
            "passed": True, "eligible_for_milestone": True, "precision": "f32", "backend": "cpu",
            "model_sha256": sha256_file(args.model), "binary_sha256": sha256_file(executable),
            "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
            "run_artifact_sha256": artifacts,
            "cases": [{**case, "passed": True, "failures": [], "output_sha256": artifacts}
                      for case in reference["cases"]],
        })
        frames = root / "frames"
        frames.mkdir()
        for i in range(64):
            (frames / f"{i:06d}.png").write_bytes(b"fixture bytes")
        inputs = {str(path): sha256_file(path) for path in sorted(frames.glob("*.png"))}
        args.recipe_script.write_text("# frozen fixture recipe\n")
        write_json(args.fixture_manifest, {
            "complete": True, "frames": 64, "warmup_frames": 16, "measured_frames": 48,
            "prompt": "truck", "max_objects": 8, "width": 2, "height": 2,
            "source": next(iter(inputs)), "source_sha256": next(iter(inputs.values())),
            "generator_sha256": sha256_file(args.recipe_script),
            "workloads": [{"id": args.workload, "expected_objects": 1, "directory": "frames",
                           "input_sha256": list(inputs.values())}],
        })
        write_json(args.qualification, {
            "passed": True, "qualified_for_performance": True,
            "reference_kind": "official-original-performance-prefix", "declared_frame_count": 64,
            "prefix_frames": 17, "workload": args.workload, "expected_objects": 1,
            "fixture_manifest_sha256": sha256_file(args.fixture_manifest),
            "checkpoint_sha256": reference["checkpoint"]["sha256"], "input_sha256": inputs,
            "source_sha256": {str(args.recipe_script): sha256_file(args.recipe_script)},
            "records": [{"frame_index": i, "active_ids": [7], "removed": [],
                         "births": [{"id": 7}] if i == 0 else [], "visible": [{"id": 7, "area": 1}]}
                        for i in range(17)],
            "emitted": [{"frame_index": i, "emitted_after_frame": i + 14} for i in range(3)],
        })
        return args, project, reference, artifacts

    def test_compiled_sources_are_recorded_and_changes_during_inference_rejected(self):
        for mutate in (False, True):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                args, project, reference, artifacts = self.fixture(Path(temporary).resolve())
                implementation = project / "src/api/model.cpp"
                original_digest = sha256_file(implementation)

                def infer(*command, stdout, **kwargs):
                    (args.output / "run").mkdir()
                    write_json(args.output / "run/manifest.json", {"simulated": True})
                    if mutate:
                        implementation.write_text("changed while inference was running\n")
                    return Mock(returncode=0, poll=Mock(return_value=0))

                with patch.object(benchmark_video, "__file__", str(project / "tools/benchmark/benchmark_video.py")), \
                     patch.object(benchmark_video.sys, "platform", "linux"), \
                     patch("tools.maintenance.artifact_snapshot.repository_root", return_value=project), \
                     patch.object(benchmark_video, "check_provenance", return_value=("f32", reference, None)), \
                     patch.object(benchmark_video, "freeze_run_artifacts", return_value=artifacts), \
                     patch.object(benchmark_video, "conditions", return_value={}), \
                     patch.object(benchmark_video.subprocess, "Popen", side_effect=infer), \
                     patch.object(benchmark_video, "analyze_run", return_value={"first_output": {}}), \
                     patch.object(benchmark_video, "process_timing", return_value={}):
                    result = benchmark_video.run(args)
                report = read_json(args.output / "benchmark.json")
                self.assertEqual(result, 1 if mutate else 0)
                self.assertEqual(report["passed"], not mutate)
                self.assertEqual(report["complete"], not mutate)
                self.assertEqual(report["source_sha256"].get(str(implementation)), original_digest)
                for path in (project / "src/models/sam3/model.hpp", project / "src/runtime/ggml/kernel.cu",
                             project / "src/CMakeLists.txt", project / "cmake/install.cmake",
                             args.fixture_manifest, args.qualification, args.validation, args.recipe_script):
                    self.assertEqual(report["source_sha256"].get(str(path)), sha256_file(path))
                if mutate:
                    self.assertTrue(any("changed during the batch" in failure and str(implementation) in failure
                                        for failure in report["failures"]), report["failures"])


if __name__ == "__main__":
    unittest.main()
