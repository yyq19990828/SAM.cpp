"""Exercise public Python commands without an inherited repository import path."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = {
    "benchmark": (
        "benchmark_video", "generate_video_benchmark",
        "generate_video_cases", "prepare_linear_probe",
        "qualify_video_benchmark",
        "quantization_benchmark",
    ),
    "convert": ("convert_sam3",),
    "maintenance": ("archive_validation",),
    "quantize": (
        "export_calibration", "prepare_coco_calibration", "screen_runtime_quantization",
        "study_activation_quantization",
    ),
    "validation": (
        "evaluate_coco_screening", "export_memory_selection_goldens",
        "export_quantized_reference", "export_reference",
        "export_video_reference", "prepare_coco_acceptance",
        "prepare_reference_source", "validate_image", "validate_linear_probes",
        "validate_video", "verify_fp8_dots", "verify_int8_dots",
        "verify_linear_probe", "verify_precision_f16", "verify_precision_q8_cache",
        "verify_q8_mmvq_dots", "verify_q8_rhs_staging", "verify_q8_mmq_rhs_staging",
        "verify_q8_mmq_dots",
    ),
    "visualization": ("render_image_comparison",),
}


class ToolEntrypointChecks(unittest.TestCase):
    def help(self, arguments, directory):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-B", *arguments, "--help"], cwd=directory,
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        return result.stdout

    def test_grouped_commands_start_outside_repository(self):
        # This also covers export_reference.py, launched as a child script by
        # export_quantized_reference without inheriting its parent's sys.path.
        with tempfile.TemporaryDirectory() as temporary:
            for group, commands in COMMANDS.items():
                for command in commands:
                    with self.subTest(command=f"{group}/{command}"):
                        self.help([str(ROOT / "tools" / group / f"{command}.py")], temporary)

    def test_legacy_and_module_entrypoints_keep_the_same_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            for group, command in (("convert", "convert_sam3"), ("validation", "validate_image"),
                                   ("validation", "export_quantized_reference")):
                with self.subTest(command=command):
                    grouped = self.help([str(ROOT / "tools" / group / f"{command}.py")], temporary)
                    legacy = self.help([str(ROOT / "tools" / f"{command}.py")], temporary)
                    module = self.help(["-m", f"tools.{group}.{command}"], ROOT)
                    self.assertEqual(grouped, legacy)
                    self.assertEqual(grouped, module)


if __name__ == "__main__":
    unittest.main()
