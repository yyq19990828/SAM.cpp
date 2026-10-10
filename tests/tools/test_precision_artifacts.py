"""Run completeness, immutable payloads, campaign boundaries and cache baselines."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.archive.precision_v2_v3.evaluate_precision import load_run, read_outputs, same_inputs
from tools.archive.precision_v2_v3.precision_acceptance import GATES_SHA256, canonical_hash
from tools.archive.precision_v2_v3.precision_artifacts import (archive_sources, campaign_check, claim_evaluation, phase_samples,
                                 source_snapshot, verify_export_artifacts)
from tools.validation.prepare_coco_acceptance import make_dataset
from tools.convert.sam3_artifacts import sha256_file, write_json
from tests.tools import test_precision_dataset
from tests.tools.test_precision_acceptance import output_fixture


class PrecisionArtifactChecks(unittest.TestCase):
    def fixture(self, root):
        annotations, used = test_precision_dataset.PrecisionDatasetChecks().fixture(root)
        dataset = make_dataset(annotations, root, [used], set(), 3, 3)
        directory = root / "run"
        directory.mkdir()
        write_json(directory / "dataset.json", dataset)
        recipe = {"schema_version": 2, "task": "image", "engine": "native", "backend": "cuda",
                  "weight_precision": "f32", "compute_mode": "f32", "feature_cache": "f32",
                  "storage_profile": "dense", "quantization_modules": [], "checkpoint_sha256": "a" * 64}
        write_json(directory / "recipe.json", recipe)
        samples = phase_samples(dataset, "development")
        outputs, lookup = [], {}
        for sample in samples:
            for index, prompt in enumerate(sample["prompts"]):
                name = sample["id"] + f"-p{index}.json"
                write_json(directory / name, output_fixture(prompt=prompt))
                outputs.append({"sample_id": sample["id"], "prompt_index": index, "prompt": prompt,
                                "file": name, "sha256": sha256_file(directory / name)})
                lookup[(sample["id"], prompt)] = (sample["coco_image_id"], int(prompt[-1]))
        manifest = {"schema_version": 2, "kind": "sam3-ranked-precision-output-v2", "complete": True,
                    "phase": "development", "diagnostic_only": True, "images": len(samples),
                    "sample_ids": [row["id"] for row in samples], "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
                    "dataset_sha256": sha256_file(directory / "dataset.json"), "input_manifest_sha256": "b" * 64,
                    "input_images": {row["id"]: row["source_sha256"] for row in samples},
                    "gates_sha256": GATES_SHA256, "campaign_sha256": None, "outputs": outputs,
                    "artifact_sha256": {}, "archived_sources": {}}
        write_json(directory / "manifest.json", manifest)
        return directory, manifest, lookup

    def test_missing_or_reordered_prompt_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest, _ = self.fixture(Path(temporary))
            load_run(directory)
            for outputs in (manifest["outputs"][:-1], list(reversed(manifest["outputs"]))):
                write_json(directory / "manifest.json", {**manifest, "outputs": outputs})
                with self.assertRaisesRegex(ValueError, "inventory"):
                    load_run(directory)

    def test_output_tampering_and_omitted_low_score_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest, lookup = self.fixture(Path(temporary))
            read_outputs(directory, manifest, lookup)
            row = manifest["outputs"][0]
            bad = output_fixture(prompt=row["prompt"])
            bad["masks"].pop()
            write_json(directory / row["file"], bad)
            with self.assertRaisesRegex(ValueError, "changed"):
                read_outputs(directory, manifest, lookup)
            row["sha256"] = sha256_file(directory / row["file"])
            with self.assertRaisesRegex(ValueError, "inventory"):
                read_outputs(directory, manifest, lookup)

    def test_mismatched_inputs_checkpoint_or_campaign_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, reference, _ = self.fixture(Path(temporary))
            for key, value in (("dataset_sha256", "0" * 64), ("campaign_sha256", "1" * 64),
                               ("input_images", {}), ("sample_ids", [])):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    same_inputs(reference, {**reference, key: value})
            candidate = copy.deepcopy(reference)
            candidate["recipe"]["checkpoint_sha256"] = "2" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                same_inputs(reference, candidate)

    def test_final_run_requires_full_frozen_campaign_and_reserve_is_forbidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest, _ = self.fixture(root)
            recipe, dataset_path = manifest["recipe"], directory / "dataset.json"
            campaign = root / "campaign.json"
            producer = root / "producer.py"
            producer.write_text("print('producer')\n")
            sources = {str(producer): sha256_file(producer)}
            write_json(campaign, {"schema_version": 2, "kind": "sam3-precision-campaign-v2", "frozen_before_evaluation": True,
                                  "gates_sha256": GATES_SHA256, "dataset_sha256": sha256_file(dataset_path),
                                  "recipe_sha256": [canonical_hash(recipe)], "artifact_sha256": sources})
            with patch("tools.archive.precision_v2_v3.precision_artifacts.source_snapshot", return_value=sources):
                self.assertEqual(campaign_check(campaign, dataset_path, recipe, "evaluation"), sha256_file(campaign))
                for path, partial in ((None, False), (campaign, True)):
                    with self.assertRaises(ValueError):
                        campaign_check(path, dataset_path, recipe, "evaluation", partial)
                with self.assertRaises(ValueError):
                    campaign_check(campaign, dataset_path, {**recipe, "feature_cache": "f16"}, "evaluation")
            with patch("tools.archive.precision_v2_v3.precision_artifacts.source_snapshot", return_value={**sources, "missing.py": "3" * 64}):
                with self.assertRaisesRegex(ValueError, "every current"):
                    campaign_check(campaign, dataset_path, recipe, "evaluation")
            claim_evaluation(campaign, recipe, root / "output")
            with self.assertRaises(FileExistsError):
                claim_evaluation(campaign, recipe, root / "different-output")
            with self.assertRaisesRegex(ValueError, "reserve"):
                phase_samples({}, "reserve")

    def test_archived_producer_remains_verifiable_after_current_tools_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live, archived = root / "live.py", root / "archived.py"
            live.write_text("original\n")
            archived.write_bytes(live.read_bytes())
            manifest = {"artifact_sha256": {str(live): sha256_file(live)}, "archived_sources": {str(live): "archived.py"}}
            live.write_text("updated\n")
            verify_export_artifacts(root, manifest)
            archived.write_text("altered evidence\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_export_artifacts(root, manifest)

    def test_migrated_tree_identity_changes_and_archive_survives_live_source_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "api").mkdir(parents=True)
            (root / "apps" / "image").mkdir(parents=True)
            (root / "support" / "image_io").mkdir(parents=True)
            (root / "third_party" / "stb").mkdir(parents=True)
            (root / "tools" / "convert").mkdir(parents=True)
            (root / "tests" / "tools").mkdir(parents=True)
            (root / "tests" / "data").mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("project(repository)\n")
            (root / "apps" / "CMakeLists.txt").write_text("# apps\n")
            (root / "support" / "image_io" / "CMakeLists.txt").write_text("# support\n")
            (root / "tools" / "CMakeLists.txt").write_text("# tools\n")
            (root / "tests" / "CMakeLists.txt").write_text("# tests\n")
            implementation = root / "src" / "api" / "model.cpp"
            implementation.write_text("int model() { return 1; }\n")
            build_file = root / "src" / "CMakeLists.txt"
            build_file.write_text("add_library(sam)\n")
            nested_tool = root / "tools" / "convert" / "convert_sam3.py"
            nested_tool.write_text("def convert():\n    return 1\n")
            legacy_entry = root / "tools" / "convert_sam3.py"
            legacy_entry.write_text("from tools.convert import convert_sam3\n")
            vendor_header = root / "third_party" / "stb" / "stb_image.h"
            vendor_header.write_text("/* vendor implementation header */\n")
            support_header = root / "support" / "image_io" / "image_io.hpp"
            support_header.write_text("struct Image {};\n")
            test_module = root / "tests" / "tools" / "test_tools.py"
            test_module.write_text("class ToolChecks: pass\n")
            golden = root / "tests" / "data" / "cases.json"
            golden.write_text("{}\n")
            with patch("tools.archive.precision_v2_v3.precision_artifacts.repository_root", return_value=root):
                before = source_snapshot()
                for path in (implementation, build_file, nested_tool, legacy_entry, vendor_header,
                             support_header, test_module, golden):
                    self.assertIn(str(path.resolve()), before)
                implementation.write_text("int model() { return 2; }\n")
                nested_tool.write_text("def convert():\n    return 2\n")
                after = source_snapshot()
                self.assertEqual(after[str(build_file.resolve())], before[str(build_file.resolve())])
                self.assertNotEqual(after[str(implementation.resolve())], before[str(implementation.resolve())])
                self.assertNotEqual(after[str(nested_tool.resolve())], before[str(nested_tool.resolve())])
                archive = root / "archive"
                archived = archive_sources(archive, after)
            self.assertEqual(archived[str(implementation.resolve())], "sources/src/api/model.cpp")
            self.assertEqual(archived[str(build_file.resolve())], "sources/src/CMakeLists.txt")
            self.assertEqual(archived[str(nested_tool.resolve())], "sources/tools/convert/convert_sam3.py")
            self.assertEqual(archived[str(vendor_header.resolve())], "sources/third_party/stb/stb_image.h")
            self.assertEqual(archived[str(support_header.resolve())], "sources/support/image_io/image_io.hpp")
            self.assertEqual(archived[str(test_module.resolve())], "sources/tests/tools/test_tools.py")
            self.assertEqual(archived[str(golden.resolve())], "sources/tests/data/cases.json")
            # The archive must stay verifiable after the live sources are gone.
            for path in (implementation, build_file, nested_tool, legacy_entry, vendor_header,
                         support_header, test_module, golden):
                path.unlink()
            manifest = {"artifact_sha256": after, "archived_sources": archived}
            verify_export_artifacts(archive, manifest)
            (archive / "sources" / "tools" / "convert" / "convert_sam3.py").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_export_artifacts(archive, manifest)


if __name__ == "__main__":
    unittest.main()
