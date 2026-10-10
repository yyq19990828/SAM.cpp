"""Task floors, non-veto diagnostics and independent v3 performance benefits."""

import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.archive.precision_v2_v3.precision_performance import assess_performance, deployment_status
from tools.archive.precision_v2_v3.freeze_precision_campaign import development_eligible, freeze
from tools.archive.precision_v2_v3.precision_artifacts import validate_evaluation_history
from tools.archive.precision_v2_v3.evaluate_precision import load_run, same_inputs
from tools.archive.precision_v2_v3.precision_acceptance import (canonical_hash, compare_objects, diagnostic_summary,
                                                   gate_identity, load_gates, object_gates,
                                                   quality_profile, version_recipe)
from tools.archive.precision_v2_v3.validate_precision_regression import compare_cases, fixed_cases
from tools.convert.sam3_artifacts import read_json, sha256_file, write_json
from tests.tools import test_precision_acceptance, test_precision_artifacts, test_precision_performance
from tests.tools.test_precision_acceptance import output_fixture


def recipe(tier="balanced", **kwargs):
    return version_recipe(test_precision_acceptance.PrecisionAcceptanceChecks().recipe(**kwargs), 3, tier)


class PrecisionV3Checks(unittest.TestCase):
    def setUp(self):
        self.gates = load_gates(policy_version=3)
        self.common = self.gates["common"]
        self.profile = self.gates["profiles"]["balanced"]

    def compare(self, reference, candidate, **kwargs):
        return compare_objects(reference, candidate, self.profile, self.common, **kwargs)

    def test_tier_is_explicit_and_independent_of_weight_compute_and_cache(self):
        profiles = [quality_profile(recipe(precision=p, compute=c, cache=k), self.gates)[1]
                    for p, c, k in (("f32", "f32", "f32"), ("q4_k", "f16", "mixed-q8_0"),
                                    ("f16", "f32", "f16"))]
        self.assertEqual(profiles, [self.profile] * 3)
        old = test_precision_acceptance.PrecisionAcceptanceChecks().recipe()
        for version, tier in ((3, None), (3, "q4_k"), (2, "compact")):
            with self.subTest(version=version, tier=tier), self.assertRaises(ValueError):
                version_recipe(old, version, tier)
        with self.assertRaises(ValueError):
            quality_profile(recipe(), load_gates())
        with self.assertRaises(ValueError):
            quality_profile({**recipe(), "quality_tier": None}, self.gates)

    def test_cache_has_both_absolute_and_separate_incremental_budgets(self):
        value = recipe("compact", precision="q4_k", cache="mixed-q8_0")
        _, absolute = quality_profile(value, self.gates)
        _, increment = quality_profile(value, self.gates, incremental=True)
        self.assertLess(increment["ap_drop_max"], absolute["ap_drop_max"])
        self.assertGreater(increment["task_mask_iou_min"], absolute["task_mask_iou_min"])
        with self.assertRaises(ValueError):
            quality_profile(recipe(), self.gates, incremental=True)

    def test_score_and_box_changes_remain_visible_without_failing_task_quality(self):
        reference = output_fixture([(0, .95, (2, 3, 20, 24))])
        candidate = copy.deepcopy(reference)
        candidate["query_scores"][0] = .7
        candidate["query_boxes"][0] = [5, 7, 23, 28]
        result = self.compare(reference, candidate)
        self.assertEqual(object_gates([result], self.profile, self.common, strict=True)["status"], "PASS")
        self.assertTrue(result["matches"][0]["task_passed"])
        self.assertFalse(result["matches"][0]["fidelity_passed"])
        diagnostics = diagnostic_summary([result])
        self.assertEqual(diagnostics["fidelity_bad_pairs"], 1)
        self.assertGreater(diagnostics["score_error_max"], .2)
        self.assertGreater(diagnostics["box_fraction_max"], .05)
        old = load_gates()
        self.assertEqual(compare_objects(reference, candidate, old["profiles"]["q6_k"], old["common"])["bad_objects"], 1)

    def test_tight_mask_fidelity_is_diagnostic_but_severe_degradation_fails(self):
        reference = output_fixture([(0, .9, (0, 0, 40, 40))])
        near = output_fixture([(3, .9, (0, 0, 37, 40))])
        result = self.compare(reference, near)
        self.assertEqual(result["bad_objects"], 0)
        self.assertEqual(result["diagnostics"]["fidelity_bad_pairs"], 1)
        degraded = output_fixture([(3, .9, (0, 0, 30, 40))])
        result = self.compare(reference, degraded)
        self.assertEqual(result["bad_objects"], 1)
        self.assertEqual(result["missing_high_objects"], 0)
        self.assertEqual(object_gates([result], self.profile, self.common, strict=True)["status"], "FAIL")

    def test_reference_gt_boundary_flip_is_reported_without_protected_veto(self):
        reference = output_fixture([(0, .95, (0, 0, 32, 40))])
        candidate = output_fixture([(0, .95, (1, 0, 32, 40))])
        gt = [output_fixture([(0, .95, (0, 0, 64, 40))])["masks"][0]["mask"]]
        result = self.compare(reference, candidate, ground_truth=gt)
        self.assertEqual(result["protected_objects"], 0)
        self.assertEqual(result["protected_misses"], 0)
        self.assertEqual(result["diagnostics"]["gt_threshold_losses"], 1)
        self.assertEqual(result["bad_objects"], 0)
        old = load_gates()
        self.assertEqual(compare_objects(reference, candidate, old["profiles"]["q6_k"], old["common"], gt)["protected_misses"], 1)

    def test_strong_gt_and_real_missing_or_duplicate_objects_still_fail(self):
        reference = output_fixture([(0, .95, (0, 0, 32, 40)), (1, .96, (32, 0, 64, 40))])
        gt = [row["mask"] for row in reference["masks"][:2]]
        result = self.compare(reference, output_fixture([(3, .95, (0, 0, 32, 40))]), ground_truth=gt)
        self.assertEqual(result["protected_objects"], 2)
        self.assertEqual(result["protected_misses"], 1)
        self.assertEqual(result["missing_high_objects"], 1)
        one = output_fixture([(0, .9, (1, 1, 10, 10))])
        duplicate = output_fixture([(0, .9, (1, 1, 10, 10)), (1, .9, (1, 1, 10, 10))])
        self.assertEqual(self.compare(one, duplicate)["extra_high_objects"], 1)

    def test_fixed_regression_allows_diagnostics_but_no_task_tail_or_new_detection(self):
        original = {row["id"]: output_fixture([(0, .95, (1, 1, 10, 10))], prompt=row["prompt"])
                    for row in fixed_cases()}
        candidate = copy.deepcopy(original)
        for value in candidate.values():
            value["query_scores"][0] = .7
        result = compare_cases(original, candidate, recipe())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["diagnostics"]["fidelity_bad_pairs"], 7)
        key = fixed_cases()[0]["id"]
        candidate[key] = output_fixture(prompt=original[key]["prompt"])
        self.assertEqual(compare_cases(original, candidate, recipe())["status"], "FAIL")
        original[key] = output_fixture(prompt=original[key]["prompt"])
        candidate[key] = output_fixture([(0, .51, (1, 1, 10, 10))], prompt=original[key]["prompt"])
        self.assertEqual(compare_cases(original, candidate, recipe())["status"], "FAIL")

    def test_valid_performance_without_large_benefit_is_not_a_quality_failure(self):
        cases, records = test_precision_performance.PrecisionPerformanceChecks().fixture(latency=.98, rss=1)
        result = assess_performance(records, cases, self.gates, "cuda")["workloads"]["full_image"]
        self.assertEqual(result["measurement_status"], "PASS")
        self.assertEqual(result["non_regression_status"], "PASS")
        self.assertEqual(result["benefit_status"], "NOT_DEMONSTRATED")
        self.assertEqual(result["performance_labels"], [])
        self.assertEqual(deployment_status("PASS", "PASS", "NOT_APPLICABLE", "PASS", [], 3), "PASS")
        self.assertEqual(deployment_status("PASS", "PASS", "NOT_APPLICABLE", "PASS", [], 2), "INCONCLUSIVE")

    def test_incomplete_quality_or_arithmetic_never_becomes_qualified_through_speed(self):
        for status in ("FAIL", "NOT_RUN", "INCONCLUSIVE"):
            with self.subTest(status=status):
                self.assertEqual(deployment_status(status, "PASS", "NOT_APPLICABLE", "PASS", ["latency"], 3), status)
                self.assertEqual(deployment_status("PASS", status, "NOT_APPLICABLE", "PASS", ["latency"], 3), status)
        with self.assertRaises(ValueError):
            deployment_status("NOT_APPLICABLE", "PASS", "NOT_APPLICABLE", "PASS", ["latency"], 3)

    def test_slowdown_and_contamination_have_different_performance_statuses(self):
        cases, records = test_precision_performance.PrecisionPerformanceChecks().fixture(latency=1.1, rss=1)
        result = assess_performance(records, cases, self.gates, "cuda")["workloads"]["full_image"]
        self.assertEqual(result["measurement_status"], "PASS")
        self.assertEqual(result["non_regression_status"], "FAIL")
        self.assertEqual(result["benefit_status"], "NOT_DEMONSTRATED")
        records[0]["other_compute_pids"] = [123]
        result = assess_performance(records, cases, self.gates, "cuda")["workloads"]["full_image"]
        self.assertEqual(result["measurement_status"], "INCONCLUSIVE")
        self.assertEqual(result["non_regression_status"], "INCONCLUSIVE")
        self.assertEqual(result["benefit_status"], "INCONCLUSIVE")

    def test_large_benefit_remains_workload_specific_and_cpu_has_no_gpu_label(self):
        cases, records = test_precision_performance.PrecisionPerformanceChecks().fixture(latency=.8, rss=1, backend="cpu")
        for row in records:
            if row["variant"] == "candidate":
                row["timing_ms"]["repeated_result"] = [100] * 20
        result = assess_performance(records, cases, self.gates, "cpu")["workloads"]
        self.assertEqual(result["full_image"]["performance_labels"], ["latency"])
        self.assertEqual(result["repeated_result"]["performance_labels"], [])
        self.assertEqual(result["full_image"]["labels"]["gpu-memory"], "NOT_APPLICABLE")


class PrecisionV3ArtifactChecks(unittest.TestCase):
    def fixture(self, root):
        directory, manifest, lookup = test_precision_artifacts.PrecisionArtifactChecks().fixture(root)
        old = copy.deepcopy(manifest)
        manifest["recipe"] = version_recipe(manifest["recipe"], 3, "balanced")
        manifest.update(schema_version=3, kind="sam3-ranked-precision-output-v3",
                        recipe_sha256=canonical_hash(manifest["recipe"]), gates_sha256=gate_identity(3)[1])
        write_json(directory / "recipe.json", manifest["recipe"])
        write_json(directory / "manifest.json", manifest)
        return directory, manifest, old

    def test_v3_run_rejects_v2_comparisons_and_tier_or_gate_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest, old = self.fixture(Path(temporary))
            self.assertEqual(load_run(directory)[0]["schema_version"], 3)
            with self.assertRaises(ValueError):
                same_inputs(old, manifest)
            for change in ({"gates_sha256": gate_identity(2)[1]},
                           {"recipe": {**manifest["recipe"], "quality_tier": "compact"}}):
                write_json(directory / "manifest.json", {**manifest, **change})
                with self.assertRaises(ValueError):
                    load_run(directory)

    def test_holdout_rejects_exposed_ids_content_and_opening_old_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, _, _ = self.fixture(Path(temporary))
            prior_path = directory / "dataset.json"
            prior = read_json(prior_path)
            history = {"complete_declaration": True, "datasets": {str(prior_path): sha256_file(prior_path)}}
            new = copy.deepcopy(prior)
            for row in new["samples"]:
                if row["split"] == "evaluation":
                    row["coco_image_id"] += 100
                    row["source_sha256"] = canonical_hash(row["coco_image_id"])
            validate_evaluation_history(history, new)
            original = next(row for row in prior["samples"] if row["split"] == "evaluation")
            for field in ("coco_image_id", "source_sha256"):
                bad = copy.deepcopy(new)
                row = next(row for row in bad["samples"] if row["split"] == "evaluation")
                row[field] = original[field]
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exposed"):
                    validate_evaluation_history(history, bad)
            bad = copy.deepcopy(new)
            next(row for row in bad["samples"] if row["split"] == "reserve")["split"] = "evaluation"
            with self.assertRaisesRegex(ValueError, "prior reserve"):
                validate_evaluation_history(history, bad)
            for declaration in (None, {}, {**history, "complete_declaration": False}):
                with self.assertRaisesRegex(ValueError, "history"):
                    validate_evaluation_history(declaration, new)
            write_json(prior_path, {**prior, "changed": True})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                validate_evaluation_history(history, new)

    def test_failed_development_and_missing_cache_evidence_cannot_consume_final(self):
        valid = {"absolute": {"status": "INCONCLUSIVE", "checks": [
                    {"name": "ap_drop", "status": "PASS"},
                    {"name": "evaluation_images", "status": "INCONCLUSIVE"}]},
                 "incremental": {"status": "NOT_APPLICABLE"}}
        development_eligible(valid, "f32")
        with self.assertRaises(ValueError):
            development_eligible(valid, "mixed-q8_0")
        for change in ({"name": "ap_drop", "status": "FAIL"},
                       {"name": "ap_drop", "status": "INCONCLUSIVE"}):
            bad = copy.deepcopy(valid)
            bad["absolute"]["checks"][0] = change
            with self.assertRaises(ValueError):
                development_eligible(bad, "f32")
        bad = {**valid, "absolute": {"status": "NOT_APPLICABLE"}}
        with self.assertRaises(ValueError):
            development_eligible(bad, "f32")

    def test_quality_only_campaign_does_not_require_performance_binary_or_cases(self):
        # Exercise freeze's orchestration with small, identity-bound synthetic
        # exports; loader coverage above checks the real on-disk manifests.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, native, _ = self.fixture(root)
            original = copy.deepcopy(native)
            original["recipe"] = {**native["recipe"], "engine": "original", "environment": {"cpu": "test"}}
            original.update(reference_kind="official-checkpoint", recipe_sha256=canonical_hash(original["recipe"]))
            binary, model = root / "export", root / "model.gguf"
            binary.write_bytes(b"synthetic executable")
            model.write_bytes(b"synthetic model")
            native["recipe"].update(environment={"cpu": "test"}, binary_sha256=sha256_file(binary), model_sha256=sha256_file(model))
            native["recipe_sha256"] = canonical_hash(native["recipe"])
            native["artifact_sha256"] = {str(p): sha256_file(p) for p in (binary, model)}
            metrics_path = root / "metrics.json"
            write_json(metrics_path, {"schema_version": 3, "kind": "sam3-precision-quality-v3", "complete": True,
                "phase": "development", "images": 1, "candidate_recipe_sha256": native["recipe_sha256"],
                "dataset_sha256": native["dataset_sha256"], "gates_sha256": gate_identity(3)[1],
                "quality_status": "INCONCLUSIVE", "artifact_sha256": {},
                "absolute": {"status": "INCONCLUSIVE", "checks": [{"name": "evaluation_images", "status": "INCONCLUSIVE"}]},
                "incremental": {"status": "NOT_APPLICABLE"}})
            selection = root / "selection.json"
            write_json(selection, {"schema_version": 3, "kind": "sam3-precision-selection-v3", "evaluation_history": {},
                "runs": [{"id": "original", "development_run": str(directory)},
                         {"id": "native", "development_run": str(directory), "development_quality": str(metrics_path)}]})
            dataset = {"samples": [{"split": "development"}] + [{"split": "evaluation"}] * 1024 + [{"split": "reserve"}] * 1024}
            rows = [(value, dataset, dataset["samples"][:1]) for value in (original, native)]
            with patch("tools.archive.precision_v2_v3.freeze_precision_campaign.load_run", side_effect=rows), \
                    patch("tools.archive.precision_v2_v3.freeze_precision_campaign.source_snapshot", side_effect=lambda: {}), \
                    patch("tools.archive.precision_v2_v3.freeze_precision_campaign.validate_evaluation_history", return_value={}):
                freeze(selection, root / "campaign.json")
            campaign = read_json(root / "campaign.json")
            self.assertIsNone(campaign["performance_cases"])
            self.assertIsNone(campaign["benchmark_binary"])
            self.assertIsNone(campaign["runs"][1]["performance_baseline"])
            self.assertEqual(campaign["kind"], "sam3-precision-campaign-v3")

    def test_evaluator_cli_reports_diagnostics_and_missing_evidence_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest, _ = self.fixture(root)
            annotations = read_json(root / "annotations.json")
            annotations["info"] = {}
            for image in annotations["images"]:
                image.update(width=64, height=48)
            for index, annotation in enumerate(annotations["annotations"], 1):
                annotation.update(id=index, bbox=[2, 3, 10, 10], segmentation=[[2, 3, 12, 3, 12, 13, 2, 13]])
            write_json(root / "annotations.json", annotations)
            dataset = read_json(directory / "dataset.json")
            dataset["provenance"]["annotations_sha256"] = sha256_file(root / "annotations.json")
            write_json(directory / "dataset.json", dataset)
            manifest["dataset_sha256"] = sha256_file(directory / "dataset.json")
            samples = {row["id"]: row for row in dataset["samples"]}
            for row in manifest["outputs"]:
                positive = int(row["prompt"].split()[-1]) in samples[row["sample_id"]]["positive_category_ids"]
                payload = output_fixture([(0, .95, (2, 3, 12, 13))] if positive else [], prompt=row["prompt"])
                write_json(directory / row["file"], payload)
                row["sha256"] = sha256_file(directory / row["file"])
            original_dir = root / "original"
            shutil.copytree(directory, original_dir)
            original = copy.deepcopy(manifest)
            original["recipe"]["engine"] = "original"
            original.update(reference_kind="official-checkpoint", recipe_sha256=canonical_hash(original["recipe"]))
            write_json(original_dir / "recipe.json", original["recipe"])
            write_json(original_dir / "manifest.json", original)
            for row in manifest["outputs"]:
                payload = read_json(directory / row["file"])
                if payload["query_scores"][0] > .5:
                    payload["query_scores"][0] = .7
                write_json(directory / row["file"], payload)
                row["sha256"] = sha256_file(directory / row["file"])
            write_json(directory / "manifest.json", manifest)
            output = root / "metrics"
            command = [sys.executable, "tools/archive/precision_v2_v3/evaluate_precision.py", "--reference", str(original_dir),
                       "--candidate", str(directory), "--annotations", str(root / "annotations.json"), "--output", str(output)]
            completed = subprocess.run(command, text=True, capture_output=True, cwd=Path(__file__).resolve().parents[2])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = read_json(output / "metrics.json")
            self.assertEqual(result["kind"], "sam3-precision-quality-v3")
            self.assertEqual(result["task_quality_status"], "INCONCLUSIVE")  # too few images/objects/area coverage
            self.assertEqual(result["qualification_status"], "NOT_RUN")
            self.assertFalse(result["diagnostics_affect_quality"])
            self.assertFalse(result["performance_required_for_qualification"])
            self.assertEqual(result["absolute"]["object_counts"]["bad_objects"], 0)
            self.assertGreater(result["absolute"]["diagnostics"]["fidelity_bad_pairs"], 0)
            checks = {row["name"]: row["status"] for row in result["absolute"]["checks"]}
            self.assertEqual(checks["ap_drop"], "PASS")
            self.assertEqual(checks["miou_drop"], "PASS")
            self.assertEqual(result["incremental_quality_status"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
