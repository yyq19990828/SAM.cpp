"""Application reference agreement is descriptive; user advice never vetoes it."""

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tools.benchmark.quantization_benchmark import (RUN_KIND, advise, compare, compare_pair, export,
                                                   load_bundle, load_cases, main, reference_kind, summarize)
from tools.convert.sam3_artifacts import read_json, sha256_file, write_json
from tools.archive.precision_v2_v3.precision_acceptance import canonical_hash
from tests.tools.test_precision_acceptance import output_fixture


def recipe(engine="native", precision="f32"):
    return {"task": "image", "engine": engine, "backend": "cuda", "weight_precision": precision,
            "compute_mode": "f32", "feature_cache": "f32", "checkpoint_sha256": "a" * 64,
            "storage_profile": "dense" if precision == "f32" else "image-modules-linear-q4_k-v1",
            "quantization_modules": [] if precision == "f32" else ["text", "decoder"]}


def bundle(root, payload, declared=None, legacy=False):
    root.mkdir()
    declared = recipe() if declared is None else declared
    write_json(root / "recipe.json", declared)
    (root / "inputs").mkdir()
    image = root / "inputs/frame-one.png"
    Image.new("RGB", (payload["width"], payload["height"]), "white").save(image)
    case = {"id": "frame-one", "image": "camera/frame.png", "prompts": [payload["prompt"]]}
    cases = root / ("dataset.json" if legacy else "cases.json")
    write_json(cases, {"schema_version": 1, "samples": [case]})
    write_json(root / "frame-one-p0.json", payload)
    manifest = {"schema_version": 3 if legacy else 1,
                "kind": "sam3-ranked-precision-output-v3" if legacy else RUN_KIND, "complete": True,
                "recipe": declared, "recipe_sha256": canonical_hash(declared), "images": 1,
                "sample_ids": ["frame-one"], "input_images": {"frame-one": sha256_file(image)},
                "inputs": {"frame-one": {"file": image.name, "sha256": sha256_file(image),
                                           "width": payload["width"], "height": payload["height"]}},
                "outputs": [{"sample_id": "frame-one", "prompt_index": 0, "prompt": payload["prompt"],
                             "file": "frame-one-p0.json", "sha256": sha256_file(root / "frame-one-p0.json")}],
                "dataset_sha256" if legacy else "cases_sha256": sha256_file(cases)}
    write_json(root / "manifest.json", manifest)
    return root


def arguments(root, advice=None):
    return argparse.Namespace(reference=root / "reference", candidate=root / "candidate", output=root / "report",
                              score_threshold=.5, matching_iou=.5, advice=advice)


class QuantizationBenchmarkChecks(unittest.TestCase):
    def test_query_permutations_and_duplicate_candidates_have_one_to_one_matches(self):
        original = output_fixture([(3, .9, (2, 3, 20, 24)), (7, .8, (30, 10, 45, 25))])
        candidate = output_fixture([(33, .9, (2, 3, 20, 24)), (1, .8, (30, 10, 45, 25)),
                                    (8, .7, (2, 3, 20, 24))])
        result = compare_pair(original, candidate, .5, .5)
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["missing_reference_queries"], [])
        self.assertEqual(len(result["added_candidate_queries"]), 1)
        self.assertEqual(result["union_mask_iou"], 1)
        self.assertEqual(summarize([result])["added_candidate_rate"], 1 / 3)

    def test_empty_reference_is_separate_from_positive_union_iou(self):
        positive = output_fixture([(0, .9, (2, 3, 20, 24))])
        missing = compare_pair(positive, output_fixture(), .5, .5)
        empty = compare_pair(output_fixture(), output_fixture(), .5, .5)
        added = compare_pair(output_fixture(), positive, .5, .5)
        result = summarize([missing, empty, added])
        self.assertEqual(result["positive_union_iou_mean"], 0)
        self.assertEqual(result["missing_reference_rate"], 1)
        self.assertEqual(result["reference_empty_pairs"], 2)
        self.assertEqual(result["reference_empty_with_candidate_objects"], 1)
        self.assertIsNone(summarize([empty])["positive_union_iou_mean"])
        self.assertEqual(advise(summarize([empty]), {"positive_union_iou_mean_min": .9})["status"], "INSUFFICIENT_DATA")

    def test_valid_bad_candidate_remains_a_completed_report_with_success_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root / "reference", output_fixture([(0, .9, (2, 3, 20, 24))]))
            bundle(root / "candidate", output_fixture(), recipe(precision="q4_k"))
            advice = root / "limits.json"
            write_json(advice, {"missing_reference_rate_max": .01})
            self.assertEqual(main(["compare", "--reference", str(root / "reference"), "--candidate", str(root / "candidate"),
                                   "--output", str(root / "report"), "--advice", str(advice)]), 0)
            report = read_json(root / "report/report.json")
            self.assertTrue(report["complete"])
            self.assertEqual(report["report_validity"], "VALID")
            self.assertEqual(report["advice"]["status"], "OUTSIDE_USER_LIMITS")
            self.assertFalse(report["advice"]["affects_exit_code"])
            self.assertEqual(report["production_accuracy"], "NOT_MEASURED")
            self.assertEqual(report["performance"]["status"], "NOT_MEASURED")
            self.assertNotIn("deployment_status", report)
            self.assertEqual(report["candidate_precision"]["kernel_precision_evidence"], "NOT_COLLECTED")

    def test_legacy_outputs_can_be_compared_without_regrading_old_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = output_fixture([(0, .9, (2, 3, 20, 24))])
            bundle(root / "reference", original, recipe("original"), legacy=True)
            bundle(root / "candidate", output_fixture(), recipe(precision="q4_k"), legacy=True)
            before = {p: sha256_file(p) for name in ("reference", "candidate") for p in (root / name).rglob("*") if p.is_file()}
            with patch("tools.archive.precision_v2_v3.precision_acceptance.quality_profile", side_effect=AssertionError("legacy gates used")):
                report = compare(arguments(root))
            self.assertEqual(report["advice"]["status"], "NOT_REQUESTED")
            self.assertEqual(report["reference_kind"], "official-checkpoint")
            self.assertEqual(report["summary"]["missing_reference_objects"], 1)
            self.assertEqual(before, {p: sha256_file(p) for p in before})

    def test_changed_payload_and_incomplete_export_are_errors(self):
        for change in ("payload", "image", "complete", "recipe", "inventory"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = bundle(Path(directory) / "run", output_fixture())
                manifest = read_json(root / "manifest.json")
                if change == "payload":
                    (root / "frame-one-p0.json").write_text("{}")
                elif change == "image":
                    (root / "inputs/frame-one.png").write_bytes(b"changed")
                elif change == "complete":
                    manifest["complete"] = False
                elif change == "recipe":
                    manifest["recipe"]["compute_mode"] = "f16"
                else:
                    manifest["outputs"] = []
                write_json(root / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    load_bundle(root)

    def test_mismatched_tokens_images_prompts_or_checkpoint_are_errors(self):
        for change in ("token", "image", "prompt", "checkpoint"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original, candidate = output_fixture(), output_fixture()
                declared = recipe(precision="q4_k")
                if change == "token":
                    candidate["token_ids"][0] = 1
                elif change == "prompt":
                    candidate["prompt"] = "different"
                elif change == "checkpoint":
                    declared["checkpoint_sha256"] = "b" * 64
                bundle(root / "reference", original)
                path = bundle(root / "candidate", candidate, declared)
                if change == "image":
                    image = path / "inputs/frame-one.png"
                    Image.new("RGB", (64, 48), "black").save(image)
                    manifest = read_json(path / "manifest.json")
                    manifest["input_images"]["frame-one"] = sha256_file(image)
                    manifest["inputs"]["frame-one"]["sha256"] = sha256_file(image)
                    write_json(path / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    compare(arguments(root))
                self.assertFalse((root / "report").exists())

    def test_user_threshold_is_explicit_and_reference_must_be_f32(self):
        value = output_fixture([(0, .6, (2, 3, 20, 24))])
        self.assertEqual(compare_pair(value, value, .7, .5)["reference_detections"], 0)
        self.assertEqual(reference_kind(recipe()), "native-f32")
        with self.assertRaises(ValueError):
            reference_kind(recipe(precision="q4_k"))
        for limits in ({}, {"unknown_min": .5}, {"missing_reference_rate_max": -1}, {"missing_reference_rate_max": True}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                advise({}, limits)

    def test_application_cases_reject_ambiguous_prompts_and_sample_ids(self):
        for rows in ([None], [{"id": "../outside", "image": "x.png", "prompts": ["object"]}],
                     [{"id": "image", "image": "x.png", "prompts": ["object", "object"]}],
                     [{"id": "image", "image": "x.png", "prompts": ["a\tb"]}]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.json"
                write_json(path, {"schema_version": 1, "samples": rows})
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_application_export_reuses_exporter_without_quality_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 48), "white").save(root / "input.png")
            cases = root / "application.json"
            write_json(cases, {"schema_version": 1, "samples": [
                {"id": "camera-one", "image": "input.png", "prompts": ["object", "another object"]}]})
            args = argparse.Namespace(cases=cases, input_root=root, output=root / "run", engine="native",
                                      backend="cuda", compute="f32", cache="f32", binary=root / "binary", model=root / "model")

            def fake_export(args, samples, inputs, identities, declared):
                self.assertNotIn("quality_tier", declared)
                self.assertNotIn("gates_sha256", declared)
                rows = []
                for sample in samples:
                    self.assertTrue((args.inputs / inputs[sample["id"]]["file"]).is_file())
                    for index, prompt in enumerate(sample["prompts"]):
                        path = args.output / f"{sample['id']}-p{index}.json"
                        write_json(path, output_fixture(prompt=prompt))
                        rows.append({"sample_id": sample["id"], "prompt_index": index, "prompt": prompt,
                                     "file": path.name, "sha256": sha256_file(path)})
                return rows, {"arithmetic_profile": "recorded-test-profile"}

            with patch("tools.benchmark.quantization_benchmark.native_recipe", return_value=recipe()), \
                    patch("tools.maintenance.artifact_snapshot.source_snapshot", return_value={}), \
                    patch("tools.maintenance.artifact_snapshot.native_snapshot", return_value={}), \
                    patch("tools.validation.export_ranked_outputs.export_native", side_effect=fake_export) as producer:
                export(args)
                broken = argparse.Namespace(**{**vars(args), "output": root / "broken-run"})
                producer.side_effect = lambda *a: ([], {})
                with self.assertRaises(ValueError):
                    export(broken)
                self.assertFalse((broken.output / "manifest.json").exists())
            manifest, outputs, _ = load_bundle(args.output)
            self.assertEqual(manifest["kind"], RUN_KIND)
            self.assertEqual(len(outputs), 2)
            self.assertEqual(manifest["kernel_precision_evidence"], "NOT_COLLECTED")
            with self.assertRaises(FileExistsError):
                export(args)


if __name__ == "__main__":
    unittest.main()
