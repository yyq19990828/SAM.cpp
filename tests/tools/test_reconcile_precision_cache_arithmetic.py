"""Frozen cache arithmetic must bind the same recipe, binary and library."""

import copy
import unittest
from pathlib import Path

from tools.validation.precision_acceptance import canonical_hash
from tools.validation.reconcile_precision_cache_arithmetic import check_recipe_alignment


class FrozenCacheAlignmentChecks(unittest.TestCase):
    def documents(self):
        binary = "/test/sam_precision_image_probe"
        library = Path("/test/libggml-cuda.so.0")
        parent_recipe = {"backend": "cuda", "weight_precision": "f32", "compute_mode": "f32",
                         "storage_profile": "dense", "quantization_modules": [], "feature_cache": "f32",
                         "binary_sha256": "binary-hash", "checkpoint_sha256": "model-hash"}
        candidate_recipe = {**parent_recipe, "feature_cache": "mixed-q8_0"}
        parent_hash, candidate_hash = map(canonical_hash, (parent_recipe, candidate_recipe))
        shared = {"complete": True, "campaign_sha256": "campaign-hash",
                  "gates_sha256": "gate-hash", "dataset_sha256": "dataset-hash",
                  "artifact_sha256": {binary: "binary-hash", str(library): "library-hash"}}
        parent = {**copy.deepcopy(shared), "recipe": parent_recipe, "recipe_sha256": parent_hash}
        candidate = {**copy.deepcopy(shared), "recipe": candidate_recipe,
                     "recipe_sha256": candidate_hash}
        quality = {**copy.deepcopy(shared), "candidate_recipe_sha256": candidate_hash,
                   "absolute_quality_status": "PASS", "incremental_quality_status": "PASS",
                   "quality_status": "PASS", "arithmetic_status": "NOT_RUN",
                   "deployment_status": "NOT_RUN"}
        performance = {**copy.deepcopy(shared), "candidate_recipe_sha256": candidate_hash,
                       "baseline_recipe_sha256": parent_hash, "quality_prerequisites_passed": True,
                       "arithmetic_status": "NOT_RUN", "deployment_status": "NOT_RUN"}
        campaign = {"frozen_before_evaluation": True,
                    "recipe_sha256": [parent_hash, candidate_hash], "gates_sha256": "gate-hash",
                    "dataset_sha256": "dataset-hash"}
        return campaign, parent, candidate, quality, performance, library

    def test_same_recipe_except_cache_is_accepted(self):
        campaign, parent, candidate, quality, performance, library = self.documents()
        result = check_recipe_alignment(campaign, parent, candidate, quality, performance,
                                        "campaign-hash", "binary-hash", library, "library-hash")
        self.assertEqual(result["candidate_recipe_sha256"], candidate["recipe_sha256"])

    def test_extra_compute_change_or_binary_mismatch_is_rejected(self):
        campaign, parent, candidate, quality, performance, library = self.documents()
        candidate["recipe"]["compute_mode"] = "f16"
        with self.assertRaisesRegex(ValueError, "F32/mixed-Q8|beyond feature cache"):
            check_recipe_alignment(campaign, parent, candidate, quality, performance,
                                   "campaign-hash", "binary-hash", library, "library-hash")
        campaign, parent, candidate, quality, performance, library = self.documents()
        candidate["artifact_sha256"][str(library)] = "different-library"
        with self.assertRaisesRegex(ValueError, "library identity"):
            check_recipe_alignment(campaign, parent, candidate, quality, performance,
                                   "campaign-hash", "binary-hash", library, "library-hash")
