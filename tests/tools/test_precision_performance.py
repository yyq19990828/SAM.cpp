"""Deployment benefits require complete independent and consistent measurements."""

import copy
import unittest

from tools.archive.precision_v2_v3.precision_acceptance import load_gates
from tools.archive.precision_v2_v3.precision_performance import ORDER, WORKLOADS, assess_performance


class PrecisionPerformanceChecks(unittest.TestCase):
    def fixture(self, latency=.8, gpu=1.0, rss=.8, backend="cuda"):
        cases = [{"id": f"case-{index}"} for index in range(8)]
        records = []
        for kind in ("latency", "memory"):
            for case in cases:
                for pair in range(3):
                    for variant in ORDER[pair]:
                        candidate = variant == "candidate"
                        pid = 10000 + len(records)
                        records.append({"case": case["id"], "pair": pair, "variant": variant, "kind": kind,
                                        "complete": True, "backend": backend, "process_id": pid,
                                        "warmups": 5, "iterations": 20, "memory_sampler_enabled": kind == "memory",
                                        "sampled_pid": pid, "gpu_samples": 200,
                                        "timing_ms": {key: [100 * (latency if candidate else 1)] * 20 for key in WORKLOADS},
                                        "rss_peak_bytes": int(1000 * (rss if candidate else 1)),
                                        "gpu_peak_bytes": int(1000 * (gpu if candidate else 1)),
                                        "other_compute_pids": [], "observer_errors": []})
        return cases, records

    def assess(self, cases, records, backend="cuda"):
        return assess_performance(records, cases, load_gates(), backend)["workloads"]["full_image"]

    def test_benefit_labels_are_independent(self):
        cases, records = self.fixture()
        result = self.assess(cases, records)
        self.assertEqual(set(result["performance_labels"]), {"latency", "host-memory"})
        cases, records = self.fixture(latency=1.0, gpu=.8, rss=1.0)
        self.assertEqual(self.assess(cases, records)["performance_labels"], ["gpu-memory"])

    def test_single_case_tail_regression_cannot_hide_in_geometric_mean(self):
        cases, records = self.fixture(latency=.8)
        for record in records:
            if record["case"] == "case-0" and record["variant"] == "candidate":
                record["timing_ms"]["full_image"][-2:] = [106, 106]
        result = self.assess(cases, records)
        self.assertLess(result["ratios"]["p95_ratio"], 1.03)
        self.assertEqual(result["labels"]["latency"], "FAIL")

    def test_between_process_threshold_crossing_is_inconclusive(self):
        cases, records = self.fixture(latency=.8)
        for record in records:
            if record["pair"] == 2 and record["variant"] == "candidate":
                record["timing_ms"]["full_image"] = [92] * 20
        result = self.assess(cases, records)
        self.assertLess(result["ratios"]["p50_ratio"], .9)
        self.assertEqual(result["labels"]["latency"], "INCONCLUSIVE")

    def test_one_process_tail_cannot_hide_behind_pooled_percentiles(self):
        cases, records = self.fixture(latency=.8)
        for record in records:
            if record["pair"] == 2 and record["case"] == "case-0" and record["variant"] == "candidate":
                record["timing_ms"]["full_image"][-2:] = [108, 108]
        result = self.assess(cases, records)
        self.assertLess(result["cases"][0]["p95_ratio"], 1.05)
        self.assertEqual(result["labels"]["latency"], "INCONCLUSIVE")

    def test_missing_reused_reordered_or_contaminated_processes_never_pass(self):
        cases, records = self.fixture()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.assess(cases, records[:-1])
        bad = copy.deepcopy(records)
        bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaisesRegex(ValueError, "order"):
            self.assess(cases, bad)
        bad = copy.deepcopy(records)
        bad[1]["process_id"] = bad[0]["process_id"]
        with self.assertRaisesRegex(ValueError, "independent"):
            self.assess(cases, bad)
        bad = copy.deepcopy(records)
        bad[0]["memory_sampler_enabled"] = True
        with self.assertRaisesRegex(ValueError, "contaminate"):
            self.assess(cases, bad)
        records[-1]["other_compute_pids"] = [2345]
        self.assertEqual(set(self.assess(cases, records)["labels"].values()), {"INCONCLUSIVE"})

    def test_peak_memory_uses_maximum_not_average(self):
        cases, records = self.fixture(gpu=.5)
        for record in records:
            if record["case"] == "case-0" and record["pair"] == 2 and record["variant"] == "candidate":
                record["gpu_peak_bytes"] = 1040
        result = self.assess(cases, records)
        self.assertEqual(result["ratios"]["gpu_peak_ratio"], 1.04)
        self.assertEqual(result["performance_labels"], [])

    def test_cpu_results_never_claim_gpu_memory_benefits(self):
        cases, records = self.fixture(backend="cpu")
        for record in records:
            record.pop("gpu_peak_bytes")
            record.pop("gpu_samples")
        result = self.assess(cases, records, "cpu")
        self.assertEqual(set(result["performance_labels"]), {"latency", "host-memory"})
        self.assertEqual(result["labels"]["gpu-memory"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
