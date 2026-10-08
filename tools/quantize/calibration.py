"""SAM calibration dataset contracts and bounded, streaming channel statistics."""

import hashlib
import re
from pathlib import Path

import numpy as np

from tools.convert.sam3_artifacts import SAM3_REVISION, artifact_path, read_json, sha256_file


def validate_dataset(value, regression_hashes=(), diagnostic=False):
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("kind") != "sam3-calibration-dataset"
            or value.get("sam3_revision") != SAM3_REVISION):
        raise ValueError("unsupported calibration dataset contract")
    seed = value.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("dataset seed must be an unsigned 32-bit integer")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("calibration dataset has no samples")
    ids, groups, hashes, paths = set(), {}, {}, {}
    calibration_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("calibration samples must be objects")
        identifier = sample.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", identifier) or identifier in ids:
            raise ValueError("invalid or duplicate calibration sample ID")
        ids.add(identifier)
        split = sample.get("split")
        if split not in ("calibration", "selection", "evaluation"):
            raise ValueError(f"{identifier}: unsupported dataset split")
        group, digest, image = sample.get("source_group"), sample.get("source_sha256"), sample.get("image")
        if not isinstance(group, str) or not group.strip() or len(group) > 256:
            raise ValueError(f"{identifier}: missing source group")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{identifier}: invalid source SHA-256")
        if (not isinstance(image, str) or not image or Path(image).is_absolute()
                or ".." in Path(image).parts):
            raise ValueError(f"{identifier}: image must be a bounded relative path")
        for mapping, key in ((groups, group), (hashes, digest), (paths, image)):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"{identifier}: source leakage across dataset splits")
            mapping[key] = split
        prompts = sample.get("prompts")
        if (not isinstance(prompts, list) or not prompts
                or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
                or len(set(prompts)) != len(prompts)):
            raise ValueError(f"{identifier}: prompts must be unique nonempty strings")
        if sample.get("transform") is not None:
            raise ValueError("calibration v1 takes original decoded images; materialize and group transforms explicitly")
        if split != "evaluation" and digest in regression_hashes and not diagnostic:
            raise ValueError("frozen regression images cannot tune calibration or parameter selection")
        calibration_count += split == "calibration"
    if not calibration_count:
        raise ValueError("dataset needs a calibration split")
    return value


def load_dataset(path, root, diagnostic=False):
    regression = read_json(Path(__file__).resolve().parents[2] / "tests/data/sam3-image-cases.json")
    excluded = {case["source_sha256"] for case in regression["cases"]}
    dataset = validate_dataset(read_json(path), excluded, diagnostic)
    # Verify every split before model allocation. Evaluation inputs are never
    # run by the exporter, but their identities must not overlap the fit set.
    for sample in dataset["samples"]:
        source = artifact_path(root, sample["image"])
        if sha256_file(source) != sample["source_sha256"]:
            raise ValueError(f"{sample['id']}: source image hash mismatch")
    return dataset


def layer_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


class ChannelStats:
    """Exact extrema/moments, plus a uniform bounded reservoir of token rows.

    Percentiles are estimates from the reservoir, not full-population tails.
    Random priorities make the retained rows independent of update chunk sizes.
    """

    def __init__(self, channels, sample_rows=128, seed=0):
        if type(channels) is not int or channels <= 0 or type(sample_rows) is not int or sample_rows <= 0:
            raise ValueError("channels and sample_rows must be positive integers")
        self.channels, self.sample_rows = channels, sample_rows
        self.rows = 0
        self.minimum = np.full(channels, np.inf, dtype=np.float64)
        self.maximum = np.full(channels, -np.inf, dtype=np.float64)
        self.total = np.zeros(channels, dtype=np.float64)
        self.squares = np.zeros(channels, dtype=np.float64)
        self.samples = np.empty((0, channels), dtype=np.float32)
        self._priorities = np.empty(0, dtype=np.float64)
        self._random = np.random.default_rng(seed)

    def update(self, values):
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError("statistics require float32 token rows with a fixed channel dimension")
        if not np.isfinite(values).all():
            raise ValueError("non-finite calibration activation")
        if not values.shape[0]:
            return
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.total += values.sum(axis=0, dtype=np.float64)
        self.squares += np.square(values, dtype=np.float64).sum(axis=0)
        self.rows += values.shape[0]
        priorities = np.concatenate((self._priorities, self._random.random(values.shape[0])))
        samples = np.concatenate((self.samples, values))
        count = min(self.sample_rows, len(priorities))
        selected = np.argpartition(priorities, len(priorities) - count)[-count:]
        selected = selected[np.argsort(priorities[selected])]
        self._priorities = priorities[selected].copy()
        self.samples = samples[selected].copy()

    def summary(self):
        if not self.rows:
            raise ValueError("cannot publish empty calibration statistics")
        quantiles = np.quantile(np.abs(self.samples), [0.99, 0.999], axis=0)
        return {"channels": self.channels, "token_rows": self.rows,
                "minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                "absmax": np.maximum(np.abs(self.minimum), np.abs(self.maximum)).tolist(),
                "mean": (self.total / self.rows).tolist(),
                "rms": np.sqrt(self.squares / self.rows).tolist(),
                "reservoir_rows": len(self.samples),
                "estimated_abs_percentile_99": quantiles[0].tolist(),
                "estimated_abs_percentile_99_9": quantiles[1].tolist()}
