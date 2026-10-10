"""Dependency-free SAM 3 image weight allocation policies shared by tools."""

import hashlib

QUANTIZATION_MODULES = ("vision", "text", "fusion", "decoder")
MAX_QUANTIZATION_MODULE_CSV_LENGTH = 256
FULL_MODULE_PROFILE_PREFIX = "image-full-linear-"
CUSTOM_MODULE_PROFILE_PREFIX = "image-modules-linear-"
MIXED_STORAGE_PROFILE = "image-mixed-linear-v1"
TENSOR_MIXED_STORAGE_PROFILE = "image-tensor-mixed-linear-v1"
MODULE_TENSOR_PREFIXES = {
    "vision": ("vit.", "neck."),
    "text": ("text.",),
    "fusion": ("fenc.",),
    "decoder": ("ddec.", "geom.", "scoring.", "seg."),
}
QUANTIZATION_PROFILES = {
    "q8_0": {"storage_profile": "image-linear-q8_0-v1", "ggml_type": "Q8_0",
             "vision_storage_profile": "image-vision-linear-q8_0-v1",
             "file_type": 7, "block_elements": 32, "block_bytes": 34},
    "q6_k": {"storage_profile": "image-linear-q6_k-v1", "ggml_type": "Q6_K",
             "vision_storage_profile": "image-vision-linear-q6_k-v1",
             "file_type": 18, "block_elements": 256, "block_bytes": 210,
             "fallback_type": "Q8_0"},
    "q5_k": {"storage_profile": "image-linear-q5_k-v1", "ggml_type": "Q5_K",
             "vision_storage_profile": "image-vision-linear-q5_k-v1",
             "file_type": 16, "block_elements": 256, "block_bytes": 176,
             "fallback_type": "Q8_0"},
    "q4_k": {"storage_profile": "image-linear-q4_k-v1", "ggml_type": "Q4_K",
             "vision_storage_profile": "image-vision-linear-q4_k-v1",
             "file_type": 14, "block_elements": 256, "block_bytes": 144,
             "fallback_type": "Q8_0"},
}
SUPPORTED_STORAGE_PROFILES = tuple(
    profile
    for item in QUANTIZATION_PROFILES.values()
    for profile in (item["storage_profile"], item["vision_storage_profile"])
)
SUPPORTED_STORAGE_PROFILES += tuple(
    f"{FULL_MODULE_PROFILE_PREFIX}{precision}-v1" for precision in QUANTIZATION_PROFILES
)


def canonical_quantization_modules(modules):
    """Validate module names and return the canonical module order."""
    if isinstance(modules, str):
        if len(modules) > MAX_QUANTIZATION_MODULE_CSV_LENGTH:
            raise ValueError("quantization modules CSV is too long")
        values = modules.split(",")
        if len(values) > len(QUANTIZATION_MODULES):
            raise ValueError(f"quantization modules may contain at most {len(QUANTIZATION_MODULES)} entries")
    else:
        try:
            iterator = iter(modules)
        except TypeError as error:
            raise ValueError("quantization modules must be a CSV string or iterable of names") from error
        values = []
        for value in iterator:
            if len(values) >= len(QUANTIZATION_MODULES):
                raise ValueError(f"quantization modules may contain at most {len(QUANTIZATION_MODULES)} entries")
            values.append(value)
    if not values:
        raise ValueError("quantization modules must be a nonempty list of module names")
    if any(not isinstance(value, str) or len(value) > MAX_QUANTIZATION_MODULE_CSV_LENGTH
           or not value.strip() for value in values):
        raise ValueError("quantization modules must contain short, nonempty module names")
    values = [value.strip() for value in values]
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate quantization module: {', '.join(sorted(duplicates))}")
    unknown = sorted(seen - set(QUANTIZATION_MODULES))
    if unknown:
        raise ValueError(f"unsupported quantization module: {', '.join(unknown)}")
    return [module for module in QUANTIZATION_MODULES if module in seen]


def _schema4_quantization_profile(precision, modules, full=False):
    base = QUANTIZATION_PROFILES[precision]
    modules = canonical_quantization_modules(modules)
    if full and modules != list(QUANTIZATION_MODULES):
        raise ValueError("full module profile requires all four quantization modules")
    profile = {key: value for key, value in base.items() if key != "vision_storage_profile"}
    profile.update(storage_profile=(f"{FULL_MODULE_PROFILE_PREFIX}{precision}-v1" if full
                                   else f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1"),
                   schema_version=4, modules=modules, modules_csv=",".join(modules),
                   profile_status="candidate" if full else "diagnostic")
    return profile


def mixed_quantization_policy(base_precision, module_precisions):
    """Resolve bounded module overrides; hash a versioned, language-neutral encoding."""
    if not isinstance(base_precision, str) or base_precision not in QUANTIZATION_PROFILES:
        raise ValueError("mixed base_precision must be q8_0/q6_k/q5_k/q4_k")
    if not isinstance(module_precisions, dict) or set(module_precisions) - set(QUANTIZATION_MODULES):
        raise ValueError("module_precisions must map only vision/text/fusion/decoder")
    allowed = ("f32", *QUANTIZATION_PROFILES)
    if any(not isinstance(value, str) or value not in allowed for value in module_precisions.values()):
        raise ValueError("mixed module formats support f32/q8_0/q6_k/q5_k/q4_k; module F16 is not implemented")
    modules = {module: module_precisions.get(module, base_precision) for module in QUANTIZATION_MODULES}
    encoded = f"sam3:{MIXED_STORAGE_PROFILE}\nbase={base_precision}\n"
    encoded += "".join(f"{module}={value}\n" for module, value in modules.items())
    return {"base_precision": base_precision, "module_precisions": modules,
            "policy_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest()}


def mixed_quantization_profile(policy, storage_profile=None):
    if isinstance(policy, dict) and "tensor_precisions" in policy:
        return tensor_mixed_quantization_profile(policy, storage_profile)
    if not isinstance(policy, dict) or set(policy) - {"base_precision", "module_precisions", "policy_sha256"}:
        raise ValueError("mixed policy requires base_precision and module_precisions only")
    resolved = mixed_quantization_policy(policy.get("base_precision"), policy.get("module_precisions"))
    if "policy_sha256" in policy and policy["policy_sha256"] != resolved["policy_sha256"]:
        raise ValueError("mixed weight policy SHA-256 disagrees with module allocation")
    if storage_profile not in (None, MIXED_STORAGE_PROFILE):
        raise ValueError("mixed weights require image-mixed-linear-v1")
    base = QUANTIZATION_PROFILES[resolved["base_precision"]]
    return {**{key: value for key, value in base.items() if key != "vision_storage_profile"},
            **resolved, "storage_profile": MIXED_STORAGE_PROFILE, "schema_version": 5,
            "modules": [module for module, value in resolved["module_precisions"].items() if value != "f32"],
            "module_precisions_csv": ",".join(f"{module}={value}" for module, value in resolved["module_precisions"].items()),
            "profile_status": "candidate"}


def tensor_mixed_quantization_profile(policy, storage_profile=None):
    from tools.quantize.tensor_policy import validate_tensor_precisions, quantization_module_for_tensor
    if set(policy) - {"base_precision", "module_precisions", "tensor_precisions", "policy_sha256"}:
        raise ValueError("unknown field in exact-tensor weight policy")
    resolved = mixed_quantization_policy(policy.get("base_precision"), policy.get("module_precisions"))
    overrides = validate_tensor_precisions(policy["tensor_precisions"])
    encoded = f"sam3:{TENSOR_MIXED_STORAGE_PROFILE}\nbase={resolved['base_precision']}\n"
    encoded += "".join(f"{module}={value}\n" for module, value in resolved["module_precisions"].items())
    encoded += "".join(f"tensor:{name}={value}\n" for name, value in overrides.items())
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if "policy_sha256" in policy and policy["policy_sha256"] != digest:
        raise ValueError("tensor weight policy SHA-256 disagrees with the resolved allocation")
    if storage_profile not in (None, TENSOR_MIXED_STORAGE_PROFILE):
        raise ValueError("tensor overrides require image-tensor-mixed-linear-v1")
    base = QUANTIZATION_PROFILES[resolved["base_precision"]]
    selected = {module for module, value in resolved["module_precisions"].items() if value != "f32"}
    selected.update(quantization_module_for_tensor(name) for name, value in overrides.items() if value != "f32")
    return {**{key: value for key, value in base.items() if key != "vision_storage_profile"},
            "base_precision": resolved["base_precision"], "module_precisions": resolved["module_precisions"],
            "tensor_precisions": overrides, "policy_sha256": digest,
            "storage_profile": TENSOR_MIXED_STORAGE_PROFILE, "schema_version": 6,
            "modules": [module for module in QUANTIZATION_MODULES if module in selected],
            "module_precisions_csv": ",".join(f"{module}={value}" for module, value in resolved["module_precisions"].items()),
            "tensor_precisions_csv": ",".join(f"{name}={value}" for name, value in overrides.items()),
            "profile_status": "candidate"}


def parse_mixed_module_precisions(csv):
    if not isinstance(csv, str) or len(csv) > MAX_QUANTIZATION_MODULE_CSV_LENGTH:
        raise ValueError("mixed module precision CSV exceeds its bound")
    fields = csv.split(",")
    if len(fields) != len(QUANTIZATION_MODULES):
        raise ValueError("mixed module precision CSV requires all four modules")
    result = {}
    for module, field in zip(QUANTIZATION_MODULES, fields):
        key, separator, value = field.partition("=")
        if key != module or not separator or value not in ("f32", *QUANTIZATION_PROFILES):
            raise ValueError("mixed module precisions must be a canonical module=format CSV")
        result[key] = value
    return result


def quantization_profile(precision, storage_profile=None, quantization_modules=None, *, mixed_policy=None):
    if precision == "mixed":
        if quantization_modules is not None:
            raise ValueError("mixed weights use module_precisions instead of a selected module list")
        return mixed_quantization_profile(mixed_policy, storage_profile)
    if mixed_policy is not None:
        raise ValueError("mixed_policy requires precision=mixed")
    try:
        base = QUANTIZATION_PROFILES[precision]
    except KeyError as error:
        raise ValueError(f"unsupported quantized precision: {precision}") from error

    if quantization_modules is not None:
        modules = canonical_quantization_modules(quantization_modules)
        if storage_profile is None:
            return _schema4_quantization_profile(precision, modules)
        if storage_profile.startswith(FULL_MODULE_PROFILE_PREFIX):
            expected_full = f"{FULL_MODULE_PROFILE_PREFIX}{precision}-v1"
            if storage_profile != expected_full:
                raise ValueError(f"storage profile {storage_profile!r} does not match precision {precision}")
            return _schema4_quantization_profile(precision, modules, full=True)
        if storage_profile == f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1":
            return _schema4_quantization_profile(precision, modules)
        raise ValueError("storage profile disagrees with the selected quantization modules")

    full_profile = f"{FULL_MODULE_PROFILE_PREFIX}{precision}-v1"
    custom_profile = f"{CUSTOM_MODULE_PROFILE_PREFIX}{precision}-v1"
    if storage_profile == full_profile:
        return _schema4_quantization_profile(precision, QUANTIZATION_MODULES, full=True)
    if storage_profile == custom_profile:
        raise ValueError("custom module storage profile requires quantization modules")

    selected = base["storage_profile"] if storage_profile is None else storage_profile
    profile = {key: value for key, value in base.items() if key != "vision_storage_profile"}
    if selected == base["storage_profile"]:
        profile.update(storage_profile=selected, quantize_text_linear=True,
                       profile_status="diagnostic", schema_version=3, modules=None)
    elif selected == base["vision_storage_profile"]:
        profile.update(storage_profile=selected, quantize_text_linear=False,
                       profile_status="candidate", schema_version=3, modules=None)
    else:
        raise ValueError(f"storage profile {selected!r} does not match precision {precision}")
    return profile
