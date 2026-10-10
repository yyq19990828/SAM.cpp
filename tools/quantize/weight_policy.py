"""Dependency-free SAM 3 image weight allocation policies shared by tools."""

QUANTIZATION_MODULES = ("vision", "text", "fusion", "decoder")
MAX_QUANTIZATION_MODULE_CSV_LENGTH = 256
FULL_MODULE_PROFILE_PREFIX = "image-full-linear-"
CUSTOM_MODULE_PROFILE_PREFIX = "image-modules-linear-"
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


def quantization_profile(precision, storage_profile=None, quantization_modules=None):
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
