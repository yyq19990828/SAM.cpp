# Third-Party Notices

## SAM 3 C++ Graph Implementation

Selected SAM 3 image/tracker graph, tokenizer, and checkpoint conversion code is adapted
from [PABannier/sam3.cpp](https://github.com/PABannier/sam3.cpp/tree/416186c501d060df7ca02989d49b38080f5f81f3),
revision `416186c501d060df7ca02989d49b38080f5f81f3`.

Copyright (c) 2025-2026 Pierre-Antoine Bannier. Distributed under the MIT license;
the original text is retained in [licenses/sam3.cpp-MIT.txt](licenses/sam3.cpp-MIT.txt).
Other model families, GUI, video decoding, and the community association,
lifecycle and memory-cache implementation are not part of this port. Tracker
math is adapted separately, with full temporal integration still pending.

## GGML

The supported dependency is official
[GGML 0.25.3](https://github.com/ggml-org/ggml/tree/353b63b439f27ab2cc19dac97ab1681ba6d2d084),
revision `353b63b439f27ab2cc19dac97ab1681ba6d2d084`. It is fetched separately or
provided by the embedding application's existing `ggml` CMake target.

SAM's [local Metal patch](cmake/patches/README.md) adds explicit FP32
matrix/attention behavior and native window partition/restoration. The window
operators are adapted from the MIT-licensed
[PABannier GGML revision `7a466633`](https://github.com/PABannier/ggml/commit/7a466633195d336e08dd5fec8b8ab42a2c5cb5e3).
CMake applies the patch to a separate source copy and verifies
the patched hashes; the shared upstream checkout remains unchanged. Caller-owned
GGML targets need this patch or an equivalent precision implementation.

Copyright (c) 2023-2026 The ggml authors. Distributed under the MIT license;
the original text is retained in [licenses/ggml-MIT.txt](licenses/ggml-MIT.txt).

## GGUF Python Tools

The conversion and validation environment uses
[`gguf` 0.19.0](https://pypi.org/project/gguf/0.19.0/), published from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/a290ce626663dae1d54f70bce3ca6d8f67aab62f/gguf-py).
It provides GGUF serialization and inspection; it is not a C++ runtime dependency.
The package's MIT license is retained in
[licenses/gguf-py-MIT.txt](licenses/gguf-py-MIT.txt). Its version is pinned in
`tools/requirements.lock`.

## stb Image Codecs

`examples/stb/stb_image.h` and `examples/stb/stb_image_write.h` are copied unchanged
from the pinned `PABannier/sam3.cpp` revision above. Sean Barrett and the stb
contributors offer these files under either the MIT license or the public-domain
Unlicense; their complete license and attribution notices remain in each header.
They compile only in the example/test image I/O translation unit and are not
included by the SAM public headers.

## SAM Model and Official Reference

The official behavioral reference is
[Meta SAM 3](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40),
revision `2345a4ad109ac29c569da749c91d84f10dc08c40`. Its SAM model license is
retained separately in [licenses/SAM-model-license.txt](licenses/SAM-model-license.txt).
Model checkpoints and upstream media are external artifacts; these C++ dependency
licenses do not grant rights to those artifacts. Conversion/reference tools
record artifact provenance, and runtime inference does not download weights.

The [visual example gallery](docs/visual-examples.md) contains comparison images
derived from the pinned Meta SAM 3 `assets/images/truck.jpg` and `groceries.jpg`,
with actual inference masks rendered by SAM.cpp. Those source-media pixels and
derived comparison images follow the separately retained [SAM license](licenses/SAM-model-license.txt),
rather than the C++ code's MIT license. Credit: Meta, official SAM 3 example
media and FP32 model reference.
