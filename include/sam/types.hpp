#ifndef SAM_CPP_SAM_TYPES_HPP
#define SAM_CPP_SAM_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sam {

enum class Backend { Auto, Cpu, Metal, Cuda };
enum class CudaComputeMode { F32, F16 };

struct BackendOptions {
    Backend backend = Backend::Auto;
    int threads = 4;
    int cuda_device = 0; // Index among devices visible to the CUDA registry.
    CudaComputeMode cuda_compute = CudaComputeMode::F32; // Opt-in reduced operands; GGUF weight storage is unchanged.
};

struct ModelInfo {
    std::string architecture;
    std::string precision; // Storage precision; arithmetic follows the backend and profile.
    Backend backend = Backend::Cpu;
    int threads = 4;
    std::size_t tensor_count = 0, weight_bytes = 0; // Loaded representation, excluding buffer padding.
    bool tokenizer_compatibility_repaired = false;
    std::string task;
    std::string profile;
    std::string storage_profile; // Versioned weight policy, separate from the temporal profile.
    std::string arithmetic_profile; // Runtime arithmetic contract, separate from weight storage; empty for legacy F32 math.
    std::vector<std::string> quantization_modules; // Quantized modules for schema-4/5 image profiles.
    std::string device_name;
    int cuda_device = -1; // Resolved visible CUDA index; -1 on other backends.
    std::string base_precision; // Schema 5 only; general.file_type describes this base.
    std::vector<std::pair<std::string, std::string>> module_precisions; // Canonical module order, including F32 modules.
    std::string policy_sha256; // Schema-5 allocation identity, separate from file/source hashes.
};

struct RuntimeStats {
    // Backend node counts exclude metadata-only views/reshapes/permutations.
    std::uint64_t vision_encodes = 0, text_encodes = 0, inferences = 0;
    std::uint64_t cpu_nodes = 0, metal_nodes = 0, graph_partitions = 0;
    std::uint64_t host_upload_bytes = 0, host_download_bytes = 0;
    std::size_t weight_buffer_bytes = 0, compute_buffer_bytes = 0;
    double image_ms = 0.0, text_ms = 0.0, inference_ms = 0.0;
    std::uint64_t blas_nodes = 0; // Subset of cpu_nodes, not additional device work.
    std::uint64_t cuda_nodes = 0;
};

struct TensorData {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    std::string layout;
};

// Borrowed interleaved RGB8. The final row need not include stride padding.
struct ImageView {
    const std::uint8_t* data = nullptr;
    std::size_t size_bytes = 0;
    int width = 0;
    int height = 0;
    std::size_t row_stride = 0;
};

struct Box {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

struct Mask {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> data;
};

struct Detection {
    Box box;
    float score = 0.0f;
    Mask mask;
    int query_index = -1;
};

struct Result {
    std::vector<Detection> detections;
};

struct VideoOptions {
    int max_objects = 8;
};

struct TrackedObject {
    std::uint64_t id = 0;
    Box box;
    float score = 0.0f;
    Mask mask;
};

struct VideoFrameResult {
    int frame_index = 0;
    std::vector<TrackedObject> objects;
};

struct VideoStats {
    RuntimeStats runtime;
    std::uint64_t accepted_frames = 0, emitted_frames = 0, tracker_calls = 0;
    std::uint64_t rejected_new_objects = 0;
    std::size_t active_objects = 0, retained_records = 0, retained_memory_bytes = 0;
    std::size_t pending_frames = 0, pending_high_water = 0, retained_records_high_water = 0;
    double frame_ms = 0.0, tracker_ms = 0.0, memory_ms = 0.0;
};

} // namespace sam

#endif // SAM_CPP_SAM_TYPES_HPP
