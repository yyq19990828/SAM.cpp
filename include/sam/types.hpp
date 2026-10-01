#ifndef SAM_CPP_SAM_TYPES_HPP
#define SAM_CPP_SAM_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sam {

enum class Backend { Auto, Cpu, Metal };

struct BackendOptions {
    Backend backend = Backend::Auto;
    int threads = 4;
};

struct ModelInfo {
    std::string architecture;
    std::string precision; // Container precision; CPU computation preserves values in F32.
    Backend backend = Backend::Cpu;
    int threads = 4;
    std::size_t tensor_count = 0, weight_bytes = 0; // Loaded representation, excluding buffer padding.
    bool tokenizer_compatibility_repaired = false;
};

struct RuntimeStats {
    std::uint64_t vision_encodes = 0, text_encodes = 0, inferences = 0;
    std::uint64_t cpu_nodes = 0, metal_nodes = 0, graph_partitions = 0;
    std::uint64_t host_upload_bytes = 0, host_download_bytes = 0;
    std::size_t weight_buffer_bytes = 0, compute_buffer_bytes = 0;
    double image_ms = 0.0, text_ms = 0.0, inference_ms = 0.0;
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

} // namespace sam

#endif // SAM_CPP_SAM_TYPES_HPP
