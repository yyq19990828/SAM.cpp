#ifndef SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_SUPPORT_HPP
#define SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_SUPPORT_HPP

#include <sam/sam.hpp>
#include "image_io.hpp"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

namespace sam_example {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path model;
    std::filesystem::path image;
    std::filesystem::path output;
    std::string text;
    sam::BackendOptions backend;
    float score_threshold = 0.5f;
    int repeat = 1;
    bool help = false;
};

inline int positive_integer(std::string_view value, const std::string& name) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result <= 0) {
        throw std::invalid_argument(name + " requires a positive integer");
    }
    return result;
}

inline int nonnegative_integer(std::string_view value, const std::string& name) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || result < 0) {
        throw std::invalid_argument(name + " requires a nonnegative integer");
    }
    return result;
}

inline Options parse_options(int argc, char** argv, bool allow_repeat = true) {
    Options options;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--help") {
            if (argc != 2) {
                throw std::invalid_argument("--help must be used alone");
            }
            options.help = true;
            return options;
        }
        if (name != "--model" && name != "--image" && name != "--text" &&
            name != "--output" && name != "--backend" && name != "--threads" &&
            name != "--cuda-device" && name != "--cuda-compute" && name != "--cpu-compute" &&
            name != "--score-threshold" && (name != "--repeat" || !allow_repeat)) {
            throw std::invalid_argument("Unknown argument: " + name);
        }
        if (!seen.insert(name).second) {
            throw std::invalid_argument("Duplicate argument: " + name);
        }
        if (++i >= argc) {
            throw std::invalid_argument("Missing value for " + name);
        }
        const std::string value = argv[i];
        if (value.empty() || value.rfind("--", 0) == 0) {
            throw std::invalid_argument("Missing value for " + name);
        }
        if (name == "--model") options.model = value;
        else if (name == "--image") options.image = value;
        else if (name == "--text") options.text = value;
        else if (name == "--output") options.output = value;
        else if (name == "--threads") options.backend.threads = positive_integer(value, name);
        else if (name == "--cuda-device") options.backend.cuda_device = nonnegative_integer(value, name);
        else if (name == "--cuda-compute") {
            if (value == "f32") options.backend.cuda_compute = sam::CudaComputeMode::F32;
            else if (value == "f16") options.backend.cuda_compute = sam::CudaComputeMode::F16;
            else throw std::invalid_argument("--cuda-compute must be f32 or f16");
        }
        else if (name == "--cpu-compute") {
            if (value == "f32") options.backend.cpu_compute = sam::CpuComputeMode::F32;
            else if (value == "native-quantized") options.backend.cpu_compute = sam::CpuComputeMode::NativeQuantized;
            else throw std::invalid_argument("--cpu-compute must be f32 or native-quantized");
        }
        else if (name == "--repeat") options.repeat = positive_integer(value, name);
        else if (name == "--backend") {
            if (value == "auto") options.backend.backend = sam::Backend::Auto;
            else if (value == "cpu") options.backend.backend = sam::Backend::Cpu;
            else if (value == "metal") options.backend.backend = sam::Backend::Metal;
            else if (value == "cuda") options.backend.backend = sam::Backend::Cuda;
            else throw std::invalid_argument("--backend must be auto, cpu, metal, or cuda");
        } else {
            char* end = nullptr;
            errno = 0;
            options.score_threshold = std::strtof(value.c_str(), &end);
            if (errno == ERANGE || value.find_first_of(" \t\r\n\v\f") != std::string::npos ||
                end != value.c_str() + value.size() || !std::isfinite(options.score_threshold) ||
                options.score_threshold < 0 || options.score_threshold > 1) {
                throw std::invalid_argument("--score-threshold must be finite and in [0,1]");
            }
        }
    }
    if (seen.count("--cuda-device") && options.backend.backend != sam::Backend::Cuda)
        throw std::invalid_argument("--cuda-device requires --backend cuda");
    if (seen.count("--cuda-compute") && options.backend.backend != sam::Backend::Cuda)
        throw std::invalid_argument("--cuda-compute requires --backend cuda");
    if (seen.count("--cpu-compute") && options.backend.backend != sam::Backend::Cpu)
        throw std::invalid_argument("--cpu-compute requires --backend cpu");
    for (const char* name : {"--model", "--image", "--text", "--output"}) {
        if (!seen.count(name)) {
            throw std::invalid_argument(std::string("Required argument: ") + name);
        }
    }
    return options;
}

inline const char* backend_name(sam::Backend backend) {
    switch (backend) {
        case sam::Backend::Auto: return "auto";
        case sam::Backend::Cpu: return "cpu";
        case sam::Backend::Metal: return "metal";
        case sam::Backend::Cuda: return "cuda";
    }
    throw std::runtime_error("Unknown resolved backend");
}

inline void write_compute_policy(std::ostream& stream, const sam::BackendOptions& options) {
    const std::string cpu = options.cpu_compute == sam::CpuComputeMode::NativeQuantized ? "native-quantized" : "f32";
    const std::string cuda = options.cuda_compute == sam::CudaComputeMode::F16 ? "f16" : "f32";
    stream << ",\"compute_mode\":\"" << (options.backend == sam::Backend::Cuda ? cuda : cpu)
           << "\",\"cpu_compute\":\"" << cpu << "\",\"cuda_compute\":\"" << cuda << '"';
}

inline std::string json_string(std::string_view value) {
    std::string result = "\"";
    const char* hex = "0123456789abcdef";
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + '"';
}

inline void write_model_profile(std::ostream& stream, const sam::ModelInfo& info) {
    stream << ",\"precision\":" << json_string(info.precision)
           << ",\"storage_profile\":" << json_string(info.storage_profile)
           << ",\"arithmetic_profile\":" << json_string(info.arithmetic_profile)
           << ",\"quantization_modules\":[";
    for (std::size_t i = 0; i < info.quantization_modules.size(); ++i) {
        if (i) stream << ',';
        stream << json_string(info.quantization_modules[i]);
    }
    stream << ']' << ",\"device_name\":" << json_string(info.device_name)
           << ",\"cuda_device\":" << info.cuda_device;
    if (info.precision == "mixed") {
        stream << ",\"base_precision\":" << json_string(info.base_precision) << ",\"module_precisions\":{";
        for (std::size_t i = 0; i < info.module_precisions.size(); ++i) {
            if (i) stream << ',';
            stream << json_string(info.module_precisions[i].first) << ':' << json_string(info.module_precisions[i].second);
        }
        stream << "},\"policy_sha256\":" << json_string(info.policy_sha256);
        if (!info.tensor_precisions.empty()) {
            stream << ",\"tensor_precisions\":{";
            for (std::size_t i = 0; i < info.tensor_precisions.size(); ++i) {
                if (i) stream << ',';
                stream << json_string(info.tensor_precisions[i].first) << ':' << json_string(info.tensor_precisions[i].second);
            }
            stream << '}';
        }
    }
}

inline std::ofstream output_file(const std::filesystem::path& path, bool binary = false) {
    std::ofstream stream(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode(0)));
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);
    return stream;
}

inline void write_tensor(const std::filesystem::path& path, const sam::TensorData& tensor) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "Reference dumps require IEEE-754 binary32");
    std::size_t count = 1;
    if (tensor.shape.empty()) throw std::runtime_error("Tensor shape is empty");
    for (const auto dimension : tensor.shape) {
        if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
                                  std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error("Tensor shape is invalid");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    if (count != tensor.values.size()) throw std::runtime_error("Tensor size does not match shape");
    auto file = output_file(path, true);
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&endian) == 1) {
        file.write(reinterpret_cast<const char*>(tensor.values.data()),
                   static_cast<std::streamsize>(count * sizeof(float)));
    } else {
        for (const float value : tensor.values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const char bytes[4] = {static_cast<char>(bits), static_cast<char>(bits >> 8),
                                   static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)};
            file.write(bytes, sizeof(bytes));
        }
    }
    file.close();
}

class OutputDirectory {
public:
    explicit OutputDirectory(const std::filesystem::path& path, bool keep_partial = false)
        : path_(path), keep_partial_(keep_partial) {
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error("Output directory already exists: " + path_.string());
        }
    }
    OutputDirectory(const OutputDirectory&) = delete;
    OutputDirectory& operator=(const OutputDirectory&) = delete;
    ~OutputDirectory() {
        if (!complete_ && !keep_partial_) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }
    }
    void complete() { complete_ = true; }

private:
    std::filesystem::path path_;
    bool complete_ = false, keep_partial_ = false;
};

inline sam::ImageView image_view(const Image& image) {
    return {image.rgb.data(), image.rgb.size(), image.width, image.height,
            static_cast<std::size_t>(image.width) * 3};
}

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

inline std::size_t process_peak_rss_bytes() {
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#else
    return 0;
#endif
}

inline void write_runtime_stats(std::ostream& stream, const sam::RuntimeStats& stats) {
    stream << "{\"vision_encodes\":" << stats.vision_encodes
           << ",\"text_encodes\":" << stats.text_encodes << ",\"inferences\":" << stats.inferences
           << ",\"cpu_nodes\":" << stats.cpu_nodes << ",\"metal_nodes\":" << stats.metal_nodes
           << ",\"blas_nodes\":" << stats.blas_nodes
           << ",\"cuda_nodes\":" << stats.cuda_nodes
           << ",\"graph_partitions\":" << stats.graph_partitions
           << ",\"host_upload_bytes\":" << stats.host_upload_bytes
           << ",\"host_download_bytes\":" << stats.host_download_bytes
           << ",\"weight_buffer_bytes\":" << stats.weight_buffer_bytes
           << ",\"compute_buffer_bytes\":" << stats.compute_buffer_bytes
           << ",\"image_ms\":" << stats.image_ms
           << ",\"text_ms\":" << stats.text_ms
           << ",\"inference_ms\":" << stats.inference_ms
           << ",\"process_peak_rss_bytes\":" << process_peak_rss_bytes() << "}";
}

inline void write_detections(std::ostream& stream, const sam::Result& result,
                             const std::filesystem::path& directory, bool raw_masks) {
    stream << "[";
    std::set<int> query_indices;
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        if (i) stream << ",";
        if (!std::isfinite(detection.score) || detection.score < 0 || detection.score > 1 ||
            !std::isfinite(detection.box.x0) ||
            !std::isfinite(detection.box.y0) || !std::isfinite(detection.box.x1) ||
            !std::isfinite(detection.box.y1)) {
            throw std::runtime_error("Inference returned a non-finite detection");
        }
        if (detection.query_index < 0 || !query_indices.insert(detection.query_index).second ||
            detection.mask.width <= 0 || detection.mask.height <= 0 ||
            static_cast<std::size_t>(detection.mask.width) > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(detection.mask.height) ||
            detection.mask.data.size() != static_cast<std::size_t>(detection.mask.width) * detection.mask.height ||
            std::any_of(detection.mask.data.begin(), detection.mask.data.end(),
                        [](std::uint8_t value) { return value > 1; })) {
            throw std::runtime_error("Inference returned an invalid mask or query index");
        }
        const std::string name = "mask-" + std::to_string(detection.query_index) +
                                 (raw_masks ? ".bin" : ".png");
        if (raw_masks) {
            auto mask_file = output_file(directory / name, true);
            mask_file.write(reinterpret_cast<const char*>(detection.mask.data.data()),
                            static_cast<std::streamsize>(detection.mask.data.size()));
            mask_file.close();
        } else {
            write_mask_png(directory / name, detection.mask.width, detection.mask.height,
                           detection.mask.data);
        }
        stream << "{\"query_index\":" << detection.query_index << ",\"score\":"
               << detection.score << ",\"box\":[" << detection.box.x0 << ','
               << detection.box.y0 << ',' << detection.box.x1 << ',' << detection.box.y1
               << "],\"mask\":{\"file\":" << json_string(name)
               << ",\"dtype\":\"uint8\",\"shape\":[" << detection.mask.height << ','
               << detection.mask.width << "]}}";
    }
    stream << "]";
}

} // namespace sam_example

#endif // SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_SUPPORT_HPP
