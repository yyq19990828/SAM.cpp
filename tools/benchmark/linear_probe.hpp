#ifndef SAM_CPP_TOOLS_BENCHMARK_LINEAR_PROBE_HPP
#define SAM_CPP_TOOLS_BENCHMARK_LINEAR_PROBE_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam_probe {

struct Input {
    std::uint32_t m = 0, n = 0, k = 0, sample_rows = 0, gelu = 0;
    float scaled_activation_max = 0;
    std::vector<float> weights, bias, scale, samples, reference;

    std::vector<float> expanded_input() const {
        std::vector<float> result(std::size_t(n) * k);
        for (std::size_t row = 0; row < n; ++row)
            std::copy_n(samples.data() + (row % sample_rows) * k, k, result.data() + row * k);
        return result;
    }
};

inline Input read_input(const std::filesystem::path& path) {
    const std::uint32_t one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1 || sizeof(float) != 4)
        throw std::runtime_error("linear probe requires a little-endian F32 host");
    std::ifstream stream(path, std::ios::binary);
    char magic[8]{};
    stream.read(magic, 8);
    if (std::string(magic, 8) != "SLPROB01") throw std::runtime_error("invalid linear probe input");
    Input input;
    for (auto* field : {&input.m, &input.n, &input.k, &input.sample_rows, &input.gelu})
        stream.read(reinterpret_cast<char*>(field), 4);
    stream.read(reinterpret_cast<char*>(&input.scaled_activation_max), 4);
    if (!stream || !input.m || !input.n || !input.k || !input.sample_rows || input.gelu > 1 ||
        input.m > 16384 || input.n > 16384 || input.k > 16384 || input.sample_rows > input.n ||
        !std::isfinite(input.scaled_activation_max) || input.scaled_activation_max < 0)
        throw std::runtime_error("invalid or excessive linear probe dimensions");
    const auto elements = std::uint64_t(input.m) * input.k + input.m + input.k +
        std::uint64_t(input.sample_rows) * (input.k + input.m);
    if (elements > (1ull << 28) || std::filesystem::file_size(path) != 32 + 4 * elements)
        throw std::runtime_error("linear probe payload length differs");
    auto read = [&](std::vector<float>& values, std::size_t count) {
        values.resize(count);
        stream.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(float)));
        if (!stream || !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("truncated or non-finite linear probe payload");
    };
    read(input.weights, std::size_t(input.m) * input.k);
    read(input.bias, input.m);
    read(input.scale, input.k);
    read(input.samples, std::size_t(input.sample_rows) * input.k);
    read(input.reference, std::size_t(input.sample_rows) * input.m);
    for (float scale : input.scale) if (scale <= 0) throw std::runtime_error("non-positive channel scale");
    return input;
}

struct Error {
    double relative_l2 = 0, max_abs = 0;
};

inline Error error(const std::vector<float>& values, const std::vector<float>& reference, std::size_t period) {
    if (values.empty() || reference.empty() || period != reference.size())
        throw std::runtime_error("invalid error reference");
    double difference_squared = 0, reference_squared = 0, maximum = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("non-finite output (including F16 overflow)");
        const double expected = reference[i % period], difference = double(values[i]) - expected;
        difference_squared += difference * difference;
        reference_squared += expected * expected;
        maximum = std::max(maximum, std::abs(difference));
    }
    return {std::sqrt(difference_squared / std::max(reference_squared, 1e-60)), maximum};
}

inline double quantile(std::vector<double> values, double fraction) {
    if (values.empty()) throw std::runtime_error("empty timing sample");
    std::sort(values.begin(), values.end());
    const double position = fraction * (values.size() - 1);
    const auto low = std::size_t(position), high = std::min(low + 1, values.size() - 1);
    return values[low] + (position - low) * (values[high] - values[low]);
}

inline double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

inline int iterations(const char* text) {
    std::size_t consumed = 0;
    const std::string value(text);
    const auto count = std::stoll(value, &consumed);
    if (consumed != value.size() || count < 1 || count > 10000)
        throw std::invalid_argument("iteration count must be an integer in [1,10000]");
    return int(count);
}

inline void write_array(std::ostream& stream, const std::vector<double>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) stream << ',';
        stream << values[i];
    }
    stream << ']';
}

inline void write_output(const std::filesystem::path& directory, const std::vector<float>& values) {
    std::ofstream stream(directory / "output.f32", std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing linear probe output");
}

inline void write_common(std::ostream& stream, const Input& input, const std::string& mode,
                         const std::vector<double>& wall_ms, const Error& numerical) {
    stream << std::setprecision(10)
        << "{\"schema_version\":1,\"diagnostic_only\":true,\"mode\":\"" << mode << "\",\"m\":" << input.m
        << ",\"n\":" << input.n << ",\"k\":" << input.k << ",\"gelu_erf\":" << (input.gelu ? "true" : "false")
        << ",\"output_vs_original_f32\":{\"relative_l2\":" << numerical.relative_l2
        << ",\"max_abs\":" << numerical.max_abs << "},\"wall_ms\":";
    write_array(stream, wall_ms);
    stream << ",\"wall_median_ms\":" << quantile(wall_ms, 0.5) << ",\"wall_p95_ms\":" << quantile(wall_ms, 0.95);
}

} // namespace sam_probe

#endif // SAM_CPP_TOOLS_BENCHMARK_LINEAR_PROBE_HPP
