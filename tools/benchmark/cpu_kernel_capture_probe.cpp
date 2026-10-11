#include "cpu_quantized_matmul.hpp"
#include "../../support/image_io/image_support.hpp"
#include "ggml-cpu.h"
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

using Callback = void (*)(const ggml_tensor*, const char*, const char*, ggml_type,
                          const void*, std::size_t, std::int64_t, void*);
using Install = void (*)(Callback, void*);

struct Event {
    std::string phase, path, tensor;
    ggml_type type;
    std::int64_t wall_us = 0;
    std::vector<char> scratch;
};

struct Capture {
    Install install = nullptr;
    std::vector<Event> events;
    std::exception_ptr error;

    Capture() {
        const auto reg = ggml_backend_reg_by_name("CPU");
        install = reinterpret_cast<Install>(ggml_backend_reg_get_proc_address(reg, "ggml_cpu_sam_set_observer_v1"));
        if (!install) throw std::runtime_error("pinned CPU kernel capture extension is unavailable");
    }
    ~Capture() { install(nullptr, nullptr); }
    void start() { events.clear(); error = {}; install(observe, this); }
    void stop() { install(nullptr, nullptr); if (error) std::rethrow_exception(error); }
    static void observe(const ggml_tensor* dst, const char* phase, const char* path, ggml_type type,
                        const void* data, std::size_t bytes, std::int64_t wall_us, void* user) noexcept {
        auto& self = *static_cast<Capture*>(user);
        try {
            if (self.error) return;
            Event row{phase, path, ggml_get_name(dst), type, wall_us, {}};
            if (bytes) {
                if (!data || bytes > 16 * 1024 * 1024) throw std::runtime_error("invalid or excessive captured CPU scratch");
                row.scratch.assign(static_cast<const char*>(data), static_cast<const char*>(data) + bytes);
            }
            self.events.push_back(std::move(row));
        } catch (...) { self.error = std::current_exception(); }
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void binary(const std::filesystem::path& path, const std::vector<T>& values) {
    auto file = sam_example::output_file(path, true);
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    require(static_cast<bool>(file), "cannot save captured CPU payload");
}

// Decode the captured wire bytes directly; the runtime's packing routine is
// not called here. Q8_K's sums are checked against its actual signed codes.
std::vector<float> decode(ggml_type type, const std::vector<char>& payload, std::size_t elements) {
    const std::size_t block = type == GGML_TYPE_Q8_0 ? 32 : 256;
    const std::size_t bytes = type == GGML_TYPE_Q8_0 ? 34 : 292;
    require(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q8_K, "unexpected captured CPU RHS type");
    require(elements % block == 0 && payload.size() == elements / block * bytes, "captured CPU RHS size differs");
    std::vector<float> result(elements);
    for (std::size_t begin = 0; begin < elements; begin += block) {
        const auto* source = payload.data() + begin / block * bytes;
        float scale = 0;
        const std::size_t offset = type == GGML_TYPE_Q8_0 ? 2 : 4;
        if (type == GGML_TYPE_Q8_0) {
            ggml_fp16_t half;
            std::memcpy(&half, source, sizeof(half));
            scale = ggml_fp16_to_fp32(half);
        } else std::memcpy(&scale, source, sizeof(scale));
        require(std::isfinite(scale), "captured CPU scale is nonfinite");
        for (std::size_t i = 0; i < block; ++i) {
            std::int8_t code;
            std::memcpy(&code, source + offset + i, 1);
            result[begin + i] = scale * code;
        }
        if (type == GGML_TYPE_Q8_K) for (std::size_t group = 0; group < 16; ++group) {
            int sum = 0;
            for (std::size_t i = 0; i < 16; ++i) {
                std::int8_t code;
                std::memcpy(&code, source + offset + group * 16 + i, 1);
                sum += code;
            }
            std::int16_t stored;
            std::memcpy(&stored, source + 260 + group * 2, sizeof(stored));
            require(stored == sum, "captured Q8_K subblock sum differs from actual codes");
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-tiled-support") {
            ggml_backend_load_all();
            const bool supported = ggml_cpu_has_avx() || ggml_cpu_has_avx2() || ggml_cpu_has_avx512_vnni();
            std::cout << "CPU tiled AVX implementation " << (supported ? "available" : "unavailable") << '\n';
            return supported ? 0 : 77;
        }
        if (argc != 3 || std::string(argv[1]) != "--output")
            throw std::invalid_argument("usage: sam_cpu_kernel_capture_probe --output NEW_DIR | --check-tiled-support");
        const std::filesystem::path output = argv[2];
        sam_example::OutputDirectory directory(output, true);
        std::ostringstream rows;
        rows << std::setprecision(12);
        std::size_t case_id = 0, captures = 0;
        double maximum_error = 0;
        for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K})
            for (const auto threads : {1, 4}) for (const auto n : {3, 33}) {
                sam_cpu_probe::Fixture fixture(type, 256, 99, n, "outlier", true, n == 3);
                sam_cpu_probe::Matmul execution(fixture, sam::CpuComputeMode::NativeQuantized, threads);
                Capture capture;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    if (repeat) {
                        for (auto& value : fixture.rhs) value *= -.5f;
                        fixture.pack_rhs(fixture.packed_rhs);
                        execution.upload_rhs(fixture.rhs);
                    }
                    capture.start();
                    execution.compute();
                    capture.stop();
                    require(capture.events.size() == 6, "kernel capture omitted a QKV packing/arithmetic boundary");
                    const auto actual = execution.read();
                    execution.check_resident_weights();
                    const auto prefix = "case-" + std::to_string(case_id++);
                    binary(output / (prefix + ".input.f32"), fixture.rhs);
                    binary(output / (prefix + ".weights.bin"), fixture.packed);
                    binary(output / (prefix + ".output.f32"), actual);
                    for (std::size_t part = 0; part < 3; ++part) {
                        const auto& packing = capture.events[part * 2];
                        const auto& arithmetic = capture.events[part * 2 + 1];
                        require(packing.phase == "packing" && arithmetic.phase == "arithmetic" &&
                            packing.path == arithmetic.path && packing.tensor == arithmetic.tensor &&
                            packing.wall_us >= 0 && arithmetic.wall_us >= 0 && arithmetic.scratch.empty(),
                            "kernel capture phase pairing differs");
                        require(packing.scratch == fixture.packed_rhs, "actual kernel scratch differs from encoder reference");
                        fixture.decoded_rhs = decode(packing.type, packing.scratch, fixture.rhs.size());
                        const auto accuracy = sam_cpu_probe::validate(fixture, actual, true);
                        maximum_error = std::max(maximum_error, accuracy.max_effective_error);
                        const auto payload = prefix + ".part-" + std::to_string(part) + ".rhs.bin";
                        binary(output / payload, packing.scratch);
                        if (captures++) rows << ',';
                        rows << "{\"case\":" << case_id - 1 << ",\"repeat\":" << repeat << ",\"part\":" << part
                             << ",\"threads\":" << threads << ",\"k\":256,\"m\":99,\"n\":" << n
                             << ",\"strided_rhs\":" << (n == 3 ? "true" : "false")
                             << ",\"weight_type\":" << sam_example::json_string(ggml_type_name(type))
                             << ",\"path\":" << sam_example::json_string(packing.path)
                             << ",\"rhs_type\":" << sam_example::json_string(ggml_type_name(packing.type))
                             << ",\"scratch_bytes\":" << packing.scratch.size() << ",\"packing_wall_us\":" << packing.wall_us
                             << ",\"arithmetic_wall_us\":" << arithmetic.wall_us
                             << ",\"payload\":" << sam_example::json_string(payload)
                             << ",\"max_decoded_scalar_error\":" << accuracy.max_effective_error << '}';
                    }
                }
                const auto count = capture.events.size();
                execution.compute();
                require(capture.events.size() == count, "CPU callback survived diagnostic removal");
            }
        auto report = sam_example::output_file(output / "report.json");
        report << std::setprecision(12)
               << "{\"schema_version\":1,\"kind\":\"sam-cpu-kernel-capture-v1\",\"complete\":true,"
                  "\"diagnostic_only\":true,\"performance_comparable\":false,"
                  "\"scope\":\"actual synchronized CPU dot/tiled RHS scratch; bounded synthetic QKV operators\","
                  "\"timing_scope\":\"kernel phase wall time including worker barriers; scratch-copy callback excluded from arithmetic; not pure instruction time\","
                  "\"ggml_commit\":" << sam_example::json_string(ggml_commit())
               << ",\"cases\":" << case_id << ",\"captures\":" << captures
               << ",\"maximum_decoded_scalar_error\":" << maximum_error << ",\"events\":[" << rows.str() << "]}\n";
        report.close();
        directory.complete();
        std::cout << "Captured " << captures << " real CPU packing/arithmetic pairs; max scalar error=" << maximum_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_cpu_kernel_capture_probe: " << error.what() << '\n';
        return 1;
    }
}
