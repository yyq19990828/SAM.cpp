#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "ggml.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <dlfcn.h>
#endif

#ifndef SAM_GGML_REVISION
#define SAM_GGML_REVISION "unknown"
#endif

#ifndef SAM_GGML_LIBRARY_PATH
#define SAM_GGML_LIBRARY_PATH "unknown"
#endif

namespace {

constexpr const char * usage =
    "usage: sam_quantize_rows --type q6_k|q5_k|q4_k --rows N --row-width N --input F32LE --output PACKED\n"
    "       sam_quantize_rows --identity";

bool parse_positive(const std::string & text, int64_t & value) {
    if (text.empty()) {
        return false;
    }
    try {
        size_t consumed = 0;
        const long long parsed = std::stoll(text, &consumed, 10);
        if (consumed != text.size() || parsed <= 0) {
            return false;
        }
        value = static_cast<int64_t>(parsed);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::string json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    escaped += "?";
                } else {
                    escaped += static_cast<char>(character);
                }
        }
    }
    return escaped;
}

bool write_exclusive(const std::filesystem::path & path, const uint8_t * data, size_t size) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        std::cerr << "cannot create output exclusively: " << std::strerror(errno) << '\n';
        return false;
    }
    bool ok = true;
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(fd, data + offset, size - offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            std::cerr << "failed writing packed output: " << std::strerror(errno) << '\n';
            ok = false;
            break;
        }
        offset += static_cast<size_t>(written);
    }
    if (ok && ::fsync(fd) != 0) {
        std::cerr << "failed syncing packed output: " << std::strerror(errno) << '\n';
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        std::cerr << "failed closing packed output: " << std::strerror(errno) << '\n';
        ok = false;
    }
    if (!ok) {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    return ok;
}

int fail(const std::string & message, int code = 2) {
    std::cerr << message << '\n' << usage << '\n';
    return code;
}

std::string quantize_library_path() {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    Dl_info info{};
    const auto address = reinterpret_cast<const void *>(
        reinterpret_cast<std::uintptr_t>(&ggml_quantize_chunk));
    if (dladdr(address, &info) != 0 && info.dli_fname != nullptr) {
        return info.dli_fname;
    }
#endif
    return SAM_GGML_LIBRARY_PATH;
}

int run(const std::map<std::string, std::string> & args) {
    const auto type_it = args.find("--type");
    const auto rows_it = args.find("--rows");
    const auto width_it = args.find("--row-width");
    const auto input_it = args.find("--input");
    const auto output_it = args.find("--output");
    if (args.size() != 5 || type_it == args.end() || rows_it == args.end() || width_it == args.end()
            || input_it == args.end() || output_it == args.end()) {
        return fail("missing or unsupported quantization arguments");
    }

    ggml_type type;
    if (type_it->second == "q6_k") {
        type = GGML_TYPE_Q6_K;
    } else if (type_it->second == "q5_k") {
        type = GGML_TYPE_Q5_K;
    } else if (type_it->second == "q4_k") {
        type = GGML_TYPE_Q4_K;
    } else {
        return fail("quantizer supports Q6_K, Q5_K, and Q4_K only");
    }

    int64_t rows = 0;
    int64_t row_width = 0;
    if (!parse_positive(rows_it->second, rows) || !parse_positive(width_it->second, row_width)) {
        return fail("rows and row width must be positive integers");
    }
    const int64_t block_size = ggml_blck_size(type);
    const size_t type_size = ggml_type_size(type);
    if (block_size <= 0 || type_size == 0 || row_width % block_size != 0) {
        return fail("row width is not divisible by the GGML block size");
    }
    if (rows > std::numeric_limits<int64_t>::max() / row_width) {
        return fail("matrix element count overflows int64");
    }
    const int64_t elements = rows * row_width;
    if (static_cast<uint64_t>(elements) > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return fail("F32 input byte count overflows size_t");
    }
    const size_t input_bytes = static_cast<size_t>(elements) * sizeof(float);
    if (input_bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return fail("F32 input byte count exceeds the stream read limit");
    }
    const size_t row_bytes = ggml_row_size(type, row_width);
    if (row_bytes == 0 || static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() / row_bytes) {
        return fail("quantized output byte count overflows size_t");
    }
    const size_t output_bytes = static_cast<size_t>(rows) * row_bytes;

    std::error_code error;
    const uintmax_t actual_input_bytes = std::filesystem::file_size(input_it->second, error);
    if (error || actual_input_bytes != input_bytes) {
        return fail("F32 input has an invalid or truncated byte count");
    }
    std::ifstream input(input_it->second, std::ios::binary);
    if (!input) {
        return fail("cannot open F32 input");
    }
    std::vector<float> values(static_cast<size_t>(elements));
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(input_bytes));
    if (!input || static_cast<size_t>(input.gcount()) != input_bytes) {
        return fail("failed reading complete F32 input");
    }
    const uint16_t endian_probe = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
        auto * bytes = reinterpret_cast<uint8_t *>(values.data());
        for (size_t offset = 0; offset < input_bytes; offset += 4) {
            std::swap(bytes[offset], bytes[offset + 3]);
            std::swap(bytes[offset + 1], bytes[offset + 2]);
        }
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            return fail("F32 input contains NaN or infinity");
        }
    }

    std::vector<uint8_t> packed(output_bytes);
    const size_t written = ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows, row_width, nullptr);
    if (written != output_bytes) {
        return fail("GGML quantizer returned an unexpected byte count", 1);
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (!ggml_validate_row_data(type, packed.data() + static_cast<size_t>(row) * row_bytes, row_bytes)) {
            return fail("GGML produced an invalid quantized row", 1);
        }
    }
    if (!write_exclusive(output_it->second, packed.data(), packed.size())) {
        return 1;
    }
    std::cout << packed.size() << '\n';
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--identity") {
        std::cout << "{\"ggml_revision\":\"" << json_escape(SAM_GGML_REVISION)
                  << "\",\"ggml_build_commit\":\"" << json_escape(ggml_commit())
                  << "\",\"ggml_version\":\"" << json_escape(ggml_version())
                  << "\",\"quantization_version\":" << GGML_QNT_VERSION
                  << ",\"ggml_library_path\":\"" << json_escape(SAM_GGML_LIBRARY_PATH)
                  << "\",\"ggml_quantize_library_path\":\"" << json_escape(quantize_library_path())
                  << "\"}\n";
        return 0;
    }
    std::map<std::string, std::string> args;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        if (key.rfind("--", 0) != 0 || !args.emplace(key, argv[index + 1]).second) {
            return fail("invalid or duplicate argument");
        }
    }
    if ((argc - 1) % 2 != 0) {
        return fail("argument is missing a value");
    }
    return run(args);
}
