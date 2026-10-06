#include "sam/internal/models/sam3/image_ops.hpp"
#include "sam/internal/models/sam3/model.hpp"
#include "sam/internal/models/sam3/tokenizer.hpp"
#include "sam/internal/models/sam3/weights.hpp"
#include "sam/internal/input_validation.hpp"
#include "sam/internal/io/gguf_reader.hpp"
#include "../examples/image_support.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_stats_aggregate_compatibility() {
    const sam::RuntimeStats stats{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.25, 12.5, 13.75};
    require(stats.host_upload_bytes == 7 && stats.host_download_bytes == 8 &&
            stats.weight_buffer_bytes == 9 && stats.compute_buffer_bytes == 10 &&
            stats.image_ms == 11.25 && stats.text_ms == 12.5 && stats.inference_ms == 13.75 &&
            stats.blas_nodes == 0 && stats.cuda_nodes == 0, "Legacy stats aggregate positions changed");
    const sam::ModelInfo info{"sam3", "q8_0", sam::Backend::Cpu, 2, 1, 34, false};
    require(info.arithmetic_profile.empty() && info.quantization_modules.empty() &&
            info.device_name.empty() && info.cuda_device == -1,
            "Legacy ModelInfo aggregate fields changed or schema-4 modules leaked into old models");
}

void test_model_quantization_modules_json() {
    sam::ModelInfo info;
    std::ostringstream stream;
    sam_example::write_model_profile(stream, info);
    require(stream.str().find(",\"quantization_modules\":[]") != std::string::npos,
            "legacy model JSON profile must emit an empty quantization module array");
    info.quantization_modules = {"vision", "text", "fusion", "decoder"};
    std::ostringstream modular;
    sam_example::write_model_profile(modular, info);
    require(modular.str().find(",\"quantization_modules\":[\"vision\",\"text\",\"fusion\",\"decoder\"]") != std::string::npos,
            "schema-4 module names were not serialized as a canonical JSON array");
    info.device_name = "NVIDIA GPU";
    info.cuda_device = 2;
    std::ostringstream device;
    sam_example::write_model_profile(device, info);
    require(device.str().find(",\"device_name\":\"NVIDIA GPU\",\"cuda_device\":2") != std::string::npos,
            "selected CUDA device identity was not serialized");
}

template<class Exception, class Function>
void rejects(Function function, const char* message) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("sam-contracts-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { require(std::filesystem::create_directory(path), "cannot create contract-test directory"); }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

void write_fixture(const std::filesystem::path& path,
                   const std::function<void(gguf_context*)>& change = {}, bool second_tensor = false) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({65536, nullptr, false}), &ggml_free);
    require(context != nullptr, "cannot allocate small GGUF fixture");
    auto* tensor = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 1, 3, 2);
    ggml_set_name(tensor, "probe.weight");
    const std::array<float, 6> values{-2, -1, 0, 1, 2, 3};
    std::memcpy(tensor->data, values.data(), sizeof(values));
    sam::internal::GgufPtr file(gguf_init_empty());
    gguf_set_val_str(file.get(), "general.architecture", "sam3");
    gguf_set_val_u32(file.get(), "sam.schema_version", 1);
    gguf_set_val_str(file.get(), "sam.task", "text_image");
    gguf_set_val_u32(file.get(), "probe.scalar", 17);
    gguf_set_val_str(file.get(), "probe.label", "labelx");
    const std::array<std::uint32_t, 2> indices{3, 5};
    gguf_set_arr_data(file.get(), "probe.indices", GGUF_TYPE_UINT32, indices.data(), indices.size());
    const char* labels[] = {"first", "second"};
    gguf_set_arr_str(file.get(), "probe.labels", labels, 2);
    gguf_add_tensor(file.get(), tensor);
    if (second_tensor) {
        ggml_tensor second = *tensor;
        ggml_set_name(&second, "other.weight");
        gguf_add_tensor(file.get(), &second);
    }
    if (change) change(file.get());
    require(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot write GGUF fixture");
}

void write_quant_fixture(const std::filesystem::path& path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({65536, nullptr, false}), &ggml_free);
    require(context != nullptr, "cannot allocate quantized GGUF fixture");
    auto* tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q8_0, 32, 2);
    ggml_set_name(tensor, "probe.quant");
    std::array<float, 32> row{};
    for (int i = 0; i < 32; ++i) row[i] = static_cast<float>(i - 16) / 16.0f;
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (int i = 0; i < 2; ++i)
        traits->from_float_ref(row.data(), static_cast<char*>(tensor->data) + i * traits->type_size, row.size());
    sam::internal::GgufPtr file(gguf_init_empty());
    gguf_add_tensor(file.get(), tensor);
    require(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot write quantized GGUF fixture");
}

void write_schema4_module_fixture(const std::filesystem::path& path, const char* modules,
                                  bool missing_modules = false, bool wrong_module_type = false,
                                  const char* profile = "image-full-linear-q8_0-v1") {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4096, nullptr, false}), &ggml_free);
    require(context != nullptr, "cannot allocate schema-4 GGUF fixture");
    auto* tensor = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, 2);
    ggml_set_name(tensor, "probe.weight");
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
    sam::internal::GgufPtr file(gguf_init_empty());
    gguf_set_val_str(file.get(), "general.architecture", "sam3");
    gguf_set_val_u32(file.get(), "sam.schema_version", 4);
    gguf_set_val_str(file.get(), "sam.task", "text_image");
    gguf_set_val_u32(file.get(), "general.file_type", 7);
    gguf_set_val_u32(file.get(), "general.quantization_version", 2);
    gguf_set_val_str(file.get(), "sam.storage_profile", profile);
    if (!missing_modules) {
        if (wrong_module_type) gguf_set_val_u32(file.get(), "sam.quantization.modules", 1);
        else gguf_set_val_str(file.get(), "sam.quantization.modules", modules);
    }
    gguf_add_tensor(file.get(), tensor);
    require(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot write schema-4 GGUF fixture");
}

std::vector<char> fixture_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void mutate_fixture(const std::filesystem::path& path, const std::function<void(std::vector<char>&)>& change) {
    auto bytes = fixture_bytes(path);
    change(bytes);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream), "cannot mutate GGUF fixture");
}

void integer_at(std::vector<char>& bytes, std::size_t offset, std::uint64_t value, std::size_t width = 8) {
    require(offset + width <= bytes.size(), "fixture mutation offset invalid");
    for (std::size_t i = 0; i < width; ++i) bytes[offset + i] = static_cast<char>(value >> (8 * i));
}

std::size_t text_at(const std::vector<char>& bytes, const std::string& text) {
    const auto found = std::search(bytes.begin(), bytes.end(), text.begin(), text.end());
    require(found != bytes.end(), "fixture text not found");
    return static_cast<std::size_t>(found - bytes.begin());
}

sam::internal::sam3::TokenizerData token_fixture() {
    sam::internal::sam3::TokenizerData data;
    for (int c = 33; c <= 126; ++c) {
        const std::string symbol(1, static_cast<char>(c));
        data.vocabulary.emplace(symbol, c - 33);
        data.vocabulary.emplace(symbol + "</w>", c - 33 + 256);
    }
    data.vocabulary.emplace("<start_of_text>", 49406);
    data.vocabulary.emplace("<end_of_text>", 49407);
    data.vocabulary.emplace("##</w>", 15483);
    data.vocabulary.emplace("#...</w>", 31491);
    data.vocabulary.emplace("#@</w>", 35062);
    data.vocabulary.emplace("'re</w>", 982);
    data.vocabulary.emplace("'s</w>", 568);
    data.vocabulary.emplace("'t</w>", 713);
    data.vocabulary.emplace("..", 608);
    data.vocabulary.emplace("...</w>", 678);
    data.vocabulary.emplace("an", 514);
    data.vocabulary.emplace("and</w>", 537);
    data.vocabulary.emplace("ck</w>", 868);
    data.vocabulary.emplace("don</w>", 847);
    data.vocabulary.emplace("el", 544);
    data.vocabulary.emplace("el</w>", 832);
    data.vocabulary.emplace("els</w>", 1645);
    data.vocabulary.emplace("fru", 3248);
    data.vocabulary.emplace("fruit</w>", 5190);
    data.vocabulary.emplace("it</w>", 585);
    data.vocabulary.emplace("on</w>", 525);
    data.vocabulary.emplace("re</w>", 810);
    data.vocabulary.emplace("ru", 681);
    data.vocabulary.emplace("tr", 635);
    data.vocabulary.emplace("tru", 931);
    data.vocabulary.emplace("truck</w>", 4629);
    data.vocabulary.emplace("we</w>", 649);
    data.vocabulary.emplace("wh", 573);
    data.vocabulary.emplace("whe", 2009);
    data.vocabulary.emplace("wheel</w>", 6744);
    data.vocabulary.emplace("wheels</w>", 8025);
    data.merges.emplace_back("a", "n");
    data.merges.emplace_back("o", "n</w>");
    data.merges.emplace_back("an", "d</w>");
    data.merges.emplace_back("e", "l");
    data.merges.emplace_back("'", "s</w>");
    data.merges.emplace_back("w", "h");
    data.merges.emplace_back("i", "t</w>");
    data.merges.emplace_back(".", ".");
    data.merges.emplace_back("t", "r");
    data.merges.emplace_back("w", "e</w>");
    data.merges.emplace_back("..", ".</w>");
    data.merges.emplace_back("r", "u");
    data.merges.emplace_back("'", "t</w>");
    data.merges.emplace_back("r", "e</w>");
    data.merges.emplace_back("e", "l</w>");
    data.merges.emplace_back("d", "on</w>");
    data.merges.emplace_back("c", "k</w>");
    data.merges.emplace_back("tr", "u");
    data.merges.emplace_back("'", "re</w>");
    data.merges.emplace_back("el", "s</w>");
    data.merges.emplace_back("wh", "e");
    data.merges.emplace_back("f", "ru");
    data.merges.emplace_back("tru", "ck</w>");
    data.merges.emplace_back("fru", "it</w>");
    data.merges.emplace_back("whe", "el</w>");
    data.merges.emplace_back("whe", "els</w>");
    data.merges.emplace_back("#", "#</w>");
    data.merges.emplace_back("#", "...</w>");
    data.merges.emplace_back("#", "@</w>");
    return data;
}


void test_tokenizer() {
    auto data = token_fixture();
    sam::internal::sam3::Tokenizer tokenizer(data);
    // Exact IDs from Meta's pinned BPE asset; only merges used by these cases
    // are retained in this offline fixture, preserving their relative ranks.
    require(tokenizer.encode("Truck's 123 wheels!") == std::vector<std::int32_t>{49406, 4629, 568, 272, 273, 274, 8025, 256, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "official ASCII tokenizer golden mismatch");
    require(tokenizer.encode("  WHEEL\n\t and fruit  ") == std::vector<std::int32_t>{49406, 6744, 537, 5190, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "official ASCII tokenizer golden mismatch");
    require(tokenizer.encode("a... 're we're don't 10") == std::vector<std::int32_t>{49406, 320, 678, 982, 649, 982, 847, 713, 272, 271, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "official ASCII tokenizer golden mismatch");
    require(tokenizer.encode("## #@ #...") == std::vector<std::int32_t>{49406, 15483, 35062, 31491, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "official ASCII tokenizer golden mismatch");
    require(tokenizer.encode("truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck truck") == std::vector<std::int32_t>{49406, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 4629, 49407}, "official ASCII tokenizer golden mismatch");
    require(tokenizer.encode("<start_of_text>truck<end_of_text>") == std::vector<std::int32_t>{49406, 49406, 4629, 49407, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "official ASCII tokenizer golden mismatch");

    auto empty = tokenizer.encode(" \t\r\n ");
    require(empty[0] == 49406 && empty[1] == 49407 && empty[2] == 0, "empty prompt padding differs");
    require(sam::internal::sam3::clean_prompt("a\vb\fc") == "ab c", "ftfy ASCII control cleaning differs");
    require(sam::internal::sam3::clean_prompt("a\r\nb\tc") == "a b c", "ASCII whitespace cleaning differs");
    for (const std::string text : {"caf\xc3\xa9", "a&AMP b", "a&not b", "a&amp; b", "&#65", "a\x01"}) {
        rejects<std::invalid_argument>([&] { tokenizer.encode(text); }, "unsupported text was accepted");
    }
    rejects<std::invalid_argument>([&] { tokenizer.validate_tokens({49406, 49407}); }, "short token input was accepted");
    auto invalid = empty;
    invalid[2] = 49408;
    rejects<std::invalid_argument>([&] { tokenizer.validate_tokens(invalid); }, "out-of-range token ID was accepted");
    invalid = empty; invalid[2] = 123;
    rejects<std::invalid_argument>([&] { tokenizer.validate_tokens(invalid); }, "nonzero padding was accepted");
    invalid = empty; invalid[0] = 0;
    rejects<std::invalid_argument>([&] { tokenizer.validate_tokens(invalid); }, "missing start token was accepted");
    invalid = empty; invalid[1] = 0;
    rejects<std::invalid_argument>([&] { tokenizer.validate_tokens(invalid); }, "missing end token was accepted");
    data.vocabulary.erase("<start_of_text>");
    rejects<std::runtime_error>([&] { sam::internal::sam3::Tokenizer bad(data); }, "incompatible tokenizer was accepted");
}

void test_images() {
    const std::vector<std::uint8_t> pixels{0, 127, 255, 255, 0, 127, 99, 99, 0, 255, 127, 127, 255, 0};
    sam::ImageView image{pixels.data(), pixels.size(), 2, 2, 8};
    const auto normalized = sam::internal::sam3::preprocess_image(image, 2);
    require(normalized.size() == 12 && normalized[0] == -1.0f && normalized[1] == 1.0f &&
            normalized[2] == -1.0f && normalized[4] == -0.0039215087890625f &&
            normalized[8] == 1.0f, "RGB stride, layout, or normalization differs");
    auto invalid = image; invalid.row_stride = 5;
    rejects<std::invalid_argument>([&] { sam::internal::validate_image(invalid); }, "undersized stride was accepted");
    invalid = image; --invalid.size_bytes;
    rejects<std::invalid_argument>([&] { sam::internal::validate_image(invalid); }, "truncated final row was accepted");
    invalid = image; invalid.height = 3; invalid.row_stride = std::numeric_limits<std::size_t>::max();
    rejects<std::invalid_argument>([&] { sam::internal::validate_image(invalid); }, "overflowing stride extent was accepted");
    invalid = image; invalid.width = 0;
    rejects<std::invalid_argument>([&] { sam::internal::validate_image(invalid); }, "empty image was accepted");
    invalid = image; invalid.data = nullptr;
    rejects<std::invalid_argument>([&] { sam::internal::validate_image(invalid); }, "null RGB buffer was accepted");
    const std::vector<std::uint8_t> edge{0,0,0, 0,0,0, 255,255,255, 255,255,255};
    const auto downsampled = sam::internal::sam3::preprocess_image({edge.data(), edge.size(), 4, 1, 12}, 2);
    const float low = -0.7176470756530762f, high = 0.7176471948623657f;
    require(downsampled[0] == low && downsampled[1] == high && downsampled[2] == low,
            "antialiased uint8 downsampling differs");
    const std::vector<std::uint8_t> small_corner{0,0,0, 0,0,0, 0,0,0, 1,1,1};
    const auto enlarged = sam::internal::sam3::preprocess_image({small_corner.data(), small_corner.size(), 2,2,6},3);
    require(enlarged[4] == -1.0f, "resize rounded intermediate passes instead of the final image");
    const std::vector<std::uint8_t> half_step{0,0,0, 1,1,1};
    const auto ties = sam::internal::sam3::preprocess_image({half_step.data(), half_step.size(), 2,1,6},3);
    require(ties[1] == -1.0f, "resize half-values were not rounded to nearest even");
    // Frozen torchvision 0.25.0 / torch 2.10.0 CUDA float-antialias results.
    // These high-coordinate samples cross a byte-rounding boundary if an
    // absolute center or filtering accumulation is rounded before subtraction.
    for (const auto shape : {std::array<int, 3>{129, 83, 71}, std::array<int, 3>{257, 129, 97}}) {
        const int width = shape[0], height = shape[1], target = shape[2];
        std::vector<std::uint8_t> synthetic(static_cast<std::size_t>(width) * height * 3);
        for (std::size_t i = 0; i < synthetic.size(); ++i)
            synthetic[i] = static_cast<std::uint8_t>((i * 97 + 19) % 256);
        const auto actual = sam::internal::sam3::preprocess_image(
            {synthetic.data(), synthetic.size(), width, height, static_cast<std::size_t>(width) * 3}, target);
        if (width == 129) {
            require(actual[target * target + 62 * target + 54] == 0.13725495338439941f,
                    "float antialias resize lost the relative-coordinate contraction");
        } else {
            require(actual[target * target + 27 * target + 92] == 0.15294122695922852f &&
                    actual[target * target + 37 * target + 46] == 0.20784318447113037f,
                    "float antialias resize changed independently frozen RGB8 rounding");
        }
    }
    rejects<std::invalid_argument>([] { sam::internal::validate_score_threshold(std::numeric_limits<float>::quiet_NaN()); }, "NaN threshold was accepted");
    rejects<std::invalid_argument>([] { sam::internal::validate_score_threshold(-0.1f); }, "negative threshold was accepted");
    rejects<std::invalid_argument>([] { sam::internal::validate_backend_options({sam::Backend::Cpu, 0}); }, "zero CPU threads were accepted");
    rejects<std::invalid_argument>([] { sam::internal::validate_backend_options({static_cast<sam::Backend>(99), 4}); }, "unknown backend was accepted");
    sam::internal::validate_backend_options({sam::Backend::Cuda, 4, 2});
    rejects<std::invalid_argument>([] { sam::internal::validate_backend_options({sam::Backend::Cuda, 4, -1}); }, "negative CUDA device was accepted");
    rejects<std::invalid_argument>([] { sam::internal::validate_backend_options({sam::Backend::Cpu, 4, 1}); }, "CUDA device index was accepted for CPU");
}

void test_results() {
    const std::vector<float> boxes{0,0.5f,1,1, 0,0.5f,1,1, 0,0.5f,1,1};
    const auto result = sam::internal::sam3::postprocess_detections(boxes, {0,1,1}, 0, {-1,1, -1,1, 0,0}, 2,1,3,1,0.25f);
    require(result.detections.size() == 2 && result.detections[0].query_index == 1 &&
            result.detections[1].query_index == 2, "strict confidence filtering or detection order differs");
    require(result.detections[0].box.x0 == -1.5f && result.detections[0].box.x1 == 1.5f,
            "source box scaling differs");
    require(result.detections[0].mask.data == std::vector<std::uint8_t>{0,0,1} &&
            result.detections[1].mask.data == std::vector<std::uint8_t>{0,0,0}, "mask interpolation/threshold differs");
    require(sam::internal::sam3::postprocess_detections(boxes, {0,1,1}, 0, {-1,1,-1,1,0,0},2,1,3,1,1).detections.empty(),
            "successful empty results differ");
    rejects<std::runtime_error>([] { sam::internal::sam3::postprocess_detections({}, {1},0,{1},1,1,1,1,0.5f); }, "mismatched output size was accepted");
    rejects<std::runtime_error>([] { sam::internal::sam3::postprocess_detections({0,0,1,1},{1},0,{std::numeric_limits<float>::infinity()},1,1,1,1,0.5f); }, "non-finite output was accepted");
}

void test_weight_failures(const std::filesystem::path& directory) {
    const auto path = directory / "container.gguf";
    rejects<std::runtime_error>([&] { sam::internal::GgufReader missing((directory / "missing.gguf").string()); },
                                "missing GGUF was accepted");
    write_fixture(path);
    sam::internal::GgufReader file(path.string());
    require(file.u32("probe.scalar") == 17 && file.string("probe.label") == "labelx", "GGUF typed metadata differs");
    const auto indices = file.array("probe.indices", GGUF_TYPE_UINT32, 2);
    std::array<std::uint32_t, 2> values{};
    std::memcpy(values.data(), gguf_get_arr_data(file.metadata(), indices), sizeof(values));
    require(values[0] == 3 && values[1] == 5, "GGUF typed array differs");
    const auto labels = file.array("probe.labels", GGUF_TYPE_STRING, 2);
    require(std::string(gguf_get_arr_str(file.metadata(), labels, 1)) == "second", "GGUF string array differs");
    require(file.tensors().size() == 1 && file.tensors()[0].dimensions == std::vector<std::int64_t>{1,3,2},
            "GGUF leading singleton or tensor dimensions differ");
    std::array<float, 6> payload{};
    file.read(file.tensors()[0].offset, reinterpret_cast<char*>(payload.data()), sizeof(payload));
    require(payload == std::array<float, 6>{-2,-1,0,1,2,3}, "GGUF tensor payload differs");
    rejects<std::runtime_error>([&] { file.u32("general.architecture"); }, "wrong scalar type reached a typed getter");
    rejects<std::runtime_error>([&] { file.array("probe.scalar", GGUF_TYPE_UINT32, 1); }, "scalar reached an array getter");
    rejects<std::runtime_error>([&] { file.array("probe.labels", GGUF_TYPE_UINT32, 2); }, "wrong array element type was accepted");
    rejects<std::runtime_error>([&] { file.array("probe.indices", GGUF_TYPE_UINT32, 3); }, "wrong array count was accepted");
    write_quant_fixture(path);
    mutate_fixture(path, [](auto& bytes) {
        const auto row_width = text_at(bytes, "probe.quant") + std::string("probe.quant").size() + 4;
        integer_at(bytes, row_width, 16);
        integer_at(bytes, row_width + 8, 4);
    });
    rejects<std::runtime_error>([&] { sam::internal::GgufReader invalid(path.string()); },
                                "a quantized tensor with a misaligned row width but block-aligned element count was accepted");
    auto reject_bytes = [&](const std::function<void(std::vector<char>&)>& change, const char* message, bool second = false) {
        write_fixture(path, {}, second);
        mutate_fixture(path, change);
        rejects<std::runtime_error>([&] { sam::internal::GgufReader invalid(path.string()); }, message);
    };
    reject_bytes([](auto& bytes) { bytes.resize(3); }, "truncated GGUF header was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, 4, 2, 4); }, "unsupported GGUF version was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, 8, 4097); }, "oversized GGUF tensor count was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, 16, 257); }, "oversized GGUF metadata count was accepted");
    reject_bytes([](auto& bytes) { bytes.pop_back(); }, "truncated final tensor padding was accepted");
    reject_bytes([](auto& bytes) { bytes.back() = 1; }, "nonzero tensor padding was accepted");
    reject_bytes([](auto& bytes) { bytes.push_back(0); }, "trailing GGUF payload was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "probe.indices") + 13 + 8, UINT64_MAX); },
                 "oversized metadata array was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "probe.label") + 11 + 4,
                                           sam::internal::GgufReader::metadata_limit + 1); },
                 "oversized metadata string was accepted");
    reject_bytes([](auto& bytes) { bytes[text_at(bytes, "labelx") + 5] = 0; }, "embedded NUL metadata value was accepted");
    reject_bytes([](auto& bytes) { bytes[text_at(bytes, "probe.weight") + 11] = 0; }, "embedded NUL tensor name was accepted");
    reject_bytes([](auto& bytes) { bytes[text_at(bytes, "probe.label") + 10] = 0; }, "embedded NUL metadata key was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "probe.weight") + 12 + 4, 0); },
                 "zero tensor dimension was accepted");
    reject_bytes([](auto& bytes) {
        const auto dimensions = text_at(bytes, "probe.weight") + 12 + 4;
        integer_at(bytes, dimensions, UINT64_C(1) << 32);
        integer_at(bytes, dimensions + 8, UINT64_C(1) << 32);
    }, "overflowing tensor shape was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "probe.weight") + 12 + 4 + 24, 255, 4); },
                 "unknown tensor type was accepted");
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "other.weight") + 12 + 4 + 24 + 4, 0); },
                 "overlapping tensor offsets were accepted", true);
    reject_bytes([](auto& bytes) { integer_at(bytes, text_at(bytes, "other.weight") + 12 + 4 + 24 + 4, UINT64_MAX); },
                 "overflowing tensor offset was accepted", true);
    reject_bytes([](auto& bytes) {
        const auto offset = text_at(bytes, "other.weight");
        const std::string duplicate = "probe.weight";
        std::copy(duplicate.begin(), duplicate.end(), bytes.begin() + offset);
    }, "duplicate GGUF tensor names were accepted", true);
    for (auto alignment : {0u, 3u, 64u}) {
        write_fixture(path, [&](gguf_context* context) { gguf_set_val_u32(context, "general.alignment", 32); });
        mutate_fixture(path, [&](auto& bytes) { integer_at(bytes, text_at(bytes, "general.alignment") + 17 + 4, alignment, 4); });
        rejects<std::runtime_error>([&] { sam::internal::GgufReader invalid(path.string()); }, "invalid GGUF alignment was accepted");
    }
    for (const auto& change : std::vector<std::function<void(gguf_context*)>>{
            [](auto* c) { gguf_set_val_u32(c, "general.architecture", 3); },
            [](auto* c) { gguf_set_val_str(c, "general.architecture", "sam2"); },
            [](auto* c) { gguf_set_val_str(c, "sam.task", "video"); },
            [](auto* c) { gguf_set_val_bool(c, "sam.schema_version", true); }}) {
        write_fixture(path, change);
        rejects<std::runtime_error>([&] { sam::internal::sam3::inspect_weights(path.string()); }, "unsupported SAM metadata was accepted");
    }
    write_fixture(path, [](auto* c) {
        gguf_set_val_u32(c, "general.file_type", 1);
        gguf_set_val_str(c, "sam.storage_profile", "visual-tracker-f32-v1");
    });
    bool rejected_profile = false;
    try { (void) sam::internal::sam3::inspect_weights(path.string()); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("storage profile") != std::string::npos,
                "image hybrid declaration was not rejected before weight validation");
        rejected_profile = true;
    }
    require(rejected_profile, "video-only hybrid declaration accepted on an image container");
    reject_bytes([](auto& bytes) { integer_at(bytes, 0, 0x73616d33, 4); }, "legacy SAM magic was accepted");
    try { sam::internal::GgufReader legacy(path.string()); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("original .pt") != std::string::npos, "legacy GGML has no reconversion guidance");
    }
}

void test_schema4_module_metadata(const std::filesystem::path& directory) {
    const auto path = directory / "schema4-modules.gguf";
    const auto full_modules = "vision,text,fusion,decoder";
    const auto expect_error = [&](const char* modules, bool missing, bool wrong_type,
                                  const char* profile, const char* expected_message) {
        write_schema4_module_fixture(path, modules, missing, wrong_type, profile);
        bool rejected = false;
        try { (void)sam::internal::sam3::inspect_weights(path.string()); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find(expected_message) != std::string::npos;
        }
        require(rejected, "schema-4 module metadata was accepted or failed at the wrong contract");
    };
    expect_error("text,vision", false, false, "image-full-linear-q8_0-v1", "canonical");
    expect_error("vision", false, false, "image-full-linear-q8_0-v1", "canonical");
    expect_error(nullptr, true, false, "image-full-linear-q8_0-v1", "sam.quantization.modules");
    expect_error(nullptr, false, true, "image-full-linear-q8_0-v1", "sam.quantization.modules");

    // Reaching the next required checkpoint key proves these canonical module
    // selections pass the profile binding. Even a diagnostic modules-profile
    // containing all four modules must remain distinct from the full preset.
    expect_error(full_modules, false, false, "image-full-linear-q8_0-v1", "sam.source.checkpoint_sha256");
    expect_error(full_modules, false, false, "image-modules-linear-q8_0-v1", "sam.source.checkpoint_sha256");
    expect_error("vision", false, false, "image-modules-linear-q8_0-v1", "sam.source.checkpoint_sha256");
}

void test_checkpoint(const std::string& path, const std::filesystem::path& directory) {
    const auto file = sam::internal::sam3::inspect_weights(path);
    sam::internal::sam3::Tokenizer tokenizer(file.tokenizer);
    require(tokenizer.encode("Truck's 123 wheels!") == std::vector<std::int32_t>{49406, 4629, 568, 272, 273, 274, 8025, 256, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            "GGUF tokenizer differs from official golden");
    sam::internal::GgufPtr corrupted(gguf_init_empty());
    gguf_set_kv(corrupted.get(), file.reader->metadata());
    const auto key = file.reader->array("tokenizer.ggml.merges", GGUF_TYPE_STRING, 48894);
    std::vector<const char*> merges;
    for (std::size_t i = 0; i < 48894; ++i) merges.push_back(gguf_get_arr_str(file.reader->metadata(), key, i));
    const auto original_first_merge = merges[0];
    merges[0] = "bad merge";
    gguf_set_arr_str(corrupted.get(), "tokenizer.ggml.merges", merges.data(), merges.size());
    for (const auto& info : file.tensors) {
        ggml_tensor tensor{};
        tensor.type = static_cast<ggml_type>(info.type);
        ggml_set_name(&tensor, info.name.c_str());
        const auto dims = sam::internal::sam3::canonical_dimensions(info.dimensions);
        std::copy(dims.begin(), dims.end(), tensor.ne);
        tensor.nb[0] = ggml_type_size(tensor.type);
        tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
        for (int d = 2; d < 4; ++d) tensor.nb[d] = tensor.nb[d - 1] * tensor.ne[d - 1];
        gguf_add_tensor(corrupted.get(), &tensor);
    }
    const auto output_path = directory / "corrupted-tokenizer.gguf";
    // Sparse holes preserve valid tensor extents without copying gigabytes.
    const auto write_sparse = [&] {
        require(gguf_write_to_file(corrupted.get(), output_path.string().c_str(), true), "cannot write corrupted GGUF metadata");
        const auto last = gguf_get_n_tensors(corrupted.get()) - 1;
        const auto bytes = gguf_get_tensor_size(corrupted.get(), last);
        const auto data_bytes = gguf_get_tensor_offset(corrupted.get(), last) + bytes + (32 - bytes % 32) % 32;
        std::fstream sparse(output_path, std::ios::binary | std::ios::in | std::ios::out);
        sparse.seekp(static_cast<std::streamoff>(gguf_get_meta_size(corrupted.get()) + data_bytes - 1));
        sparse.put(0); sparse.close();
    };
    write_sparse();
    rejects<std::runtime_error>([&] { sam::internal::sam3::inspect_weights(output_path.string()); },
                                "corrupted GGUF tokenizer merge sequence was accepted");
    if (!file.quantized && !file.storage_profile.empty()) {
        merges[0] = original_first_merge;
        gguf_set_arr_str(corrupted.get(), "tokenizer.ggml.merges", merges.data(), merges.size());
        for (const auto* profile : {"unknown", "visual-tracker-f32-v1"}) {
            gguf_set_val_str(corrupted.get(), "sam.storage_profile", profile);
            gguf_set_val_u32(corrupted.get(), "general.file_type", profile == std::string("unknown") ? 1 : 0);
            write_sparse();
            rejects<std::runtime_error>([&] { sam::internal::sam3::inspect_weights(output_path.string()); },
                                        "hybrid precision misdeclaration was accepted");
        }
        gguf_set_val_u32(corrupted.get(), "general.file_type", 1);
        gguf_set_tensor_type(corrupted.get(), "vit.blocks.0.attn.qkv.weight", GGML_TYPE_F16);
        write_sparse();
        bool rejected = false;
        try { (void) sam::internal::sam3::inspect_weights(output_path.string()); }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("tensor precision") == std::string::npos)
                throw std::runtime_error(std::string("unexpected hybrid type rejection: ") + error.what());
            rejected = true;
        }
        require(rejected, "F16 visual weight was accepted in the hybrid profile");
    }
    if (file.quantized && !file.modular_quantized && file.precision == "q8_0") {
        const auto* q6_profile = sam::internal::sam3::image_quantization_profile("image-linear-q6_k-v1");
        sam::internal::GgufPtr malformed(gguf_init_empty());
        gguf_set_kv(malformed.get(), file.reader->metadata());
        gguf_set_val_str(malformed.get(), "sam.storage_profile", q6_profile->name);
        gguf_set_val_u32(malformed.get(), "general.file_type", q6_profile->file_type);
        for (const auto& info : file.tensors) {
            const auto shape = sam::internal::sam3::canonical_dimensions(info.dimensions);
            std::array<std::int64_t, 4> dimensions = shape;
            if (info.name == "vit.blocks.0.mlp.lin2.weight") dimensions[0] = 4864;
            const std::vector<std::int64_t> logical_shape(dimensions.begin(), dimensions.end());
            ggml_tensor tensor{};
            tensor.type = sam::internal::sam3::image_quantized_tensor_type(info.name, logical_shape, *q6_profile);
            ggml_set_name(&tensor, info.name.c_str());
            std::copy(dimensions.begin(), dimensions.end(), tensor.ne);
            const auto block = static_cast<std::uint64_t>(ggml_blck_size(tensor.type));
            tensor.nb[0] = ggml_type_size(tensor.type);
            tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / block);
            for (int dimension = 2; dimension < GGML_MAX_DIMS; ++dimension)
                tensor.nb[dimension] = tensor.nb[dimension - 1] * tensor.ne[dimension - 1];
            gguf_add_tensor(malformed.get(), &tensor);
        }
        const auto output_path = directory / "mis-shaped-q6-model.gguf";
        require(gguf_write_to_file(malformed.get(), output_path.string().c_str(), true),
                "cannot write malformed sparse K-profile fixture");
        const auto last = gguf_get_n_tensors(malformed.get()) - 1;
        const auto last_size = gguf_get_tensor_size(malformed.get(), last);
        const auto data_bytes = gguf_get_tensor_offset(malformed.get(), last) + last_size + (32 - last_size % 32) % 32;
        std::fstream sparse(output_path, std::ios::binary | std::ios::in | std::ios::out);
        sparse.seekp(static_cast<std::streamoff>(gguf_get_meta_size(malformed.get()) + data_bytes - 1));
        sparse.put(0);
        sparse.close();
        require(static_cast<bool>(sparse), "cannot size malformed sparse K-profile fixture");
        bool shape_rejected = false;
        try {
            (void)sam::internal::sam3::load_state(output_path.string(), {sam::Backend::Cpu, 2});
        } catch (const std::runtime_error& error) {
            shape_rejected = std::string(error.what()).find("tensor shape") != std::string::npos;
        }
        require(shape_rejected, "malformed K row shape reached quantized tensor construction or backend allocation");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        TemporaryDirectory temporary;
        test_stats_aggregate_compatibility();
        test_model_quantization_modules_json();
        test_tokenizer(); test_images(); test_results(); test_weight_failures(temporary.path);
        test_schema4_module_metadata(temporary.path);
        if (argc > 1) {
            test_checkpoint(argv[1], temporary.path);
        }
        std::cout << "SAM contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
