#include "common/sha256.hpp"
#include "common/sha256.hpp"
#include "models/sam3/weights.hpp"
#include "models/sam3/tensors.hpp"
#include "runtime/ggml.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace sam::internal;
using namespace sam::internal::sam3;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("sam-mixed-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { require(std::filesystem::create_directory(path), "cannot create mixed fixture directory"); }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

const std::string module_csv = "vision=q4_k,text=q8_0,fusion=q6_k,decoder=q5_k";
const std::string encoded_policy = "sam3:image-mixed-linear-v1\nbase=q4_k\nvision=q4_k\ntext=q8_0\nfusion=q6_k\ndecoder=q5_k\n";

void write_fixture(const std::filesystem::path& path, const std::function<void(gguf_context*)>& change = {}, bool wrong_type = false) {
    auto context = make_context(16);
    GgufPtr file(gguf_init_empty());
    gguf_set_val_str(file.get(), "general.architecture", "sam3");
    gguf_set_val_str(file.get(), "sam.task", "text_image");
    gguf_set_val_u32(file.get(), "sam.schema_version", 5);
    gguf_set_val_u32(file.get(), "general.file_type", 14);
    gguf_set_val_u32(file.get(), "general.quantization_version", 2);
    gguf_set_val_str(file.get(), "sam.storage_profile", "image-mixed-linear-v1");
    gguf_set_val_str(file.get(), "sam.quantization.base_precision", "q4_k");
    gguf_set_val_str(file.get(), "sam.quantization.module_precisions", module_csv.c_str());
    gguf_set_val_str(file.get(), "sam.quantization.policy_sha256", sha256(encoded_policy).c_str());
    const auto policy = parse_image_mixed_policy("q4_k", module_csv, sha256(encoded_policy));
    std::vector<std::vector<char>> payloads;
    for (const auto* name : {"vit.blocks.0.attn.qkv.weight", "text.resizer.weight", "fenc.layers.0.linear1.weight",
                             "ddec.layers.0.linear1.weight", "text.token_embed.weight", "vit.blocks.0.mlp.lin2.weight"}) {
        const std::int64_t width = std::string(name).find("lin2") != std::string::npos ? 4736 : 256;
        const auto type = wrong_type && std::string(name) == "vit.blocks.0.attn.qkv.weight" ? GGML_TYPE_Q8_0 :
            image_mixed_quantized_tensor_type(name, {width, 2}, policy);
        auto* tensor = ggml_new_tensor_2d(context.get(), type, width, 2);
        ggml_set_name(tensor, name);
        std::vector<float> values(static_cast<std::size_t>(width * 2));
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.013f;
        payloads.emplace_back(ggml_nbytes(tensor));
        auto& bytes = payloads.back();
        if (type == GGML_TYPE_F32) std::memcpy(bytes.data(), values.data(), bytes.size());
        else {
            const auto* traits = ggml_get_type_traits(type);
            const auto row_bytes = ggml_row_size(type, width);
            for (int row = 0; row < 2; ++row)
                traits->from_float_ref(values.data() + row * width, bytes.data() + row * row_bytes, width);
        }
        tensor->data = bytes.data();
        gguf_add_tensor(file.get(), tensor);
    }
    if (change) change(file.get());
    require(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot write mixed fixture");
}

void check_fixture_cpu(const std::string& path) {
    GgufReader reader(path);
    const auto policy = read_image_mixed_policy(reader);
    const auto tensors = reader.tensors();
    require(!tensors.empty(), "mixed fixture is empty");
    // This is a small arithmetic test, not the complete 1133-tensor SAM adapter.
    for (const auto& tensor : tensors) {
        require(tensor.dimensions.size() <= 2 && tensor.dimensions[0] <= 4736 &&
                (tensor.dimensions.size() == 1 || tensor.dimensions[1] <= 16), "mixed arithmetic fixture is too large");
        require(tensor.type == image_mixed_quantized_tensor_type(tensor.name, tensor.dimensions, policy),
                "mixed fixture type differs from the independent native policy");
    }
    const bool packed = std::any_of(tensors.begin(), tensors.end(), [](const TensorInfo& tensor) {
        return ggml_is_quantized(static_cast<ggml_type>(tensor.type));
    });
    GgmlRuntime runtime({sam::Backend::Cpu, 2}, !packed, packed);
    auto weights = make_context(tensors.size() + 2);
    std::vector<ggml_tensor*> stored;
    for (const auto& tensor : tensors) {
        auto* weight = ggml_new_tensor_2d(weights.get(), static_cast<ggml_type>(tensor.type), tensor.dimensions[0],
                                        tensor.dimensions.size() == 2 ? tensor.dimensions[1] : 1);
        ggml_set_name(weight, tensor.name.c_str());
        require(ggml_nbytes(weight) == tensor.size_bytes, "mixed fixture byte layout differs");
        stored.push_back(weight);
    }
    BufferPtr buffer(ggml_backend_alloc_ctx_tensors(weights.get(), runtime.weights_backend()));
    require(static_cast<bool>(buffer), "cannot allocate small mixed weights");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    sam::RuntimeStats stats;
    GraphExecution execution(runtime, 16 * tensors.size(), stats);
    std::vector<ggml_tensor*> inputs, outputs;
    std::vector<std::vector<float>> decoded, rhs_values;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        auto* weight = stored[i];
        std::vector<char> bytes(static_cast<std::size_t>(info.size_bytes));
        reader.read(info.offset, bytes.data(), bytes.size());
        const auto width = weight->ne[0], rows = weight->ne[1];
        decoded.emplace_back(static_cast<std::size_t>(width * rows));
        if (ggml_is_quantized(weight->type)) {
            validate_quantized_payload(weight->type, info.dimensions, bytes.data(), bytes.size());
            const auto* traits = ggml_get_type_traits(weight->type);
            for (std::int64_t row = 0; row < rows; ++row)
                traits->to_float(bytes.data() + row * ggml_row_size(weight->type, width), decoded.back().data() + row * width, width);
        } else std::memcpy(decoded.back().data(), bytes.data(), bytes.size());
        require(std::all_of(decoded.back().begin(), decoded.back().end(), [](float value) { return std::isfinite(value); }),
                "mixed fixture decoded non-finite weights");
        ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
        inputs.push_back(input_tensor(execution.context(), ("mixed.input." + std::to_string(i)).c_str(), width, 3));
        outputs.push_back(ggml_mul_mat(execution.context(), weight, inputs.back()));
        require(ggml_prec_set_acc(outputs.back(), GGML_PREC_F32), "mixed CPU F32 hint rejected");
        execution.output(outputs.back());
        rhs_values.emplace_back(static_cast<std::size_t>(width * 3));
        for (std::size_t k = 0; k < rhs_values.back().size(); ++k)
            rhs_values.back()[k] = k == 0 ? 128.0f : static_cast<float>(static_cast<int>(k % 23) - 11) * 0.017f;
    }
    execution.allocate();
    for (std::size_t i = 0; i < inputs.size(); ++i) upload(inputs[i], rhs_values[i], stats);
    execution.compute();
    double maximum_error = 0;
    for (std::size_t i = 0; i < stored.size(); ++i) {
        const auto actual = download(outputs[i], stats);
        const auto width = stored[i]->ne[0], rows = stored[i]->ne[1];
        for (std::int64_t column = 0; column < 3; ++column) {
            for (std::int64_t row = 0; row < rows; ++row) {
                double expected = 0;
                for (std::int64_t k = 0; k < width; ++k)
                    expected += static_cast<double>(decoded[i][row * width + k]) * rhs_values[i][column * width + k];
                const auto error = std::abs(static_cast<double>(actual[column * rows + row]) - expected);
                maximum_error = std::max(maximum_error, error);
                require(std::isfinite(actual[column * rows + row]) && error <= 1e-4 + std::abs(expected) * 1e-5,
                        "mixed CPU product differs from decoded scalar reference");
            }
        }
        require(stored[i]->type == tensors[i].type, "mixed CPU execution replaced resident weight storage");
    }
    require(stats.cpu_nodes > 0 && stats.cuda_nodes == 0 && stats.metal_nodes == 0, "mixed arithmetic escaped CPU");
    std::cout << "mixed CPU tensors=" << tensors.size() << " maximum_absolute_error=" << maximum_error << '\n';
}

void check_metadata(const std::filesystem::path& path) {
    require(sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 empty vector differs");
    require(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 short vector differs");
    require(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "SHA-256 two-block vector differs");
    const std::vector<std::function<void(gguf_context*)>> invalid{
        [](auto* file) { gguf_set_val_u32(file, "sam.schema_version", 4); },
        [](auto* file) { gguf_set_val_str(file, "sam.storage_profile", "image-full-linear-q4_k-v1"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.base_precision", "f16"); },
        [](auto* file) { gguf_set_val_u32(file, "sam.quantization.module_precisions", 1); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.module_precisions", "text=q8_0,vision=q4_k,fusion=q6_k,decoder=q5_k"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.module_precisions", "vision=q4_k,text=q8_0,fusion=q6_k"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.module_precisions", "vision=q4_k,text=q8_0,fusion=q6_k,decoder=q5_k,"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.module_precisions", "vision=q4_k,text=q8_0,fusion=q6_k,decoder=f16"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.modules", "vision"); },
        [](auto* file) { gguf_set_val_str(file, "sam.quantization.policy_sha256", std::string(64, '0').c_str()); },
        [](auto* file) { gguf_set_val_u32(file, "general.file_type", 7); },
        [](auto* file) { gguf_set_val_u32(file, "general.quantization_version", 1); },
        [](auto* file) { gguf_remove_key(file, "sam.quantization.policy_sha256"); }};
    for (const auto& change : invalid) {
        write_fixture(path, change);
        bool rejected = false;
        try { GgufReader reader(path.string()); (void)read_image_mixed_policy(reader); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid mixed GGUF metadata passed native validation");
        // The model loader must reject this metadata before allocating weights.
        rejected = false;
        try { (void)inspect_weights(path.string()); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid mixed GGUF metadata passed adapter inspection");
    }
    write_fixture(path);
    const auto policy = parse_image_mixed_policy("q4_k", module_csv, sha256(encoded_policy));
    require(image_mixed_quantized_tensor_type("geom.points_direct_project.weight", {2, 256}, policy) == GGML_TYPE_F32 &&
            image_mixed_quantized_tensor_type("text.token_embed.weight", {256, 2}, policy) == GGML_TYPE_F32 &&
            image_mixed_quantized_tensor_type("ddec.presence_token_head.layers.2.weight", {256}, policy) == GGML_TYPE_F32,
            "mixed policy quantized a protected tensor");
    auto context = make_context(4096);
    sam3_model model;
    model.ctx = context.get();
    model.weight_type = GGML_TYPE_F32;
    sam3_register_tensors(model);
    std::map<ggml_type, int> counts;
    for (const auto& entry : model.tensors) {
        const auto* tensor = entry.second;
        std::vector<std::int64_t> shape{tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        while (shape.size() > 1 && shape.back() == 1) shape.pop_back();
        ++counts[image_mixed_quantized_tensor_type(entry.first, shape, policy)];
    }
    require(counts == std::map<ggml_type, int>{{GGML_TYPE_F32, 785}, {GGML_TYPE_Q4_K, 96},
            {GGML_TYPE_Q8_0, 129}, {GGML_TYPE_Q6_K, 36}, {GGML_TYPE_Q5_K, 87}},
            "mixed native canonical inventory differs from the Python format contract");
    write_fixture(path, {}, true);
    bool rejected = false;
    try { check_fixture_cpu(path.string()); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "mixed CPU accepted a stored Q8 matrix assigned Q4 by metadata");
    write_fixture(path);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--fixture") {
            check_fixture_cpu(argv[2]);
            return 0;
        }
        require(argc == 1, "usage: test_mixed_quantization [--fixture tiny-converted.gguf]");
        TemporaryDirectory directory;
        const auto path = directory.path / "mixed.gguf";
        check_metadata(path);
        check_fixture_cpu(path.string());
        std::cout << "mixed weight metadata and CPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
