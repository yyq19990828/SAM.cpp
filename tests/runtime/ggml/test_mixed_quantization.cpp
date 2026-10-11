#include "common/sha256.hpp"
#include "common/sha256.hpp"
#include "models/sam3/weights.hpp"
#include "models/sam3/tensors.hpp"
#include "runtime/ggml.hpp"
#include "models/sam3/model.hpp"
#include "backend_test_support.hpp"
#include "../../../tools/benchmark/execution_cost.hpp"
#include "../../../tools/benchmark/cpu_quantized_matmul.hpp"

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

const std::string f16_module_csv = "vision=f16,text=q8_0,fusion=q6_k,decoder=q4_k";
const std::string f16_tensor_csv = "text.resizer.weight=f16,vit.blocks.0.attn.qkv.weight=q5_k";
const std::string f16_encoded = "sam3:image-mixed-f16-linear-v1\nbase=q4_k\nvision=f16\ntext=q8_0\nfusion=q6_k\ndecoder=q4_k\n"
                                "tensor:text.resizer.weight=f16\ntensor:vit.blocks.0.attn.qkv.weight=q5_k\n";

void write_fixture(const std::filesystem::path& path, const std::function<void(gguf_context*)>& change = {},
                   bool wrong_type = false, bool mixed_f16 = false) {
    auto context = make_context(16);
    GgufPtr file(gguf_init_empty());
    gguf_set_val_str(file.get(), "general.architecture", "sam3");
    gguf_set_val_str(file.get(), "sam.task", "text_image");
    gguf_set_val_u32(file.get(), "sam.schema_version", mixed_f16 ? 7 : 5);
    gguf_set_val_u32(file.get(), "general.file_type", 14);
    gguf_set_val_u32(file.get(), "general.quantization_version", 2);
    const std::string profile = mixed_f16 ? "image-mixed-f16-linear-v1" : "image-mixed-linear-v1";
    gguf_set_val_str(file.get(), "sam.storage_profile", profile.c_str());
    gguf_set_val_str(file.get(), "sam.quantization.base_precision", "q4_k");
    gguf_set_val_str(file.get(), "sam.quantization.module_precisions", (mixed_f16 ? f16_module_csv : module_csv).c_str());
    const auto digest = sha256(mixed_f16 ? f16_encoded : encoded_policy);
    gguf_set_val_str(file.get(), "sam.quantization.policy_sha256", digest.c_str());
    if (mixed_f16) gguf_set_val_str(file.get(), "sam.quantization.tensor_precisions", f16_tensor_csv.c_str());
    const auto policy = parse_image_mixed_policy("q4_k", mixed_f16 ? f16_module_csv : module_csv,
                                                digest, mixed_f16 ? f16_tensor_csv : "", profile);
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

void check_fixture(const std::string& path, const std::string& cost_report = {}, sam::Backend backend = sam::Backend::Cpu,
                       sam::CpuComputeMode compute = sam::CpuComputeMode::F32) {
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
    sam::BackendOptions options{backend, 2};
    options.cpu_compute = compute;
    GgmlRuntime runtime(options, !packed, packed);
    const auto observer = std::make_shared<sam_cost::Observer>();
    observer->stage = "bounded_mixed_linear_fixture";
    if (backend == sam::Backend::Cpu) runtime.set_graph_observer(observer);
    auto weights = make_context(tensors.size() + 2);
    std::vector<ggml_tensor*> stored;
    for (const auto& tensor : tensors) {
        const auto allocated_type = runtime.allocated_weight_type(static_cast<ggml_type>(tensor.type));
        auto* weight = ggml_new_tensor_2d(weights.get(), allocated_type, tensor.dimensions[0],
                                        tensor.dimensions.size() == 2 ? tensor.dimensions[1] : 1);
        ggml_set_name(weight, tensor.name.c_str());
        const auto expected_bytes = tensor.type == GGML_TYPE_F16 && backend == sam::Backend::Cpu ? tensor.size_bytes * 2 : tensor.size_bytes;
        require(ggml_nbytes(weight) == expected_bytes, "mixed fixture byte layout differs");
        require(allocated_type == (tensor.type == GGML_TYPE_F16 && backend == sam::Backend::Cpu ? GGML_TYPE_F32 : tensor.type),
                "mixed fixture violated per-tensor device storage policy");
        stored.push_back(weight);
    }
    BufferPtr buffer(ggml_backend_alloc_ctx_tensors(weights.get(), runtime.weights_backend()));
    require(static_cast<bool>(buffer), "cannot allocate small mixed weights");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    sam::RuntimeStats stats;
    GraphExecution execution(runtime, 16 * tensors.size(), stats);
    std::vector<ggml_tensor*> inputs, outputs;
    std::vector<std::vector<float>> decoded, rhs_values;
    std::vector<std::vector<char>> resident_bytes;
    std::vector<std::int64_t> first_rows;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        auto* weight = stored[i];
        std::vector<char> bytes(static_cast<std::size_t>(info.size_bytes));
        reader.read(info.offset, bytes.data(), bytes.size());
        const auto width = weight->ne[0], rows = weight->ne[1];
        decoded.emplace_back(static_cast<std::size_t>(width * rows));
        const auto source_type = static_cast<ggml_type>(info.type);
        if (ggml_is_quantized(source_type)) {
            validate_quantized_payload(source_type, info.dimensions, bytes.data(), bytes.size());
            const auto* traits = ggml_get_type_traits(source_type);
            for (std::int64_t row = 0; row < rows; ++row)
                traits->to_float(bytes.data() + row * ggml_row_size(source_type, width), decoded.back().data() + row * width, width);
        } else if (source_type == GGML_TYPE_F16) {
            validate_f16_payload(bytes.data(), bytes.size());
            decoded.back() = promote_f16(bytes);
        } else std::memcpy(decoded.back().data(), bytes.data(), bytes.size());
        require(std::all_of(decoded.back().begin(), decoded.back().end(), [](float value) { return std::isfinite(value); }),
                "mixed fixture decoded non-finite weights");
        if (source_type == GGML_TYPE_F16 && weight->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(weight, decoded.back().data(), 0, decoded.back().size() * sizeof(float));
            resident_bytes.emplace_back(ggml_nbytes(weight));
            std::memcpy(resident_bytes.back().data(), decoded.back().data(), resident_bytes.back().size());
        } else {
            ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
            resident_bytes.push_back(bytes);
        }
        inputs.push_back(input_tensor(execution.context(), ("mixed.input." + std::to_string(i)).c_str(), width, 3));
        const bool qkv_view = info.name.find(".attn.qkv.weight") != std::string::npos && rows > 1;
        first_rows.push_back(qkv_view ? 1 : 0);
        auto* left = qkv_view ? ggml_view_2d(execution.context(), weight, width, 1, weight->nb[1], weight->nb[1]) : weight;
        outputs.push_back(ggml_mul_mat(execution.context(), left, inputs.back()));
        require(ggml_prec_set_acc(outputs.back(), GGML_PREC_F32), "mixed F32 hint rejected");
        execution.output(outputs.back());
        rhs_values.emplace_back(static_cast<std::size_t>(width * 3));
        for (std::size_t k = 0; k < rhs_values.back().size(); ++k) {
            // CUDA packed Q can stage RHS as Q8_1, including a half block sum.
            // These Q operands have exact unit scale and exact half sums, so
            // the decoded scalar oracle remains strict. F16/F32 operands use
            // non-half-exact activations to detect unintended RHS narrowing.
            rhs_values.back()[k] = backend == sam::Backend::Cuda && ggml_is_quantized(source_type) ?
                static_cast<float>(static_cast<int>(k % 3) - 1) * 127.0f :
                k == 0 ? 128.0f : static_cast<float>(static_cast<int>(k % 23) - 11) * 0.017f;
        }
    }
    execution.allocate();
    double maximum_error = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (pass) for (auto& value : rhs_values[i])
                value = backend == sam::Backend::Cuda && ggml_is_quantized(stored[i]->type) ?
                    value * -0.5f : value * -0.73f + 0.001f;
            upload(inputs[i], rhs_values[i], stats, &runtime);
        }
        execution.compute();
        for (std::size_t i = 0; i < stored.size(); ++i) {
            const auto actual = download(outputs[i], stats, &runtime);
            const auto width = stored[i]->ne[0], rows = outputs[i]->ne[0];
            auto reference_rhs = rhs_values[i];
            if (compute == sam::CpuComputeMode::NativeQuantized && ggml_is_quantized(stored[i]->type)) {
                const auto rhs_type = cpu_quantized_rhs_type(stored[i]->type);
                const auto pack = ggml_get_type_traits_cpu(rhs_type)->from_float;
                require(pack != nullptr, "CPU mixed RHS reference encoder unavailable");
                const auto row_bytes = ggml_row_size(rhs_type, width);
                std::vector<char> packed_rhs(row_bytes * 3);
                for (std::int64_t column = 0; column < 3; ++column) {
                    pack(rhs_values[i].data() + column * width, packed_rhs.data() + column * row_bytes, width);
                }
                sam_cpu_probe::decode_cpu_rhs(rhs_type, packed_rhs, reference_rhs);
            }
            for (std::int64_t column = 0; column < 3; ++column) {
                for (std::int64_t row = 0; row < rows; ++row) {
                    double expected = 0;
                    for (std::int64_t k = 0; k < width; ++k)
                        expected += static_cast<double>(decoded[i][(row + first_rows[i]) * width + k]) * reference_rhs[column * width + k];
                    const auto error = std::abs(static_cast<double>(actual[column * rows + row]) - expected);
                    maximum_error = std::max(maximum_error, error);
                    if (!std::isfinite(actual[column * rows + row]) || error > 1e-4 + std::abs(expected) * 1e-5)
                        std::cerr << "mixed product mismatch: tensor=" << tensors[i].name << " stored=" << ggml_type_name(static_cast<ggml_type>(tensors[i].type))
                                  << " allocated=" << ggml_type_name(stored[i]->type) << " pass=" << pass << " expected=" << expected
                                  << " actual=" << actual[column * rows + row] << " error=" << error << '\n';
                    require(std::isfinite(actual[column * rows + row]) && error <= 1e-4 + std::abs(expected) * 1e-5,
                            "mixed product differs from decoded scalar reference");
                }
            }
            require(stored[i]->type == runtime.allocated_weight_type(static_cast<ggml_type>(tensors[i].type)),
                    "mixed execution replaced resident weight storage");
            std::vector<char> after(ggml_nbytes(stored[i]));
            ggml_backend_tensor_get(stored[i], after.data(), 0, after.size());
            require(after == resident_bytes[i], "mixed execution changed resident weight bytes");
        }
    }
    const bool placed = backend == sam::Backend::Cpu ?
        stats.cpu_nodes > 0 && stats.cuda_nodes == 0 && stats.metal_nodes == 0 :
        backend == sam::Backend::Cuda ?
        stats.cuda_nodes > 0 && stats.cpu_nodes == 0 && stats.blas_nodes == 0 && stats.metal_nodes == 0 :
        backend == sam::Backend::Metal && stats.metal_nodes > 0 && stats.cpu_nodes == 0 &&
        stats.blas_nodes == 0 && stats.cuda_nodes == 0;
    require(placed,
            "mixed arithmetic escaped the requested backend");
    if (compute == sam::CpuComputeMode::NativeQuantized) {
        std::size_t packed_calls = 0;
        for (const auto& node : observer->nodes) {
            require(node.category != "quantized_to_f32", "native mixed graph decoded Q weights");
            if (node.category == "matmul" && !node.inputs.empty() && node.inputs.front().type.front() == 'q') {
                require(node.backend == "CPU" && node.calls == 2, "native mixed Q placement or graph reuse differs");
                packed_calls += node.calls;
            }
        }
        require(packed_calls > 0, "native mixed graph omitted packed Q products");
    }
    if (!cost_report.empty()) {
        require(!std::filesystem::exists(cost_report), "cost fixture report must use a new file");
        auto output = sam_example::output_file(cost_report);
        observer->write(output);
        output << '\n';
    }
    const auto* backend_name = backend == sam::Backend::Cpu ? "CPU" : backend == sam::Backend::Cuda ? "CUDA" : "Metal";
    std::cout << "mixed " << backend_name << " tensors=" << tensors.size()
              << " maximum_absolute_error=" << maximum_error << '\n';
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
    try { check_fixture(path.string()); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "mixed CPU accepted a stored Q8 matrix assigned Q4 by metadata");
    write_fixture(path);
}

void check_tensor_policy(const std::filesystem::path& path) {
    const std::string tensors = "ddec.layers.0.linear1.weight=f32,fenc.layers.0.linear1.weight=q5_k,"
                                "text.resizer.weight=q4_k,vit.blocks.0.attn.qkv.weight=q6_k";
    const std::string identity = "c290f907480a3b3cadc5304989e2d4463ba0e8ca1e414a18bcf63bcfca17f471";
    const auto policy = parse_image_mixed_policy("q4_k", module_csv, identity, tensors);
    require(policy.tensor_precisions.size() == 4, "native tensor policy lost an override");
    require(image_mixed_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", {256, 2}, policy) == GGML_TYPE_Q6_K &&
            image_mixed_quantized_tensor_type("text.resizer.weight", {256, 2}, policy) == GGML_TYPE_Q4_K &&
            image_mixed_quantized_tensor_type("fenc.layers.0.linear1.weight", {256, 2}, policy) == GGML_TYPE_Q5_K &&
            image_mixed_quantized_tensor_type("ddec.layers.0.linear1.weight", {256, 2}, policy) == GGML_TYPE_F32,
            "native tensor precedence differs from the Python contract");
    const std::vector<std::string> invalid{
        "vit.blocks.0.attn.qkv.weight=f16", "vit.blocks.0.attn.qkv.weight=q6_k,vit.blocks.0.attn.qkv.weight=f32",
        "vit.blocks.01.attn.qkv.weight=f32", "vit.blocks.32.attn.qkv.weight=f32", "text.token_embed.weight=f32",
        "vit.blocks.0.attn.qkv.weight=q6_k,text.resizer.weight=q4_k", "vit.blocks.*.attn.qkv.weight=f32"};
    for (const auto& value : invalid) {
        bool rejected = false;
        try { (void)parse_image_mixed_policy("q4_k", module_csv, identity, value); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid exact tensor policy was accepted");
    }
    auto tensor_metadata = [&](gguf_context* file) {
        gguf_set_val_u32(file, "sam.schema_version", 6);
        gguf_set_val_str(file, "sam.storage_profile", "image-tensor-mixed-linear-v1");
        gguf_set_val_str(file, "sam.quantization.tensor_precisions", tensors.c_str());
        gguf_set_val_str(file, "sam.quantization.policy_sha256", identity.c_str());
    };
    write_fixture(path, tensor_metadata);
    {
        GgufReader reader(path.string());
        require(read_image_mixed_policy(reader).tensor_precisions == policy.tensor_precisions,
                "native schema-6 metadata readback differs");
    }
    for (const auto& change : std::vector<std::function<void(gguf_context*)>>{
            [](auto* file) { gguf_set_val_u32(file, "sam.schema_version", 5); },
            [](auto* file) { gguf_remove_key(file, "sam.quantization.tensor_precisions"); },
            [](auto* file) { gguf_set_val_str(file, "sam.quantization.tensor_precisions", ""); },
            [](auto* file) { gguf_set_val_str(file, "sam.quantization.policy_sha256", std::string(64, '0').c_str()); }}) {
        write_fixture(path, [&](auto* file) { tensor_metadata(file); change(file); });
        bool rejected = false;
        try { GgufReader reader(path.string()); (void)read_image_mixed_policy(reader); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "altered schema-6 metadata passed preallocation validation");
    }
}

void check_f16_policy(const std::filesystem::path& path) {
    const auto policy = parse_image_mixed_policy("q4_k", f16_module_csv, sha256(f16_encoded), f16_tensor_csv,
                                                 "image-mixed-f16-linear-v1");
    require(image_mixed_quantized_tensor_type("vit.blocks.0.mlp.lin2.weight", {4736, 2}, policy) == GGML_TYPE_F16 &&
            image_mixed_quantized_tensor_type("vit.blocks.0.attn.qkv.weight", {256, 2}, policy) == GGML_TYPE_Q5_K &&
            image_mixed_quantized_tensor_type("text.resizer.weight", {256, 2}, policy) == GGML_TYPE_F16 &&
            image_mixed_quantized_tensor_type("text.token_embed.weight", {256, 2}, policy) == GGML_TYPE_F32,
            "native F16 policy lost precedence, row eligibility or protection");
    for (const auto& change : std::vector<std::function<void(gguf_context*)>>{
            [](auto* file) { gguf_set_val_u32(file, "sam.schema_version", 6); },
            [](auto* file) { gguf_set_val_str(file, "sam.storage_profile", "image-mixed-linear-v1"); },
            [](auto* file) { gguf_set_val_str(file, "sam.quantization.tensor_precisions", ""); },
            [](auto* file) { gguf_set_val_str(file, "sam.quantization.policy_sha256", std::string(64, '0').c_str()); }}) {
        write_fixture(path, change, false, true);
        bool rejected = false;
        try { GgufReader reader(path.string()); (void)read_image_mixed_policy(reader); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "altered schema-7 metadata passed native validation");
    }
    for (ggml_fp16_t bad : {ggml_fp16_t(0x7c00), ggml_fp16_t(0xfc00), ggml_fp16_t(0x7e00)}) {
        bool rejected = false;
        try { validate_f16_payload(reinterpret_cast<const char*>(&bad), sizeof(bad)); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "non-finite F16 weight payload passed validation");
    }
    write_fixture(path, {}, false, true);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "cuda" || std::string(argv[1]) == "metal")) {
            const bool cuda = std::string(argv[1]) == "cuda";
            if (!(cuda ? sam::test::cuda_available() : sam::test::metal_available())) {
                std::cout << (cuda ? "CUDA" : "Metal") << " device unavailable\n";
                return 77;
            }
            const auto backend = cuda ? sam::Backend::Cuda : sam::Backend::Metal;
            TemporaryDirectory directory;
            const auto path = directory.path / "mixed.gguf";
            write_fixture(path);
            check_fixture(path.string(), "", backend);
            check_f16_policy(path);
            check_fixture(path.string(), "", backend);
            return 0;
        }
        if ((argc == 3 || argc == 5) && std::string(argv[1]) == "--fixture") {
            require(argc == 3 || std::string(argv[3]) == "--cost-report", "expected --cost-report NEW_FILE");
            check_fixture(argv[2], argc == 5 ? argv[4] : "");
            return 0;
        }
        require(argc == 1, "usage: test_mixed_quantization [cuda|metal|--fixture tiny-converted.gguf [--cost-report NEW_FILE]]");
        TemporaryDirectory directory;
        const auto path = directory.path / "mixed.gguf";
        check_metadata(path);
        check_fixture(path.string());
        check_tensor_policy(path);
        check_f16_policy(path);
        check_fixture(path.string());
        check_fixture(path.string(), "", sam::Backend::Cpu, sam::CpuComputeMode::NativeQuantized);
        std::cout << "mixed weight metadata and CPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
