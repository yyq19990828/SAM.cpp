#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP

#include "../state.hpp"
#include "mask_decoder.hpp"
#include "memory_encoder.hpp"
#include "memory_attention.hpp"
#include "memory_selection.hpp"
#include "mask_ops.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

inline constexpr int sam3_tracker_graph_batch_limit = 8;
inline constexpr std::size_t sam3_tracker_fused_graph_capacity = 4096;
inline constexpr std::size_t sam3_tracker_decoder_graph_capacity = 1024;
inline constexpr std::size_t sam3_tracker_memory_graph_capacity = 512;
inline constexpr std::size_t sam3_tracker_pointer_graph_capacity = 128;

inline std::size_t sam3_tracker_workspace_capacity(const GgmlRuntime& runtime) {
    return sam3_tracker_fused_graph_capacity * (runtime.quantized_cpu_f32_weights() ? 2 : 1);
}

struct TrackerResidentUploadState {
    template<class Upload>
    void ensure_core(Upload&& upload) {
        if (core_uploaded_) return;
        upload();
        core_uploaded_ = true;
    }

    template<class Upload>
    void ensure_high(Upload&& upload) {
        if (high_uploaded_) return;
        upload();
        high_uploaded_ = true;
    }

    void reset_frame() noexcept { core_uploaded_ = high_uploaded_ = false; }
    void release_high() noexcept { high_uploaded_ = false; }
    bool core_uploaded() const noexcept { return core_uploaded_; }
    bool high_uploaded() const noexcept { return high_uploaded_; }

private:
    bool core_uploaded_ = false;
    bool high_uploaded_ = false;
};

struct TrackerRecord {
    std::vector<ggml_bf16_t> memory;
    std::vector<float> pointer;
    float object_logit = 0, iou = 0;
};

struct TrackerPrediction {
    std::vector<float> mask, pointer, conditioned, decoder_masks, decoder_iou;
    float object_logit = 0, iou = 0, decoder_object_logit = 0;
    int mask_index = 0, pointer_index = 0;
};

struct TrackerPropagationInput {
    const std::map<int, TrackerRecord>* records = nullptr;
    const MemorySelection* selection = nullptr;
};

struct TrackerGraphShape {
    enum class Stage : std::uint8_t { pointer_time, propagation, condition, decoder, memory_encoder } stage;
    Backend backend = Backend::Cpu;
    ggml_type activation_type = GGML_TYPE_F32;
    std::string storage_profile;
    std::string arithmetic_profile;
    std::vector<std::int64_t> dimensions;
    std::vector<std::size_t> strides;
    std::vector<int> memory_position_order;
    int spatial_count = 0, pointer_count = 0, batch = 1;
    bool seed = false, present = true, diagnostics = true, unrounded_output = false;
    bool high_resident = false;
    bool none_resident = false;

    bool operator==(const TrackerGraphShape& other) const {
        return stage == other.stage && backend == other.backend && activation_type == other.activation_type &&
            storage_profile == other.storage_profile &&
            arithmetic_profile == other.arithmetic_profile && dimensions == other.dimensions &&
            strides == other.strides && memory_position_order == other.memory_position_order &&
            spatial_count == other.spatial_count &&
            pointer_count == other.pointer_count && batch == other.batch && seed == other.seed &&
            present == other.present && diagnostics == other.diagnostics &&
            unrounded_output == other.unrounded_output && high_resident == other.high_resident &&
            none_resident == other.none_resident;
    }
};

struct TrackerNoneSerialPolicy {
    const void* runtime_identity = nullptr;
    Backend backend = Backend::Cpu;
    std::string storage_profile, arithmetic_profile;
    int spatial = 0, pointers = 0;
    std::size_t budget = 0;
};

inline bool tracker_none_serial_policy_matches(const std::optional<TrackerNoneSerialPolicy>& policy,
        const void* runtime_identity, Backend backend, const std::string& storage_profile,
        const std::string& arithmetic_profile, int spatial, int pointers, std::size_t budget) {
    return policy && policy->runtime_identity == runtime_identity && policy->backend == backend &&
        policy->storage_profile == storage_profile && policy->arithmetic_profile == arithmetic_profile &&
        policy->spatial == spatial && policy->pointers == pointers && policy->budget == budget;
}

inline void tracker_shape_tensor(TrackerGraphShape& shape, std::initializer_list<std::int64_t> dimensions) {
    std::size_t stride = sizeof(float);
    for (const auto dimension : dimensions) {
        shape.dimensions.push_back(dimension);
        shape.strides.push_back(stride);
        if (dimension > 0 && stride <= std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimension))
            stride *= static_cast<std::size_t>(dimension);
        else
            stride = 0;
    }
    shape.dimensions.push_back(0); // Tensor boundary; prevents ambiguous flattened layouts.
    shape.strides.push_back(0);
}

template<class Callback>
inline void sam3_for_each_tracker_chunk(int total, int& chunk_limit, Callback&& callback) {
    if (total < 0) throw std::invalid_argument("tracker chunk total cannot be negative");
    int begin = 0;
    while (begin < total) {
        if (chunk_limit <= 0) throw std::invalid_argument("tracker chunk limit must be positive");
        const int count = std::min(chunk_limit, total - begin);
        callback(begin, count);
        begin += count; // The callback may change chunk_limit while recursing.
    }
}

class TrackerGraphSlot {
public:
    GraphExecution& ensure(GgmlRuntime& runtime, RuntimeStats& stats,
                           const std::shared_ptr<GraphWorkspace>& workspace,
                           const TrackerGraphShape& shape, std::size_t graph_size,
                           bool& built, double& build_ms) {
        built = !execution_ || graph_size_ != graph_size || !(shape_ == shape);
        if (built) {
            execution_.reset();
            required_workspace_bytes_.reset();
            const auto start = std::chrono::steady_clock::now();
            execution_ = std::make_unique<GraphExecution>(runtime, graph_size, stats, workspace);
            build_ms += elapsed_ms(start);
            graph_size_ = graph_size;
            shape_ = shape;
        }
        return *execution_;
    }

    void reset() { execution_.reset(); graph_size_ = 0; shape_ = {}; required_workspace_bytes_.reset(); }
    GraphExecution* get() const { return execution_.get(); }
    std::size_t required_workspace_bytes() {
        if (!execution_) throw std::runtime_error("tracker graph slot is empty");
        if (!required_workspace_bytes_) required_workspace_bytes_ = execution_->required_workspace_bytes();
        return *required_workspace_bytes_;
    }

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    std::unique_ptr<GraphExecution> execution_;
    std::size_t graph_size_ = 0;
    TrackerGraphShape shape_{};
    std::optional<std::size_t> required_workspace_bytes_;
};

class TrackerExecution {
public:
    TrackerExecution(ModelState& model, RuntimeStats& stats)
        : model_(model), stats_(stats), workspace_(std::make_shared<GraphWorkspace>(
              *model.runtime, sam3_tracker_workspace_capacity(*model.runtime))) {
        const auto& weights = model_.definition.weights;
        const auto gaussian = read_weight(weights.sam_pe.pe_gaussian);
        const auto no_point = read_weight(weights.sam_pe.not_a_point_embed);
        no_object_ = read_weight(weights.no_obj_ptr);
        memory_tpos_ = read_weight(weights.tensors.at("mem_enc.tpos_enc"));
        const auto memory_position = sam3_sinusoidal_pe_2d(72, 72, 64);
        memory_position_bank_.reserve(memory_position.size() * 7);
        for (int position = 0; position < 7; ++position)
            for (std::size_t i = 0; i < memory_position.size(); ++i)
                memory_position_bank_.push_back(memory_position[i] + memory_tpos_[position * 64 + i % 64]);
        dense_position_.resize(256 * 72 * 72);
        rope_.resize(256 * 72 * 72);
        for (int y = 0; y < 72; ++y) for (int x = 0; x < 72; ++x) {
            const auto pixel = y * 72 + x;
            for (int k = 0; k < 128; ++k) {
                const float phase = ((2.0f * (x + 0.5f) / 72 - 1) * gaussian[k] +
                                     (2.0f * (y + 0.5f) / 72 - 1) * gaussian[128 + k]) * 6.283185307179586f;
                dense_position_[pixel * 256 + k] = std::sin(phase);
                dense_position_[pixel * 256 + 128 + k] = std::cos(phase);
                const float angle = (k < 64 ? x : y) / std::pow(10000.0f, 4.0f * (k % 64) / 256);
                rope_[pixel * 256 + k * 2] = std::cos(angle);
                rope_[pixel * 256 + k * 2 + 1] = std::sin(angle);
            }
        }
        no_point_sparse_ = no_point;
        no_point_sparse_.insert(no_point_sparse_.end(), no_point.begin(), no_point.end());
        make_resident_context();
    }

    void begin_frame(const ImageFeatures& image) {
        if (active_image_) throw std::runtime_error("tracker frame is already active");
        if (image.tracker[0].size() != 256u * 288 * 288 ||
            image.tracker[1].size() != 256u * 144 * 144 ||
            image.tracker[2].size() != 256u * 72 * 72 || image.position.size() != 256u * 72 * 72)
            throw std::runtime_error("tracker frame has noncanonical feature shapes");
        high_resident_ = high_residency_allowed_;
        none_resident_ = false;
        resident_uploads_.reset_frame();
        active_image_ = &image;
        frame_serial_budget_ = 0;
        frame_serial_required_budget_ = 0;
    }

    void end_frame() noexcept {
        workspace_->release_storage();
        if (resident_buffer_) {
            clear_context(resident_context_.get());
            resident_buffer_.reset();
        }
        if (high_buffer_) {
            clear_context(high_context_.get());
            high_buffer_.reset();
        }
        resident_uploads_.reset_frame();
        active_image_ = nullptr;
        high_resident_ = high_residency_allowed_;
        none_resident_ = false;
    }

    std::vector<TrackerPrediction> propagate_batch(const ImageFeatures& image,
                                                    const std::vector<TrackerPropagationInput>& inputs,
                                                    int frame_count) {
        validate_frame(image);
        if (inputs.empty()) return {};
        const auto spatial_count = inputs.front().selection ? inputs.front().selection->spatial.size() : 0;
        const auto pointer_count = inputs.front().selection ? inputs.front().selection->pointers.size() : 0;
        if (!spatial_count) throw std::runtime_error("tracker has no conditioning memory");
        for (const auto& input : inputs)
            if (!input.records || !input.selection || input.selection->spatial.size() != spatial_count ||
                input.selection->pointers.size() != pointer_count)
                throw std::invalid_argument("tracker propagation batch has incompatible memory shapes");

        const auto batch = static_cast<int>(inputs.size());
        const auto baseline = serial_workspace_budget(static_cast<int>(spatial_count),
                                                      static_cast<int>(pointer_count));
        frame_serial_budget_ = std::max(frame_serial_budget_, baseline);
        frame_serial_required_budget_ = std::max(frame_serial_required_budget_, serial_required_budget_);
        const int spatial = static_cast<int>(spatial_count);
        const int pointers = static_cast<int>(pointer_count);
        const bool sticky_none_serial = tracker_none_serial_policy_matches(none_serial_policy_,
            model_.runtime.get(), model_.runtime->backend(),
            model_.model_info.precision + ":" + model_.model_info.storage_profile,
            model_.runtime->arithmetic_profile(), spatial, pointers, baseline);
        if (sticky_none_serial) {
            if (!none_resident_ && (resident_buffer_ || high_buffer_ || workspace_->allocated_bytes()))
                demote_none_residency();
            none_resident_ = true;
            high_resident_ = false;
            max_fused_batch_ = sam3_tracker_graph_batch_limit;
            fusion_disabled_ = true;
        } else {
            select_resident_mode(baseline);
            require_frame(image);
        }
        configure_batch_policy(spatial, pointers, resident_buffer_bytes(), sticky_none_serial);
        const auto pointer_positions = project_pointer_positions(inputs, frame_count, baseline);
        validate_memory_payload_inputs(inputs);
        auto result = run_propagation_batch(image, inputs, pointer_positions, static_cast<int>(spatial_count),
                                            static_cast<int>(pointer_count), batch, baseline);
        return result;
    }

    // Serial seed/reconditioning path. It intentionally keeps its own decoder shape.
    TrackerPrediction decode(const ImageFeatures& image, const std::vector<float>& conditioned,
                             const std::vector<float>* seed_mask = nullptr,
                             std::size_t budget_override = 0) {
        validate_frame(image);
        const auto budget = seed_mask ? seed_memory_workspace_budget() :
            (budget_override ? budget_override :
             (frame_serial_budget_ ? frame_serial_budget_ : seed_memory_workspace_budget()));
        select_resident_mode(budget);
        require_frame(image);
        const auto& weights = model_.definition.weights;
        const bool seed = seed_mask != nullptr;
        auto key = decoder_shape(seed, 1);
        if (apply_none_resident_policy(key, budget)) key = decoder_shape(seed, 1);
        bool built = false;
        auto& graph = decoder_graph_.ensure(*model_.runtime, stats_, workspace_, key,
                                            sam3_tracker_decoder_graph_capacity,
                                            built, graph_build_ms_);
        if (built) {
            const auto graph_start = std::chrono::steady_clock::now();
            auto* ctx = graph.context();
            auto* current = seed && !none_resident_ ? resident_tracker_[2] :
                input_tensor(ctx, seed ? "sam_seed_current" : "sam_current", 256, 72, 72);
            auto* position = none_resident_ ? input_tensor(ctx, "sam_position", 256, 72, 72) : resident_dense_position_;
            auto* sparse = none_resident_ ? input_tensor(ctx, "sam_sparse", 256, 2) : resident_no_point_;
            auto* first = high_resident_ ? resident_tracker_[0] : input_tensor(ctx, "sam_first_neck", 256, 288, 288);
            auto* second = high_resident_ ? resident_tracker_[1] : input_tensor(ctx, "sam_second_neck", 256, 144, 144);
            ggml_tensor* seed_tensor = nullptr;
            ggml_tensor* dense = nullptr;
            if (seed) {
                seed_tensor = input_tensor(ctx, "sam_seed_mask", 1152, 1152);
                dense = sam3_conv_2d(ctx, weights.tensors.at("trk_mask_ds.weight"), seed_tensor, 4, 4, 0, 0);
                dense = ggml_add(ctx, dense, weights.tensors.at("trk_mask_ds.bias"));
                for (int stage = 0; stage < 2; ++stage) {
                    dense = sam3_conv_2d(ctx, weights.sam_pe.mask_ds_conv_w[stage], dense, 2, 2, 0, 0);
                    const auto channels = weights.sam_pe.mask_ds_conv_b[stage]->ne[0];
                    dense = ggml_add(ctx, dense, ggml_reshape_4d(ctx, weights.sam_pe.mask_ds_conv_b[stage], 1, 1, channels, 1));
                    dense = ggml_cont(ctx, ggml_permute(ctx, dense, 1, 2, 0, 3));
                    dense = sam3_layer_norm_2d(ctx, dense, weights.sam_pe.mask_ds_norm_w[stage], weights.sam_pe.mask_ds_norm_b[stage]);
                    dense = ggml_gelu_erf(ctx, dense);
                    dense = ggml_cont(ctx, ggml_permute(ctx, dense, 2, 0, 1, 3));
                }
                dense = sam3_conv_2d_sk_p0(ctx, weights.sam_pe.mask_ds_conv_w[2], dense);
                dense = ggml_add(ctx, dense, ggml_reshape_4d(ctx, weights.sam_pe.mask_ds_conv_b[2], 1, 1, 256, 1));
                dense = ggml_cont(ctx, ggml_permute(ctx, dense, 1, 2, 0, 3));
            } else {
                dense = ggml_repeat(ctx, ggml_reshape_4d(ctx, weights.sam_pe.no_mask_embed, 256, 1, 1, 1), current);
            }
            auto output = sam3_build_sam_dec_graph(ctx, weights, current, position, sparse, dense, first, second);
            auto* pointers = sam3_mlp_forward(ctx, output.mask_tokens, weights.obj_ptr_proj_w, weights.obj_ptr_proj_b, 3);
            graph.output(output.masks); graph.output(output.iou_pred); graph.output(output.obj_score); graph.output(pointers);
            decoder_inputs_ = {seed ? nullptr : current, seed_tensor,
                               high_resident_ ? nullptr : first, high_resident_ ? nullptr : second,
                               seed && none_resident_ ? current : nullptr,
                               none_resident_ ? position : nullptr, none_resident_ ? sparse : nullptr};
            decoder_masks_ = output.masks;
            decoder_iou_ = output.iou_pred;
            decoder_object_ = output.obj_score;
            decoder_pointer_ = pointers;
            graph_build_ms_ += elapsed_ms(graph_start);
        }

        const auto required = decoder_graph_.required_workspace_bytes();
        if (required > budget || resident_buffer_bytes() > budget - std::min(required, budget)) {
            if (high_resident_) {
                demote_high_residency();
                return decode(image, conditioned, seed_mask, budget_override);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, budget);
                demote_none_residency();
                return decode(image, conditioned, seed_mask, budget_override);
            }
            ++budget_failures_;
            throw budget_error("decoder", required, budget);
        }
        std::size_t actual_arena = required;
        if (!allocate_under_budget(graph, required, budget, &actual_arena)) {
            if (high_resident_) {
                demote_high_residency();
                return decode(image, conditioned, seed_mask, budget_override);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, budget);
                demote_none_residency();
                return decode(image, conditioned, seed_mask, budget_override);
            }
            ++budget_failures_;
            throw budget_error("decoder", actual_arena, budget);
        }
        if (!seed) upload_tracked(decoder_inputs_.conditioned, conditioned);
        else if (none_resident_) upload_tracked(decoder_inputs_.frame_current, image.tracker[2]);
        if (seed) upload_tracked(decoder_inputs_.seed, *seed_mask);
        if (none_resident_) {
            upload_tracked(decoder_inputs_.position, dense_position_);
            upload_tracked(decoder_inputs_.sparse, no_point_sparse_);
        }
        if (!high_resident_) {
            upload_tracked(decoder_inputs_.first, image.tracker[0]);
            upload_tracked(decoder_inputs_.second, image.tracker[1]);
        }
        if (!none_resident_) {
            ensure_core_resident_uploaded(image);
            if (high_resident_) ensure_high_resident_uploaded(image);
        }
        graph.compute();
        return finish_prediction(conditioned, download_tracked(decoder_masks_), download_tracked(decoder_iou_),
                                 download_tracked(decoder_object_), download_tracked(decoder_pointer_), seed_mask);
    }

    std::vector<ggml_bf16_t> encode_memory(const ImageFeatures& image, const std::vector<float>& mask,
                                           bool present, std::vector<float>* unrounded = nullptr) {
        validate_frame(image);
        const auto budget = frame_serial_budget_ ? frame_serial_budget_ : seed_memory_workspace_budget();
        select_resident_mode(budget);
        require_frame(image);
        auto key = memory_shape(present, unrounded != nullptr);
        if (apply_none_resident_policy(key, budget)) key = memory_shape(present, unrounded != nullptr);
        bool built = false;
        auto& graph = memory_graph_.ensure(*model_.runtime, stats_, workspace_, key,
                                           sam3_tracker_memory_graph_capacity,
                                           built, graph_build_ms_);
        if (built) {
            const auto graph_start = std::chrono::steady_clock::now();
            auto* ctx = graph.context();
            memory_mask_input_ = input_tensor(ctx, "memory_mask", 1152, 1152);
            memory_pixels_input_ = none_resident_ ? input_tensor(ctx, "memory_pixels", 256, 72, 72) : resident_tracker_[2];
            auto* output = build_memory_encoder(ctx, model_.definition.weights, memory_mask_input_, memory_pixels_input_, present);
            graph.output(output);
            memory_output_ = output;
            graph_build_ms_ += elapsed_ms(graph_start);
        }
        const auto required = memory_graph_.required_workspace_bytes();
        if (required > budget || resident_buffer_bytes() > budget - std::min(required, budget)) {
            if (high_resident_) demote_high_residency();
            if (required > budget || resident_buffer_bytes() > budget - std::min(required, budget)) {
                if (!none_resident_) {
                    remember_none_resident_policy(key, budget);
                    demote_none_residency();
                    return encode_memory(image, mask, present, unrounded);
                }
                ++budget_failures_;
                throw budget_error("memory encoder", required, budget);
            }
        }
        std::size_t actual_arena = required;
        if (!allocate_under_budget(graph, required, budget, &actual_arena)) {
            if (high_resident_) {
                demote_high_residency();
                return encode_memory(image, mask, present, unrounded);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, budget);
                demote_none_residency();
                return encode_memory(image, mask, present, unrounded);
            }
            ++budget_failures_;
            throw budget_error("memory encoder", actual_arena, budget);
        }
        upload_tracked(memory_mask_input_, mask);
        if (none_resident_) upload_tracked(memory_pixels_input_, image.tracker[2]);
        else ensure_core_resident_uploaded(image);
        graph.compute();
        auto values = download_tracked(memory_output_);
        if (unrounded) *unrounded = values;
        std::vector<ggml_bf16_t> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) result[i] = ggml_fp32_to_bf16(values[i]);
        return result;
    }

    const std::vector<float>& no_object_pointer() const { return no_object_; }
    const GraphDiagnostics& diagnostics() const { return workspace_->diagnostics(); }
    std::size_t resident_buffer_bytes() const {
        return (resident_buffer_ ? ggml_backend_buffer_get_size(resident_buffer_.get()) : 0) +
            (high_buffer_ ? ggml_backend_buffer_get_size(high_buffer_.get()) : 0);
    }
    std::size_t workspace_live_bytes() const { return workspace_->allocated_bytes(); }
    std::size_t model_buffer_bytes() const { return ggml_backend_buffer_get_size(model_.buffer.get()); }
    std::size_t batch_splits() const { return batch_splits_; }
    std::size_t serial_fallbacks() const { return serial_fallbacks_; }
    double graph_build_ms() const { return graph_build_ms_; }
    double upload_ms() const { return upload_ms_; }
    double download_ms() const { return download_ms_; }

    std::string diagnostics_json() const {
        const auto& d = diagnostics();
        const auto graph_builds = d.graph_builds + probe_diagnostics_.graph_builds;
        const auto graph_reuses = d.graph_reuses + probe_diagnostics_.graph_reuses;
        const auto graph_binds = d.graph_binds + probe_diagnostics_.graph_binds;
        const auto reserve_probes = d.reserve_probes + probe_diagnostics_.reserve_probes;
        const auto compute_calls = d.compute_calls + probe_diagnostics_.compute_calls;
        const auto workspace_peak = std::max(d.workspace_peak_bytes, probe_diagnostics_.workspace_peak_bytes);
        std::ostringstream out;
        out << std::setprecision(9)
            << "{\"scope\":\"tracker-session-cumulative\",\"graph_builds\":" << graph_builds
            << ",\"graph_reuses\":" << graph_reuses
            << ",\"graph_binds\":" << graph_binds
            << ",\"reserve_probes\":" << reserve_probes
            << ",\"compute_calls\":" << compute_calls
            << ",\"workspace_peak_bytes\":" << workspace_peak
            << ",\"resident_buffer_bytes\":" << resident_buffer_bytes()
            << ",\"resident_peak_bytes\":" << resident_peak_bytes_
            << ",\"high_fpn_resident\":" << (high_resident_ ? "true" : "false")
            << ",\"none_resident\":" << (none_resident_ ? "true" : "false")
            << ",\"resident_demotions\":" << resident_demotions_
            << ",\"none_resident_fallbacks\":" << none_resident_fallbacks_
            << ",\"memory_payload_peak_bytes\":" << memory_payload_peak_bytes_
            << ",\"budget_failures\":" << budget_failures_
            << ",\"workspace_live_bytes\":" << workspace_live_bytes()
            << ",\"model_weight_buffer_bytes\":" << model_buffer_bytes()
            << ",\"model_tracker_live_backend_bytes\":" << resident_buffer_bytes() + workspace_live_bytes() + model_buffer_bytes()
            << ",\"batch_splits\":" << batch_splits_
            << ",\"serial_fallbacks\":" << serial_fallbacks_
            << ",\"max_fused_batch\":" << max_fused_batch_
            << ",\"fusion_disabled\":" << (fusion_disabled_ ? "true" : "false")
            << ",\"serial_workspace_budget_bytes\":" <<
                (frame_serial_budget_ ? frame_serial_budget_ : seed_memory_budget_)
            << ",\"serial_workspace_required_bytes\":" <<
                (frame_serial_budget_ ? frame_serial_required_budget_ : seed_memory_required_budget_)
            << ",\"graph_build_ms\":" << graph_build_ms_
            << ",\"bind_ms\":" << d.bind_ms + probe_diagnostics_.bind_ms
            << ",\"reserve_ms\":" << d.reserve_ms + probe_diagnostics_.reserve_ms
            << ",\"compute_ms_includes_sync\":true,\"download_ms_may_include_wait\":true"
            << ",\"compute_ms\":" << d.compute_ms + probe_diagnostics_.compute_ms
            << ",\"upload_ms\":" << upload_ms_ << ",\"download_ms\":" << download_ms_
            << ",\"gpu_timestamps\":false}";
        return out.str();
    }

private:
    struct MemoryPayload { std::vector<float> memory, pointer_position; };
    struct WorkspaceProbe { std::size_t required = 0, actual = 0; };
    struct NoneResidentPolicy { TrackerGraphShape low_shape; std::size_t budget = 0; };
    struct DecoderInputs {
        ggml_tensor* conditioned = nullptr;
        ggml_tensor* seed = nullptr;
        ggml_tensor* first = nullptr;
        ggml_tensor* second = nullptr;
        ggml_tensor* frame_current = nullptr;
        ggml_tensor* position = nullptr;
        ggml_tensor* sparse = nullptr;
    };

    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    void accumulate_probe_diagnostics(const GraphDiagnostics& diagnostics) {
        probe_diagnostics_.graph_builds += diagnostics.graph_builds;
        probe_diagnostics_.graph_reuses += diagnostics.graph_reuses;
        probe_diagnostics_.graph_binds += diagnostics.graph_binds;
        probe_diagnostics_.reserve_probes += diagnostics.reserve_probes;
        probe_diagnostics_.compute_calls += diagnostics.compute_calls;
        probe_diagnostics_.workspace_peak_bytes = std::max(probe_diagnostics_.workspace_peak_bytes,
                                                           diagnostics.workspace_peak_bytes);
        probe_diagnostics_.bind_ms += diagnostics.bind_ms;
        probe_diagnostics_.reserve_ms += diagnostics.reserve_ms;
        probe_diagnostics_.compute_ms += diagnostics.compute_ms;
    }

    std::vector<float> read_weight(ggml_tensor* tensor) {
        if (!tensor || tensor->type != GGML_TYPE_F32) throw std::runtime_error("tracker constant must be F32");
        const auto start = std::chrono::steady_clock::now();
        auto values = download(tensor, stats_);
        download_ms_ += elapsed_ms(start);
        return values;
    }

    void upload_tracked(ggml_tensor* tensor, const std::vector<float>& values) {
        const auto start = std::chrono::steady_clock::now();
        upload(tensor, values, stats_);
        upload_ms_ += elapsed_ms(start);
    }

    std::vector<float> download_tracked(ggml_tensor* tensor) {
        const auto start = std::chrono::steady_clock::now();
        auto values = download(tensor, stats_);
        download_ms_ += elapsed_ms(start);
        return values;
    }

    void make_resident_context() {
        resident_context_ = make_context(6);
        high_context_ = make_context(2);
        for (int i = 0; i < 2; ++i) {
            const int size = 288 >> i;
            resident_tracker_[i] = ggml_new_tensor_4d(high_context_.get(), GGML_TYPE_F32, 256, size, size, 1);
            ggml_set_name(resident_tracker_[i], ("tracker_frame_" + std::to_string(i)).c_str());
        }
        resident_tracker_[2] = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 72, 72, 1);
        ggml_set_name(resident_tracker_[2], "tracker_frame_2");
        resident_position_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 5184, 1, 1);
        resident_memory_position_bank_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32,
                                                            64, 5184, 7, 1);
        resident_dense_position_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 72, 72, 1);
        resident_rope_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 2, 128, 5184, 1);
        resident_no_point_ = ggml_new_tensor_4d(resident_context_.get(), GGML_TYPE_F32, 256, 2, 1, 1);
        ggml_set_name(resident_position_, "tracker_frame_position");
        ggml_set_name(resident_memory_position_bank_, "tracker_memory_position_bank");
        ggml_set_name(resident_dense_position_, "tracker_dense_position");
        ggml_set_name(resident_rope_, "tracker_rope_frequencies");
        ggml_set_name(resident_no_point_, "tracker_no_point_prompt");
    }

    void validate_frame(const ImageFeatures& image) const {
        if (!active_image_ || active_image_ != &image)
            throw std::runtime_error("tracker execution requires the current resident frame");
    }

    std::size_t context_buffer_bytes(const ggml_context* context) const {
        const auto buffer_type = ggml_backend_get_default_buffer_type(model_.runtime->weights_backend());
        return ggml_backend_alloc_ctx_tensors_from_buft_size(const_cast<ggml_context*>(context), buffer_type);
    }

    void select_resident_mode(std::size_t budget) {
        if (none_resident_ || resident_buffer_) return;
        const auto core_bytes = context_buffer_bytes(resident_context_.get());
        const auto high_bytes = high_residency_allowed_ ? context_buffer_bytes(high_context_.get()) : 0;
        if (core_bytes > budget) {
            demote_none_residency();
            return;
        }
        if (high_residency_allowed_ && high_bytes > budget - core_bytes) {
            high_resident_ = false;
            high_residency_allowed_ = false;
            ++resident_demotions_;
        } else {
            high_resident_ = high_residency_allowed_;
        }
    }

    void require_frame(const ImageFeatures& image) {
        validate_frame(image);
        if (none_resident_) return;
        if (!resident_buffer_) {
            auto buffer = ggml_backend_alloc_ctx_tensors(resident_context_.get(), model_.runtime->weights_backend());
            if (!buffer) throw std::runtime_error("failed to allocate tracker resident frame inputs");
            resident_buffer_.reset(buffer);
            ggml_backend_buffer_set_usage(resident_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (high_resident_ && !high_buffer_) {
            auto high_buffer = ggml_backend_alloc_ctx_tensors(high_context_.get(), model_.runtime->weights_backend());
            if (!high_buffer) throw std::runtime_error("failed to allocate tracker high-resolution frame inputs");
            high_buffer_.reset(high_buffer);
            ggml_backend_buffer_set_usage(high_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        resident_peak_bytes_ = std::max(resident_peak_bytes_, resident_buffer_bytes());
    }

    void ensure_core_resident_uploaded(const ImageFeatures& image) {
        if (!resident_buffer_) throw std::runtime_error("tracker core resident frame was not allocated");
        resident_uploads_.ensure_core([&] {
            upload_tracked(resident_tracker_[2], image.tracker[2]);
            upload_tracked(resident_position_, image.position);
            upload_tracked(resident_memory_position_bank_, memory_position_bank_);
            upload_tracked(resident_dense_position_, dense_position_);
            upload_tracked(resident_rope_, rope_);
            upload_tracked(resident_no_point_, no_point_sparse_);
        });
    }

    void ensure_high_resident_uploaded(const ImageFeatures& image) {
        if (!high_buffer_) throw std::runtime_error("tracker high-resolution resident frame was not allocated");
        resident_uploads_.ensure_high([&] {
            upload_tracked(resident_tracker_[0], image.tracker[0]);
            upload_tracked(resident_tracker_[1], image.tracker[1]);
        });
    }

    static void clear_context(ggml_context* context) noexcept {
        if (!context) return;
        for (auto* tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
            tensor->extra = nullptr;
        }
    }

    void demote_high_residency() noexcept {
        if (!high_resident_) return;
        // Cached propagation/decoder graphs can hold these external tensors as
        // sources, so drop those graphs before releasing their storage.
        propagation_graph_.reset(); decoder_graph_.reset();
        conditioned_ = propagation_masks_ = propagation_iou_ = propagation_object_ =
            propagation_pointer_ = propagation_pointer_position_ = propagation_first_input_ =
            propagation_second_input_ = nullptr;
        decoder_inputs_ = {};
        decoder_masks_ = decoder_iou_ = decoder_object_ = decoder_pointer_ = nullptr;
        clear_context(high_context_.get());
        high_buffer_.reset();
        resident_uploads_.release_high();
        high_resident_ = false;
        high_residency_allowed_ = false;
        ++resident_demotions_;
        policy_resident_bytes_ = resident_buffer_bytes();
    }

    void demote_none_residency() noexcept {
        if (none_resident_) return;
        // Flush scheduler references before releasing either resident context.
        // Cached graph metadata remains intact and gets a distinct none-resident
        // key; a later frame reuses the original resident tensor metadata.
        workspace_->release_storage();
        clear_context(resident_context_.get());
        clear_context(high_context_.get());
        resident_buffer_.reset();
        high_buffer_.reset();
        resident_uploads_.reset_frame();
        high_resident_ = false;
        none_resident_ = true;
        ++none_resident_fallbacks_;
        policy_resident_bytes_ = 0;
        max_fused_batch_ = sam3_tracker_graph_batch_limit;
        fusion_disabled_ = false;
    }

    bool apply_none_resident_policy(const TrackerGraphShape& low_shape, std::size_t budget) {
        if (!none_resident_policy_ || none_resident_policy_->budget != budget ||
            !(none_resident_policy_->low_shape == low_shape)) return false;
        demote_none_residency();
        return true;
    }

    void remember_none_resident_policy(const TrackerGraphShape& low_shape, std::size_t budget) {
        none_resident_policy_ = NoneResidentPolicy{low_shape, budget};
    }

    void remember_none_serial_policy(int spatial, int pointers, std::size_t budget) {
        none_serial_policy_ = TrackerNoneSerialPolicy{model_.runtime.get(), model_.runtime->backend(),
            model_.model_info.precision + ":" + model_.model_info.storage_profile,
            model_.runtime->arithmetic_profile(), spatial, pointers, budget};
    }

    TrackerGraphShape shape(TrackerGraphShape::Stage stage, int spatial, int pointers, int batch,
                            bool seed = false, bool present = true, bool unrounded = false) const {
        TrackerGraphShape value;
        value.stage = stage; value.backend = model_.runtime->backend();
        value.storage_profile = model_.model_info.precision + ":" + model_.model_info.storage_profile;
        value.arithmetic_profile = model_.runtime->arithmetic_profile();
        value.spatial_count = spatial; value.pointer_count = pointers; value.batch = batch;
        value.seed = seed; value.present = present; value.unrounded_output = unrounded;
        value.high_resident = high_resident_ &&
            (stage == TrackerGraphShape::Stage::propagation || stage == TrackerGraphShape::Stage::decoder);
        value.none_resident = none_resident_;
        return value;
    }

    static std::vector<float> sine_positions(const std::vector<TrackerPropagationInput>& inputs, int frame_count) {
        std::vector<float> values;
        const auto pointers = inputs.front().selection->pointers.size();
        if (!pointers) return values;
        const int maximum = std::min(frame_count, 16);
        if (maximum <= 1) throw std::runtime_error("tracker pointer temporal range is invalid");
        values.reserve(inputs.size() * pointers * 256);
        for (const auto& input : inputs) for (const auto& selected : input.selection->pointers) {
            const float relative = static_cast<float>(selected.position) / (maximum - 1);
            for (int phase = 0; phase < 2; ++phase) for (int k = 0; k < 128; ++k) {
                const float angle = relative / std::pow(10000.0f, 2.0f * (k / 2) / 128);
                values.push_back(phase ? std::cos(angle) : std::sin(angle));
            }
        }
        return values;
    }

    std::vector<float> tiled_rope(int spatial) const {
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(spatial) * rope_.size());
        for (int i = 0; i < spatial; ++i) values.insert(values.end(), rope_.begin(), rope_.end());
        return values;
    }

    std::vector<float> repeated_no_point(int batch) const {
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(batch) * no_point_sparse_.size());
        for (int i = 0; i < batch; ++i) values.insert(values.end(), no_point_sparse_.begin(), no_point_sparse_.end());
        return values;
    }

    std::vector<float> project_pointer_positions(const std::vector<TrackerPropagationInput>& inputs,
                                                 int frame_count, std::size_t serial_budget) {
        const auto pointer_count = static_cast<int>(inputs.front().selection->pointers.size());
        if (!pointer_count) return {};
        const auto batch = static_cast<int>(inputs.size());
        if (fusion_disabled_ && batch > 1) {
            std::vector<float> result;
            result.reserve(inputs.size() * static_cast<std::size_t>(pointer_count) * 64);
            for (int begin = 0; begin < batch; ++begin) {
                std::vector<TrackerPropagationInput> one{inputs[static_cast<std::size_t>(begin)]};
                auto values = project_pointer_positions(one, frame_count, serial_budget);
                if (values.size() != static_cast<std::size_t>(pointer_count) * 64)
                    throw std::runtime_error("serial pointer projection returned the wrong object count");
                result.insert(result.end(), values.begin(), values.end());
            }
            if (result.size() != inputs.size() * static_cast<std::size_t>(pointer_count) * 64)
                throw std::runtime_error("serial pointer projection omitted an object");
            return result;
        }
        if (batch > max_fused_batch_) {
            std::vector<float> result;
            result.reserve(inputs.size() * static_cast<std::size_t>(pointer_count) * 64);
            sam3_for_each_tracker_chunk(batch, max_fused_batch_, [&](int begin, int count) {
                std::vector<TrackerPropagationInput> slice(inputs.begin() + begin, inputs.begin() + begin + count);
                auto values = project_pointer_positions(slice, frame_count, serial_budget);
                const auto expected = static_cast<std::size_t>(count) * pointer_count * 64;
                if (values.size() != expected)
                    throw std::runtime_error("batched pointer projection returned the wrong chunk size");
                result.insert(result.end(), values.begin(), values.end());
            });
            if (result.size() != inputs.size() * static_cast<std::size_t>(pointer_count) * 64)
                throw std::runtime_error("batched pointer projection omitted an object");
            return result;
        }
        auto key = shape(TrackerGraphShape::Stage::pointer_time, 0, pointer_count, batch);
        tracker_shape_tensor(key, {256, pointer_count, batch, 1});
        tracker_shape_tensor(key, {64, pointer_count, batch, 1});
        if (apply_none_resident_policy(key, serial_budget)) {
            key = shape(TrackerGraphShape::Stage::pointer_time, 0, pointer_count, batch);
            tracker_shape_tensor(key, {256, pointer_count, batch, 1});
            tracker_shape_tensor(key, {64, pointer_count, batch, 1});
        }
        bool built = false;
        auto& graph = pointer_graph_.ensure(*model_.runtime, stats_, workspace_, key,
                                            sam3_tracker_pointer_graph_capacity,
                                            built, graph_build_ms_);
        if (built) {
            const auto graph_start = std::chrono::steady_clock::now();
            auto* ctx = graph.context();
            pointer_input_ = input_tensor(ctx, "pointer_temporal_input", 256, pointer_count, batch);
            pointer_output_ = ggml_add(ctx, ggml_mul_mat(ctx, model_.definition.weights.obj_ptr_tpos_w, pointer_input_),
                                       model_.definition.weights.obj_ptr_tpos_b);
            ggml_set_name(pointer_output_, "pointer_temporal_output");
            graph.output(pointer_output_);
            graph_build_ms_ += elapsed_ms(graph_start);
        }
        const auto required = pointer_graph_.required_workspace_bytes();
        bool over_budget = required > serial_budget || resident_buffer_bytes() >
            serial_budget - std::min(required, serial_budget);
        if (over_budget && batch > 1) {
            ++batch_splits_;
            max_fused_batch_ = std::min(max_fused_batch_, std::max(1, batch / 2));
            pointer_graph_.reset(); pointer_input_ = pointer_output_ = nullptr;
            return project_pointer_positions(inputs, frame_count, serial_budget);
        }
        if (over_budget) {
            if (high_resident_) {
                demote_high_residency();
                over_budget = required > serial_budget || resident_buffer_bytes() >
                    serial_budget - std::min(required, serial_budget);
            }
            if (over_budget && !none_resident_) {
                remember_none_resident_policy(key, serial_budget);
                demote_none_residency();
                return project_pointer_positions(inputs, frame_count, serial_budget);
            }
            if (over_budget) {
                ++budget_failures_;
                throw budget_error("pointer projection", required, serial_budget);
            }
        }
        std::size_t actual_arena = required;
        if (!allocate_under_budget(graph, required, serial_budget, &actual_arena)) {
            if (batch > 1) {
                ++batch_splits_;
                max_fused_batch_ = std::min(max_fused_batch_, std::max(1, batch / 2));
                pointer_graph_.reset(); pointer_input_ = pointer_output_ = nullptr;
                return project_pointer_positions(inputs, frame_count, serial_budget);
            }
            if (high_resident_) {
                demote_high_residency();
                return project_pointer_positions(inputs, frame_count, serial_budget);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, serial_budget);
                demote_none_residency();
                return project_pointer_positions(inputs, frame_count, serial_budget);
            }
            ++budget_failures_;
            throw budget_error("pointer projection", actual_arena, serial_budget);
        }
        const auto values = sine_positions(inputs, frame_count);
        upload_tracked(pointer_input_, values);
        graph.compute();
        return download_tracked(pointer_output_);
    }

    void validate_memory_payload_inputs(const std::vector<TrackerPropagationInput>& inputs) const {
        for (const auto& input : inputs) {
            for (const auto& selected : input.selection->spatial)
                (void)input.records->at(selected.frame);
            for (const auto& selected : input.selection->pointers) {
                const auto& pointer = input.records->at(selected.frame).pointer;
                if (pointer.size() != 256)
                    throw std::runtime_error("tracker pointer has an invalid feature width");
            }
        }
    }

    static std::vector<float> slice_pointer_positions(const std::vector<float>& source,
            int pointers, std::size_t begin, std::size_t count) {
        if (!pointers) {
            if (!source.empty()) throw std::runtime_error("tracker pointer-position payload is unexpectedly nonempty");
            return {};
        }
        const auto row = static_cast<std::size_t>(pointers) * 64;
        if (source.size() % row || begin > source.size() / row || count > source.size() / row - begin)
            throw std::runtime_error("tracker pointer-position slice is out of range");
        const auto first = begin * row;
        return std::vector<float>(source.begin() + first, source.begin() + first + count * row);
    }

    MemoryPayload make_memory_payload(const std::vector<TrackerPropagationInput>& inputs,
                                      const std::vector<float>& projected) {
        const auto spatial_count = inputs.front().selection->spatial.size();
        const auto pointer_count = inputs.front().selection->pointers.size();
        const auto batch = inputs.size();
        const auto tokens = spatial_count * 5184 + pointer_count * 4;
        MemoryPayload result;
        result.memory.reserve(batch * tokens * 64);
        result.pointer_position.reserve(batch * pointer_count * 4 * 64);
        for (std::size_t b = 0; b < batch; ++b) {
            const auto& input = inputs[b];
            for (const auto& selected : input.selection->spatial) {
                const auto& record = input.records->at(selected.frame);
                for (const auto value : record.memory) result.memory.push_back(ggml_bf16_to_fp32(value));
            }
            for (std::size_t i = 0; i < input.selection->pointers.size(); ++i) {
                const auto& selected = input.selection->pointers[i];
                const auto& pointer = input.records->at(selected.frame).pointer;
                result.memory.insert(result.memory.end(), pointer.begin(), pointer.end());
                const auto projected_offset = (b * pointer_count + i) * 64;
                for (int token = 0; token < 4; ++token)
                    result.pointer_position.insert(result.pointer_position.end(),
                        projected.begin() + projected_offset, projected.begin() + projected_offset + 64);
            }
        }
        const auto bytes = (result.memory.capacity() + result.pointer_position.capacity()) * sizeof(float);
        memory_payload_peak_bytes_ = std::max(memory_payload_peak_bytes_, bytes);
        return result;
    }

    std::vector<float> memory_position_payload(const std::vector<TrackerPropagationInput>& inputs,
                                               const MemoryPayload& payload) const {
        if (inputs.empty()) return {};
        const auto spatial_count = inputs.front().selection->spatial.size();
        const auto pointer_count = inputs.front().selection->pointers.size();
        const auto batch = inputs.size();
        constexpr std::size_t memory_plane_size = 64u * 5184;
        const auto pointer_row = pointer_count * 4 * 64;
        std::vector<float> values;
        values.reserve(batch * (spatial_count * memory_plane_size + pointer_row));
        for (std::size_t b = 0; b < batch; ++b) {
            for (const auto& selected : inputs[b].selection->spatial) {
                if (selected.position < 0 || selected.position > 6)
                    throw std::runtime_error("tracker memory position is outside its resident table");
                const auto position = static_cast<std::size_t>(6 - selected.position);
                const auto first = memory_position_bank_.begin() + position * memory_plane_size;
                values.insert(values.end(), first, first + memory_plane_size);
            }
            if (pointer_row) {
                const auto first = payload.pointer_position.begin() + b * pointer_row;
                values.insert(values.end(), first, first + pointer_row);
            }
        }
        return values;
    }

    TrackerGraphShape propagation_shape(int spatial, int pointers, int batch,
                                        const std::vector<TrackerPropagationInput>& inputs) const {
        const auto tokens = spatial * 5184 + pointers * 4;
        auto key = shape(TrackerGraphShape::Stage::propagation, spatial, pointers, batch, false, true, true);
        if (!none_resident_) {
            for (const auto& input : inputs) {
                for (const auto& selected : input.selection->spatial)
                    key.memory_position_order.push_back(selected.position);
                key.memory_position_order.push_back(-1);
            }
        }
        std::vector<std::array<std::int64_t, 4>> layouts;
        if (none_resident_) {
            layouts = {{256, 5184, 1, 1}, {256, 5184, 1, 1}, {64, tokens, batch, 1},
                {64, tokens, batch, 1}, {2, 128, 5184, 1},
                {2, 128, static_cast<std::int64_t>(spatial) * 5184, 1},
                {256, 72, 72, batch}, {256, 72, 72, 1}, {256, 2, batch, 1},
                {256, 288, 288, 1}, {256, 144, 144, 1}};
        } else {
            const auto layout_count = pointers ? 12u : 11u;
            layouts = {{256, 5184, 1, 1}, {256, 5184, 1, 1}, {64, tokens, batch, 1},
                {2, 128, 5184, 1}, {2, 128, static_cast<std::int64_t>(spatial) * 5184, 1},
                {64, 5184, 7, 1}, {256, 72, 72, batch}, {256, 72, 72, 1}, {256, 2, batch, 1},
                {256, 288, 288, 1}, {256, 144, 144, 1}};
            if (pointers) layouts.insert(layouts.begin() + 3,
                std::array<std::int64_t, 4>{64, pointers * 4, batch, 1});
            if (layouts.size() != layout_count) throw std::runtime_error("tracker propagation layout key is inconsistent");
        }
        for (const auto& dims : layouts)
            tracker_shape_tensor(key, {dims[0], dims[1], dims[2], dims[3]});
        return key;
    }

    ggml_tensor* build_memory_positions(ggml_context* ctx,
            const std::vector<TrackerPropagationInput>& inputs, ggml_tensor* pointer_position) {
        ggml_tensor* batch_position = nullptr;
        for (std::size_t b = 0; b < inputs.size(); ++b) {
            ggml_tensor* object_position = nullptr;
            for (const auto& selected : inputs[b].selection->spatial) {
                if (selected.position < 0 || selected.position > 6)
                    throw std::runtime_error("tracker memory position is outside its resident table");
                const auto offset = static_cast<std::size_t>(6 - selected.position) * resident_memory_position_bank_->nb[2];
                auto* view = ggml_view_3d(ctx, resident_memory_position_bank_, 64, 5184, 1,
                    resident_memory_position_bank_->nb[1], resident_memory_position_bank_->nb[2], offset);
                object_position = object_position ? ggml_concat(ctx, object_position, view, 1) : view;
            }
            if (pointer_position) {
                const auto pointer_tokens = static_cast<std::int64_t>(inputs[b].selection->pointers.size() * 4);
                auto* view = ggml_view_3d(ctx, pointer_position, 64, pointer_tokens, 1,
                    pointer_position->nb[1], pointer_position->nb[2], b * pointer_position->nb[2]);
                object_position = ggml_concat(ctx, object_position, view, 1);
            }
            if (!object_position) throw std::runtime_error("tracker memory positions are empty");
            batch_position = batch_position ? ggml_concat(ctx, batch_position, object_position, 2) : object_position;
        }
        return batch_position;
    }

    void build_propagation_graph(GraphExecution& graph, int spatial, int pointers, int batch,
                                 const std::vector<TrackerPropagationInput>& inputs) {
        const auto& weights = model_.definition.weights;
        const auto tokens = spatial * 5184 + pointers * 4;
        auto* ctx = graph.context();
        auto* current = none_resident_ ? input_tensor(ctx, "tracker_current", 256, 5184) :
            ggml_reshape_3d(ctx, resident_tracker_[2], 256, 5184, 1);
        auto* current_position = none_resident_ ? input_tensor(ctx, "tracker_current_position", 256, 5184) :
            ggml_reshape_3d(ctx, resident_position_, 256, 5184, 1);
        auto* memory = input_tensor(ctx, "tracker_memory", 64, tokens, batch);
        propagation_pointer_position_ = pointers && !none_resident_
            ? input_tensor(ctx, "tracker_pointer_position", 64, pointers * 4, batch) : nullptr;
        ggml_tensor* memory_position = nullptr;
        ggml_tensor* rope = nullptr;
        ggml_tensor* key_rope = nullptr;
        if (none_resident_) {
            propagation_memory_position_input_ = input_tensor(ctx, "tracker_memory_position", 64, tokens, batch);
            memory_position = propagation_memory_position_input_;
            propagation_rope_input_ = input_tensor(ctx, "tracker_rope", 2, 128, 5184);
            propagation_key_rope_input_ = input_tensor(ctx, "tracker_key_rope", 2, 128, spatial * 5184);
            rope = propagation_rope_input_;
            key_rope = propagation_key_rope_input_;
            propagation_current_input_ = current;
            propagation_current_position_input_ = current_position;
        } else {
            const std::vector<TrackerPropagationInput> request = inputs;
            memory_position = build_memory_positions(ctx, request, propagation_pointer_position_);
            rope = resident_rope_;
            key_rope = resident_rope_;
            if (spatial > 1) {
                auto* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 128, spatial * 5184);
                key_rope = ggml_repeat(ctx, resident_rope_, target);
            }
        }
        auto* conditioned = sam3_build_mem_attn_graph(ctx, weights, current, current_position, memory,
            memory_position, rope, key_rope, pointers * 4);
        conditioned_ = conditioned;

        auto* conditioned_spatial = ggml_reshape_4d(ctx, conditioned, 256, 72, 72, batch);
        auto* dense = ggml_repeat(ctx,
            ggml_reshape_4d(ctx, weights.sam_pe.no_mask_embed, 256, 1, 1, 1), conditioned_spatial);
        ggml_tensor* sparse = none_resident_ ?
            input_tensor(ctx, "tracker_sparse", 256, 2, batch) : resident_no_point_;
        if (none_resident_) {
            propagation_dense_position_input_ = input_tensor(ctx, "tracker_dense_position", 256, 72, 72);
            propagation_sparse_input_ = sparse;
        } else if (batch > 1) {
            auto* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, batch);
            sparse = ggml_repeat(ctx, resident_no_point_, target);
        }
        auto* dense_position = none_resident_ ? propagation_dense_position_input_ : resident_dense_position_;
        auto* first = high_resident_ ? resident_tracker_[0] : input_tensor(ctx, "tracker_first_neck", 256, 288, 288);
        auto* second = high_resident_ ? resident_tracker_[1] : input_tensor(ctx, "tracker_second_neck", 256, 144, 144);
        propagation_first_input_ = high_resident_ ? nullptr : first;
        propagation_second_input_ = high_resident_ ? nullptr : second;
        auto output = sam3_build_sam_dec_graph(ctx, weights, conditioned_spatial,
            dense_position, sparse, dense, first, second);
        propagation_masks_ = output.masks;
        propagation_iou_ = output.iou_pred;
        propagation_object_ = output.obj_score;
        propagation_pointer_ = sam3_mlp_forward(ctx, output.mask_tokens,
            weights.obj_ptr_proj_w, weights.obj_ptr_proj_b, 3);
        graph.output(conditioned_); graph.output(propagation_masks_); graph.output(propagation_iou_);
        graph.output(propagation_object_); graph.output(propagation_pointer_);
    }

    std::vector<TrackerPrediction> run_propagation_batch(const ImageFeatures& image,
            const std::vector<TrackerPropagationInput>& inputs, const std::vector<float>& pointer_positions,
            int spatial, int pointers, int batch, std::size_t serial_budget) {
        if (batch <= 0) throw std::invalid_argument("tracker propagation batch is empty");
        if (static_cast<std::size_t>(batch) != inputs.size())
            throw std::invalid_argument("tracker propagation batch size does not match its inputs");
        if (fusion_disabled_)
            return run_serial_batch(image, inputs, pointer_positions, pointers, serial_budget);
        if (batch > max_fused_batch_) {
            std::vector<TrackerPrediction> result;
            result.reserve(static_cast<std::size_t>(batch));
            sam3_for_each_tracker_chunk(batch, max_fused_batch_, [&](int begin, int count) {
                std::vector<TrackerPropagationInput> slice(inputs.begin() + begin, inputs.begin() + begin + count);
                auto positions = slice_pointer_positions(pointer_positions, pointers,
                    static_cast<std::size_t>(begin), static_cast<std::size_t>(count));
                auto values = run_propagation_batch(image, slice,
                    positions, spatial, pointers, count, serial_budget);
                if (values.size() != static_cast<std::size_t>(count))
                    throw std::runtime_error("tracker propagation returned the wrong chunk size");
                result.insert(result.end(), std::make_move_iterator(values.begin()),
                              std::make_move_iterator(values.end()));
            });
            if (result.size() != static_cast<std::size_t>(batch))
                throw std::runtime_error("tracker propagation omitted an object");
            return result;
        }
        auto key = propagation_shape(spatial, pointers, batch, inputs);
        if (apply_none_resident_policy(key, serial_budget))
            key = propagation_shape(spatial, pointers, batch, inputs);
        bool built = false;
        auto& graph = propagation_graph_.ensure(*model_.runtime, stats_, workspace_, key,
                                                 sam3_tracker_fused_graph_capacity,
                                                 built, graph_build_ms_);
        if (built) {
            const auto graph_start = std::chrono::steady_clock::now();
            build_propagation_graph(graph, spatial, pointers, batch, inputs);
            graph_build_ms_ += elapsed_ms(graph_start);
        }
        const auto required = propagation_graph_.required_workspace_bytes();
        const auto resident = resident_buffer_bytes();
        bool over_budget = required > serial_budget || resident > serial_budget - std::min(required, serial_budget);
        if (over_budget && batch > 1) {
            ++batch_splits_;
            max_fused_batch_ = std::min(max_fused_batch_, std::max(1, batch / 2));
            propagation_graph_.reset(); conditioned_ = propagation_masks_ = propagation_iou_ =
                propagation_object_ = propagation_pointer_ = nullptr;
            return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
        }
        if (over_budget) {
            if (high_resident_) {
                demote_high_residency();
                return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, serial_budget);
                demote_none_residency();
                return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
            }
            ++serial_fallbacks_;
            remember_none_serial_policy(spatial, pointers, serial_budget);
            fusion_disabled_ = true;
            propagation_graph_.reset(); conditioned_ = propagation_masks_ = propagation_iou_ =
                propagation_object_ = propagation_pointer_ = nullptr;
            return run_serial_batch(image, inputs, pointer_positions, pointers, serial_budget);
        }
        std::size_t actual_arena = required;
        if (!allocate_under_budget(graph, required, serial_budget, &actual_arena)) {
            if (batch > 1) {
                ++batch_splits_;
                max_fused_batch_ = std::min(max_fused_batch_, std::max(1, batch / 2));
                propagation_graph_.reset(); conditioned_ = propagation_masks_ = propagation_iou_ =
                    propagation_object_ = propagation_pointer_ = nullptr;
                return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
            }
            if (high_resident_) {
                demote_high_residency();
                return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, serial_budget);
                demote_none_residency();
                return run_propagation_batch(image, inputs, pointer_positions, spatial, pointers, batch, serial_budget);
            }
            ++serial_fallbacks_;
            remember_none_serial_policy(spatial, pointers, serial_budget);
            fusion_disabled_ = true;
            propagation_graph_.reset(); conditioned_ = propagation_masks_ = propagation_iou_ =
                propagation_object_ = propagation_pointer_ = nullptr;
            return run_serial_batch(image, inputs, pointer_positions, pointers, serial_budget);
        }
        const auto payload = make_memory_payload(inputs, pointer_positions);
        upload_tracked(ggml_get_tensor(graph.context(), "tracker_memory"), payload.memory);
        if (none_resident_) {
            upload_tracked(propagation_current_input_, image.tracker[2]);
            upload_tracked(propagation_current_position_input_, image.position);
            upload_tracked(propagation_memory_position_input_, memory_position_payload(inputs, payload));
            upload_tracked(propagation_rope_input_, rope_);
            upload_tracked(propagation_key_rope_input_, tiled_rope(spatial));
            upload_tracked(propagation_dense_position_input_, dense_position_);
            upload_tracked(propagation_sparse_input_, repeated_no_point(batch));
            upload_tracked(propagation_first_input_, image.tracker[0]);
            upload_tracked(propagation_second_input_, image.tracker[1]);
        } else if (pointers) {
            upload_tracked(propagation_pointer_position_, payload.pointer_position);
        }
        if (!none_resident_ && !high_resident_) {
            upload_tracked(propagation_first_input_, image.tracker[0]);
            upload_tracked(propagation_second_input_, image.tracker[1]);
        }
        if (!none_resident_) {
            ensure_core_resident_uploaded(image);
            if (high_resident_) ensure_high_resident_uploaded(image);
        }
        graph.compute();
        const auto conditioned = download_tracked(conditioned_);
        const auto masks = download_tracked(propagation_masks_);
        const auto iou = download_tracked(propagation_iou_);
        const auto object = download_tracked(propagation_object_);
        const auto pointers_output = download_tracked(propagation_pointer_);
        return split_predictions(conditioned, masks, iou, object, pointers_output, batch);
    }

    std::vector<TrackerPrediction> split_predictions(const std::vector<float>& conditioned,
            const std::vector<float>& masks, const std::vector<float>& iou,
            const std::vector<float>& object, const std::vector<float>& pointers, int batch) const {
        constexpr std::size_t conditioned_size = 256u * 72 * 72;
        constexpr std::size_t mask_size = 288u * 288;
        std::vector<TrackerPrediction> result;
        result.reserve(static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) {
            TrackerPrediction prediction;
            const auto condition_offset = static_cast<std::size_t>(b) * conditioned_size;
            prediction.conditioned.assign(conditioned.begin() + condition_offset,
                                          conditioned.begin() + condition_offset + conditioned_size);
            const auto mask_offset = static_cast<std::size_t>(b) * 4 * mask_size;
            prediction.decoder_masks.assign(masks.begin() + mask_offset,
                                            masks.begin() + mask_offset + 4 * mask_size);
            prediction.decoder_iou.assign(iou.begin() + static_cast<std::size_t>(b) * 4,
                                          iou.begin() + static_cast<std::size_t>(b + 1) * 4);
            prediction.object_logit = object.at(static_cast<std::size_t>(b));
            prediction.decoder_object_logit = prediction.object_logit;
            int best = 1;
            for (int i = 2; i < 4; ++i)
                if (prediction.decoder_iou[i] > prediction.decoder_iou[best]) best = i;
            prediction.iou = prediction.decoder_iou[best];
            prediction.mask_index = best;
            prediction.pointer_index = best;
            const auto mask_start = static_cast<std::size_t>(best) * mask_size;
            prediction.mask.assign(prediction.decoder_masks.begin() + mask_start,
                                   prediction.decoder_masks.begin() + mask_start + mask_size);
            const auto pointer_start = static_cast<std::size_t>(b) * 4 * 256 + static_cast<std::size_t>(best) * 256;
            prediction.pointer.assign(pointers.begin() + pointer_start, pointers.begin() + pointer_start + 256);
            if (prediction.object_logit <= 0) {
                std::fill(prediction.mask.begin(), prediction.mask.end(), -1024.0f);
                prediction.pointer = no_object_;
            }
            result.push_back(std::move(prediction));
        }
        return result;
    }

    std::size_t serial_workspace_budget(int spatial, int pointers) {
        if (budget_spatial_ == spatial && budget_pointers_ == pointers && serial_budget_) return serial_budget_;
        release_frame_storage_for_probe();
        budget_spatial_ = spatial; budget_pointers_ = pointers;
        serial_budget_ = 0;
        serial_required_budget_ = 0;
        const auto pointer = probe_pointer_workspace(pointers);
        const auto condition = probe_condition_workspace(spatial, pointers);
        const auto decoder = probe_decoder_workspace(false);
        const auto seed_decoder = probe_decoder_workspace(true);
        const auto memory = probe_memory_workspace(false);
        const auto absent_memory = probe_memory_workspace(true);
        for (const auto& probe : {pointer, condition, decoder, seed_decoder, memory, absent_memory}) {
            serial_budget_ = std::max(serial_budget_, probe.actual);
            serial_required_budget_ = std::max(serial_required_budget_, probe.required);
        }
        return serial_budget_;
    }

    std::size_t seed_memory_workspace_budget() {
        if (seed_memory_budget_) return seed_memory_budget_;
        release_frame_storage_for_probe();
        seed_memory_required_budget_ = 0;
        const auto decoder = probe_decoder_workspace(false);
        const auto seed_decoder = probe_decoder_workspace(true);
        const auto memory = probe_memory_workspace(false);
        const auto absent_memory = probe_memory_workspace(true);
        for (const auto& probe : {decoder, seed_decoder, memory, absent_memory}) {
            seed_memory_budget_ = std::max(seed_memory_budget_, probe.actual);
            seed_memory_required_budget_ = std::max(seed_memory_required_budget_, probe.required);
        }
        return seed_memory_budget_;
    }

    std::runtime_error budget_error(const char* stage, std::size_t arena, std::size_t budget) const {
        std::ostringstream message;
        message << "tracker " << stage << " exceeds serial workspace budget: resident="
                << resident_buffer_bytes() << " arena=" << arena << " budget=" << budget;
        return std::runtime_error(message.str());
    }

    bool allocate_under_budget(GraphExecution& graph, std::size_t required,
                               std::size_t budget, std::size_t* actual_bytes = nullptr) {
        const auto resident = resident_buffer_bytes();
        if (actual_bytes) *actual_bytes = required;
        if (resident > budget || required > budget - resident) return false;
        const auto arena_limit = budget - resident;
        // A shared scheduler retains its largest previous allocation. Drop that
        // arena before binding a smaller shape when its live peak would exceed
        // this workload's serial baseline.
        if (workspace_->allocated_bytes() > arena_limit) workspace_->release_storage();
        graph.allocate();
        const auto actual = workspace_->allocated_bytes();
        if (actual_bytes) *actual_bytes = actual;
        if (actual > arena_limit) {
            workspace_->release_storage();
            return false;
        }
        return true;
    }

    void release_frame_storage_for_probe() noexcept {
        workspace_->release_storage();
        if (resident_buffer_) {
            clear_context(resident_context_.get());
            resident_buffer_.reset();
        }
        if (high_buffer_) {
            clear_context(high_context_.get());
            high_buffer_.reset();
        }
        resident_uploads_.reset_frame();
    }

    void configure_batch_policy(int spatial, int pointers, std::size_t resident_bytes,
                                bool preserve_none_serial = false) {
        if (policy_spatial_ == spatial && policy_pointers_ == pointers &&
            policy_resident_bytes_ == resident_bytes) return;
        policy_spatial_ = spatial;
        policy_pointers_ = pointers;
        policy_resident_bytes_ = resident_bytes;
        if (preserve_none_serial) return;
        max_fused_batch_ = sam3_tracker_graph_batch_limit;
        fusion_disabled_ = false;
    }

    WorkspaceProbe probe_actual_workspace(GraphExecution& graph) {
        try {
            const auto required = graph.required_workspace_bytes();
            graph.allocate();
            const auto diagnostics = graph.diagnostics();
            accumulate_probe_diagnostics(diagnostics);
            const auto actual = diagnostics.workspace_peak_bytes;
            return {required, actual};
        } catch (...) {
            accumulate_probe_diagnostics(graph.diagnostics());
            throw;
        }
    }

    WorkspaceProbe probe_pointer_workspace(int pointers) {
        if (!pointers) return {};
        const auto build_start = std::chrono::steady_clock::now();
        GraphExecution graph(*model_.runtime, 512, stats_);
        auto* input = input_tensor(graph.context(), "probe_pointer_input", 256, pointers);
        auto* output = ggml_add(graph.context(), ggml_mul_mat(graph.context(), model_.definition.weights.obj_ptr_tpos_w, input),
                                model_.definition.weights.obj_ptr_tpos_b);
        graph.output(output);
        graph_build_ms_ += elapsed_ms(build_start);
        return probe_actual_workspace(graph);
    }

    WorkspaceProbe probe_condition_workspace(int spatial, int pointers) {
        const auto tokens = spatial * 5184 + pointers * 4;
        const auto build_start = std::chrono::steady_clock::now();
        GraphExecution graph(*model_.runtime, 16384, stats_);
        auto* ctx = graph.context();
        auto* current = input_tensor(ctx, "probe_current", 256, 5184);
        auto* current_position = input_tensor(ctx, "probe_current_position", 256, 5184);
        auto* memory = input_tensor(ctx, "probe_memory", 64, tokens);
        auto* position = input_tensor(ctx, "probe_memory_position", 64, tokens);
        auto* rope = input_tensor(ctx, "probe_rope", 2, 128, 5184);
        auto* key_rope = input_tensor(ctx, "probe_key_rope", 2, 128, spatial * 5184);
        auto* output = sam3_build_mem_attn_graph(ctx, model_.definition.weights, current, current_position,
                                                  memory, position, rope, key_rope, pointers * 4);
        graph.output(output);
        graph_build_ms_ += elapsed_ms(build_start);
        return probe_actual_workspace(graph);
    }

    WorkspaceProbe probe_decoder_workspace(bool seed) {
        const auto& weights = model_.definition.weights;
        const auto build_start = std::chrono::steady_clock::now();
        GraphExecution graph(*model_.runtime, 8192, stats_);
        auto* ctx = graph.context();
        auto* current = input_tensor(ctx, "probe_decoder_current", 256, 72, 72);
        auto* position = input_tensor(ctx, "probe_decoder_position", 256, 72, 72);
        auto* sparse = input_tensor(ctx, "probe_decoder_sparse", 256, 2);
        auto* first = input_tensor(ctx, "probe_decoder_first", 256, 288, 288);
        auto* second = input_tensor(ctx, "probe_decoder_second", 256, 144, 144);
        ggml_tensor* dense = nullptr;
        if (seed) {
            auto* input = input_tensor(ctx, "probe_seed_mask", 1152, 1152);
            dense = sam3_conv_2d(ctx, weights.tensors.at("trk_mask_ds.weight"), input, 4, 4, 0, 0);
            dense = ggml_add(ctx, dense, weights.tensors.at("trk_mask_ds.bias"));
            for (int stage = 0; stage < 2; ++stage) {
                dense = sam3_conv_2d(ctx, weights.sam_pe.mask_ds_conv_w[stage], dense, 2, 2, 0, 0);
                const auto channels = weights.sam_pe.mask_ds_conv_b[stage]->ne[0];
                dense = ggml_add(ctx, dense, ggml_reshape_4d(ctx, weights.sam_pe.mask_ds_conv_b[stage], 1, 1, channels, 1));
                dense = ggml_cont(ctx, ggml_permute(ctx, dense, 1, 2, 0, 3));
                dense = sam3_layer_norm_2d(ctx, dense, weights.sam_pe.mask_ds_norm_w[stage], weights.sam_pe.mask_ds_norm_b[stage]);
                dense = ggml_gelu_erf(ctx, dense);
                dense = ggml_cont(ctx, ggml_permute(ctx, dense, 2, 0, 1, 3));
            }
            dense = sam3_conv_2d_sk_p0(ctx, weights.sam_pe.mask_ds_conv_w[2], dense);
            dense = ggml_add(ctx, dense, ggml_reshape_4d(ctx, weights.sam_pe.mask_ds_conv_b[2], 1, 1, 256, 1));
            dense = ggml_cont(ctx, ggml_permute(ctx, dense, 1, 2, 0, 3));
        } else {
            dense = ggml_repeat(ctx, ggml_reshape_4d(ctx, weights.sam_pe.no_mask_embed, 256, 1, 1, 1), current);
        }
        auto output = sam3_build_sam_dec_graph(ctx, weights, current, position, sparse, dense, first, second);
        auto* pointer = sam3_mlp_forward(ctx, output.mask_tokens, weights.obj_ptr_proj_w, weights.obj_ptr_proj_b, 3);
        graph.output(output.masks); graph.output(output.iou_pred); graph.output(output.obj_score); graph.output(pointer);
        graph_build_ms_ += elapsed_ms(build_start);
        return probe_actual_workspace(graph);
    }

    WorkspaceProbe probe_memory_workspace(bool present) {
        const auto build_start = std::chrono::steady_clock::now();
        GraphExecution graph(*model_.runtime, 4096, stats_);
        auto* ctx = graph.context();
        auto* mask = input_tensor(ctx, "probe_memory_mask", 1152, 1152);
        auto* pixels = input_tensor(ctx, "probe_memory_pixels", 256, 72, 72);
        auto* output = build_memory_encoder(ctx, model_.definition.weights, mask, pixels, present);
        graph.output(output);
        graph_build_ms_ += elapsed_ms(build_start);
        return probe_actual_workspace(graph);
    }

    std::vector<TrackerPrediction> run_serial_batch(const ImageFeatures& image,
            const std::vector<TrackerPropagationInput>& inputs, const std::vector<float>& pointer_positions,
            int pointers, std::size_t serial_budget) {
        std::vector<TrackerPrediction> result;
        result.reserve(inputs.size());
        for (std::size_t b = 0; b < inputs.size(); ++b) {
            std::vector<float> conditioned;
            {
                const std::vector<TrackerPropagationInput> one{inputs[b]};
                const auto one_pointer_positions = slice_pointer_positions(pointer_positions, pointers, b, 1);
                const auto payload = make_memory_payload(one, one_pointer_positions);
                conditioned = condition_one(image, *inputs[b].selection, payload, serial_budget);
            }
            result.push_back(decode(image, conditioned, nullptr, serial_budget));
        }
        return result;
    }

    std::vector<float> condition_one(const ImageFeatures& image, const MemorySelection& selection,
                                     const MemoryPayload& payload, std::size_t budget_override) {
        require_frame(image);
        const int pointer_count = static_cast<int>(selection.pointers.size());
        const int spatial_count = static_cast<int>(selection.spatial.size());
        const auto budget = budget_override ? budget_override :
            (frame_serial_budget_ ? frame_serial_budget_ : seed_memory_workspace_budget());
        auto key = condition_shape(spatial_count, pointer_count, selection);
        if (apply_none_resident_policy(key, budget)) key = condition_shape(spatial_count, pointer_count, selection);
        bool built = false;
        auto& graph = propagation_graph_.ensure(*model_.runtime, stats_, workspace_, key,
                                                 sam3_tracker_fused_graph_capacity,
                                                 built, graph_build_ms_);
        if (built) {
            const auto graph_start = std::chrono::steady_clock::now();
            auto* ctx = graph.context();
            auto* current = none_resident_ ? input_tensor(ctx, "tracker_current", 256, 5184) :
                ggml_reshape_3d(ctx, resident_tracker_[2], 256, 5184, 1);
            auto* current_position = none_resident_ ? input_tensor(ctx, "tracker_current_position", 256, 5184) :
                ggml_reshape_3d(ctx, resident_position_, 256, 5184, 1);
            auto* memory = input_tensor(ctx, "tracker_memory", 64, spatial_count * 5184 + pointer_count * 4);
            propagation_pointer_position_ = pointer_count && !none_resident_
                ? input_tensor(ctx, "tracker_pointer_position", 64, pointer_count * 4) : nullptr;
            ggml_tensor* memory_position = nullptr;
            ggml_tensor* rope = nullptr;
            ggml_tensor* key_rope = nullptr;
            if (none_resident_) {
                propagation_memory_position_input_ = input_tensor(ctx, "tracker_memory_position", 64,
                    spatial_count * 5184 + pointer_count * 4);
                propagation_rope_input_ = input_tensor(ctx, "tracker_rope", 2, 128, 5184);
                propagation_key_rope_input_ = input_tensor(ctx, "tracker_key_rope", 2, 128, spatial_count * 5184);
                memory_position = propagation_memory_position_input_;
                rope = propagation_rope_input_;
                key_rope = propagation_key_rope_input_;
                propagation_current_input_ = current;
                propagation_current_position_input_ = current_position;
            } else {
                const std::vector<TrackerPropagationInput> request{{nullptr, &selection}};
                memory_position = build_memory_positions(ctx, request, propagation_pointer_position_);
                rope = resident_rope_;
                key_rope = resident_rope_;
                if (spatial_count > 1) {
                    auto* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 128, spatial_count * 5184);
                    key_rope = ggml_repeat(ctx, resident_rope_, target);
                }
            }
            conditioned_ = sam3_build_mem_attn_graph(ctx, model_.definition.weights, current, current_position,
                memory, memory_position, rope, key_rope, pointer_count * 4);
            graph.output(conditioned_);
            propagation_memory_ = memory;
            graph_build_ms_ += elapsed_ms(graph_start);
        }
        const auto required = propagation_graph_.required_workspace_bytes();
        if (required > budget || resident_buffer_bytes() > budget - std::min(required, budget)) {
            if (high_resident_) {
                demote_high_residency();
                return condition_one(image, selection, payload, budget_override);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, budget);
                demote_none_residency();
                return condition_one(image, selection, payload, budget_override);
            }
            ++budget_failures_;
            throw budget_error("condition", required, budget);
        }
        std::size_t actual_arena = required;
        if (!allocate_under_budget(graph, required, budget, &actual_arena)) {
            if (high_resident_) {
                demote_high_residency();
                return condition_one(image, selection, payload, budget_override);
            }
            if (!none_resident_) {
                remember_none_resident_policy(key, budget);
                demote_none_residency();
                return condition_one(image, selection, payload, budget_override);
            }
            ++budget_failures_;
            throw budget_error("condition", actual_arena, budget);
        }
        upload_tracked(propagation_memory_, payload.memory);
        if (none_resident_) {
            upload_tracked(propagation_current_input_, image.tracker[2]);
            upload_tracked(propagation_current_position_input_, image.position);
            const std::vector<TrackerPropagationInput> request{{nullptr, &selection}};
            upload_tracked(propagation_memory_position_input_, memory_position_payload(request, payload));
            upload_tracked(propagation_rope_input_, rope_);
            upload_tracked(propagation_key_rope_input_, tiled_rope(spatial_count));
        } else if (pointer_count) {
            upload_tracked(propagation_pointer_position_, payload.pointer_position);
        }
        if (!none_resident_) ensure_core_resident_uploaded(image);
        graph.compute();
        return download_tracked(conditioned_);
    }

    TrackerGraphShape condition_shape(int spatial, int pointers, const MemorySelection& selection) const {
        const auto tokens = spatial * 5184 + pointers * 4;
        auto key = shape(TrackerGraphShape::Stage::condition, spatial, pointers, 1);
        if (!none_resident_)
            for (const auto& selected : selection.spatial) key.memory_position_order.push_back(selected.position);
        std::vector<std::array<std::int64_t, 4>> layouts;
        if (none_resident_) {
            layouts = {{256, 5184, 1, 1}, {256, 5184, 1, 1}, {64, tokens, 1, 1},
                {64, tokens, 1, 1}, {2, 128, 5184, 1},
                {2, 128, static_cast<std::int64_t>(spatial) * 5184, 1}};
        } else {
            const auto layout_count = pointers ? 7u : 6u;
            layouts = {{256, 5184, 1, 1}, {256, 5184, 1, 1}, {64, tokens, 1, 1},
                {2, 128, 5184, 1}, {2, 128, static_cast<std::int64_t>(spatial) * 5184, 1},
                {64, 5184, 7, 1}};
            if (pointers) layouts.insert(layouts.begin() + 3,
                std::array<std::int64_t, 4>{64, pointers * 4, 1, 1});
            if (layouts.size() != layout_count) throw std::runtime_error("tracker condition layout key is inconsistent");
        }
        for (const auto& dims : layouts)
            tracker_shape_tensor(key, {dims[0], dims[1], dims[2], dims[3]});
        return key;
    }

    TrackerGraphShape decoder_shape(bool seed, int batch) const {
        auto key = shape(TrackerGraphShape::Stage::decoder, 0, 0, batch, seed);
        tracker_shape_tensor(key, {256, 72, 72, batch});
        tracker_shape_tensor(key, {256, 72, 72, 1});
        tracker_shape_tensor(key, {256, 2, batch, 1});
        tracker_shape_tensor(key, {256, 288, 288, 1});
        tracker_shape_tensor(key, {256, 144, 144, 1});
        if (seed) tracker_shape_tensor(key, {1152, 1152, 1, 1});
        return key;
    }

    TrackerGraphShape memory_shape(bool present, bool unrounded) const {
        auto key = shape(TrackerGraphShape::Stage::memory_encoder, 0, 0, 1, false, present, unrounded);
        tracker_shape_tensor(key, {1152, 1152, 1, 1});
        tracker_shape_tensor(key, {256, 72, 72, 1});
        return key;
    }

    TrackerPrediction finish_prediction(const std::vector<float>& conditioned,
            std::vector<float> masks, std::vector<float> iou, std::vector<float> object,
            std::vector<float> projected, const std::vector<float>* seed_mask) const {
        TrackerPrediction result;
        result.conditioned = conditioned;
        result.decoder_masks = std::move(masks);
        result.decoder_iou = std::move(iou);
        result.object_logit = object.front(); result.decoder_object_logit = result.object_logit;
        int best = 1;
        for (int i = 2; i < 4; ++i) if (result.decoder_iou[i] > result.decoder_iou[best]) best = i;
        int selected = best, pointer_token = best;
        if (seed_mask) {
            std::size_t intersection = 0, total = 0;
            for (int i = 0; i < 288 * 288; ++i) {
                intersection += result.decoder_masks[i] > 0.05f;
                total += result.decoder_masks[i] > -0.05f;
            }
            if (!total || static_cast<float>(intersection) / total >= 0.98f) selected = 0;
            pointer_token = 0;
        }
        result.iou = result.decoder_iou[selected];
        result.mask_index = selected; result.pointer_index = pointer_token;
        constexpr std::size_t mask_size = 288u * 288;
        result.mask.assign(result.decoder_masks.begin() + selected * mask_size,
                           result.decoder_masks.begin() + (selected + 1) * mask_size);
        result.pointer.assign(projected.begin() + pointer_token * 256,
                              projected.begin() + (pointer_token + 1) * 256);
        if (result.object_logit <= 0) {
            std::fill(result.mask.begin(), result.mask.end(), -1024.0f);
            result.pointer = no_object_;
        }
        return result;
    }

    ModelState& model_;
    RuntimeStats& stats_;
    std::shared_ptr<GraphWorkspace> workspace_;
    GraphDiagnostics probe_diagnostics_{};
    ContextPtr resident_context_;
    BufferPtr resident_buffer_;
    ContextPtr high_context_;
    BufferPtr high_buffer_;
    TrackerResidentUploadState resident_uploads_;
    std::array<ggml_tensor*, 3> resident_tracker_{};
    ggml_tensor* resident_position_ = nullptr;
    ggml_tensor* resident_memory_position_bank_ = nullptr;
    ggml_tensor* resident_dense_position_ = nullptr;
    ggml_tensor* resident_rope_ = nullptr;
    ggml_tensor* resident_no_point_ = nullptr;
    const ImageFeatures* active_image_ = nullptr;

    TrackerGraphSlot pointer_graph_, propagation_graph_, decoder_graph_, memory_graph_;
    ggml_tensor *pointer_input_ = nullptr, *pointer_output_ = nullptr;
    ggml_tensor *propagation_memory_ = nullptr, *propagation_pointer_position_ = nullptr;
    ggml_tensor *propagation_current_input_ = nullptr, *propagation_current_position_input_ = nullptr;
    ggml_tensor *propagation_memory_position_input_ = nullptr, *propagation_rope_input_ = nullptr;
    ggml_tensor *propagation_key_rope_input_ = nullptr, *propagation_dense_position_input_ = nullptr;
    ggml_tensor *propagation_sparse_input_ = nullptr;
    ggml_tensor *propagation_first_input_ = nullptr, *propagation_second_input_ = nullptr;
    ggml_tensor *conditioned_ = nullptr, *propagation_masks_ = nullptr, *propagation_iou_ = nullptr;
    ggml_tensor *propagation_object_ = nullptr, *propagation_pointer_ = nullptr;
    DecoderInputs decoder_inputs_{};
    ggml_tensor *decoder_masks_ = nullptr, *decoder_iou_ = nullptr, *decoder_object_ = nullptr, *decoder_pointer_ = nullptr;
    ggml_tensor *memory_mask_input_ = nullptr, *memory_pixels_input_ = nullptr, *memory_output_ = nullptr;

    std::vector<float> no_object_, no_point_sparse_, memory_tpos_, memory_position_bank_;
    std::vector<float> dense_position_, rope_;
    int budget_spatial_ = -1, budget_pointers_ = -1;
    int policy_spatial_ = -1, policy_pointers_ = -1, max_fused_batch_ = sam3_tracker_graph_batch_limit;
    std::size_t policy_resident_bytes_ = 0;
    bool fusion_disabled_ = false;
    bool high_resident_ = true, high_residency_allowed_ = true, none_resident_ = false;
    std::optional<NoneResidentPolicy> none_resident_policy_;
    std::optional<TrackerNoneSerialPolicy> none_serial_policy_;
    std::size_t serial_budget_ = 0, serial_required_budget_ = 0;
    std::size_t batch_splits_ = 0, serial_fallbacks_ = 0;
    std::size_t frame_serial_budget_ = 0, frame_serial_required_budget_ = 0;
    std::size_t seed_memory_budget_ = 0, seed_memory_required_budget_ = 0;
    std::size_t resident_peak_bytes_ = 0, resident_demotions_ = 0, none_resident_fallbacks_ = 0;
    std::size_t budget_failures_ = 0;
    std::size_t memory_payload_peak_bytes_ = 0;
    double graph_build_ms_ = 0, upload_ms_ = 0, download_ms_ = 0;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP
