#ifndef SAM_CPP_SRC_MODELS_SAM3_VIDEO_WORKSPACE_POLICY_HPP
#define SAM_CPP_SRC_MODELS_SAM3_VIDEO_WORKSPACE_POLICY_HPP

#include "../state.hpp"
#include "frame_storage.hpp"
#include "graph_cache.hpp"
#include "mask_decoder.hpp"
#include "memory_attention.hpp"
#include "memory_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace sam::internal::sam3 {

// Splits oversized propagation batches without losing the caller's mutable
// limit: recursive fallbacks may lower it and the next chunk must observe it.
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

// Sticky decision for the serial propagation path. It binds the exact runtime
// identity, backend, profiles, memory shape and budget it was measured with.
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

// Owns the serial workspace budget, its probe measurements, batch splitting
// decisions and the sticky none-resident/none-serial policies. The coordinator
// keeps the build order; this class only answers budget and policy questions.
class TrackerWorkspacePolicy {
public:
    struct WorkspaceProbe {
        std::size_t required = 0, actual = 0;
    };

    TrackerWorkspacePolicy(ModelState& model, RuntimeStats& stats, GraphWorkspace& workspace,
                           TrackerFrameStorage& frames, TrackerGraphCache& graphs)
        : model_(model), stats_(stats), workspace_(workspace), frames_(frames), graphs_(graphs) {}

    bool none_serial_policy_matches(int spatial, int pointers, std::size_t budget) const {
        return tracker_none_serial_policy_matches(none_serial_policy_, model_.runtime.get(), model_.runtime->backend(),
            model_.model_info.precision + ":" + model_.model_info.storage_profile,
            model_.runtime->arithmetic_profile(), spatial, pointers, budget);
    }

    void remember_none_serial_policy(int spatial, int pointers, std::size_t budget) {
        none_serial_policy_ = TrackerNoneSerialPolicy{model_.runtime.get(), model_.runtime->backend(),
            model_.model_info.precision + ":" + model_.model_info.storage_profile,
            model_.runtime->arithmetic_profile(), spatial, pointers, budget};
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

    void select_resident_mode(std::size_t budget) {
        if (frames_.select_resident_mode(budget) != TrackerFrameStorage::ResidentDecision::demote_none) return;
        demote_none_residency();
    }

    void demote_none_residency() noexcept {
        if (frames_.none_resident()) return;
        // Flush scheduler references before releasing either resident context.
        // Cached graph metadata remains intact and gets a distinct none-resident
        // key; a later frame reuses the original resident tensor metadata.
        workspace_.release_storage();
        frames_.force_none_residency();
        policy_resident_bytes_ = 0;
        max_fused_batch_ = sam3_tracker_graph_batch_limit;
        fusion_disabled_ = false;
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
                << frames_.resident_bytes() << " arena=" << arena << " budget=" << budget;
        return std::runtime_error(message.str());
    }

    bool allocate_under_budget(GraphExecution& graph, std::size_t required,
                               std::size_t budget, std::size_t* actual_bytes = nullptr) {
        const auto resident = frames_.resident_bytes();
        if (actual_bytes) *actual_bytes = required;
        if (resident > budget || required > budget - resident) return false;
        const auto arena_limit = budget - resident;
        // Different backend arenas can grow in opposite directions between
        // stages. Compare their combined retained/required sizes before binding.
        graph.allocate(arena_limit);
        const auto actual = workspace_.allocated_bytes();
        if (actual_bytes) *actual_bytes = actual;
        if (actual > arena_limit) {
            workspace_.release_storage();
            return false;
        }
        return true;
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

    bool fusion_disabled() const { return fusion_disabled_; }
    void set_fusion_disabled(bool value) { fusion_disabled_ = value; }
    int max_fused_batch() const { return max_fused_batch_; }
    void reset_fused_batch() { max_fused_batch_ = sam3_tracker_graph_batch_limit; }
    void clamp_fused_batch(int batch) {
        max_fused_batch_ = std::min(max_fused_batch_, std::max(1, batch / 2));
    }
    template<class Callback>
    void for_each_batch(int total, Callback&& callback) {
        sam3_for_each_tracker_chunk(total, max_fused_batch_, std::forward<Callback>(callback));
    }

    // Per-frame serial budget: the frame-level limit is reset at each frame,
    // while the measured seed/memory fallback stays cached.
    void clear_frame_budget() noexcept {
        frame_serial_budget_ = 0;
        frame_serial_required_budget_ = 0;
    }
    void note_frame_budget(std::size_t budget, std::size_t required) {
        frame_serial_budget_ = std::max(frame_serial_budget_, budget);
        frame_serial_required_budget_ = std::max(frame_serial_required_budget_, required);
    }
    std::size_t frame_serial_budget() const { return frame_serial_budget_; }
    std::size_t effective_serial_budget() {
        return frame_serial_budget_ ? frame_serial_budget_ : seed_memory_workspace_budget();
    }

    void count_batch_split() { ++batch_splits_; }
    void count_serial_fallback() { ++serial_fallbacks_; }
    void count_budget_failure() { ++budget_failures_; }
    std::size_t batch_splits() const { return batch_splits_; }
    std::size_t serial_fallbacks() const { return serial_fallbacks_; }
    std::size_t budget_failures() const { return budget_failures_; }

    void set_policy_resident_bytes(std::size_t bytes) { policy_resident_bytes_ = bytes; }
    std::size_t serial_budget_bytes() const {
        return frame_serial_budget_ ? frame_serial_budget_ : seed_memory_budget_;
    }
    std::size_t serial_required_bytes() const {
        return frame_serial_budget_ ? frame_serial_required_budget_ : seed_memory_required_budget_;
    }
    std::size_t serial_required_budget() const { return serial_required_budget_; }
    const GraphDiagnostics& probe_diagnostics() const { return probe_diagnostics_; }

private:
    struct NoneResidentPolicy { TrackerGraphShape low_shape; std::size_t budget = 0; };

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

    void release_frame_storage_for_probe() noexcept {
        workspace_.release_storage();
        frames_.release_storage();
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
        graphs_.add_auxiliary_build_ms(elapsed_ms(build_start));
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
                                                  memory, position, rope, key_rope, pointers * 4, model_.runtime->attention_policy());
        graph.output(output);
        graphs_.add_auxiliary_build_ms(elapsed_ms(build_start));
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
        graphs_.add_auxiliary_build_ms(elapsed_ms(build_start));
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
        graphs_.add_auxiliary_build_ms(elapsed_ms(build_start));
        return probe_actual_workspace(graph);
    }

    ModelState& model_;
    RuntimeStats& stats_;
    GraphWorkspace& workspace_;
    TrackerFrameStorage& frames_;
    TrackerGraphCache& graphs_;
    GraphDiagnostics probe_diagnostics_{};
    std::optional<NoneResidentPolicy> none_resident_policy_;
    std::optional<TrackerNoneSerialPolicy> none_serial_policy_;
    int budget_spatial_ = -1, budget_pointers_ = -1;
    int policy_spatial_ = -1, policy_pointers_ = -1;
    std::size_t policy_resident_bytes_ = 0;
    int max_fused_batch_ = sam3_tracker_graph_batch_limit;
    bool fusion_disabled_ = false;
    std::size_t serial_budget_ = 0, serial_required_budget_ = 0;
    std::size_t seed_memory_budget_ = 0, seed_memory_required_budget_ = 0;
    std::size_t frame_serial_budget_ = 0, frame_serial_required_budget_ = 0;
    std::size_t batch_splits_ = 0, serial_fallbacks_ = 0;
    std::size_t budget_failures_ = 0;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_VIDEO_WORKSPACE_POLICY_HPP
