#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP

#include "../state.hpp"
#include "mask_decoder.hpp"
#include "memory_encoder.hpp"
#include "memory_attention.hpp"
#include "memory_selection.hpp"
#include "mask_ops.hpp"
#include <array>
#include <map>

namespace sam::internal::sam3 {

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

class TrackerExecution {
public:
    TrackerExecution(ModelState& model, RuntimeStats& stats) : model_(model), stats_(stats) {
        const auto& weights = model_.definition.weights;
        gaussian_ = read_weight(weights.sam_pe.pe_gaussian);
        no_point_ = read_weight(weights.sam_pe.not_a_point_embed);
        no_object_ = read_weight(weights.no_obj_ptr);
        memory_tpos_ = read_weight(weights.tensors.at("mem_enc.tpos_enc"));
        memory_position_ = sam3_sinusoidal_pe_2d(72, 72, 64);
        dense_position_.resize(256 * 72 * 72);
        rope_.resize(256 * 72 * 72);
        for (int y = 0; y < 72; ++y) for (int x = 0; x < 72; ++x) {
            const auto pixel = y * 72 + x;
            for (int k = 0; k < 128; ++k) {
                const float phase = ((2.0f * (x + 0.5f) / 72 - 1) * gaussian_[k] +
                                     (2.0f * (y + 0.5f) / 72 - 1) * gaussian_[128 + k]) * 6.283185307179586f;
                dense_position_[pixel * 256 + k] = std::sin(phase);
                dense_position_[pixel * 256 + 128 + k] = std::cos(phase);
                const float angle = (k < 64 ? x : y) / std::pow(10000.0f, 4.0f * (k % 64) / 256);
                rope_[pixel * 256 + k * 2] = std::cos(angle);
                rope_[pixel * 256 + k * 2 + 1] = std::sin(angle);
            }
        }
    }

    std::vector<float> condition(const ImageFeatures& image, const std::map<int, TrackerRecord>& records,
                                 const MemorySelection& selection, int frame_count) {
        const auto& weights = model_.definition.weights;
        if (selection.spatial.empty()) throw std::runtime_error("tracker has no conditioning memory");
        std::vector<float> memory, position, key_rope;
        for (const auto& selected : selection.spatial) {
            const auto& record = records.at(selected.frame);
            for (const auto value : record.memory) memory.push_back(ggml_bf16_to_fp32(value));
            for (std::size_t i = 0; i < memory_position_.size(); ++i)
                position.push_back(memory_position_[i] + memory_tpos_[(6 - selected.position) * 64 + i % 64]);
            key_rope.insert(key_rope.end(), rope_.begin(), rope_.end());
        }
        if (!selection.pointers.empty()) {
            GraphExecution graph(*model_.runtime, 256, stats_);
            auto* ctx = graph.context();
            auto* input = input_tensor(ctx, "pointer_temporal_input", 256, selection.pointers.size());
            auto* output = ggml_add(ctx, ggml_mul_mat(ctx, weights.obj_ptr_tpos_w, input), weights.obj_ptr_tpos_b);
            graph.output(output); graph.allocate();
            std::vector<float> sine;
            const int maximum = std::min(frame_count, 16);
            for (const auto& selected : selection.pointers) {
                const float relative = static_cast<float>(selected.position) / (maximum - 1);
                for (int phase = 0; phase < 2; ++phase) for (int k = 0; k < 128; ++k) {
                    const float angle = relative / std::pow(10000.0f, 2.0f * (k / 2) / 128);
                    sine.push_back(phase ? std::cos(angle) : std::sin(angle));
                }
            }
            upload(input, sine, stats_); graph.compute();
            const auto projected = download(output, stats_);
            for (std::size_t i = 0; i < selection.pointers.size(); ++i) {
                const auto& pointer = records.at(selection.pointers[i].frame).pointer;
                memory.insert(memory.end(), pointer.begin(), pointer.end());
                for (int token = 0; token < 4; ++token)
                    position.insert(position.end(), projected.begin() + i * 64, projected.begin() + (i + 1) * 64);
            }
        }
        GraphExecution graph(*model_.runtime, 16384, stats_);
        auto* ctx = graph.context();
        auto* current = input_tensor(ctx, "tracker_current", 256, 5184);
        auto* current_position = input_tensor(ctx, "tracker_position", 256, 5184);
        auto* prompt = input_tensor(ctx, "tracker_memory", 64, memory.size() / 64);
        auto* prompt_position = input_tensor(ctx, "tracker_memory_position", 64, memory.size() / 64);
        auto* frequencies = input_tensor(ctx, "tracker_rope", 2, 128, 5184);
        auto* key_frequencies = input_tensor(ctx, "tracker_key_rope", 2, 128, key_rope.size() / 256);
        auto* output = sam3_build_mem_attn_graph(ctx, weights, current, current_position, prompt,
            prompt_position, frequencies, key_frequencies, static_cast<int>(selection.pointers.size() * 4));
        graph.output(output); graph.allocate();
        upload(current, image.tracker[2], stats_); upload(current_position, image.position, stats_);
        upload(prompt, memory, stats_); upload(prompt_position, position, stats_);
        upload(frequencies, rope_, stats_); upload(key_frequencies, key_rope, stats_);
        graph.compute();
        return download(output, stats_);
    }

    TrackerPrediction decode(const ImageFeatures& image, std::vector<float> conditioned,
                             const std::vector<float>* seed_mask = nullptr) {
        const auto& weights = model_.definition.weights;
        GraphExecution graph(*model_.runtime, 8192, stats_);
        auto* ctx = graph.context();
        auto* current = input_tensor(ctx, "sam_current", 256, 72, 72);
        auto* position = input_tensor(ctx, "sam_dense_position", 256, 72, 72);
        auto* sparse = input_tensor(ctx, "sam_sparse", 256, 2);
        auto* first = input_tensor(ctx, "sam_first_neck", 256, 288, 288);
        auto* second = input_tensor(ctx, "sam_second_neck", 256, 144, 144);
        ggml_tensor* seed = nullptr;
        ggml_tensor* dense = nullptr;
        if (seed_mask) {
            seed = input_tensor(ctx, "sam_seed_mask", 1152, 1152);
            dense = sam3_conv_2d(ctx, weights.tensors.at("trk_mask_ds.weight"), seed, 4, 4, 0, 0);
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
        for (auto* tensor : {output.masks, output.iou_pred, output.obj_score, pointers}) graph.output(tensor);
        graph.allocate();
        auto sparse_values = no_point_; sparse_values.insert(sparse_values.end(), no_point_.begin(), no_point_.end());
        upload(current, conditioned, stats_); upload(position, dense_position_, stats_); upload(sparse, sparse_values, stats_);
        upload(first, image.tracker[0], stats_); upload(second, image.tracker[1], stats_);
        if (seed) upload(seed, *seed_mask, stats_);
        graph.compute();
        TrackerPrediction result;
        result.conditioned = std::move(conditioned);
        result.decoder_masks = download(output.masks, stats_); result.decoder_iou = download(output.iou_pred, stats_);
        result.object_logit = download(output.obj_score, stats_).front();
        result.decoder_object_logit = result.object_logit;
        const auto projected = download(pointers, stats_);
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
        result.mask.assign(result.decoder_masks.begin() + selected * 288 * 288,
                           result.decoder_masks.begin() + (selected + 1) * 288 * 288);
        result.pointer.assign(projected.begin() + pointer_token * 256, projected.begin() + (pointer_token + 1) * 256);
        if (result.object_logit <= 0) {
            std::fill(result.mask.begin(), result.mask.end(), -1024.0f); result.pointer = no_object_;
        }
        return result;
    }

    std::vector<ggml_bf16_t> encode_memory(const ImageFeatures& image, const std::vector<float>& mask,
                                          bool present, std::vector<float>* unrounded = nullptr) {
        GraphExecution graph(*model_.runtime, 4096, stats_);
        auto* ctx = graph.context();
        auto* input = input_tensor(ctx, "memory_mask", 1152, 1152);
        auto* pixels = input_tensor(ctx, "memory_pixels", 256, 72, 72);
        auto* output = build_memory_encoder(ctx, model_.definition.weights, input, pixels, present);
        graph.output(output); graph.allocate();
        upload(input, mask, stats_); upload(pixels, image.tracker[2], stats_); graph.compute();
        auto values = download(output, stats_);
        if (unrounded) *unrounded = values;
        std::vector<ggml_bf16_t> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) result[i] = ggml_fp32_to_bf16(values[i]);
        return result;
    }

    const std::vector<float>& no_object_pointer() const { return no_object_; }

private:
    std::vector<float> read_weight(ggml_tensor* tensor) {
        if (!tensor || tensor->type != GGML_TYPE_F32) throw std::runtime_error("tracker constant must be F32");
        return download(tensor, stats_);
    }
    ModelState& model_;
    RuntimeStats& stats_;
    std::vector<float> gaussian_, no_point_, no_object_, memory_tpos_, memory_position_, dense_position_, rope_;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TRACKING_EXECUTION_HPP
