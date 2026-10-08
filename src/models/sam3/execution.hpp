#ifndef SAM_CPP_SRC_MODELS_SAM3_EXECUTION_HPP
#define SAM_CPP_SRC_MODELS_SAM3_EXECUTION_HPP

#include "architecture.hpp"
#include "vision.hpp"
#include "text_encoder.hpp"
#include "prompt_encoder.hpp"
#include "fusion_encoder.hpp"
#include "detector.hpp"
#include "mask_decoder.hpp"
#include "video/preprocessing.hpp"
#include "runtime/ggml.hpp"
#include "sam/types.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

struct ImageFeatures {
    std::vector<float> preprocessed;
    std::array<HostTensor, 3> vision;
    std::array<std::vector<float>, 3> tracker;
    std::vector<float> position, geometry;
    int width = 0, height = 0;
};

struct Prediction {
    std::vector<float> text, fusion, boxes, class_logits, mask_logits;
    float presence_logit = 0.0f;
};

struct PreparedPrompt {
    std::vector<float> text, tokens, attention_bias, validity;
};

struct DetectorOutput {
    std::vector<float> boxes, class_logits, query_features;
    float presence_logit = 0.0f;
};

inline double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The concrete SAM 3 image implementation keeps independently callable stages.
struct ModelDefinition {
    sam3_model weights;

    ImageFeatures encode_image(GgmlRuntime& runtime, std::vector<float> pixels,
                               int width, int height, RuntimeStats& stats, bool video = false) const {
        const auto start = std::chrono::steady_clock::now();
        const auto& hp = weights.hparams;
        if (video && runtime.feature_cache_type() != GGML_TYPE_F32)
            throw std::invalid_argument("experimental image feature caches do not apply to video");
        ImageFeatures features;
        features.width = width;
        features.height = height;
        features.preprocessed = std::move(pixels);
        {
            GraphExecution execution(runtime, 16384, stats);
            auto* ctx = execution.context();
            auto* input = input_tensor(ctx, "input_image", hp.img_size, hp.img_size, 3);
            const auto columns_type = runtime.convolution_columns_type();
            auto* vit = sam3_build_vit_graph(ctx, input, weights, columns_type);
            ggml_tensor* neck[4] = {};
            sam3_build_neck_graph(ctx, vit, weights.neck_det, neck, columns_type);
            ggml_tensor* cached_neck[3] = {};
            for (int i = 0; i < 3; ++i) {
                // The low-resolution feature drives fusion/detection. Q8_0 is
                // limited to the two mask-detail levels by the validated recipe.
                const auto type = i == 2 && runtime.feature_cache_type() == GGML_TYPE_Q8_0
                    ? GGML_TYPE_F32 : runtime.feature_cache_type();
                cached_neck[i] = type == GGML_TYPE_F32 ? neck[i] : ggml_cast(ctx, neck[i], type);
                execution.output(cached_neck[i]);
            }
            ggml_tensor* tracker_neck[4] = {};
            if (video) {
                if (!weights.neck_trk.scales[0].deconv1_w)
                    throw std::runtime_error("video encoding requires full tracker weights");
                sam3_build_neck_graph(ctx, vit, weights.neck_trk, tracker_neck, columns_type);
                for (int i = 0; i < 3; ++i) execution.output(tracker_neck[i]);
            }
            execution.allocate();
            upload(input, features.preprocessed, stats);
            execution.compute();
            for (int i = 0; i < 3; ++i) features.vision[i] = HostTensor::download(cached_neck[i], stats);
            if (video) for (int i = 0; i < 3; ++i) {
                features.tracker[i] = download(tracker_neck[i], stats);
                round_bf16_storage(features.tracker[i]);
            }
        }
        const int h = hp.n_img_embd(), d = hp.neck_dim;
        features.position = sam3_sinusoidal_pe_2d(h, h, d);
        {
            GraphExecution execution(runtime, 4096, stats);
            auto* ctx = execution.context();
            auto image = cached_input(ctx, "geometry_image", features.vision[2], d, h * h);
            auto* position = input_tensor(ctx, "geometry_position", d, h * h);
            auto output = sam3_build_geom_enc_graph(ctx, weights, image.values, position);
            execution.output(output.geo_feats);
            execution.allocate();
            features.vision[2].upload_to(image.stored, stats);
            upload(position, features.position, stats);
            execution.compute();
            features.geometry = download(output.geo_feats, stats);
        }
        ++stats.vision_encodes;
        stats.image_ms = elapsed_ms(start);
        return features;
    }

    std::vector<float> encode_text(GgmlRuntime& runtime, const std::vector<std::int32_t>& ids,
                                   RuntimeStats& stats) const {
        const auto start = std::chrono::steady_clock::now();
        const int length = weights.hparams.text_ctx_len;
        GraphExecution execution(runtime, 16384, stats);
        auto* ctx = execution.context();
        auto* input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length);
        ggml_set_input(input);
        auto* output = sam3_build_text_encoder_graph(ctx, input, weights);
        execution.output(output);
        execution.allocate();
        ggml_backend_tensor_set(input, ids.data(), 0, ids.size() * sizeof(std::int32_t));
        stats.host_upload_bytes += ids.size() * sizeof(std::int32_t);
        std::vector<ggml_fp16_t> causal(static_cast<std::size_t>(length) * length);
        sam3_fill_causal_mask(causal.data(), length);
        auto* mask = ggml_get_tensor(ctx, "causal_mask");
        if (!mask || !mask->buffer) throw std::runtime_error("SAM text causal mask is unallocated");
        ggml_backend_tensor_set(mask, causal.data(), 0, causal.size() * sizeof(ggml_fp16_t));
        stats.host_upload_bytes += causal.size() * sizeof(ggml_fp16_t);
        execution.compute();
        auto values = download(output, stats);
        ++stats.text_encodes;
        stats.text_ms = elapsed_ms(start);
        return values;
    }

    PreparedPrompt prepare_prompt(const ImageFeatures& image,
                                  const std::vector<std::int32_t>& ids,
                                  std::vector<float> text) const {
        const int length = weights.hparams.text_ctx_len;
        const int prompt_length = length + 1;
        PreparedPrompt prompt;
        prompt.text = std::move(text);
        prompt.tokens = prompt.text;
        prompt.tokens.insert(prompt.tokens.end(), image.geometry.begin(), image.geometry.end());
        prompt.attention_bias.assign(prompt_length, 0.0f);
        prompt.validity.assign(prompt_length, 0.0f);
        int valid_count = 1; // Empty geometry CLS is a valid prompt token.
        for (int i = 0; i < length; ++i) if (ids[i] != 0) ++valid_count;
        const float scale = static_cast<float>(prompt_length) / valid_count;
        for (int i = 0; i < length; ++i) {
            prompt.attention_bias[i] = ids[i] ? 0.0f : -1.0e9f;
            prompt.validity[i] = ids[i] ? scale : 0.0f;
        }
        prompt.validity[length] = scale;
        return prompt;
    }

    std::vector<float> fuse(GgmlRuntime& runtime, const ImageFeatures& image,
                            const PreparedPrompt& prompt, RuntimeStats& stats) const {
        const auto& hp = weights.hparams;
        const int d = hp.neck_dim, h = hp.n_img_embd(), prompt_length = hp.text_ctx_len + 1;
        GraphExecution execution(runtime, 16384, stats);
        auto* ctx = execution.context();
        auto input = cached_input(ctx, "fusion_image", image.vision[2], d, h * h);
        auto* position = input_tensor(ctx, "fusion_position", d, h * h);
        auto* tokens = input_tensor(ctx, "fusion_prompt", d, prompt_length);
        auto* bias = input_tensor(ctx, "fusion_bias", prompt_length);
        auto* output = sam3_build_fenc_graph(ctx, weights, input.values, tokens, position, bias);
        execution.output(output);
        execution.allocate();
        image.vision[2].upload_to(input.stored, stats);
        upload(position, image.position, stats);
        upload(tokens, prompt.tokens, stats);
        upload(bias, prompt.attention_bias, stats);
        execution.compute();
        return download(output, stats);
    }

    DetectorOutput detect(GgmlRuntime& runtime, const ImageFeatures& image,
                          const PreparedPrompt& prompt, const std::vector<float>& fusion,
                          RuntimeStats& stats) const {
        const auto& hp = weights.hparams;
        const int d = hp.neck_dim, h = hp.n_img_embd(), prompt_length = hp.text_ctx_len + 1;
        DetectorOutput detection;
        GraphExecution execution(runtime, 65536, stats);
        auto* ctx = execution.context();
        auto* encoded = input_tensor(ctx, "decoder_encoded", d, h * h);
        auto* position = input_tensor(ctx, "decoder_position", d, h * h);
        auto* tokens = input_tensor(ctx, "decoder_prompt", d, prompt_length);
        auto* sine = input_tensor(ctx, "sine_dim_t", 1, 64);
        auto* rpb = input_tensor(ctx, "rpb_coords", h);
        auto* bias = input_tensor(ctx, "text_attn_bias", prompt_length);
        auto* valid = input_tensor(ctx, "text_valid_mask", prompt_length);
        auto output = sam3_build_ddec_graph(ctx, weights, encoded, position, tokens, sine, rpb, bias, valid);
        execution.output(output.class_scores);
        execution.output(output.pred_boxes);
        execution.output(output.presence_score);
        execution.output(output.queries);
        execution.allocate();
        upload(encoded, fusion, stats);
        upload(position, image.position, stats);
        upload(tokens, prompt.tokens, stats);
        std::vector<float> sine_values(64), coordinates(h);
        for (int i = 0; i < 64; ++i)
            sine_values[i] = 2.0f * 3.14159265358979323846f / std::pow(10000.0f, 2.0f * i / 128.0f);
        for (int i = 0; i < h; ++i) coordinates[i] = static_cast<float>(i) / h;
        upload(sine, sine_values, stats);
        upload(rpb, coordinates, stats);
        upload(bias, prompt.attention_bias, stats);
        upload(valid, prompt.validity, stats);
        initialize_detector_zero_inputs(ctx, stats);
        execution.compute();
        detection.class_logits = download(output.class_scores, stats);
        detection.boxes = download(output.pred_boxes, stats);
        detection.presence_logit = download(output.presence_score, stats).front();
        detection.query_features = download(output.queries, stats);
        return detection;
    }

    std::vector<float> decode_masks(GgmlRuntime& runtime, const ImageFeatures& image,
                                   const PreparedPrompt& prompt, const std::vector<float>& fusion,
                                   const DetectorOutput& detection, RuntimeStats& stats) const {
        const auto& hp = weights.hparams;
        const int d = hp.neck_dim, h = hp.n_img_embd();
        const int queries = hp.ddec_num_queries, prompt_length = hp.text_ctx_len + 1;
        GraphExecution execution(runtime, 32768, stats);
        auto* ctx = execution.context();
        auto* encoded = input_tensor(ctx, "segmentation_encoded", d, h * h);
        ggml_tensor* neck[3] = {};
        CachedInput cached_neck[3] = {};
        for (int i = 0; i < 3; ++i) {
            const int size = h * (4 >> i);
            const std::string name = "segmentation_vision_" + std::to_string(i);
            cached_neck[i] = cached_input(ctx, name.c_str(), image.vision[i], d, size, size);
            neck[i] = cached_neck[i].values;
        }
        auto* objects = input_tensor(ctx, "segmentation_queries", d, queries);
        auto* tokens = input_tensor(ctx, "segmentation_prompt", d, prompt_length);
        auto* bias = input_tensor(ctx, "segmentation_bias", prompt_length);
        auto* output = sam3_build_seg_head_graph(ctx, weights, encoded, neck, objects, tokens, bias,
                                                  runtime.convolution_columns_type());
        execution.output(output);
        execution.allocate();
        upload(encoded, fusion, stats);
        // Lowest-resolution FPN is replaced by fusion output inside the mask
        // head and has no allocation; only upload inputs that are used.
        for (int i = 0; i < 3; ++i) if (cached_neck[i].stored->buffer)
            image.vision[i].upload_to(cached_neck[i].stored, stats);
        std::vector<float> object_features(detection.query_features.begin() + d, detection.query_features.end());
        upload(objects, object_features, stats);
        upload(tokens, prompt.tokens, stats);
        upload(bias, prompt.attention_bias, stats);
        execution.compute();
        return download(output, stats);
    }

    // Shared builders keep fusion and detector intermediates on the device.
    // Final diagnostic tensors remain owned host snapshots, as in staged execution.
    Prediction predict_joint(GgmlRuntime& runtime, const ImageFeatures& image,
                             const std::vector<std::int32_t>& ids, std::vector<float> text,
                             RuntimeStats& stats) const {
        const auto start = std::chrono::steady_clock::now();
        const auto& hp = weights.hparams;
        const int d = hp.neck_dim, h = hp.n_img_embd(), prompt_length = hp.text_ctx_len + 1;
        auto prompt = prepare_prompt(image, ids, std::move(text));
        GraphExecution execution(runtime, 65536, stats);
        auto* ctx = execution.context();
        auto input = cached_input(ctx, "joint_image", image.vision[2], d, h * h);
        auto* position = input_tensor(ctx, "joint_position", d, h * h);
        auto* tokens = input_tensor(ctx, "joint_prompt", d, prompt_length);
        auto* bias = input_tensor(ctx, "joint_bias", prompt_length);
        auto* valid = input_tensor(ctx, "joint_valid", prompt_length);
        auto* sine = input_tensor(ctx, "sine_dim_t", 1, 64);
        auto* rpb = input_tensor(ctx, "rpb_coords", h);
        auto* fusion = sam3_build_fenc_graph(ctx, weights, input.values, tokens, position, bias);
        auto detection = sam3_build_ddec_graph(ctx, weights, fusion, position, tokens, sine, rpb, bias, valid);
        ggml_tensor* neck[3] = {};
        CachedInput cached_neck[3] = {};
        for (int i = 0; i < 3; ++i) {
            const int size = h * (4 >> i);
            cached_neck[i] = cached_input(ctx, ("joint_vision_" + std::to_string(i)).c_str(), image.vision[i], d, size, size);
            neck[i] = cached_neck[i].values;
        }
        auto* objects = ggml_cont(ctx, ggml_view_2d(ctx, detection.queries, d, hp.ddec_num_queries,
            detection.queries->nb[1], d * sizeof(float)));
        auto* masks = sam3_build_seg_head_graph(ctx, weights, fusion, neck, objects, tokens, bias,
                                                 runtime.convolution_columns_type());
        execution.output(fusion);
        execution.output(detection.class_scores);
        execution.output(detection.pred_boxes);
        execution.output(detection.presence_score);
        execution.output(masks);
        execution.allocate();
        image.vision[2].upload_to(input.stored, stats);
        upload(position, image.position, stats);
        upload(tokens, prompt.tokens, stats);
        upload(bias, prompt.attention_bias, stats);
        upload(valid, prompt.validity, stats);
        for (int i = 0; i < 3; ++i) if (cached_neck[i].stored->buffer)
            image.vision[i].upload_to(cached_neck[i].stored, stats);
        std::vector<float> sine_values(64), coordinates(h);
        for (int i = 0; i < 64; ++i)
            sine_values[i] = 2.0f * 3.14159265358979323846f / std::pow(10000.0f, 2.0f * i / 128.0f);
        for (int i = 0; i < h; ++i) coordinates[i] = static_cast<float>(i) / h;
        upload(sine, sine_values, stats);
        upload(rpb, coordinates, stats);
        initialize_detector_zero_inputs(ctx, stats);
        execution.compute();
        Prediction prediction;
        prediction.text = std::move(prompt.text);
        prediction.fusion = download(fusion, stats);
        prediction.boxes = download(detection.pred_boxes, stats);
        prediction.class_logits = download(detection.class_scores, stats);
        prediction.presence_logit = download(detection.presence_score, stats).front();
        prediction.mask_logits = download(masks, stats);
        ++stats.inferences;
        stats.inference_ms = elapsed_ms(start);
        return prediction;
    }

    Prediction predict(GgmlRuntime& runtime, const ImageFeatures& image,
                       const std::vector<std::int32_t>& ids, std::vector<float> text,
                       RuntimeStats& stats) const {
        if (runtime.combine_graph_stages())
            return predict_joint(runtime, image, ids, std::move(text), stats);
        const auto start = std::chrono::steady_clock::now();
        auto prompt = prepare_prompt(image, ids, std::move(text));
        Prediction prediction;
        prediction.text = std::move(prompt.text);
        prediction.fusion = fuse(runtime, image, prompt, stats);
        auto detection = detect(runtime, image, prompt, prediction.fusion, stats);
        prediction.boxes = std::move(detection.boxes);
        prediction.class_logits = std::move(detection.class_logits);
        prediction.presence_logit = detection.presence_logit;
        prediction.mask_logits = decode_masks(runtime, image, prompt, prediction.fusion, detection, stats);
        ++stats.inferences;
        stats.inference_ms = elapsed_ms(start);
        return prediction;
    }
};

inline TensorData feature_tensor(const std::vector<float>& values, int channels, int height, int width) {
    TensorData tensor{{1, channels, height, width}, std::vector<float>(values.size()), "NCHW"};
    for (int pixel = 0; pixel < height * width; ++pixel)
        for (int channel = 0; channel < channels; ++channel)
            tensor.values[static_cast<std::size_t>(channel) * height * width + pixel] =
                values[static_cast<std::size_t>(pixel) * channels + channel];
    return tensor;
}

inline TensorData feature_tensor(const HostTensor& cache, int channels, int height, int width) {
    cache.validate_shape(channels, width, height, 1);
    return feature_tensor(cache.as_f32(), channels, height, width);
}

} // namespace sam::internal::sam3

#endif // SAM_CPP_SRC_MODELS_SAM3_EXECUTION_HPP
