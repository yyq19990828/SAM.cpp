#include "execution_cost.hpp"
#include <models/sam3/model.hpp>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto options = sam_example::parse_options(argc, argv, false);
        if (options.help) {
            std::cout << "Usage: sam_execution_cost_probe --model FILE --image FILE --text PROMPT --output NEW_DIR\n"
                         "  --backend cpu [--threads N] [--cpu-compute f32|native-quantized]\n"
                         "One observed CPU image inference; serialized node timings are diagnostics, not a speed benchmark.\n";
            return 0;
        }
        if (options.backend.backend != sam::Backend::Cpu || options.backend.cuda_compute != sam::CudaComputeMode::F32)
            throw std::invalid_argument("execution cost probe requires explicit --backend cpu");
        sam_example::OutputDirectory destination(options.output, true);
        const auto image = sam_example::read_image(options.image);
        const auto load_start = sam_example::Clock::now();
        const auto state = sam::internal::sam3::load_state(options.model.string(), options.backend);
        const auto load_ms = sam_example::elapsed_ms(load_start);
        if (state->model_info.task != "text_image") throw std::invalid_argument("cost probe requires image weights");
        const auto observer = std::make_shared<sam_cost::Observer>();
        state->runtime->set_graph_observer(observer);
        sam::internal::sam3::ImageSession session(state);
        observer->stage = "image_encoding";
        session.set_image(sam_example::image_view(image));
        observer->stage = "text_and_prediction";
        const auto result = session.segment_text(options.text, options.score_threshold);
        state->runtime->set_graph_observer({});
        auto output = sam_example::output_file(options.output / "execution-cost.json");
        output << "{\"schema_version\":1,\"kind\":\"sam-execution-cost-run-v1\",\"complete\":true,"
                  "\"diagnostic_only\":true,\"performance_comparable\":false,\"workload\":\"one image and one prompt\","
                  "\"model\":" << sam_example::json_string(options.model.string())
               << ",\"model_sha256\":\"NOT_COLLECTED\",\"threads\":" << options.backend.threads
               << ",\"image\":" << sam_example::json_string(options.image.string())
               << ",\"prompt\":" << sam_example::json_string(options.text)
               << ",\"model_load_wall_ms\":" << load_ms << ",\"rss_peak_bytes\":" << sam_example::process_peak_rss_bytes()
               << ",\"execution_cost\":";
        observer->write(output);
        output << ",\"runtime\":";
        sam_example::write_runtime_stats(output, session.stats());
        sam_example::write_model_profile(output, state->model_info);
        sam_example::write_compute_policy(output, options.backend);
        output << ",\"detections\":" << result.detections.size() << "}\n";
        output.close();
        destination.complete();
        std::cout << "CPU diagnostic costs: " << options.output / "execution-cost.json" << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam_execution_cost_probe: " << error.what() << '\n';
        return 1;
    }
}
