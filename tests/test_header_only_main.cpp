#include <sam/sam.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

bool missing_model_from_other_translation_unit(const char* path);

int main() {
    try {
        const auto path = std::filesystem::temp_directory_path() /
                          "sam-header-only-intentionally-missing-checkpoint.gguf";
        if (std::filesystem::exists(path)) throw std::runtime_error("Test checkpoint path already exists");
        bool rejected = false;
        try {
            (void) sam::Model::load(path.string(), {sam::Backend::Cpu, 1});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected || !missing_model_from_other_translation_unit(path.string().c_str())) {
            throw std::runtime_error("Missing model was accepted");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "header-only check: " << error.what() << '\n';
        return 1;
    }
}
