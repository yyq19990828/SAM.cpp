#include <sam/sam.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <type_traits>

// Public ownership contract: models are copyable/movable value handles over
// shared state; sessions are unique and neither copyable nor movable.
static_assert(std::is_copy_constructible<sam::Model>::value, "Model must stay copyable");
static_assert(std::is_copy_assignable<sam::Model>::value, "Model must stay copy-assignable");
static_assert(std::is_move_constructible<sam::Model>::value, "Model must stay movable");
static_assert(std::is_move_assignable<sam::Model>::value, "Model must stay move-assignable");
static_assert(!std::is_copy_constructible<sam::ImageSession>::value, "ImageSession must not be copyable");
static_assert(!std::is_copy_assignable<sam::ImageSession>::value, "ImageSession must not be copy-assignable");
static_assert(!std::is_move_constructible<sam::ImageSession>::value, "ImageSession must not become movable");
static_assert(!std::is_move_assignable<sam::ImageSession>::value, "ImageSession must not become move-assignable");
static_assert(!std::is_copy_constructible<sam::VideoSession>::value, "VideoSession must not be copyable");
static_assert(!std::is_copy_assignable<sam::VideoSession>::value, "VideoSession must not be copy-assignable");
static_assert(!std::is_move_constructible<sam::VideoSession>::value, "VideoSession must not become movable");
static_assert(!std::is_move_assignable<sam::VideoSession>::value, "VideoSession must not become move-assignable");

bool missing_model_from_other_translation_unit(const char* path);

int main() {
    try {
        const auto path = std::filesystem::temp_directory_path() /
                          "sam-two-tu-intentionally-missing-checkpoint.gguf";
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
        std::cerr << "two-TU check: " << error.what() << '\n';
        return 1;
    }
}
