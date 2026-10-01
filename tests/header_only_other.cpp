#include <sam/sam.hpp>

#include <stdexcept>

bool missing_model_from_other_translation_unit(const char* path) {
    try {
        (void) sam::Model::load(path, {sam::Backend::Cpu, 1});
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}
