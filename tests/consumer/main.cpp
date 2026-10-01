#include <sam/sam.hpp>

#include <iostream>
#include <stdexcept>

int main() {
    try {
        (void) sam::Model::load("sam-consumer-intentionally-missing-checkpoint.gguf", {sam::Backend::Cpu, 1});
        std::cerr << "Consumer unexpectedly accepted a missing checkpoint\n";
        return 1;
    } catch (const std::runtime_error&) {
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
