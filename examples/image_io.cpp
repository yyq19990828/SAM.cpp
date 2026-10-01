#include "image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace sam_example {

Image read_image(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.string().c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!pixels) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("Cannot decode image '" + path.string() + "': " +
                                 (reason ? reason : "unknown image error"));
    }
    if (width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / 3 /
                                           static_cast<std::size_t>(height)) {
        throw std::runtime_error("Decoded image dimensions overflow");
    }
    const std::size_t size = static_cast<std::size_t>(width) * height * 3;
    return {width, height, {pixels.get(), pixels.get() + size}};
}

void write_mask_png(const std::filesystem::path& path, int width, int height,
                    const std::vector<std::uint8_t>& mask) {
    if (width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() /
                                           static_cast<std::size_t>(height) ||
        mask.size() != static_cast<std::size_t>(width) * height) {
        throw std::runtime_error("Invalid mask dimensions");
    }
    std::vector<std::uint8_t> pixels(mask.size());
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] > 1) {
            throw std::runtime_error("Mask values must be binary");
        }
        pixels[i] = mask[i] ? 255 : 0;
    }
    if (!stbi_write_png(path.string().c_str(), width, height, 1, pixels.data(), width)) {
        throw std::runtime_error("Cannot write mask '" + path.string() + "'");
    }
}

} // namespace sam_example
