#ifndef SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_IO_HPP
#define SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_IO_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sam_example {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

Image read_image(const std::filesystem::path& path);
void write_mask_png(const std::filesystem::path& path, int width, int height,
                    const std::vector<std::uint8_t>& mask);

} // namespace sam_example

#endif // SAM_CPP_SUPPORT_IMAGE_IO_IMAGE_IO_HPP
