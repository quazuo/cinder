module;

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

export module StbImage;

export {
    using ::stbi_write_png;
    using ::stbi_set_flip_vertically_on_load;
    using ::stbi_loadf;
    using ::stbi_load;
    using ::stbi_image_free;

    using ::STBI_grey;
    using ::STBI_rgb_alpha;
}
