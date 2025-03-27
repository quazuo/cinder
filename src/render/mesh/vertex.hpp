#pragma once

#include <concepts>

#include "src/render/libs.hpp"

namespace zrx {
template<typename T>
concept VertexLike = requires {
    { T::get_binding_descriptions() } -> std::same_as<vector<vk::VertexInputBindingDescription>>;
    { T::get_attribute_descriptions() } -> std::same_as<vector<vk::VertexInputAttributeDescription>>;
};

struct ModelVertex {
    glm::vec3 pos;
    glm::vec2 tex_coord;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    // this is implemented to allow using `Vertex` as a key in an `unordered_map`.
    bool operator==(const ModelVertex &other) const {
        return pos == other.pos
               && tex_coord == other.tex_coord
               && tangent == other.tangent
               && bitangent == other.bitangent;
    }

    static vector<vk::VertexInputBindingDescription> get_binding_descriptions();

    static vector<vk::VertexInputAttributeDescription> get_attribute_descriptions();
};

struct SkyboxVertex {
    glm::vec3 pos;

    static vector<vk::VertexInputBindingDescription> get_binding_descriptions();

    static vector<vk::VertexInputAttributeDescription> get_attribute_descriptions();
};

// vertices of the skybox cube.
// might change this to be generated in a more smart way... but it's good enough for now
static const vector<SkyboxVertex> skybox_vertices = {
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{-1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}}
};

struct ScreenSpaceQuadVertex {
    glm::vec2 pos;
    glm::vec2 tex_coord;

    static vector<vk::VertexInputBindingDescription> get_binding_descriptions();

    static vector<vk::VertexInputAttributeDescription> get_attribute_descriptions();
};

static const vector<ScreenSpaceQuadVertex> screen_space_quad_vertices = {
    {{-1, -1}, {0, 1}},
    {{1, -1}, {1, 1}},
    {{1, 1}, {1, 0}},

    {{-1, -1}, {0, 1}},
    {{1, 1}, {1, 0}},
    {{-1, 1}, {0, 0}},
};
} // zrx

// as mentioned above, this is implemented to allow using `Vertex` as a key in an `unordered_map`.
template<>
struct std::hash<zrx::ModelVertex> {
    size_t operator()(zrx::ModelVertex const &vertex) const noexcept {
        return (hash<glm::vec3>()(vertex.pos) >> 1) ^
               (hash<glm::vec2>()(vertex.tex_coord) << 1) ^
               (hash<glm::vec3>()(vertex.normal) << 1) ^
               (hash<glm::vec3>()(vertex.tangent) << 1) ^
               (hash<glm::vec3>()(vertex.bitangent) << 1);
    }
};
