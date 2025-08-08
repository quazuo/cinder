module;

export module Cinder.Render.Mesh:Vertex;

import std;
import glm;
import vulkan_hpp;

import Cinder.Globals;

export namespace zrx {
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

    static auto get_binding_descriptions() -> vector<vk::VertexInputBindingDescription>;

    static auto get_attribute_descriptions() -> vector<vk::VertexInputAttributeDescription>;
};

struct SkyboxVertex {
    glm::vec3 pos;

    static auto get_binding_descriptions() -> vector<vk::VertexInputBindingDescription>;

    static auto get_attribute_descriptions() -> vector<vk::VertexInputAttributeDescription>;
};

// vertices of the skybox cube.
// might change this to be generated in a more smart way... but it's good enough for now
const vector<SkyboxVertex> skybox_vertices = {
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

    static auto get_binding_descriptions() -> vector<vk::VertexInputBindingDescription>;

    static auto get_attribute_descriptions() -> vector<vk::VertexInputAttributeDescription>;
};

const vector<ScreenSpaceQuadVertex> screen_space_quad_vertices = {
    {{-1, -1}, {0, 1}},
    {{1, -1}, {1, 1}},
    {{1, 1}, {1, 0}},

    {{-1, -1}, {0, 1}},
    {{1, 1}, {1, 0}},
    {{-1, 1}, {0, 0}},
};
} // zrx

// as mentioned above, this is implemented to allow using `Vertex` as a key in an `unordered_map`.
// todo - fix it
template<>
struct std::hash<zrx::ModelVertex> {
    auto operator()(zrx::ModelVertex const &vertex) const noexcept -> size_t {
        return 0;
        // return (hash<glm::vec3>()(vertex.pos) >> 1) ^
        //        (hash<glm::vec2>()(vertex.tex_coord) << 1) ^
        //        (hash<glm::vec3>()(vertex.normal) << 1) ^
        //        (hash<glm::vec3>()(vertex.tangent) << 1) ^
        //        (hash<glm::vec3>()(vertex.bitangent) << 1);
    }
};
