module;

export module Cinder.Render.Mesh:Vertex;

import std;
import glm;
import vulkan;

import Cinder.Globals;

export namespace zrx {
struct ModelVertex {
    glm::vec3 pos;
    glm::vec2 tex_coord;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;

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
    {{-1, 1, -1}},
    {{-1, -1, -1}},
    {{1, -1, -1}},
    {{1, -1, -1}},
    {{1, 1, -1}},
    {{-1, 1, -1}},

    {{-1, -1, 1}},
    {{-1, -1, -1}},
    {{-1, 1, -1}},
    {{-1, 1, -1}},
    {{-1, 1, 1}},
    {{-1, -1, 1}},

    {{1, -1, -1}},
    {{1, -1, 1}},
    {{1, 1, 1}},
    {{1, 1, 1}},
    {{1, 1, -1}},
    {{1, -1, -1}},

    {{-1, -1, 1}},
    {{-1, 1, 1}},
    {{1, 1, 1}},
    {{1, 1, 1}},
    {{1, -1, 1}},
    {{-1, -1, 1}},

    {{-1, 1, -1}},
    {{1, 1, -1}},
    {{1, 1, 1}},
    {{1, 1, 1}},
    {{-1, 1, 1}},
    {{-1, 1, -1}},

    {{-1, -1, -1}},
    {{-1, -1, 1}},
    {{1, -1, -1}},
    {{1, -1, -1}},
    {{-1, -1, 1}},
    {{1, -1, 1}}
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
