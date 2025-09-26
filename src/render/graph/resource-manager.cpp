module;

module Cinder.Render.Graph;

import glm;
import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

namespace zrx {
namespace glsl {
#include "src/render/glsl_to_cpp.inl"
#include "shaders/utils/material.glsl"
}

ResourceManager::ResourceManager(const RendererContext& ctx, const uint32_t max_bindless_handles) : renderer_ctx(ctx) {
    for (uint32_t i = 0; i < max_bindless_handles; i++) {
        free_bindless_handles.push(i);
    }

    for (uint32_t i = 0; i < MATERIAL_MAX_COUNT; i++) {
        free_model_mat_handles.push(i);
    }
}
} // zrx
