module;

module Cinder.Render.Graph;

import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

namespace zrx {
ResourceManager::ResourceManager(const RendererContext& ctx, const uint32_t max_bindless_handles) : renderer_ctx(ctx) {
    for (uint32_t i = 0; i < max_bindless_handles; i++) {
        free_bindless_handles.push(i);
    }
}
}
