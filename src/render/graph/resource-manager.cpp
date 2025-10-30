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

void ResourceManager::add_model_materials(const ResourceHandle handle, const Model& model) {
    vector<ModelMaterialHandle> material_handles;

    for (size_t i = 0; i < model.get_materials().size(); i++) {
        const Material& material = model.get_materials()[i];
        const ModelMaterialHandle new_mat_handle = get_new_handle(free_model_mat_handles);

        material_handles.emplace_back(new_mat_handle);

        const auto base_color_handle = material.base_color ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;
        const auto normal_handle     = material.normal     ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;
        const auto orm_handle        = material.orm        ? get_new_handle(free_bindless_handles) : EMPTY_TEXTURE_BINDLESS_HANDLE;

        materials.emplace(new_mat_handle, MaterialTextureHandles {
            .base_color = base_color_handle,
            .normal = normal_handle,
            .orm = orm_handle,
        });
    }

    models_to_materials.emplace(handle, material_handles);
}

void ResourceManager::recreate(const ResourceHandle handle) {
    switch (handle_to_kind_mapping.at(handle)) {
        case ResourceKind::TEXTURE:
            textures.insert_or_assign(handle, texture_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::GRAPHICS_PIPELINE:
            graphics_pipelines.insert_or_assign(handle, graphics_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        case ResourceKind::COMPUTE_PIPELINE:
            compute_pipelines.insert_or_assign(handle, compute_pipeline_builders.at(handle).create(renderer_ctx.get()));
            break;
        default:
            throw std::runtime_error("unsupported resource type in ResourceManager::recreate");
    }
}

void ResourceManager::reload_all_pipelines() {
    for (const auto& [handle, builder] : graphics_pipeline_builders) {
        graphics_pipelines.insert_or_assign(handle, builder.create(renderer_ctx.get()));
    }

    for (const auto& [handle, builder] : compute_pipeline_builders) {
        compute_pipelines.insert_or_assign(handle, builder.create(renderer_ctx.get()));
    }
}

auto ResourceManager::get_name(const ResourceHandle handle) const -> const std::string& {
    if (handle == FINAL_IMAGE_HANDLE) return FINAL_IMAGE_NAME;
    return resource_names.at(handle);
}

auto ResourceManager::get_model_mat_tex_handles(const ResourceHandle handle) const -> vector<MaterialTextureHandles> {
    vector<MaterialTextureHandles> result;

    for (auto& material_handle : models_to_materials.at(handle)) {
        result.emplace_back(materials.at(material_handle));
    }

    return result;
}
} // zrx
