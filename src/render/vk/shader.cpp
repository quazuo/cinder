#include "shader.hpp"

#include <fstream>
#include <functional>

#include "descriptor.hpp"
#include "SPIRV-Reflect/spirv_reflect.h"

// using BindingDescMap = std::map<uint32_t, std::map<uint32_t, vk::DescriptorSetLayoutBinding> >;
//
// [[nodiscard]]
// static BindingDescMap extract_binding_descs(const SpvReflectShaderModule &module, const vk::ShaderStageFlags stage_flags) {
//     BindingDescMap binding_descs;
//
//     for (const auto &set: enumerate_spv_objects<SpvReflectDescriptorSet>(&module, spvReflectEnumerateDescriptorSets)) {
//         binding_descs[set->set] = {};
//
//         for (uint32_t binding_idx = 0; binding_idx < set->binding_count; binding_idx++) {
//             const auto binding = set->bindings[binding_idx];
//
//             const vk::DescriptorSetLayoutBinding binding_desc {
//                 .binding = binding->binding,
//                 .descriptorType = static_cast<vk::DescriptorType>(binding->
//                     descriptor_type),
//                 .descriptorCount = binding->count,
//                 .stageFlags = stage_flags,
//             };
//
//             binding_descs.at(set->set).emplace(binding->binding, binding_desc);
//         }
//     }
//
//     return binding_descs;
// }
//
// static BindingDescMap merge_binding_descs(BindingDescMap desc_map_a, const BindingDescMap &desc_map_b) {
//     for (const auto &[set_idx, bindings]: desc_map_b) {
//         if (!desc_map_a.contains(set_idx)) {
//             desc_map_a[set_idx] = bindings;
//             continue;
//         }
//
//         for (const auto &[binding, desc]: bindings) {
//             auto &merged_bindings = desc_map_a.at(set_idx);
//
//             if (merged_bindings.contains(binding)) {
//                 auto &existing_binding = merged_bindings[binding];
//
//                 if (existing_binding.descriptorType != desc.descriptorType) {
//                     throw std::runtime_error("descriptor type mismatch in merge_binding_descs()");
//                 }
//
//                 if (existing_binding.descriptorCount != desc.descriptorCount) {
//                     throw std::runtime_error("descriptor count mismatch in merge_binding_descs()");
//                 }
//
//                 existing_binding.stageFlags |= desc.stageFlags;
//             } else {
//                 merged_bindings[binding] = desc;
//             }
//         }
//     }
//
//     return desc_map_a;
// }
//
// const vector<vk::raii::DescriptorSetLayout>&
// zrx::SpirvShaderPair::get_descriptor_set_layouts(const RendererContext &ctx) {
//     if (!cached_descriptor_set_layouts.empty()) {
//         return cached_descriptor_set_layouts;
//     }
//
//     const auto vert_file_buffer = read_file(vert_path);
//     const auto frag_file_buffer = read_file(frag_path);
//
//     SpvReflectShaderModule vert_module, frag_module;
//     check_spv_result(spvReflectCreateShaderModule(vert_file_buffer.size(), vert_file_buffer.data(), &vert_module));
//     check_spv_result(spvReflectCreateShaderModule(frag_file_buffer.size(), frag_file_buffer.data(), &frag_module));
//
//     auto vert_binding_descs         = extract_binding_descs(vert_module, vk::ShaderStageFlagBits::eVertex);
//     auto frag_binding_descs         = extract_binding_descs(frag_module, vk::ShaderStageFlagBits::eFragment);
//     const auto merged_binding_descs = merge_binding_descs(vert_binding_descs, frag_binding_descs);
//
//     const uint32_t max_set_idx = merged_binding_descs.rbegin()->first;
//     cached_descriptor_set_layouts.resize(max_set_idx);
//
//     for (const auto &[set_idx, binding_map]: merged_binding_descs) {
//         vector<vk::DescriptorSetLayoutBinding> bindings;
//
//         for (const auto &[_, binding_desc]: binding_map) {
//             bindings.push_back(binding_desc);
//         }
//
//         const vk::DescriptorSetLayoutCreateInfo set_layout_info{
//             .bindingCount = static_cast<uint32_t>(bindings.size()),
//             .pBindings = bindings.data(),
//         };
//
//         cached_descriptor_set_layouts[set_idx] = {*ctx.device, set_layout_info};
//     }
//
//     spvReflectDestroyShaderModule(&vert_module);
//     spvReflectDestroyShaderModule(&frag_module);
//
//     return cached_descriptor_set_layouts;
// }
