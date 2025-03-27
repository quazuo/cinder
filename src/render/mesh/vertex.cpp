#include "vertex.hpp"

namespace zrx {
vector<vk::VertexInputBindingDescription> ModelVertex::get_binding_descriptions() {
    return {
        {
            .binding = 0u,
            .stride = static_cast<uint32_t>(sizeof(ModelVertex)),
            .inputRate = vk::VertexInputRate::eVertex
        },
        {
            .binding = 1u,
            .stride = static_cast<uint32_t>(sizeof(glm::mat4)),
            .inputRate = vk::VertexInputRate::eInstance
        }
    };
}

vector<vk::VertexInputAttributeDescription> ModelVertex::get_attribute_descriptions() {
    return {
        {
            .location = 0U,
            .binding = 0U,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ModelVertex, pos)),
        },
        {
            .location = 1U,
            .binding = 0U,
            .format = vk::Format::eR32G32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ModelVertex, tex_coord)),
        },
        {
            .location = 2U,
            .binding = 0U,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ModelVertex, normal)),
        },
        {
            .location = 3U,
            .binding = 0U,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ModelVertex, tangent)),
        },
        {
            .location = 4U,
            .binding = 0U,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ModelVertex, bitangent)),
        },
        {
            .location = 5U,
            .binding = 1U,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = 0U,
        },
        {
            .location = 6U,
            .binding = 1U,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = static_cast<uint32_t>(sizeof(glm::vec4)),
        },
        {
            .location = 7U,
            .binding = 1U,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = static_cast<uint32_t>(2 * sizeof(glm::vec4)),
        },
        {
            .location = 8U,
            .binding = 1U,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = static_cast<uint32_t>(3 * sizeof(glm::vec4)),
        },
    };
}

vector<vk::VertexInputBindingDescription> SkyboxVertex::get_binding_descriptions() {
    return {
        {
            .binding = static_cast<uint32_t>(0U),
            .stride = static_cast<uint32_t>(sizeof(SkyboxVertex)),
            .inputRate = vk::VertexInputRate::eVertex
        }
    };
}

vector<vk::VertexInputAttributeDescription> SkyboxVertex::get_attribute_descriptions() {
    return {
        {
            .location = 0U,
            .binding = 0U,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(SkyboxVertex, pos)),
        },
    };
}

vector<vk::VertexInputBindingDescription> ScreenSpaceQuadVertex::get_binding_descriptions() {
    return {
        {
            .binding = static_cast<uint32_t>(0U),
            .stride = static_cast<uint32_t>(sizeof(SkyboxVertex)),
            .inputRate = vk::VertexInputRate::eVertex
        }
    };
}

vector<vk::VertexInputAttributeDescription> ScreenSpaceQuadVertex::get_attribute_descriptions() {
    return {
        {
            .location = 0U,
            .binding = 0U,
            .format = vk::Format::eR32G32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ScreenSpaceQuadVertex, pos)),
        },
        {
            .location = 1U,
            .binding = 0U,
            .format = vk::Format::eR32G32Sfloat,
            .offset = static_cast<uint32_t>(offsetof(ScreenSpaceQuadVertex, tex_coord)),
        },
    };
}
} // zrx
