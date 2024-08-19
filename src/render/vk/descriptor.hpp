#pragma once

#include <utility>
#include <variant>

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "accel-struct.hpp"
#include "ctx.hpp"

namespace zrx {
/**
 * Builder class streamlining descriptor set layout creation.
 *
 * Methods which add bindings are order-dependent and the order in which they are called
 * defines which binding is used for a given resource, i.e. first call to `addBinding` will
 * use binding 0, second will use binding 1, and so on.
 */
class DescriptorLayoutBuilder {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

public:
    DescriptorLayoutBuilder &addBinding(vk::DescriptorType type, vk::ShaderStageFlags stages,
                                        uint32_t descriptorCount = 1);

    DescriptorLayoutBuilder &addRepeatedBindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages,
                                                 uint32_t descriptorCount = 1);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext &ctx);
};

struct Resource {
    vk::ShaderStageFlags scope;
    vk::DescriptorType type;
    uint32_t descriptorCount;

protected:
    // not instantiable as-is, only possible with derived classes
    explicit Resource(const vk::ShaderStageFlags scope, const vk::DescriptorType type,
                      const uint32_t descriptorCount = 1)
        : scope(scope), type(type), descriptorCount(descriptorCount) {
    }
};

struct TextureResource : Resource {
    std::vector<std::optional<std::reference_wrapper<const Texture> > > textures;

    explicit TextureResource(const Texture &texture, const vk::ShaderStageFlags scope,
                             const vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler)
        : Resource(scope, type, 1), textures({texture}) {
    }

    explicit TextureResource(
        const std::initializer_list<std::optional<std::reference_wrapper<const Texture> > > textures,
        const vk::ShaderStageFlags scope,
        const vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler)
        : Resource(scope, type, textures.size()), textures(textures) {
    }

    TextureResource(const uint32_t descriptorCount, const vk::ShaderStageFlags scope, const vk::DescriptorType type)
        : Resource(scope, type, descriptorCount), textures(descriptorCount) {
    }
};

struct BufferResource : Resource {
    std::reference_wrapper<const Buffer> buffer;

    BufferResource(const Buffer &buffer, const vk::ShaderStageFlags scope,
                   const vk::DescriptorType type = vk::DescriptorType::eUniformBuffer)
        : Resource(scope, type), buffer(buffer) {
    }
};

struct AccelStructureResource : Resource {
    std::reference_wrapper<const AccelerationStructure> accelStructure;

    AccelStructureResource(const AccelerationStructure &accelStructure, const vk::ShaderStageFlags scope)
        : Resource(scope, vk::DescriptorType::eAccelerationStructureKHR), accelStructure(accelStructure) {
    }
};

template<typename... Ts>
struct are_all_resources : std::conjunction<std::is_base_of<Resource, Ts>...> {
};

/**
 * Convenience wrapper around Vulkan descriptor sets, mainly to pair them together with related layouts,
 * as well as provide an easy way to update them in a performant way.
 */
template<typename... Ts>
    requires are_all_resources<Ts...>::value
class DescriptorSet : public std::tuple<Ts...> {
    std::reference_wrapper<const RendererContext> ctx;
    shared_ptr<vk::raii::DescriptorSetLayout> layout;
    unique_ptr<vk::raii::DescriptorSet> set;

    struct DescriptorUpdate {
        uint32_t binding{};
        uint32_t arrayElement{};
        vk::DescriptorType type{};
        std::variant<
            vk::DescriptorBufferInfo,
            vk::DescriptorImageInfo,
            vk::WriteDescriptorSetAccelerationStructureKHR> info;
    };

    std::vector<DescriptorUpdate> queuedUpdates;

public:
    // DescriptorSet(const RendererContext &ctx, decltype(layout) l, vk::raii::DescriptorSet &&s)
    //     : ctx(ctx), layout(std::move(l)), set(make_unique<vk::raii::DescriptorSet>(std::move(s))) {
    //     static_assert(are_all_resources<Ts...>());
    // }

    explicit DescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool, Ts... elems)
        : std::tuple<Ts...>(elems...), ctx(ctx) {
        createLayout();
        createSet(pool);
    }

    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    [[nodiscard]] const vk::raii::DescriptorSetLayout &getLayout() const { return *layout; }

    /**
     * Queues an update to a given binding in this descriptor set, referencing a buffer.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queueUpdate(const uint32_t binding, const Buffer &buffer, const vk::DescriptorType type,
                               const vk::DeviceSize size, const vk::DeviceSize offset = 0,
                               const uint32_t arrayElement                            = 0) {
        static_assert(std::is_same<BufferResource, Ts[binding]>());

        const vk::DescriptorBufferInfo bufferInfo{
            .buffer = *buffer,
            .offset = offset,
            .range = size,
        };

        queuedUpdates.emplace_back(DescriptorUpdate{
            .binding = binding,
            .arrayElement = arrayElement,
            .type = type,
            .info = bufferInfo,
        });

        return *this;
    }

    /**
     * Queues an update to a given binding in this descriptor set, referencing a texture.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queueUpdate(const uint32_t binding, const Texture &texture,
                               const vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler,
                               const uint32_t arrayElement   = 0) {
        const vk::DescriptorImageInfo imageInfo{
            .sampler = *texture.getSampler(),
            .imageView = **texture.getImage().getView(ctx),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        queuedUpdates.emplace_back(DescriptorUpdate{
            .binding = binding,
            .arrayElement = arrayElement,
            .type = type,
            .info = imageInfo,
        });

        return *this;
    }

    /**
     * Queues an update to a given binding in this descriptor set, referencing a raw storage image.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queueUpdate(const uint32_t binding, const vk::raii::ImageView &view,
                               const uint32_t arrayElement = 0) {
        const vk::DescriptorImageInfo imageInfo{
            .imageView = *view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        queuedUpdates.emplace_back(DescriptorUpdate{
            .binding = binding,
            .arrayElement = arrayElement,
            .type = vk::DescriptorType::eStorageImage,
            .info = imageInfo,
        });

        return *this;
    }

    /**
    * Queues an update to a given binding in this descriptor set, referencing an acceleration structure.
    * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
    */
    DescriptorSet &queueUpdate(const uint32_t binding, const AccelerationStructure &accel,
                               const uint32_t arrayElement = 0) {
        const vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{
            .accelerationStructureCount = 1u,
            .pAccelerationStructures = &**accel, // todo - dangling pointer?
        };

        queuedUpdates.emplace_back(DescriptorUpdate{
            .binding = binding,
            .arrayElement = arrayElement,
            .type = vk::DescriptorType::eAccelerationStructureKHR,
            .info = accelInfo,
        });

        return *this;
    }

    void commitUpdates() {
        std::vector<vk::WriteDescriptorSet> descriptorWrites;

        for (const auto &update: queuedUpdates) {
            vk::WriteDescriptorSet write{
                .dstSet = **set,
                .dstBinding = update.binding,
                .dstArrayElement = update.arrayElement,
                .descriptorCount = 1,
                .descriptorType = update.type,
            };

            if (std::holds_alternative<vk::DescriptorBufferInfo>(update.info)) {
                write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(update.info);
            } else if (std::holds_alternative<vk::DescriptorImageInfo>(update.info)) {
                write.pImageInfo = &std::get<vk::DescriptorImageInfo>(update.info);
            } else if (std::holds_alternative<vk::WriteDescriptorSetAccelerationStructureKHR>(update.info)) {
                write.pNext = &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(update.info);
            } else {
                throw std::runtime_error("unexpected variant in DescriptorSet::commitUpdates");
            }

            descriptorWrites.emplace_back(write);
        }

        ctx.get().device->updateDescriptorSets(descriptorWrites, nullptr);

        queuedUpdates.clear();
    }

    /**
     * Immediately updates a single binding in this descriptor set, referencing a buffer.
     */
    void updateBinding(const uint32_t binding, const Buffer &buffer, const vk::DescriptorType type,
                       const vk::DeviceSize size, const vk::DeviceSize offset = 0,
                       const uint32_t arrayElement                            = 0) const {
        const vk::DescriptorBufferInfo bufferInfo{
            .buffer = *buffer,
            .offset = offset,
            .range = size,
        };

        const vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = binding,
            .dstArrayElement = arrayElement,
            .descriptorCount = 1,
            .descriptorType = type,
            .pBufferInfo = &bufferInfo,
        };

        ctx.get().device->updateDescriptorSets(write, nullptr);
    }

    /**
     * Immediately updates a single binding in this descriptor set, referencing a texture.
     */
    void updateBinding(const uint32_t binding, const Texture &texture,
                       const vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler,
                       const uint32_t arrayElement   = 0) const {
        const vk::DescriptorImageInfo imageInfo{
            .sampler = *texture.getSampler(),
            .imageView = **texture.getImage().getView(ctx),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = binding,
            .dstArrayElement = arrayElement,
            .descriptorCount = 1,
            .descriptorType = type,
            .pImageInfo = &imageInfo,
        };

        ctx.get().device->updateDescriptorSets(write, nullptr);
    }

    /**
     * Immediately updates a single binding in this descriptor set, referencing a raw storage image.
     */
    void updateBinding(const uint32_t binding, const vk::raii::ImageView &view,
                       const uint32_t arrayElement = 0) const {
        const vk::DescriptorImageInfo imageInfo{
            .imageView = *view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = binding,
            .dstArrayElement = arrayElement,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &imageInfo,
        };

        ctx.get().device->updateDescriptorSets(write, nullptr);
    }

    /**
     * Immediately updates a single binding in this descriptor set, referencing an acceleration structure.
     */
    void updateBinding(const uint32_t binding, const AccelerationStructure &accel,
                       const uint32_t arrayElement = 0) const {
        const vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &**accel,
        };

        const vk::WriteDescriptorSet write{
            .pNext = &accelInfo,
            .dstSet = **set,
            .dstBinding = binding,
            .dstArrayElement = arrayElement,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        };

        ctx.get().device->updateDescriptorSets(write, nullptr);
    }

private:
    void createLayout() {
        auto bindings = std::apply([](auto &&... elems) {
            return std::vector<vk::DescriptorSetLayoutBinding>{
                makeBinding(std::forward<decltype(elems)>(elems))...
            };
        }, std::forward<std::tuple<Ts...> >(*this));

        for (uint32_t i = 0; i < bindings.size(); i++) {
            bindings[i].binding = i;
        }

        const vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        layout = make_shared<vk::raii::DescriptorSetLayout>(*ctx.get().device, setLayoutInfo);
    }

    [[nodiscard]] static vk::DescriptorSetLayoutBinding makeBinding(const Resource &res) {
        return vk::DescriptorSetLayoutBinding{
            .descriptorType = res.type,
            .descriptorCount = res.descriptorCount,
            .stageFlags = res.scope,
        };
    }

    void createSet(const vk::raii::DescriptorPool &pool) {
        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *pool,
            .descriptorSetCount = 1u,
            .pSetLayouts = &**layout,
        };

        std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.get().device->allocateDescriptorSets(allocInfo);

        set = make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[0]));
    }
};

template<typename... Ts>
class DescriptorSets : std::vector<DescriptorSet<Ts> > {
public:
    DescriptorSets(Ts... elems) {
    }
};

static void usage_example() {
    const RendererContext ctx;
    const unique_ptr<vk::raii::DescriptorPool> pool;

    const unique_ptr<Texture> tex;
    const unique_ptr<Buffer> buf;

    const TextureResource desc1{*tex, vk::ShaderStageFlagBits::eFragment};
    const BufferResource desc2{*buf, vk::ShaderStageFlagBits::eVertex};

    const DescriptorSet set{ctx, *pool, desc1, desc2};

    // może tak?
    // set[0].updateBinding(*tex);

    set.updateBinding(0, *tex);
    set.updateBinding(1, *buf, vk::DescriptorType::eUniformBuffer, 16);

    set.updateBinding(1, *tex);
}

namespace utils::desc {
    template<typename... Ts>
    [[nodiscard]] std::vector<DescriptorSet<Ts...> >
    createDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                         const shared_ptr<vk::raii::DescriptorSetLayout> &layout, const uint32_t count) {
        const std::vector setLayouts(count, **layout);

        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *pool,
            .descriptorSetCount = count,
            .pSetLayouts = setLayouts.data(),
        };

        std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

        std::vector<DescriptorSet<Ts...> > finalSets;

        for (size_t i = 0; i < count; i++) {
            finalSets.emplace_back(layout, std::move(descriptorSets[i]));
        }

        return finalSets;
    }

    template<typename... Ts>
    [[nodiscard]] std::vector<DescriptorSet<Ts...> >
    createDescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                        const shared_ptr<vk::raii::DescriptorSetLayout> &layout) {
        return createDescriptorSets<Ts>(ctx, pool, layout, 1);
    }
} // utils::desc
} // zrx
