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
template<typename ResourceType>
struct DefaultDescriptorType {
};

template<>
struct DefaultDescriptorType<Texture> {
    static constexpr auto type = vk::DescriptorType::eCombinedImageSampler;
};

template<>
struct DefaultDescriptorType<Buffer> {
    static constexpr auto type = vk::DescriptorType::eUniformBuffer;
};

template<>
struct DefaultDescriptorType<BufferSlice> {
    static constexpr auto type = vk::DescriptorType::eUniformBuffer;
};

template<>
struct DefaultDescriptorType<AccelerationStructure> {
    static constexpr auto type = vk::DescriptorType::eAccelerationStructureKHR;
};

template<typename T>
struct is_valid_for_resource : std::disjunction<
            std::is_same<T, Texture>,
            std::is_same<T, Buffer>,
            std::is_same<T, BufferSlice>,
            std::is_same<T, AccelerationStructure> > {
};

template<typename T>
    requires is_valid_for_resource<T>::value
struct ResourcePack final {
    using ResourceSlot = std::optional<std::reference_wrapper<const T> >;

    vk::ShaderStageFlags scope;
    vk::DescriptorType type;
    uint32_t descriptorCount;
    std::vector<ResourceSlot> resources;

    ResourcePack(const uint32_t descriptorCount, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type = DefaultDescriptorType<T>::type)
        : scope(scope), type(type), descriptorCount(descriptorCount), resources(descriptorCount) {
    }

    ResourcePack(const T &resource, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type = DefaultDescriptorType<T>::type)
        : scope(scope), type(type), descriptorCount(1), resources({resource}) {
    }

    ResourcePack(const std::initializer_list<ResourceSlot> resources, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type = DefaultDescriptorType<T>::type)
        : scope(scope), type(type), descriptorCount(resources.size()), resources(resources) {
    }
};

template<int N, typename... Ts>
using NthTypeOf = std::tuple_element_t<N, std::tuple<Ts...> >;

/**
 * Convenience wrapper around Vulkan descriptor sets, mainly to pair them together with related layouts,
 * as well as provide an easy way to update them in a performant way.
 */
template<typename... Ts>
    requires std::conjunction_v<is_valid_for_resource<Ts>...>
class DescriptorSet {
    std::reference_wrapper<const RendererContext> ctx;
    std::tuple<ResourcePack<Ts>...> packs;
    shared_ptr<vk::raii::DescriptorSetLayout> layout;
    unique_ptr<vk::raii::DescriptorSet> set;

    using WriteInfo = std::variant<
        vk::DescriptorBufferInfo,
        vk::DescriptorImageInfo,
        vk::WriteDescriptorSetAccelerationStructureKHR>;

    struct DescriptorUpdate {
        uint32_t binding{};
        uint32_t arrayElement{};
        vk::DescriptorType type{};
        WriteInfo info;
    };

    std::vector<DescriptorUpdate> queuedUpdates;

public:
    explicit DescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool, ResourcePack<Ts>... elems)
        : ctx(ctx), packs(elems...) {
        createLayout();
        createSet(pool);
        doFullUpdate();
    }

private:
    explicit DescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                           const shared_ptr<vk::raii::DescriptorSetLayout> &layout, ResourcePack<Ts>... elems)
        : ctx(ctx), packs(elems...), layout(layout) {
        createSet(pool);
        doFullUpdate();
    }

public:
    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    [[nodiscard]] const vk::raii::DescriptorSetLayout &getLayout() const { return *layout; }

    /**
     * Queues an update to a given binding in this descriptor set.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    template<uint32_t Binding, typename ResourceType>
        requires std::is_same_v<ResourceType, NthTypeOf<Binding, Ts...> >
    DescriptorSet &queueUpdate(const ResourceType &resource, const uint32_t arrayElement = 0) {
        const auto &pack = std::get<Binding>(packs);

        DescriptorUpdate update{
            .binding = Binding,
            .arrayElement = arrayElement,
            .type = pack.type,
            .info = makeWriteInfo<Binding, ResourceType>(resource)
        };

        queuedUpdates.push_back(update);

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
     * Immediately updates a single binding in this descriptor set.
     */
    template<uint32_t Binding, typename ResourceType>
        requires std::is_same_v<ResourceType, NthTypeOf<Binding, Ts...> >
    DescriptorSet &updateBinding(const ResourceType &resource, const uint32_t arrayElement = 0) {
        const auto &pack = std::get<Binding>(packs);

        if (arrayElement >= pack.descriptorCount) {
            throw std::invalid_argument("descriptor set array element out of bounds");
        }

        vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = Binding,
            .dstArrayElement = arrayElement,
            .descriptorCount = 1,
            .descriptorType = pack.type,
        };

        const WriteInfo info = makeWriteInfo<Binding, ResourceType>(resource);

        if constexpr (std::is_same_v<ResourceType, Buffer> || std::is_same_v<ResourceType, BufferSlice>) {
            write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(info);
        } else if constexpr (std::is_same_v<ResourceType, Texture>) {
            write.pImageInfo = &std::get<vk::DescriptorImageInfo>(info);
        } else if constexpr (std::is_same_v<ResourceType, AccelerationStructure>) {
            write.pNext = &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(info);
        } else if constexpr (true) {
            throw std::runtime_error("unimplemented resource type handling");
        }

        ctx.get().device->updateDescriptorSets(write, nullptr);

        return *this;
    }

    template<uint32_t Binding, typename ResourceType>
    WriteInfo makeWriteInfo(const ResourceType &resource) {
        const auto &pack = std::get<Binding>(packs);

        if constexpr (std::is_same_v<ResourceType, Buffer> || std::is_same_v<ResourceType, BufferSlice>) {
            return vk::DescriptorBufferInfo{
                .buffer = *resource,
                .range = resource.getSize(),
            };
        } else if constexpr (std::is_same_v<ResourceType, Texture>) {
            const auto imageLayout = pack.type == vk::DescriptorType::eCombinedImageSampler
                                         ? vk::ImageLayout::eShaderReadOnlyOptimal
                                         : vk::ImageLayout::eGeneral;

            return vk::DescriptorImageInfo{
                .sampler = *resource.getSampler(),
                .imageView = **resource.getImage().getView(ctx),
                .imageLayout = imageLayout,
            };
        } else if constexpr (std::is_same_v<ResourceType, AccelerationStructure>) {
            return vk::WriteDescriptorSetAccelerationStructureKHR{
                .accelerationStructureCount = 1u,
                .pAccelerationStructures = &**resource, // todo - dangling pointer?
            };
        }

        throw std::runtime_error("unimplemented resource type handling");
    }

private:
    void createLayout() {
        auto bindings = std::apply([](auto &&... elems) {
            return std::vector<vk::DescriptorSetLayoutBinding>{
                makeBinding(std::forward<decltype(elems)>(elems))...
            };
        }, packs);

        for (uint32_t i = 0; i < bindings.size(); i++) {
            bindings[i].binding = i;
        }

        const vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        layout = make_shared<vk::raii::DescriptorSetLayout>(*ctx.get().device, setLayoutInfo);
    }

    template<typename ResourceType>
    [[nodiscard]] static vk::DescriptorSetLayoutBinding makeBinding(const ResourcePack<ResourceType> &pack) {
        return vk::DescriptorSetLayoutBinding{
            .descriptorType = pack.type,
            .descriptorCount = pack.descriptorCount,
            .stageFlags = pack.scope,
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

    template<size_t I = 0>
    void updateSlot() {
        auto pack = std::get<I>(packs);

        for (uint32_t i = 0; i < pack.descriptorCount; i++) {
            const auto &resource = pack.resources[i];

            if (resource.has_value()) {
                queueUpdate<I>(resource->get(), i);
            }
        }

        if constexpr (I + 1 < sizeof...(Ts)) {
            updateSlot<I + 1>();
        }
    }

    void doFullUpdate() {
        updateSlot();
        commitUpdates();
    }
};

template<typename... Ts>
class DescriptorSets {
    std::vector<DescriptorSet<Ts...> > sets;

public:
    DescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool) {
        // todo - implement so that the sets share layouts
    }

    [[nodiscard]] DescriptorSet<Ts...> &operator[](const size_t index) const { return sets[index]; }
};

static void usage_example() {
    const RendererContext ctx;
    const unique_ptr<vk::raii::DescriptorPool> pool;

    const unique_ptr<Texture> tex;
    const unique_ptr<Buffer> buf;

    const ResourcePack<Texture> desc1{*tex, vk::ShaderStageFlagBits::eFragment};
    const ResourcePack<Buffer> desc2{*buf, vk::ShaderStageFlagBits::eVertex};

    DescriptorSet set{ctx, *pool, desc1, desc2};

    set.updateBinding<0>(*tex);
    set.updateBinding<1>(*buf);

    // compile error
    // set.updateBinding<1>(*tex);

    // DescriptorSets<Texture, Buffer> sets{
    //     ctx,
    //     *pool,
    //     {desc1, desc2},
    //     {desc1, desc2}
    // };
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
