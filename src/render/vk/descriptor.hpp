#pragma once

#include <utility>
#include <variant>
#include <algorithm>
#include <numeric>

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"
#include "src/render/graph.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "accel-struct.hpp"
#include "ctx.hpp"
#include "src/utils/logger.hpp"

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
    using ResourceSlot = std::optional<reference_wrapper<const T> >;

    vk::ShaderStageFlags scope;
    vk::DescriptorType type;
    vk::DescriptorBindingFlags flags;
    uint32_t descriptor_count;
    vector<ResourceSlot> resources;

    ResourcePack(const uint32_t descriptor_count, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type          = DefaultDescriptorType<T>::type,
                 const vk::DescriptorBindingFlags flags = {})
        : scope(scope), type(type), flags(flags), descriptor_count(descriptor_count), resources(descriptor_count) {
    }

    ResourcePack(const T &resource, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type          = DefaultDescriptorType<T>::type,
                 const vk::DescriptorBindingFlags flags = {})
        : scope(scope), type(type), flags(flags), descriptor_count(1), resources({resource}) {
    }

    ResourcePack(const std::initializer_list<ResourceSlot> resources, const vk::ShaderStageFlags scope,
                 const vk::DescriptorType type          = DefaultDescriptorType<T>::type,
                 const vk::DescriptorBindingFlags flags = {})
        : scope(scope), type(type), flags(flags), descriptor_count(resources.size()), resources(resources) {
    }
};

template<int N, typename... Ts>
using NthTypeOf = std::tuple_element_t<N, std::tuple<Ts...> >;

/**
 * Fixed (or in other words, static) version of `DescriptorSet`.
 * Enables compile-time checks to typing of its bindings and doesn't allow changes in layout.
 */
template<typename... Ts>
    requires std::conjunction_v<is_valid_for_resource<Ts>...>
class FixedDescriptorSet {
    reference_wrapper<const RendererContext> ctx;
    std::tuple<ResourcePack<Ts>...> packs;
    shared_ptr<vk::raii::DescriptorSetLayout> layout;
    unique_ptr<vk::raii::DescriptorSet> set;

    using WriteInfo = std::variant<
        vk::DescriptorBufferInfo,
        vk::DescriptorImageInfo,
        vk::WriteDescriptorSetAccelerationStructureKHR>;

    struct DescriptorUpdate {
        uint32_t binding{};
        uint32_t array_element{};
        vk::DescriptorType type{};
        WriteInfo info;
    };

    vector<DescriptorUpdate> queued_updates;

public:
    explicit FixedDescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                ResourcePack<Ts>... elems)
        : ctx(ctx), packs(elems...) {
        create_layout();
        create_set(pool);
        do_full_update();
    }

private:
    explicit FixedDescriptorSet(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                const shared_ptr<vk::raii::DescriptorSetLayout> &layout, ResourcePack<Ts>... elems)
        : ctx(ctx), packs(elems...), layout(layout) {
        create_set(pool);
        do_full_update();
    }

public:
    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    [[nodiscard]] const vk::raii::DescriptorSetLayout &get_layout() const { return *layout; }

    /**
     * Queues an update to a given binding in this descriptor set.
     * To actually push the update, `commit_updates` must be called after all desired updates are queued.
     */
    template<uint32_t Binding, typename ResourceType>
        requires std::is_same_v<ResourceType, NthTypeOf<Binding, Ts...> >
    FixedDescriptorSet &queue_update(const ResourceType &resource, const uint32_t array_element = 0) {
        const auto &pack = std::get<Binding>(packs);

        DescriptorUpdate update{
            .binding = Binding,
            .array_element = array_element,
            .type = pack.type,
            .info = make_write_info<Binding, ResourceType>(resource)
        };

        queued_updates.push_back(update);

        return *this;
    }

    void commit_updates() {
        vector<vk::WriteDescriptorSet> descriptor_writes;

        for (const auto &update: queued_updates) {
            vk::WriteDescriptorSet write{
                .dstSet = **set,
                .dstBinding = update.binding,
                .dstArrayElement = update.array_element,
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
                Logger::error("unexpected variant in DescriptorSet::commitUpdates");
            }

            descriptor_writes.emplace_back(write);
        }

        ctx.get().device->updateDescriptorSets(descriptor_writes, nullptr);

        queued_updates.clear();
    }

    /**
     * Immediately updates a single binding in this descriptor set.
     */
    template<uint32_t Binding, typename ResourceType>
        requires std::is_same_v<ResourceType, NthTypeOf<Binding, Ts...> >
    FixedDescriptorSet &update_binding(const ResourceType &resource, const uint32_t array_element = 0) {
        const auto &pack = std::get<Binding>(packs);

        if (array_element >= pack.descriptor_count) {
            Logger::error("descriptor set array element out of bounds");
        }

        vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = Binding,
            .dstArrayElement = array_element,
            .descriptorCount = 1,
            .descriptorType = pack.type,
        };

        const WriteInfo info = make_write_info<Binding, ResourceType>(resource);

        if constexpr (std::is_same_v<ResourceType, Buffer> || std::is_same_v<ResourceType, BufferSlice>) {
            write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(info);
        } else if constexpr (std::is_same_v<ResourceType, Texture>) {
            write.pImageInfo = &std::get<vk::DescriptorImageInfo>(info);
        } else if constexpr (std::is_same_v<ResourceType, AccelerationStructure>) {
            write.pNext = &std::get<vk::WriteDescriptorSetAccelerationStructureKHR>(info);
        } else if constexpr (true) {
            Logger::error("unimplemented resource type handling");
        }

        ctx.get().device->updateDescriptorSets(write, nullptr);

        return *this;
    }

    template<uint32_t Binding, typename ResourceType>
    [[nodiscard]] WriteInfo make_write_info(const ResourceType &resource) {
        const auto &pack = std::get<Binding>(packs);

        if constexpr (std::is_same_v<ResourceType, Buffer>) {
            return vk::DescriptorBufferInfo{
                .buffer = *resource,
                .range = resource.get_size(),
            };
        } else if constexpr (std::is_same_v<ResourceType, BufferSlice>) {
            return vk::DescriptorBufferInfo{
                .buffer = **resource,
                .offset = resource.offset,
                .range = resource.size,
            };
        } else if constexpr (std::is_same_v<ResourceType, Texture>) {
            const auto image_layout = pack.type == vk::DescriptorType::eCombinedImageSampler
                                          ? vk::ImageLayout::eShaderReadOnlyOptimal
                                          : vk::ImageLayout::eGeneral;

            return vk::DescriptorImageInfo{
                .sampler = *resource.get_sampler(),
                .imageView = **resource.get_image().get_view(ctx),
                .imageLayout = image_layout,
            };
        } else if constexpr (std::is_same_v<ResourceType, AccelerationStructure>) {
            return vk::WriteDescriptorSetAccelerationStructureKHR{
                .accelerationStructureCount = 1u,
                .pAccelerationStructures = &**resource, // todo - dangling pointer?
            };
        }

        Logger::error("unimplemented resource type handling");
    }

private:
    void create_layout() {
        auto bindings = std::apply([](auto &&... elems) {
            return vector<vk::DescriptorSetLayoutBinding>{
                make_binding(std::forward<decltype(elems)>(elems))...
            };
        }, packs);

        for (uint32_t i = 0; i < bindings.size(); i++) {
            bindings[i].binding = i;
        }

        auto binding_flags = std::apply([](auto &&... elems) {
            return vector<vk::DescriptorBindingFlags>{
                extract_flags(std::forward<decltype(elems)>(elems))...
            };
        }, packs);

        const vk::StructureChain set_layout_info_chain{
            vk::DescriptorSetLayoutCreateInfo{
                .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data(),
            },
            vk::DescriptorSetLayoutBindingFlagsCreateInfo{
                .bindingCount = static_cast<uint32_t>(binding_flags.size()),
                .pBindingFlags = binding_flags.data(),
            }
        };

        layout = make_shared<vk::raii::DescriptorSetLayout>(
            *ctx.get().device,
            set_layout_info_chain.get<vk::DescriptorSetLayoutCreateInfo>()
        );
    }

    template<typename ResourceType>
    [[nodiscard]] static vk::DescriptorSetLayoutBinding make_binding(const ResourcePack<ResourceType> &pack) {
        return vk::DescriptorSetLayoutBinding{
            .descriptorType = pack.type,
            .descriptorCount = pack.descriptor_count,
            .stageFlags = pack.scope,
        };
    }

    template<typename ResourceType>
    [[nodiscard]] static vk::DescriptorBindingFlags extract_flags(const ResourcePack<ResourceType> &pack) {
        return pack.flags;
    }

    void create_set(const vk::raii::DescriptorPool &pool) {
        const vk::DescriptorSetAllocateInfo alloc_info{
            .descriptorPool = *pool,
            .descriptorSetCount = 1u,
            .pSetLayouts = &**layout,
        };

        vector<vk::raii::DescriptorSet> descriptor_sets = ctx.get().device->allocateDescriptorSets(alloc_info);

        set = make_unique<vk::raii::DescriptorSet>(std::move(descriptor_sets[0]));
    }

    template<size_t I = 0>
    void update_slot() {
        auto pack = std::get<I>(packs);

        for (uint32_t i = 0; i < pack.descriptor_count; i++) {
            const auto &resource = pack.resources[i];

            if (resource.has_value()) {
                queue_update<I>(resource->get(), i);
            }
        }

        if constexpr (I + 1 < sizeof...(Ts)) {
            update_slot<I + 1>();
        }
    }

    void do_full_update() {
        update_slot();
        commit_updates();
    }
};

template<typename... Ts>
class FixedDescriptorSets {
    vector<FixedDescriptorSet<Ts...> > sets;

public:
    FixedDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool) {
        Logger::error("unimplemented");
        // todo - implement so that the sets share layouts
    }

    [[nodiscard]] FixedDescriptorSet<Ts...> &operator[](const size_t index) const { return sets[index]; }
};

/**
 * Builder class streamlining descriptor set layout creation.
 *
 * Methods which add bindings are order-dependent and the order in which they are called
 * defines which binding is used for a given resource, i.e. first call to `addBinding` will
 * use binding 0, second will use binding 1, and so on.
 */
class DescriptorLayoutBuilder {
    vector<vk::DescriptorSetLayoutBinding> bindings;

public:
    DescriptorLayoutBuilder &add_binding(vk::DescriptorType type, vk::ShaderStageFlags stages,
                                        uint32_t descriptor_count = 1);

    DescriptorLayoutBuilder &add_repeated_bindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages,
                                                 uint32_t descriptor_count = 1);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext &ctx);
};

/**
 * Convenience wrapper around Vulkan descriptor sets, mainly to pair them together with related layouts,
 * as well as provide an easy way to update them in a performant way.
 */
class DescriptorSet {
    shared_ptr<vk::raii::DescriptorSetLayout> layout;
    unique_ptr<vk::raii::DescriptorSet> set;

    struct DescriptorUpdate {
        uint32_t binding{};
        uint32_t array_element{};
        vk::DescriptorType type{};
        std::variant<
            vk::DescriptorBufferInfo,
            vk::DescriptorImageInfo,
            vk::WriteDescriptorSetAccelerationStructureKHR> info;
    };

    vector<DescriptorUpdate> queued_updates;

public:
    explicit DescriptorSet(decltype(layout) l, vk::raii::DescriptorSet &&s)
        : layout(std::move(l)), set(make_unique<vk::raii::DescriptorSet>(std::move(s))) {
    }

    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    [[nodiscard]] const vk::raii::DescriptorSetLayout &get_layout() const { return *layout; }

    /**
     * Queues an update to a given binding in this descriptor set, referencing a buffer.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queue_update(uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                                vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t array_element = 0);

    /**
     * Queues an update to a given binding in this descriptor set, referencing a texture.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queue_update(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                                vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler,
                                uint32_t array_element  = 0);

    /**
     * Queues an update to a given binding in this descriptor set, referencing a raw storage image.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queue_update(uint32_t binding, const vk::raii::ImageView &view, uint32_t array_element = 0);

    /**
    * Queues an update to a given binding in this descriptor set, referencing an acceleration structure.
    * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
    */
    DescriptorSet &queue_update(uint32_t binding, const AccelerationStructure &accel, uint32_t array_element = 0);

    void commit_updates(const RendererContext &ctx);

    /**
     * Immediately updates a single binding in this descriptor set, referencing a buffer.
     */
    void update_binding(const RendererContext &ctx, uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                        vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t array_element = 0) const;

    /**
     * Immediately updates a single binding in this descriptor set, referencing a texture.
     */
    void update_binding(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                        vk::DescriptorType type = vk::DescriptorType::eCombinedImageSampler,
                        uint32_t array_element  = 0) const;

    /**
     * Immediately updates a single binding in this descriptor set, referencing a raw storage image.
     */
    void update_binding(const RendererContext &ctx, uint32_t binding, const vk::raii::ImageView &view,
                        uint32_t array_element = 0) const;

    /**
     * Immediately updates a single binding in this descriptor set, referencing an acceleration structure.
     */
    void update_binding(const RendererContext &ctx, uint32_t binding, const AccelerationStructure &accel,
                        uint32_t array_element = 0) const;
};

namespace utils::desc {
    template<typename... Ts>
    [[nodiscard]] vector<FixedDescriptorSet<Ts...> >
    create_fixed_descriptor_sets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                 const shared_ptr<vk::raii::DescriptorSetLayout> &layout, const uint32_t count) {
        const vector set_layouts(count, **layout);

        const vk::DescriptorSetAllocateInfo alloc_info{
            .descriptorPool = *pool,
            .descriptorSetCount = count,
            .pSetLayouts = set_layouts.data(),
        };

        vector<vk::raii::DescriptorSet> descriptor_sets = ctx.device->allocateDescriptorSets(alloc_info);

        vector<FixedDescriptorSet<Ts...> > final_sets;

        for (size_t i = 0; i < count; i++) {
            final_sets.emplace_back(layout, std::move(descriptor_sets[i]));
        }

        return final_sets;
    }

    template<typename... Ts>
    [[nodiscard]] vector<FixedDescriptorSet<Ts...> >
    create_fixed_descriptor_set(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                const shared_ptr<vk::raii::DescriptorSetLayout> &layout) {
        return create_fixed_descriptor_sets<Ts>(ctx, pool, layout, 1);
    }

    [[nodiscard]] vector<DescriptorSet>
    create_descriptor_sets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                           const shared_ptr<vk::raii::DescriptorSetLayout> &layout, uint32_t count);

    [[nodiscard]] DescriptorSet
    create_descriptor_set(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                          const shared_ptr<vk::raii::DescriptorSetLayout> &layout);
} // utils::desc
} // zrx
