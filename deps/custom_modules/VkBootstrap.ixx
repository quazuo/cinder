module;

#include "../vk-bootstrap/VkBootstrap.h"

export module VkBootstrap;

export namespace vkb {
    using vkb::Instance;
    using vkb::InstanceBuilder;
    using vkb::PhysicalDevice;
    using vkb::PhysicalDeviceSelector;
    using vkb::Device;
    using vkb::DeviceBuilder;
    using vkb::QueueType;

    using vkb::to_string_message_severity;
    using vkb::to_string_message_type;
}
