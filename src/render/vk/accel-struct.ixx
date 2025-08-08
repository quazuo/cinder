module;

export module Cinder.Render.Vulkan:AccelStruct;

import vulkan_hpp;
import std;

import :Buffer;

import Cinder.Globals;

export namespace zrx {
class AccelerationStructure {
    unique_ptr<vk::raii::AccelerationStructureKHR> handle;
    unique_ptr<Buffer> buffer;

public:
    AccelerationStructure(unique_ptr<vk::raii::AccelerationStructureKHR>&& handle, unique_ptr<Buffer>&& buffer)
    : handle(std::move(handle)), buffer(std::move(buffer)) {}

    const vk::raii::AccelerationStructureKHR& operator*() const { return *handle; }

    const Buffer& getBuffer() const { return *buffer; }
};
} // zrx
