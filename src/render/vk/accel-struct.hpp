#pragma once

#include "src/render/libs.hpp"
#include "src/render/globals.hpp"

class Buffer;

class AccelerationStructure {
    unique_ptr<vk::raii::AccelerationStructureKHR> handle;
    unique_ptr<Buffer> buffer;

public:
    AccelerationStructure(unique_ptr<vk::raii::AccelerationStructureKHR>&& handle, unique_ptr<Buffer>&& buffer);

    [[nodiscard]] const vk::raii::AccelerationStructureKHR& operator*() const { return *handle; }

    [[nodiscard]] const Buffer& getBuffer() const { return *buffer; }
};
