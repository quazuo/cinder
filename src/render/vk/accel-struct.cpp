//
// Created by macie on 04.08.2024.
//

#include "accel-struct.hpp"

#include "buffer.hpp"

AccelerationStructure::AccelerationStructure(unique_ptr<vk::raii::AccelerationStructureKHR> &&handle,
                                             unique_ptr<Buffer> &&buffer)
    : handle(std::move(handle)), buffer(std::move(buffer)) {
}
