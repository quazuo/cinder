#include "accel-struct.hpp"

#include "buffer.hpp"

namespace zrx {
AccelerationStructure::AccelerationStructure(unique_ptr<vk::raii::AccelerationStructureKHR> &&handle,
                                             unique_ptr<Buffer> &&buffer)
    : handle(std::move(handle)), buffer(std::move(buffer)) {
}
}
