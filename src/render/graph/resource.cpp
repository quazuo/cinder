#include "resource.h"

namespace zrx {
[[nodiscard]] std::set<ResourceHandle> GraphicsPipelineDesc::get_bound_resources_set() const {
    std::set<ResourceHandle> result;
    result.insert(used_resources.begin(), used_resources.end());
    return result;
}
} // zrx
