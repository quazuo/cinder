module;

export module Cinder.Utils:Flags;

import std;

export namespace zrx {

template <typename T>
struct enable_bitmask_operators : std::false_type {};

template <typename BitsType>
    requires std::is_enum_v<BitsType>
constexpr bool enable_bitmask_operators_v = enable_bitmask_operators<BitsType>::value;

template <typename BitsType>
    requires enable_bitmask_operators_v<BitsType>
constexpr BitsType operator|(const BitsType a, const BitsType b) {
    using U = std::underlying_type_t<BitsType>;
    return static_cast<BitsType>(static_cast<U>(a) | static_cast<U>(b));
}

template <typename BitsType>
    requires enable_bitmask_operators_v<BitsType>
constexpr BitsType operator&(const BitsType a, const BitsType b) {
    using U = std::underlying_type_t<BitsType>;
    return static_cast<BitsType>(static_cast<U>(a) & static_cast<U>(b));
}

template <typename BitsType>
    requires enable_bitmask_operators_v<BitsType>
constexpr bool operator!(const BitsType a) noexcept {
    using U = std::underlying_type_t<BitsType>;
    return static_cast<U>(a) == 0;
}

} // zrx
