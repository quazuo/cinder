module;

export module Cinder.Utils:EnumMap;

import std;

template<typename T>
concept enum_has_count_member = requires
{
    T::Count;
};

export namespace zrx {

template<typename K, typename V>
    requires std::is_enum_v<K> && enum_has_count_member<K>
class enum_map {
    class ValueProxy {
        std::reference_wrapper<std::optional<V>> value_ref;

        ValueProxy(std::optional<V>& value_ref) : value_ref(value_ref) {}

        friend enum_map;

    public:
        void operator=(V&& val) { value_ref.get() = val; }

        operator V() const { if (value_ref.get()) return *value_ref.get(); throw std::runtime_error(""); }
    };

    class ConstValueProxy {
        std::reference_wrapper<const std::optional<V>> value_ref;

        ConstValueProxy(const std::optional<V>& value_ref) : value_ref(value_ref) {}

        friend enum_map;

    public:
        operator V() const { if (value_ref.get()) return *value_ref.get(); throw std::runtime_error(""); }
    };

    std::array<std::optional<V>, K::Count> arr;

public:
    auto operator[](const K& key) -> ValueProxy {
        return ValueProxy(arr[key]);
    }

    auto operator[](const K& key) const -> ConstValueProxy {
        return ConstValueProxy(arr[key]);
    }
};

} // zrx