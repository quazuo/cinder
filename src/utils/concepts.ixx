module;

export module Cinder.Utils:Concepts;

import std;

export namespace zrx {

template <typename T, typename... Ts>
concept is_one_of = (std::same_as<T, Ts> || ...);

} // zrx