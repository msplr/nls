#pragma once

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <type_traits>

template <typename T> requires std::is_base_of_v<Eigen::DenseBase<T>, T>
struct fmt::formatter<T> : ostream_formatter {};
