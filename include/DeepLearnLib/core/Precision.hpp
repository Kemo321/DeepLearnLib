/**
 * @file Precision.hpp
 * @brief Element type stored by @c dl::Tensor.
 *
 * Training uses float32 only. Matrix products in cuBLAS and convolutions in
 * cuDNN may still run in TF32; that does not add another dtype here.
 */

#pragma once

#include <cstddef>

namespace dl
{

/** Element type of a tensor. The only value is float32. */
enum class Dtype
{
    Float32
};

/** Size in bytes of one element of @p dtype. */
[[nodiscard]] constexpr auto element_size(Dtype) -> std::size_t
{
    return sizeof(float);
}

/** Short name used in error text. Float32 is @c "fp32". */
[[nodiscard]] auto dtype_name(Dtype dtype) -> const char*;

}
