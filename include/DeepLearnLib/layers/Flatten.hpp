/**
 * @file Flatten.hpp
 * @brief Rank change that shares the underlying allocation.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <vector>

/**
 * @brief Collapse every dimension after the batch into one feature axis.
 *
 * No kernel is launched. The forward result is a view, and backward restores
 * the shape recorded during forward, also as a view. A rank-2 input is returned
 * unchanged.
 */
class Flatten : public Layer
{
public:
    Flatten() = default;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    std::vector<int> input_shape_cache_;
};
