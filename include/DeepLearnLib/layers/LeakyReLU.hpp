/**
 * @file LeakyReLU.hpp
 * @brief Elementwise Leaky ReLU, implemented as a small CUDA kernel.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <optional>

/**
 * @brief @c y = x when @c x > 0, and @c y = slope * x otherwise.
 *
 * The slope is fixed. It is not a parameter and it is not updated by @c step.
 * The default slope used by the models in this project is 0.1. The forward
 * input is kept so the backward pass can choose the matching multiplier.
 */
class LeakyReLU : public Layer
{
public:
    /** @param slope_val Multiplier for negative inputs. */
    explicit LeakyReLU(float slope_val = 0.1F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool input_cache_ready_ { false };
    float slope_;
};
