/**
 * @file Losses.hpp
 * @brief Mean softmax cross-entropy.
 *
 * The loss is not a @c Layer. It has no parameters and does not run SGD.
 * Softmax is computed inside the loss, so models return logits.
 */

#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

/**
 * @brief Mean cross-entropy of logits against one-hot targets.
 *
 * Both tensors are rank-2, shaped @c [batch, classes], and must live on the GPU.
 * Softmax subtracts the row maximum. A row whose maximum is not finite gets a
 * uniform distribution. Probabilities passed to the logarithm are clamped away
 * from zero. The returned gradient is @c (probability - target) / batch.
 */
class CrossEntropyLoss
{
public:
    /**
     * @brief Mean cross-entropy over the batch.
     * @param target One-hot labels.
     * @param prediction Logits.
     * @return A scalar GPU tensor. The call reads the summed row losses back to the
     *         host, so it synchronizes the current stream.
     */
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;

    /**
     * @brief Gradient of @c loss with respect to @p prediction.
     * @return A GPU tensor with the same shape as @p prediction. This call does not
     *         copy the gradient back to the host.
     */
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};
