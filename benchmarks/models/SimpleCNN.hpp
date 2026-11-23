/**
 * @file SimpleCNN.hpp
 * @brief Two-block convolution network used for MNIST and CIFAR-10.
 *
 * The stack is fixed: Conv2d 3x3 (16, pad 1), Leaky ReLU 0.1, MaxPool 2x2,
 * Conv2d 3x3 (32, pad 1), Leaky ReLU 0.1, MaxPool 2x2, Flatten, FullyConnected.
 * Two pools divide the spatial side by four, so a 28x28 input yields 1568
 * features and a 32x32 input yields 2048.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * @brief Image classifier that returns class logits.
 *
 * The model is not part of @c DeepLearnLib. It is linked from @c DeepLearnModels.
 */
class SimpleCNN
{
public:
    /**
     * @param num_classes Width of the logit vector.
     * @param image_size Height and width of the input. Square.
     * @param in_channels Channels in the input. 1 for MNIST, 3 for CIFAR-10.
     */
    explicit SimpleCNN(int num_classes, int image_size = 32, int in_channels = 3);

    /**
     * @brief Forward through the stack.
     * @return Logits shaped @c [batch, num_classes].
     */
    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    /** Layers in forward order, for a @c Network or a training step. */
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;

    /** Logit width. */
    [[nodiscard]] auto num_classes() const -> int;

    /** Spatial size expected from the loader. */
    [[nodiscard]] auto image_size() const -> int;

    /** Input channels. */
    [[nodiscard]] auto in_channels() const -> int;

private:
    int num_classes_;
    int image_size_;
    int in_channels_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
