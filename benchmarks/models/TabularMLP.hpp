/**
 * @file TabularMLP.hpp
 * @brief One-hidden-layer perceptron for a table row.
 *
 * The stack is FullyConnected, Leaky ReLU 0.1, FullyConnected. Iris uses
 * 4-16-3. Wisconsin uses 30-32-2. The model returns logits.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * @brief Tabular classifier built outside the core library.
 */
class TabularMLP
{
public:
    /**
     * @param in_features Columns of one row, excluding the label.
     * @param hidden_size Width of the hidden fully-connected layer.
     * @param num_classes Width of the logit vector.
     */
    TabularMLP(int in_features, int hidden_size, int num_classes);

    /**
     * @brief Forward through the three layers.
     * @param input_tensor @c [batch, in_features].
     * @param stream CUDA stream. Zero is the legacy default stream.
     * @return Logits shaped @c [batch, num_classes].
     */
    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    /** Layers in forward order. */
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;

    /** Input features. */
    [[nodiscard]] auto in_features() const -> int;

    /** Hidden width. */
    [[nodiscard]] auto hidden_size() const -> int;

    /** Logit width. */
    [[nodiscard]] auto num_classes() const -> int;

private:
    int in_features_;
    int hidden_size_;
    int num_classes_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
