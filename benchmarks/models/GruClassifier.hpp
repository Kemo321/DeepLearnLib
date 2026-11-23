/**
 * @file GruClassifier.hpp
 * @brief GRU plus a linear head on the last hidden state.
 *
 * ECG200 uses one input feature, 96 time steps, a hidden size of 32, and two
 * classes. The dense layer reads only the final state.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * @brief Sequence classifier built outside the core library.
 */
class GruClassifier
{
public:
    /**
     * @param input_size Features per time step.
     * @param hidden_size GRU hidden size, also the dense layer's input width.
     * @param num_classes Width of the logit vector.
     */
    GruClassifier(int input_size, int hidden_size, int num_classes);

    /**
     * @brief Forward through the GRU and the class head.
     * @param input_tensor @c [batch, time, input_size].
     * @param stream CUDA stream. Zero is the legacy default stream.
     * @return Logits shaped @c [batch, num_classes].
     */
    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    /** The GRU layer followed by the fully-connected head. */
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;

    /** Features per time step. */
    [[nodiscard]] auto input_size() const -> int;

    /** GRU hidden size. */
    [[nodiscard]] auto hidden_size() const -> int;

    /** Logit width. */
    [[nodiscard]] auto num_classes() const -> int;

private:
    int input_size_;
    int hidden_size_;
    int num_classes_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
