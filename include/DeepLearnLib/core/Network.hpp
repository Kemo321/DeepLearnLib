/**
 * @file Network.hpp
 * @brief Ordered stack of layers, plus the binary checkpoint format.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Forward evaluation and checkpoint I/O for a list of layers.
 *
 * @c forward does not compute a loss and does not update weights, so the same
 * object serves training and the forward-latency benchmark. The loss and the
 * SGD step live in the training program: forward, loss, backward from the last
 * layer to the first, then @c Layer::step on each layer.
 *
 * A checkpoint is a host-endian binary file with no magic number. It stores
 * the layer count, then for each layer the parameter count and, for every
 * parameter, the name, rank, dimensions, and row-major float32 values.
 * Optimizer state and the epoch number are not stored. @c load requires a
 * network with the same number of layers.
 */
class Network
{
public:
    /**
     * @brief Take ownership of @p layers_vector and copy optimizer settings onto each layer.
     * @param layers_vector Layers in forward order. A null entry is rejected.
     * @param learning_rate Initial step size written to every layer.
     * @param gradient_clip Absolute clip bound. The default is zero, which disables clipping.
     */
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate,
        float gradient_clip = dl::kDefaultGradientClip);

    /**
     * @brief Pass @p input_tensor through every layer.
     * @return Logits from the last layer.
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;

    /**
     * @brief Write parameters of every layer to @p path.
     * @throws std::runtime_error if the file cannot be created.
     */
    void save(const std::string& path);

    /**
     * @brief Replace parameters from a file written by @c save.
     * @throws std::runtime_error if the file is missing, truncated, or has a different layer count.
     */
    void load(const std::string& path);

    /** Replace the clip bound and copy it onto every layer. */
    void set_gradient_clip(float abs_bound);

    /** Clip bound currently stored on the network. */
    [[nodiscard]] auto gradient_clip() const -> float;

    /**
     * @brief Return @p gradient with each element clamped when clipping is enabled.
     * @return @p gradient unchanged when the bound is zero.
     */
    [[nodiscard]] auto clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor;

    /** Ask every layer to clamp its stored parameter gradients. */
    void clip_parameter_gradients(cudaStream_t stream = 0);

private:
    auto sync_layer_optimizer_state() -> void;

    std::vector<std::shared_ptr<Layer>> layers_;
    float gradient_clip_ { dl::kDefaultGradientClip };
    mutable std::optional<dl::Tensor> loss_grad_clip_cache_;
};
