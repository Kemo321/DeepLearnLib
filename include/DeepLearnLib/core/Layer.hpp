/**
 * @file Layer.hpp
 * @brief Common contract for a trainable layer.
 *
 * A network is an ordered list of layers. Each layer computes its own forward
 * and backward pass. There is no autograd graph. The training program calls
 * @c backward on every layer before it calls @c step, so a gradient is taken
 * with respect to the weights from the forward pass.
 */

#pragma once

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"

#include <cmath>
#include <map>
#include <string>

/**
 * @brief One stage of a network.
 *
 * Subclasses with parameters expose them as views named @c weights and
 * @c bias, or @c weight_space for GRU. Layers without parameters
 * (Leaky ReLU, max pooling, flatten) leave @c step empty.
 *
 * The training program writes @c learning_rate, @c momentum, @c weight_decay,
 * and @c gradient_clip before an epoch. Layers do not read the experiment file.
 */
class Layer
{
public:
    /** Step size used by @c step. The training program overwrites this each epoch. */
    float learning_rate = 0.001F;

    /**
     * Absolute bound applied inside the SGD kernel after weight decay.
     * Zero disables clipping.
     */
    float gradient_clip = 0.0F;

    /** Momentum coefficient. Zero selects plain SGD and skips the velocity buffer. */
    float momentum = 0.0F;

    /** L2 penalty added to the gradient inside the SGD kernel, not to the logged loss. */
    float weight_decay = 0.0005F;

    virtual ~Layer() = default;

    /** Subsequent @c forward calls keep the state required by @c backward. */
    virtual void train()
    {
        is_training_ = true;
    }

    /**
     * Subsequent @c forward calls may drop training-only state.
     * Among the layers in this library only GRU changes its cuDNN path.
     */
    virtual void eval()
    {
        is_training_ = false;
    }

    /**
     * @brief Run the layer and keep whatever @c backward needs.
     * @param input_tensor Activation from the previous layer. Not modified.
     * @param stream CUDA stream. Zero is the legacy default stream.
     * @return Output activation. Often a view of a buffer cached for this shape.
     */
    [[nodiscard]] virtual auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor = 0;

    /**
     * @brief Propagate the upstream gradient and, when the layer has weights, store theirs.
     * @param output_error_derivative Gradient of the loss with respect to this layer's output.
     * @param stream CUDA stream. Zero is the legacy default stream.
     * @return Gradient of the loss with respect to this layer's input.
     * @throws std::runtime_error if @c forward was not called for this batch.
     */
    [[nodiscard]] virtual auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor = 0;

    /**
     * @brief Apply SGD to the parameters stored by the matching @c backward.
     *
     * Weight decay and optional clipping happen inside the update kernel.
     * The default implementation is a no-op for layers without parameters.
     */
    virtual void step(cudaStream_t stream = 0)
    {
        (void)stream;
    }

    /**
     * @brief Clamp stored parameter gradients to the closed interval from @c -abs_bound to @c abs_bound.
     *
     * The training loop relies on the clip inside @c step instead. This hook
     * remains for tests that bound a gradient before the update.
     */
    virtual void clip_gradients(float abs_bound, cudaStream_t stream = 0)
    {
        (void)abs_bound;
        (void)stream;
    }

    /** Learning rate currently stored on the layer. */
    [[nodiscard]] auto step_learning_rate() const -> float
    {
        return learning_rate;
    }

    /** Positive clip bound, or zero when clipping is off. */
    [[nodiscard]] auto parameter_clip_bound() const -> float
    {
        return gradient_clip > 0.0F ? gradient_clip : 0.0F;
    }

    /**
     * @brief Parameters as views, keyed by name.
     *
     * The map is ordered, so names are saved lexicographically.
     * An empty map means the layer has no parameters.
     */
    virtual auto get_parameters() -> std::map<std::string, dl::Tensor>
    {
        return {};
    }

    /**
     * @brief Copy values into the parameters named by @p params.
     * @throws std::out_of_range if an expected name is missing.
     * @throws std::runtime_error if a tensor has a different number of elements.
     */
    virtual void set_parameters(const std::map<std::string, dl::Tensor>& params)
    {
        (void)params;
    }

    /** Record the device the layer's parameters live on. */
    virtual auto to(dl::Device device) -> void
    {
        device_ = device;
    }

protected:
    bool is_training_ = true;
    dl::Device device_ = dl::Device::GPU;
};
