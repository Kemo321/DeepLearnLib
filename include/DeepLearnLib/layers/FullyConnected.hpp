/**
 * @file FullyConnected.hpp
 * @brief Dense layer: GEMM through cuBLAS, bias through a CUDA kernel.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <map>
#include <optional>
#include <string>

/**
 * @brief Affine map @c y = x W + b for a batch of row vectors.
 *
 * @c x is @c [batch, input_size], @c W is @c [input_size, output_size], and
 * @c b is added to every row. The product is written into a buffer reused for
 * the same shape. Parameters are exposed as @c weights and @c bias.
 * @c inertia_val is the cuBLAS beta used when accumulating the weight gradient,
 * so a value of zero overwrites the previous gradient.
 */
class FullyConnected : public Layer
{
public:
    /**
     * @param input_size Number of incoming features.
     * @param output_size Number of outgoing features.
     * @param inertia_val Scale applied to a gradient already stored from an earlier backward.
     */
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    /** Features consumed from each sample. */
    [[nodiscard]] auto input_size() const -> int
    {
        return input_size_;
    }

    /** Features produced for each sample. */
    [[nodiscard]] auto output_size() const -> int
    {
        return output_size_;
    }

private:
    dl::Tensor weights_;
    dl::Tensor biases_;
    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool input_cache_ready_ { false };
    dl::Tensor weights_gradient_;
    dl::Tensor biases_gradient_;
    std::optional<dl::Tensor> weights_velocity_;
    std::optional<dl::Tensor> biases_velocity_;
    int input_size_;
    int output_size_;
    float inertia_;
};
