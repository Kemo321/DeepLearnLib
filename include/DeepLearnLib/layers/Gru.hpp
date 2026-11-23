/**
 * @file Gru.hpp
 * @brief Single-layer unidirectional GRU backed by cuDNN.
 *
 * The public header does not include the cuDNN RNN descriptors. Those live in
 * the private implementation, so a caller can include this file without pulling
 * in the RNN API.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief GRU that returns only the hidden state after the last time step.
 *
 * The input is @c [batch, time, features]. The output is @c [batch, hidden_size].
 * The initial hidden state is zero. Parameters are one cuDNN weight-space tensor
 * named @c weight_space, updated with the same SGD rule as the other layers.
 * For a hidden size of at most 128 the forward path uses
 * @c CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H. Arithmetic stays on cuDNN's default
 * math mode so the gate order matches LibTorch. @c eval() switches cuDNN to its
 * inference path.
 */
class Gru : public Layer
{
public:
    /**
     * @param input_size Features per time step.
     * @param hidden_size Length of the hidden state.
     */
    Gru(int input_size, int hidden_size);
    ~Gru() override;

    Gru(const Gru&) = delete;
    auto operator=(const Gru&) -> Gru& = delete;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    /**
     * @brief One gate matrix copied out of the cuDNN weight space.
     *
     * Used to seed the matching LibTorch GRU with the same initial values.
     * This is not the checkpoint format. Checkpoints store @c weight_space.
     */
    struct LinearLayer
    {
        int rows = 0;
        int cols = 0;
        std::vector<float> matrix;
        std::vector<float> bias;
    };

    /**
     * @brief Copy the gate matrices to the host.
     *
     * Synchronizes the device. The order is the order cuDNN reports for this
     * weight space.
     */
    [[nodiscard]] auto export_linear_layers() const -> std::vector<LinearLayer>;

    /** Features per time step. */
    [[nodiscard]] auto input_size() const -> int;

    /** Hidden-state length. */
    [[nodiscard]] auto hidden_size() const -> int;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
