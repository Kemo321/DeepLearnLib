/**
 * @file Conv2d.hpp
 * @brief 2D convolution through cuDNN, plus the descriptor wrappers it needs.
 *
 * Bias is applied in the same cuDNN call as the convolution, with an identity
 * activation. Leaky ReLU stays a separate layer. Algorithms are chosen on the
 * first batch of a given shape and reused until the shape changes.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <cudnn.h>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dl
{

/**
 * @brief Throw if @p status is not @c CUDNN_STATUS_SUCCESS.
 * The message names @p file and @p line of the call site.
 */
inline auto check_cudnn(cudnnStatus_t status, const char* file, int line) -> void
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        const std::string message = std::string("cuDNN error at ") + file + ":" + std::to_string(line) + ": "
            + cudnnGetErrorString(status);
        log_error_message(message);
        throw std::runtime_error(message);
    }
}

#ifndef CHECK_CUDNN
/** Check a cuDNN call and report this source location on failure. */
#define CHECK_CUDNN(call) ::dl::check_cudnn((call), __FILE__, __LINE__)
#endif

/** Process-wide cuDNN handle. Created once and shared by every layer. */
class CudnnContext
{
public:
    /** Native handle. Valid for the rest of the process. */
    static auto handle() -> cudnnHandle_t;

    CudnnContext(const CudnnContext&) = delete;
    auto operator=(const CudnnContext&) -> CudnnContext& = delete;
    CudnnContext(CudnnContext&&) = delete;
    auto operator=(CudnnContext&&) -> CudnnContext& = delete;

private:
    CudnnContext();
    ~CudnnContext();

    cudnnHandle_t handle_ { nullptr };
};

/** @c CudnnContext::handle without naming the singleton. */
auto get_cudnn_handle() -> cudnnHandle_t;

/** Bind @p stream on the process-wide cuDNN handle. */
inline auto bind_cudnn_stream(cudaStream_t stream) -> void
{
    CHECK_CUDNN(cudnnSetStream(get_cudnn_handle(), stream));
}

/** RAII wrapper around @c cudnnTensorDescriptor_t in NCHW layout. */
class CudnnTensorDescriptor
{
public:
    CudnnTensorDescriptor();
    ~CudnnTensorDescriptor();

    CudnnTensorDescriptor(const CudnnTensorDescriptor&) = delete;
    auto operator=(const CudnnTensorDescriptor&) -> CudnnTensorDescriptor& = delete;
    CudnnTensorDescriptor(CudnnTensorDescriptor&& other) noexcept;
    auto operator=(CudnnTensorDescriptor&& other) noexcept -> CudnnTensorDescriptor&;

    /** Native descriptor. Owned by this object. */
    auto get() const -> cudnnTensorDescriptor_t;

    /** Describe a contiguous NCHW tensor. */
    auto set_nchw(int n, int c, int h, int w, cudnnDataType_t data_type = CUDNN_DATA_FLOAT) -> void;

private:
    cudnnTensorDescriptor_t desc_ { nullptr };
};

/** RAII wrapper around @c cudnnFilterDescriptor_t. */
class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor();
    ~CudnnFilterDescriptor();

    CudnnFilterDescriptor(const CudnnFilterDescriptor&) = delete;
    auto operator=(const CudnnFilterDescriptor&) -> CudnnFilterDescriptor& = delete;
    CudnnFilterDescriptor(CudnnFilterDescriptor&& other) noexcept;
    auto operator=(CudnnFilterDescriptor&& other) noexcept -> CudnnFilterDescriptor&;

    /** Native descriptor. Owned by this object. */
    auto get() const -> cudnnFilterDescriptor_t;

    /** Describe a KCRS filter: outputs, inputs, kernel height, kernel width. */
    auto set_nchw(int out_channels, int in_channels, int kernel_h, int kernel_w,
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT) -> void;

private:
    cudnnFilterDescriptor_t desc_ { nullptr };
};

/** RAII wrapper around @c cudnnConvolutionDescriptor_t for a square 2D kernel. */
class CudnnConvolutionDescriptor
{
public:
    CudnnConvolutionDescriptor();
    ~CudnnConvolutionDescriptor();

    CudnnConvolutionDescriptor(const CudnnConvolutionDescriptor&) = delete;
    auto operator=(const CudnnConvolutionDescriptor&) -> CudnnConvolutionDescriptor& = delete;
    CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&& other) noexcept;
    auto operator=(CudnnConvolutionDescriptor&& other) noexcept -> CudnnConvolutionDescriptor&;

    /** Native descriptor. Owned by this object. */
    auto get() const -> cudnnConvolutionDescriptor_t;

    /** Padding and stride, shared by both spatial axes. Cross-correlation, not a flipped kernel. */
    auto set_2d(int padding, int stride, cudnnDataType_t compute_type = CUDNN_DATA_FLOAT) -> void;

    /** Select the cuDNN math mode, including tensor-core variants. */
    auto set_math_type(cudnnMathType_t math_type) -> void;

private:
    cudnnConvolutionDescriptor_t desc_ { nullptr };
};

/** RAII wrapper around @c cudnnActivationDescriptor_t. Convolution uses identity. */
class CudnnActivationDescriptor
{
public:
    CudnnActivationDescriptor();
    ~CudnnActivationDescriptor();

    CudnnActivationDescriptor(const CudnnActivationDescriptor&) = delete;
    auto operator=(const CudnnActivationDescriptor&) -> CudnnActivationDescriptor& = delete;
    CudnnActivationDescriptor(CudnnActivationDescriptor&& other) noexcept;
    auto operator=(CudnnActivationDescriptor&& other) noexcept -> CudnnActivationDescriptor&;

    /** Native descriptor. Owned by this object. */
    auto get() const -> cudnnActivationDescriptor_t;

    /** @param mode cuDNN activation. Convolution passes identity.
     *  @param nan_opt How cuDNN treats NaN inputs.
     *  @param coef Slope or ceiling passed through to cuDNN. Unused for identity. */
    auto set(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_opt, double coef) -> void;

private:
    cudnnActivationDescriptor_t desc_ { nullptr };
};

/**
 * @brief Device scratch buffer grown to the largest size a layer has requested.
 *
 * A later call with a smaller size keeps the existing allocation.
 */
class CudaWorkspace
{
public:
    /** Allocate when @p bytes exceeds the current capacity. Zero is a no-op. */
    auto ensure(size_t bytes) -> void;

    /** Device pointer, or null when nothing has been allocated. */
    auto get() const -> void*;

    /** Capacity in bytes. */
    auto size() const -> size_t;

private:
    struct Deleter
    {
        void operator()(void* pointer) const;
    };

    std::unique_ptr<void, Deleter> ptr_;
    size_t bytes_ { 0 };
};

}

/** cuDNN type for a library dtype. Float32 maps to @c CUDNN_DATA_FLOAT. */
[[nodiscard]] inline auto cudnn_data_type(dl::Dtype) -> cudnnDataType_t
{
    return CUDNN_DATA_FLOAT;
}

/**
 * @brief Square 2D convolution.
 *
 * The input is NCHW. Weights are exposed as @c weights and @c bias.
 * @c inertia_val is the scale applied to a weight gradient already stored
 * from an earlier backward; zero overwrites it. Forward and both backward
 * algorithms come from cuDNN's search the first time a shape is seen.
 */
class Conv2d : public Layer
{
public:
    /**
     * @param in_channels Channels of the input activation.
     * @param out_channels Number of filters.
     * @param kernel_size Height and width of each filter.
     * @param stride_val Step shared by both spatial axes.
     * @param padding_val Zeros added on each spatial border.
     * @param inertia_val Scale of a gradient accumulated from a previous backward.
     */
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
        float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;

    /**
     * @brief Convolution into a caller-owned @p output.
     * Used when the caller already holds the cached activation buffer.
     */
    auto forward_into(const dl::Tensor& input_tensor, dl::Tensor& output, cudaStream_t stream = 0) -> void;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    /** Channels expected on the input. */
    [[nodiscard]] auto in_channels() const -> int
    {
        return in_channels_;
    }

    /** Number of filters. */
    [[nodiscard]] auto out_channels() const -> int
    {
        return out_channels_;
    }

    /** Spatial size of each filter. */
    [[nodiscard]] auto kernel_size() const -> int
    {
        return kernel_size_;
    }

private:
    auto configure_io_descriptors(int batch, int height, int width) -> void;
    auto select_algorithms() -> void;
    auto ensure_workspace(size_t bytes) -> void;

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

    std::vector<int> input_shape_cache_;
    std::vector<int> output_shape_cache_;

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    float inertia_;

    dl::CudnnTensorDescriptor input_desc_;
    dl::CudnnTensorDescriptor output_desc_;
    dl::CudnnTensorDescriptor bias_desc_;
    dl::CudnnFilterDescriptor filter_desc_;
    dl::CudnnConvolutionDescriptor conv_desc_;
    dl::CudnnActivationDescriptor activation_desc_;
    dl::CudaWorkspace workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_ { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM };
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_ { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 };
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_ { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 };
    bool algorithms_selected_ { false };
};
