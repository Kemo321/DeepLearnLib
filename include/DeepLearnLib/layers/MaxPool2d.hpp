/**
 * @file MaxPool2d.hpp
 * @brief 2D max pooling through cuDNN.
 */

#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Conv2d.hpp"

#include <cudnn.h>
#include <optional>
#include <vector>

namespace dl
{

/**
 * @brief RAII wrapper around @c cudnnPoolingDescriptor_t.
 *
 * The descriptor is configured for non-overlapping 2D max pooling.
 */
class CudnnPoolingDescriptor
{
public:
    CudnnPoolingDescriptor();
    ~CudnnPoolingDescriptor();

    CudnnPoolingDescriptor(const CudnnPoolingDescriptor&) = delete;
    auto operator=(const CudnnPoolingDescriptor&) -> CudnnPoolingDescriptor& = delete;
    CudnnPoolingDescriptor(CudnnPoolingDescriptor&& other) noexcept;
    auto operator=(CudnnPoolingDescriptor&& other) noexcept -> CudnnPoolingDescriptor&;

    /** Native handle. Owned by this object. */
    auto get() const -> cudnnPoolingDescriptor_t;

    /**
     * @brief Select 2D max pooling.
     * @param window Spatial window. The same value is used for height and width.
     * @param stride Step between windows.
     * @param padding Zeros added on each spatial border.
     */
    auto set_max_2d(int window, int stride, int padding = 0) -> void;

private:
    cudnnPoolingDescriptor_t desc_ { nullptr };
};

}

/**
 * @brief Downsample an NCHW activation by keeping the maximum of each window.
 *
 * The layer has no parameters. Backward sends the upstream gradient to the
 * argmax location inside each window and writes zero elsewhere. The forward
 * input is retained for that routing. Channel count is unchanged.
 */
class MaxPool2d : public Layer
{
public:
    /**
     * @param kernel_size_val Window height and width.
     * @param stride_val Step between windows.
     */
    MaxPool2d(int kernel_size_val, int stride_val);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    auto configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void;

    int kernel_size_;
    int stride_;

    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool caches_ready_ { false };
    std::vector<int> input_shape_cache_;
    std::vector<int> output_shape_cache_;

    dl::CudnnPoolingDescriptor pooling_desc_;
    dl::CudnnTensorDescriptor input_desc_;
    dl::CudnnTensorDescriptor output_desc_;
    bool descriptors_configured_ { false };
};
