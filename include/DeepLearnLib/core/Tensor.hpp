/**
 * @file Tensor.hpp
 * @brief Float32 tensor, CUDA status checks, and the process-wide cuBLAS handle.
 *
 * A tensor is a shape plus a single allocation. Copying is deleted so a large
 * block is not duplicated by accident. A view is another tensor that shares
 * the allocation and may use a different shape. Training tensors live on the
 * GPU. The host is used to load a batch, read a loss or logits, and save weights.
 */

#pragma once

#ifndef DEEPLEARNLIB_ENABLE_CUDA
#define DEEPLEARNLIB_ENABLE_CUDA 1
#endif

#include <algorithm>
#include <cstddef>
#if DEEPLEARNLIB_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearnLib/core/Precision.hpp"

namespace dl
{

/** Forward declaration of the logger hook defined in @c Logger.hpp. */
auto log_error_message(const std::string& message) -> void;

/** Where the elements of a tensor are stored. */
enum class Device
{
    CPU,
    GPU
};

/** Deleter for a host allocation created with @c ::operator new. */
struct CpuDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            ::operator delete(static_cast<void*>(ptr));
        }
    }
};

#if DEEPLEARNLIB_ENABLE_CUDA

/** Deleter for a device allocation created with @c cudaMalloc. */
struct CudaDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
};

/**
 * @brief Throw if @p status is not @c cudaSuccess.
 * @param status Value returned by the CUDA call.
 * @param file Source file of the call, usually @c __FILE__.
 * @param line Source line of the call, usually @c __LINE__.
 */
inline auto check_cuda(cudaError_t status, const char* file, int line) -> void
{
    if (status != cudaSuccess)
    {
        const std::string message = std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": "
            + cudaGetErrorString(status);
        log_error_message(message);
        throw std::runtime_error(message);
    }
}

/**
 * @brief Throw if @p status is not @c CUBLAS_STATUS_SUCCESS.
 * The message includes @p file and @p line of the call site.
 */
inline auto check_cublas(cublasStatus_t status, const char* file, int line) -> void
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        const std::string message = std::string("cuBLAS error at ") + file + ":" + std::to_string(line) + ": "
            + cublasGetStatusString(status);
        log_error_message(message);
        throw std::runtime_error(message);
    }
}

/** Check a CUDA call and report this source location on failure. */
#define CHECK_CUDA(call) ::dl::check_cuda((call), __FILE__, __LINE__)
/** Check a cuBLAS call and report this source location on failure. */
#define CHECK_CUBLAS(call) ::dl::check_cublas((call), __FILE__, __LINE__)

/**
 * @brief Process-wide cuBLAS handle.
 *
 * The handle is created on first use and is not opened again per layer.
 * Copy and move are deleted.
 */
class CublasContext
{
public:
    /** Native handle. Valid for the rest of the process. */
    static auto handle() -> cublasHandle_t;

    CublasContext(const CublasContext&) = delete;
    auto operator=(const CublasContext&) -> CublasContext& = delete;
    CublasContext(CublasContext&&) = delete;
    auto operator=(CublasContext&&) -> CublasContext& = delete;

private:
    CublasContext();
    ~CublasContext();

    cublasHandle_t handle_ { nullptr };
    void* workspace_ { nullptr };
};

/** @c CublasContext::handle without naming the singleton. */
auto get_cublas_handle() -> cublasHandle_t;

/** Stream currently bound for library launches. Null means the legacy stream. */
[[nodiscard]] auto current_stream() -> cudaStream_t;

/** Bind @p stream for later launches that do not take an explicit stream. */
auto set_current_stream(cudaStream_t stream) -> void;

/**
 * @brief Set the current stream for the lifetime of this object, then restore the previous one.
 */
class StreamGuard
{
public:
    explicit StreamGuard(cudaStream_t stream);
    ~StreamGuard();

    StreamGuard(const StreamGuard&) = delete;
    auto operator=(const StreamGuard&) -> StreamGuard& = delete;
    StreamGuard(StreamGuard&&) = delete;
    auto operator=(StreamGuard&&) -> StreamGuard& = delete;

private:
    cudaStream_t previous_ { 0 };
};

/**
 * @brief Deleter for pinned host memory.
 *
 * The destructor synchronizes @c stream before @c cudaFreeHost, so the buffer
 * is not released while a copy from it is still queued.
 */
struct PinnedHostDeleter
{
    cudaStream_t stream { 0 };

    void operator()(float* ptr) const
    {
        if (ptr == nullptr)
        {
            return;
        }
        static_cast<void>(cudaStreamSynchronize(stream));
        static_cast<void>(cudaFreeHost(ptr));
    }
};

/**
 * @brief A non-blocking CUDA stream destroyed with this object.
 *
 * The destructor synchronizes the stream before destroying it.
 * The image training path uses one of these to upload the next batch.
 */
class UniqueCudaStream
{
public:
    UniqueCudaStream()
    {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~UniqueCudaStream()
    {
        if (stream_ != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(stream_));
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    UniqueCudaStream(const UniqueCudaStream&) = delete;
    auto operator=(const UniqueCudaStream&) -> UniqueCudaStream& = delete;
    UniqueCudaStream(UniqueCudaStream&&) = delete;
    auto operator=(UniqueCudaStream&&) -> UniqueCudaStream& = delete;

    /** Native stream. Owned by this object. */
    [[nodiscard]] auto get() const -> cudaStream_t
    {
        return stream_;
    }

private:
    cudaStream_t stream_ { nullptr };
};

/** Device-to-device copy on @c current_stream. A zero-byte copy is a no-op. */
inline auto memcpy_d2d_on_current(void* dst, const void* src, size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }
    CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, current_stream()));
}
#endif

/**
 * @brief Dense float32 array with an explicit shape and device.
 *
 * The constructor that takes an existing @c shared_ptr builds a view: the
 * allocation is shared and freed when the last tensor that references it dies.
 * Elementwise operators that return a tensor allocate a result. Methods whose
 * names end in @c _ write into @c *this. GPU math throws if an operand is on
 * the CPU, if the shapes cannot be combined, or if a pointer is null.
 */
class Tensor
{
public:
    /** Empty tensor. No allocation. */
    Tensor();

    /**
     * @brief Allocate a contiguous tensor.
     * @param shape Dimensions. The product is the number of elements.
     * @param device @c CPU or @c GPU. The training path uses @c GPU.
     * @param dtype Must be float32.
     */
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU, Dtype dtype = Dtype::Float32);

    /**
     * @brief View over an existing allocation.
     * @param shape Shape of this view.
     * @param strides Stride of each dimension, in elements.
     * @param data Allocation shared with the tensor that created it.
     * @param device Device of @p data.
     * @param dtype Element type of @p data.
     */
    Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device,
        Dtype dtype = Dtype::Float32);

    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    auto operator=(const Tensor&) -> Tensor& = delete;

    Tensor(Tensor&&) noexcept = default;
    auto operator=(Tensor&&) noexcept -> Tensor& = default;

    /** Dimensions, batch first. */
    auto get_shape() const -> const std::vector<int>&;

    /** Stride of each dimension, in elements. */
    auto get_strides() const -> const std::vector<int>&;

    /** Number of elements, the product of @c get_shape. */
    auto get_size() const -> size_t;

    /** @c CPU or @c GPU. */
    auto get_device() const -> Device;

    /** Element type. Always float32. */
    auto get_dtype() const -> Dtype;

    /** Bytes occupied by one element. */
    auto element_size() const -> std::size_t;

    /** @c get_size times @c element_size. */
    auto nbytes() const -> std::size_t;

    /** Pointer to the first element. Host or device, matching @c get_device. */
    auto get_data() const -> const float*;

    /** Mutable pointer to the first element. */
    auto data() -> float*;

    /** Const pointer to the first element. */
    auto data() const -> const float*;
#if DEEPLEARNLIB_ENABLE_CUDA

    /**
     * @brief Return a view when @p dtype is the tensor's own dtype.
     * Any other dtype is rejected. No conversion kernel exists.
     */
    auto to_dtype(Dtype dtype, cudaStream_t stream = 0) const -> Tensor;
#endif

    /**
     * @brief Matrix product that allocates its result.
     * @param other Right-hand matrix.
     * @param transpose_a Transpose @c *this before the product. The transpose is logical.
     * @param transpose_b Transpose @p other before the product.
     */
    auto matmul(const Tensor& other, bool transpose_a = false, bool transpose_b = false) const -> Tensor;

    /**
     * @brief Matrix product written into @p out.
     * @param other Right-hand matrix.
     * @param out Destination buffer. Reused by layers so a step does not allocate.
     * @param transpose_a Transpose @c *this before the product.
     * @param transpose_b Transpose @p other before the product.
     * @param beta Scale already stored in @p out. Zero overwrites it.
     *        The product itself uses TF32 tensor-core math.
     * @return @p out.
     */
    auto matmul_into(const Tensor& other, Tensor& out, bool transpose_a = false, bool transpose_b = false,
        float beta = 0.0F) const -> Tensor&;

    /**
     * @brief Reuse @p slot when its shape, device, and dtype already match.
     * @return The tensor stored in @p slot after the check.
     *
     * Layers call this so a repeated batch shape does not call @c cudaMalloc again.
     */
    static auto ensure(std::optional<Tensor>& slot, const std::vector<int>& shape, Device device,
        Dtype dtype = Dtype::Float32) -> Tensor&;

    /** Elementwise sum. Allocates the result. Both operands must be contiguous GPU tensors of equal size. */
    auto operator+(const Tensor& other) const -> Tensor;

    /** Elementwise difference. Allocates the result. */
    auto operator-(const Tensor& other) const -> Tensor;

    /** Elementwise product. Allocates the result. */
    auto operator*(const Tensor& other) const -> Tensor;

    /** Scale every element. Allocates the result. */
    auto operator*(float scalar) const -> Tensor;

    /** Add @p scalar to every element. Allocates the result. */
    auto operator+(float scalar) const -> Tensor;

    /** Add @p other into @c *this. */
    auto add_(const Tensor& other) -> Tensor&;

    /** Scale @c *this by @p scalar. */
    auto mul_(float scalar) -> Tensor&;

    /** Write the elementwise product of @c *this and @p other into @p out. */
    auto mul_into(const Tensor& other, Tensor& out) const -> Tensor&;

    /** Add @c scale * other into @c *this. */
    auto add_scaled_(const Tensor& other, float scale) -> Tensor&;

    /**
     * @brief In-place SGD step.
     *
     * For each element, @c update = grad + decay * weight, then the update is
     * clamped to @c [-clip, clip] when @p clip is positive, then
     * @c weight -= lr * update.
     */
    auto sgd_update_(const Tensor& grad, float lr, float decay, float clip = 0.0F) -> Tensor&;

    /**
     * @brief In-place SGD step with momentum.
     *
     * The clipped, weight-decayed gradient is added to @c momentum * velocity.
     * The weight then moves by @c lr times that velocity. @p velocity is updated in place.
     */
    auto sgd_momentum_update_(const Tensor& grad, Tensor& velocity, float lr, float momentum, float decay,
        float clip = 0.0F) -> Tensor&;

    /**
     * @brief Add a bias row to every row of a @c [batch, features] tensor.
     * @p bias has one element per feature.
     */
    auto add_row_(const Tensor& bias) -> Tensor&;

    /**
     * @brief Sum the rows of @p matrix into this vector.
     * @param matrix Rank-2 tensor shaped @c [batch, features].
     * @param beta Scale applied to the values already stored here. Zero replaces them with the sum.
     */
    auto add_sum_rows_(const Tensor& matrix, float beta = 0.0F) -> Tensor&;

    /** Elementwise clamp to @c [lo, hi]. Allocates the result. */
    auto clamp(float lo, float hi) const -> Tensor;

    /** Elementwise clamp of @c *this to @c [lo, hi]. */
    auto clamp_(float lo, float hi) -> Tensor&;

    /** True when any element is NaN or infinity. The read synchronizes. */
    [[nodiscard]] auto has_non_finite() const -> bool;

    /** Throw if @c has_non_finite is true. @p context is included in the message. */
    auto assert_finite(const char* context) const -> void;

    /**
     * @brief Sum of all elements, as a one-element GPU tensor.
     * @param dim Only a full reduction (@c -1) is implemented.
     * @throws std::runtime_error if @p dim is not @c -1.
     */
    auto sum(int dim = -1) const -> Tensor;

    /**
     * @brief Reshape without copying.
     * A single @c -1 infers that dimension from the element count.
     * @throws std::runtime_error if the element count does not match.
     */
    auto view(const std::vector<int>& new_shape) const -> Tensor;

    /** View with the same shape as @c *this. The allocation is shared. */
    [[nodiscard]] auto as_view() const -> Tensor;

    /** Contiguous transpose of a rank-2 tensor. The result is a new allocation. */
    auto transpose() const -> Tensor;

    /** Contiguous tensor of zeros with the shape, device, and dtype of @p other. */
    static auto zeros_like(const Tensor& other) -> Tensor;

    /** One-line shape, device, and dtype description for error messages. */
    [[nodiscard]] auto describe() const -> std::string;

#if DEEPLEARNLIB_ENABLE_CUDA

    /**
     * @brief Copy elements to a host vector.
     * The copy is asynchronous on @p stream and the call waits for that stream.
     */
    auto to_host(cudaStream_t stream = 0) const -> std::vector<float>;

    /**
     * @brief Upload @p host_data into a new tensor.
     * @param shape Dimensions of the new tensor.
     * @param host_data Row-major values. The length must equal the product of @p shape.
     * @param device Destination. @c GPU copies through the given stream.
     * @param stream Stream used for the upload. Ignored when @p device is @c CPU.
     * @param dtype Element type. Only float32 is accepted.
     */
    static auto from_host(const std::vector<int>& shape, const std::vector<float>& host_data,
        Device device = Device::GPU, cudaStream_t stream = 0, Dtype dtype = Dtype::Float32) -> Tensor;

    /** @c from_host for a pointer. The pointer must address @c product(shape) floats. */
    static auto from_host(const std::vector<int>& shape, const float* host_data, Device device = Device::GPU,
        cudaStream_t stream = 0, Dtype dtype = Dtype::Float32) -> Tensor;
#else
    auto to_host() const -> std::vector<float>;
    static auto from_host(const std::vector<int>& shape, const std::vector<float>& host_data,
        Device device = Device::GPU, Dtype dtype = Dtype::Float32) -> Tensor;
    static auto from_host(const std::vector<int>& shape, const float* host_data, Device device = Device::GPU,
        Dtype dtype = Dtype::Float32) -> Tensor;
#endif

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    Device device_;
    Dtype dtype_ { Dtype::Float32 };

    std::shared_ptr<float> data_;
#if DEEPLEARNLIB_ENABLE_CUDA
    std::unique_ptr<float, PinnedHostDeleter> h2d_staging_;
#endif

    auto compute_strides() -> void;
    auto is_contiguous() const -> bool;
    auto ensure_gpu(const char* op_name) const -> void;
    auto ensure_binary_op(const Tensor& other, const char* op_name) const -> void;
};

/** Render @p shape as @c [d0, d1, ...]. */
inline auto format_shape(const std::vector<int>& shape) -> std::string
{
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index)
    {
        if (index > 0)
        {
            text += ", ";
        }
        text += std::to_string(shape[index]);
    }
    text += "]";
    return text;
}

}
