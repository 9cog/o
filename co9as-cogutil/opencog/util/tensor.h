/*
 * opencog/util/tensor.h
 *
 * Copyright (C) 2025 OpenCog Foundation
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _OPENCOG_TENSOR_H
#define _OPENCOG_TENSOR_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencog/util/oc_assert.h>

/** \addtogroup grp_cogutil
 *  @{
 */

namespace opencog
{

/**
 * @brief ATen-style tensor class for numerical computations.
 *
 * This is a lightweight, header-only tensor implementation inspired by
 * PyTorch's ATen library. It provides multi-dimensional array operations
 * with broadcasting support and common mathematical functions.
 *
 * @tparam T The scalar type (default: double)
 *
 * Example usage:
 * @code
 *   Tensor<double> a = Tensor<double>::zeros({3, 4});
 *   Tensor<double> b = Tensor<double>::ones({3, 4});
 *   Tensor<double> c = a + b;
 *   Tensor<double> d = c.matmul(Tensor<double>::randn({4, 2}));
 * @endcode
 */
template<typename T = double>
class Tensor
{
public:
    using value_type = T;
    using size_type = size_t;
    using shape_type = std::vector<size_t>;
    using storage_type = std::shared_ptr<std::vector<T>>;

private:
    storage_type _data;      // Shared pointer for copy-on-write semantics
    shape_type _shape;       // Dimensions of the tensor
    shape_type _strides;     // Strides for indexing
    size_t _offset;          // Offset into storage (for views)
    size_t _numel;           // Total number of elements

    // Compute strides from shape (row-major order)
    static shape_type compute_strides(const shape_type& shape)
    {
        shape_type strides(shape.size());
        if (shape.empty()) return strides;

        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    // Compute total number of elements
    static size_t compute_numel(const shape_type& shape)
    {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(),
                               size_t(1), std::multiplies<size_t>());
    }

    // Convert multi-dimensional index to linear index
    size_t linear_index(const std::vector<size_t>& indices) const
    {
        OC_ASSERT(indices.size() == _shape.size(),
                  "Tensor: index dimension mismatch");
        size_t idx = _offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            OC_ASSERT(indices[i] < _shape[i],
                      "Tensor: index out of bounds at dimension %zu", i);
            idx += indices[i] * _strides[i];
        }
        return idx;
    }

    // Ensure unique ownership of data (for copy-on-write)
    void ensure_unique()
    {
        if (!_data.unique()) {
            auto new_data = std::make_shared<std::vector<T>>(_numel);
            // Copy only the relevant elements
            for (size_t i = 0; i < _numel; ++i) {
                (*new_data)[i] = (*_data)[_offset + i];
            }
            _data = new_data;
            _offset = 0;
            _strides = compute_strides(_shape);
        }
    }

public:
    // ===================================================================
    // Constructors
    // ===================================================================

    /// Default constructor: creates an empty tensor
    Tensor()
        : _data(std::make_shared<std::vector<T>>()),
          _shape(), _strides(), _offset(0), _numel(0)
    {}

    /// Construct tensor with given shape, uninitialized
    explicit Tensor(const shape_type& shape)
        : _shape(shape),
          _strides(compute_strides(shape)),
          _offset(0),
          _numel(compute_numel(shape))
    {
        _data = std::make_shared<std::vector<T>>(_numel);
    }

    /// Construct tensor with given shape and fill value
    Tensor(const shape_type& shape, T fill_value)
        : _shape(shape),
          _strides(compute_strides(shape)),
          _offset(0),
          _numel(compute_numel(shape))
    {
        _data = std::make_shared<std::vector<T>>(_numel, fill_value);
    }

    /// Construct 1D tensor from initializer list
    Tensor(std::initializer_list<T> values)
        : _shape({values.size()}),
          _strides({1}),
          _offset(0),
          _numel(values.size())
    {
        _data = std::make_shared<std::vector<T>>(values);
    }

    /// Construct tensor from vector with shape
    Tensor(const std::vector<T>& values, const shape_type& shape)
        : _shape(shape),
          _strides(compute_strides(shape)),
          _offset(0),
          _numel(compute_numel(shape))
    {
        OC_ASSERT(values.size() == _numel,
                  "Tensor: data size mismatch with shape");
        _data = std::make_shared<std::vector<T>>(values);
    }

    // Copy and move constructors/assignment (use defaults with shared_ptr)
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // ===================================================================
    // Static factory methods (ATen-style)
    // ===================================================================

    /// Create a tensor filled with zeros
    static Tensor zeros(const shape_type& shape)
    {
        return Tensor(shape, T(0));
    }

    /// Create a tensor filled with ones
    static Tensor ones(const shape_type& shape)
    {
        return Tensor(shape, T(1));
    }

    /// Create a tensor filled with a specific value
    static Tensor full(const shape_type& shape, T value)
    {
        return Tensor(shape, value);
    }

    /// Create a tensor with uninitialized values
    static Tensor empty(const shape_type& shape)
    {
        return Tensor(shape);
    }

    /// Create a 1D tensor with values from start to end (exclusive)
    static Tensor arange(T start, T end, T step = T(1))
    {
        OC_ASSERT(step != T(0), "Tensor::arange: step cannot be zero");
        OC_ASSERT((step > 0 && start < end) || (step < 0 && start > end),
                  "Tensor::arange: invalid range");

        std::vector<T> values;
        for (T v = start; (step > 0) ? (v < end) : (v > end); v += step) {
            values.push_back(v);
        }
        return Tensor(values, {values.size()});
    }

    /// Create a 1D tensor with evenly spaced values
    static Tensor linspace(T start, T end, size_t steps)
    {
        OC_ASSERT(steps > 0, "Tensor::linspace: steps must be positive");
        std::vector<T> values(steps);
        if (steps == 1) {
            values[0] = start;
        } else {
            T delta = (end - start) / static_cast<T>(steps - 1);
            for (size_t i = 0; i < steps; ++i) {
                values[i] = start + static_cast<T>(i) * delta;
            }
        }
        return Tensor(values, {steps});
    }

    /// Create an identity matrix
    static Tensor eye(size_t n)
    {
        Tensor result = zeros({n, n});
        for (size_t i = 0; i < n; ++i) {
            result.at({i, i}) = T(1);
        }
        return result;
    }

    /// Create a tensor with random values from uniform distribution [0, 1)
    static Tensor rand(const shape_type& shape)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(T(0), T(1));

        Tensor result(shape);
        for (size_t i = 0; i < result._numel; ++i) {
            (*result._data)[i] = dist(gen);
        }
        return result;
    }

    /// Create a tensor with random values from standard normal distribution
    static Tensor randn(const shape_type& shape)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<T> dist(T(0), T(1));

        Tensor result(shape);
        for (size_t i = 0; i < result._numel; ++i) {
            (*result._data)[i] = dist(gen);
        }
        return result;
    }

    /// Create a tensor with random integers in [low, high)
    static Tensor randint(T low, T high, const shape_type& shape)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<long long> dist(
            static_cast<long long>(low),
            static_cast<long long>(high) - 1
        );

        Tensor result(shape);
        for (size_t i = 0; i < result._numel; ++i) {
            (*result._data)[i] = static_cast<T>(dist(gen));
        }
        return result;
    }

    // ===================================================================
    // Shape and size information
    // ===================================================================

    /// Get the shape of the tensor
    const shape_type& shape() const { return _shape; }

    /// Get the size of a specific dimension
    size_t size(size_t dim) const
    {
        OC_ASSERT(dim < _shape.size(), "Tensor::size: dimension out of range");
        return _shape[dim];
    }

    /// Get the number of dimensions
    size_t ndim() const { return _shape.size(); }

    /// Get the total number of elements
    size_t numel() const { return _numel; }

    /// Check if tensor is empty
    bool empty() const { return _numel == 0; }

    /// Get the strides
    const shape_type& strides() const { return _strides; }

    // ===================================================================
    // Element access
    // ===================================================================

    /// Access element at index (const)
    const T& at(const std::vector<size_t>& indices) const
    {
        return (*_data)[linear_index(indices)];
    }

    /// Access element at index (mutable)
    T& at(const std::vector<size_t>& indices)
    {
        ensure_unique();
        return (*_data)[linear_index(indices)];
    }

    /// Access element using variadic indices (const)
    template<typename... Indices>
    const T& operator()(Indices... indices) const
    {
        return at({static_cast<size_t>(indices)...});
    }

    /// Access element using variadic indices (mutable)
    template<typename... Indices>
    T& operator()(Indices... indices)
    {
        return at({static_cast<size_t>(indices)...});
    }

    /// Get underlying data pointer (const)
    const T* data() const
    {
        return _data->data() + _offset;
    }

    /// Get underlying data pointer (mutable)
    T* data()
    {
        ensure_unique();
        return _data->data() + _offset;
    }

    /// Convert scalar tensor to value
    T item() const
    {
        OC_ASSERT(_numel == 1, "Tensor::item: tensor must have exactly one element");
        return (*_data)[_offset];
    }

    // ===================================================================
    // Reshape operations
    // ===================================================================

    /// Reshape the tensor to a new shape (returns new tensor)
    Tensor reshape(const shape_type& new_shape) const
    {
        size_t new_numel = compute_numel(new_shape);
        OC_ASSERT(new_numel == _numel,
                  "Tensor::reshape: total elements must match");

        Tensor result;
        result._data = _data;
        result._shape = new_shape;
        result._strides = compute_strides(new_shape);
        result._offset = _offset;
        result._numel = new_numel;
        return result;
    }

    /// Flatten to 1D tensor
    Tensor flatten() const
    {
        return reshape({_numel});
    }

    /// Squeeze dimensions of size 1
    Tensor squeeze() const
    {
        shape_type new_shape;
        for (size_t dim : _shape) {
            if (dim != 1) new_shape.push_back(dim);
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }

    /// Unsqueeze: add dimension of size 1 at position
    Tensor unsqueeze(size_t dim) const
    {
        OC_ASSERT(dim <= _shape.size(),
                  "Tensor::unsqueeze: dimension out of range");
        shape_type new_shape = _shape;
        new_shape.insert(new_shape.begin() + dim, 1);
        return reshape(new_shape);
    }

    /// Transpose 2D tensor
    Tensor t() const
    {
        OC_ASSERT(_shape.size() == 2, "Tensor::t: only 2D tensors supported");
        Tensor result({_shape[1], _shape[0]});
        for (size_t i = 0; i < _shape[0]; ++i) {
            for (size_t j = 0; j < _shape[1]; ++j) {
                result.at({j, i}) = at({i, j});
            }
        }
        return result;
    }

    /// Transpose with dimension permutation
    Tensor permute(const std::vector<size_t>& dims) const
    {
        OC_ASSERT(dims.size() == _shape.size(),
                  "Tensor::permute: dimension count mismatch");

        shape_type new_shape(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            new_shape[i] = _shape[dims[i]];
        }

        Tensor result(new_shape);
        std::vector<size_t> src_idx(_shape.size(), 0);
        std::vector<size_t> dst_idx(_shape.size(), 0);

        for (size_t i = 0; i < _numel; ++i) {
            // Convert linear index to source indices
            size_t tmp = i;
            for (int d = static_cast<int>(_shape.size()) - 1; d >= 0; --d) {
                src_idx[d] = tmp % _shape[d];
                tmp /= _shape[d];
            }
            // Permute indices
            for (size_t d = 0; d < dims.size(); ++d) {
                dst_idx[d] = src_idx[dims[d]];
            }
            result.at(dst_idx) = at(src_idx);
        }
        return result;
    }

    // ===================================================================
    // Clone and copy operations
    // ===================================================================

    /// Create a deep copy
    Tensor clone() const
    {
        Tensor result(_shape);
        std::copy(data(), data() + _numel, result.data());
        return result;
    }

    /// Copy data from another tensor (in-place)
    Tensor& copy_(const Tensor& other)
    {
        OC_ASSERT(_shape == other._shape,
                  "Tensor::copy_: shape mismatch");
        ensure_unique();
        std::copy(other.data(), other.data() + _numel, data());
        return *this;
    }

    /// Fill with value (in-place)
    Tensor& fill_(T value)
    {
        ensure_unique();
        std::fill(data(), data() + _numel, value);
        return *this;
    }

    /// Fill with zeros (in-place)
    Tensor& zero_()
    {
        return fill_(T(0));
    }

    // ===================================================================
    // Arithmetic operations (element-wise)
    // ===================================================================

    /// Unary negation
    Tensor operator-() const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = -(*_data)[_offset + i];
        }
        return result;
    }

    /// Element-wise addition
    Tensor operator+(const Tensor& other) const
    {
        OC_ASSERT(_shape == other._shape,
                  "Tensor::operator+: shape mismatch");
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] + (*other._data)[other._offset + i];
        }
        return result;
    }

    /// Scalar addition
    Tensor operator+(T scalar) const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] + scalar;
        }
        return result;
    }

    /// Element-wise subtraction
    Tensor operator-(const Tensor& other) const
    {
        OC_ASSERT(_shape == other._shape,
                  "Tensor::operator-: shape mismatch");
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] - (*other._data)[other._offset + i];
        }
        return result;
    }

    /// Scalar subtraction
    Tensor operator-(T scalar) const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] - scalar;
        }
        return result;
    }

    /// Element-wise multiplication
    Tensor operator*(const Tensor& other) const
    {
        OC_ASSERT(_shape == other._shape,
                  "Tensor::operator*: shape mismatch");
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] * (*other._data)[other._offset + i];
        }
        return result;
    }

    /// Scalar multiplication
    Tensor operator*(T scalar) const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] * scalar;
        }
        return result;
    }

    /// Element-wise division
    Tensor operator/(const Tensor& other) const
    {
        OC_ASSERT(_shape == other._shape,
                  "Tensor::operator/: shape mismatch");
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] / (*other._data)[other._offset + i];
        }
        return result;
    }

    /// Scalar division
    Tensor operator/(T scalar) const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = (*_data)[_offset + i] / scalar;
        }
        return result;
    }

    // In-place operations
    Tensor& operator+=(const Tensor& other)
    {
        OC_ASSERT(_shape == other._shape, "Tensor::operator+=: shape mismatch");
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] += (*other._data)[other._offset + i];
        }
        return *this;
    }

    Tensor& operator+=(T scalar)
    {
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] += scalar;
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other)
    {
        OC_ASSERT(_shape == other._shape, "Tensor::operator-=: shape mismatch");
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] -= (*other._data)[other._offset + i];
        }
        return *this;
    }

    Tensor& operator-=(T scalar)
    {
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] -= scalar;
        }
        return *this;
    }

    Tensor& operator*=(const Tensor& other)
    {
        OC_ASSERT(_shape == other._shape, "Tensor::operator*=: shape mismatch");
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] *= (*other._data)[other._offset + i];
        }
        return *this;
    }

    Tensor& operator*=(T scalar)
    {
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] *= scalar;
        }
        return *this;
    }

    Tensor& operator/=(const Tensor& other)
    {
        OC_ASSERT(_shape == other._shape, "Tensor::operator/=: shape mismatch");
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] /= (*other._data)[other._offset + i];
        }
        return *this;
    }

    Tensor& operator/=(T scalar)
    {
        ensure_unique();
        for (size_t i = 0; i < _numel; ++i) {
            (*_data)[_offset + i] /= scalar;
        }
        return *this;
    }

    // ===================================================================
    // Mathematical functions (element-wise)
    // ===================================================================

    /// Apply function to each element
    template<typename Func>
    Tensor apply(Func func) const
    {
        Tensor result(_shape);
        for (size_t i = 0; i < _numel; ++i) {
            (*result._data)[i] = func((*_data)[_offset + i]);
        }
        return result;
    }

    /// Absolute value
    Tensor abs() const { return apply([](T x) { return std::abs(x); }); }

    /// Square root
    Tensor sqrt() const { return apply([](T x) { return std::sqrt(x); }); }

    /// Exponential
    Tensor exp() const { return apply([](T x) { return std::exp(x); }); }

    /// Natural logarithm
    Tensor log() const { return apply([](T x) { return std::log(x); }); }

    /// Log base 2
    Tensor log2() const { return apply([](T x) { return std::log2(x); }); }

    /// Log base 10
    Tensor log10() const { return apply([](T x) { return std::log10(x); }); }

    /// Power
    Tensor pow(T exponent) const
    {
        return apply([exponent](T x) { return std::pow(x, exponent); });
    }

    /// Square
    Tensor square() const { return apply([](T x) { return x * x; }); }

    /// Sine
    Tensor sin() const { return apply([](T x) { return std::sin(x); }); }

    /// Cosine
    Tensor cos() const { return apply([](T x) { return std::cos(x); }); }

    /// Tangent
    Tensor tan() const { return apply([](T x) { return std::tan(x); }); }

    /// Hyperbolic sine
    Tensor sinh() const { return apply([](T x) { return std::sinh(x); }); }

    /// Hyperbolic cosine
    Tensor cosh() const { return apply([](T x) { return std::cosh(x); }); }

    /// Hyperbolic tangent (tanh)
    Tensor tanh() const { return apply([](T x) { return std::tanh(x); }); }

    /// Sigmoid activation
    Tensor sigmoid() const
    {
        return apply([](T x) { return T(1) / (T(1) + std::exp(-x)); });
    }

    /// ReLU activation
    Tensor relu() const
    {
        return apply([](T x) { return std::max(T(0), x); });
    }

    /// Leaky ReLU activation
    Tensor leaky_relu(T negative_slope = T(0.01)) const
    {
        return apply([negative_slope](T x) {
            return x >= T(0) ? x : negative_slope * x;
        });
    }

    /// Softplus activation
    Tensor softplus() const
    {
        return apply([](T x) { return std::log1p(std::exp(x)); });
    }

    /// Clamp values to range
    Tensor clamp(T min_val, T max_val) const
    {
        return apply([min_val, max_val](T x) {
            return std::max(min_val, std::min(max_val, x));
        });
    }

    /// Floor
    Tensor floor() const { return apply([](T x) { return std::floor(x); }); }

    /// Ceiling
    Tensor ceil() const { return apply([](T x) { return std::ceil(x); }); }

    /// Round
    Tensor round() const { return apply([](T x) { return std::round(x); }); }

    /// Sign (-1, 0, or 1)
    Tensor sign() const
    {
        return apply([](T x) {
            return (T(0) < x) - (x < T(0));
        });
    }

    // ===================================================================
    // Reduction operations
    // ===================================================================

    /// Sum of all elements
    T sum() const
    {
        T result = T(0);
        for (size_t i = 0; i < _numel; ++i) {
            result += (*_data)[_offset + i];
        }
        return result;
    }

    /// Mean of all elements
    T mean() const
    {
        return sum() / static_cast<T>(_numel);
    }

    /// Variance of all elements
    T var(bool unbiased = true) const
    {
        T m = mean();
        T sq_sum = T(0);
        for (size_t i = 0; i < _numel; ++i) {
            T diff = (*_data)[_offset + i] - m;
            sq_sum += diff * diff;
        }
        size_t denom = unbiased ? (_numel - 1) : _numel;
        return sq_sum / static_cast<T>(denom);
    }

    /// Standard deviation
    T std(bool unbiased = true) const
    {
        return std::sqrt(var(unbiased));
    }

    /// Product of all elements
    T prod() const
    {
        T result = T(1);
        for (size_t i = 0; i < _numel; ++i) {
            result *= (*_data)[_offset + i];
        }
        return result;
    }

    /// Maximum value
    T max() const
    {
        OC_ASSERT(_numel > 0, "Tensor::max: empty tensor");
        T result = (*_data)[_offset];
        for (size_t i = 1; i < _numel; ++i) {
            result = std::max(result, (*_data)[_offset + i]);
        }
        return result;
    }

    /// Minimum value
    T min() const
    {
        OC_ASSERT(_numel > 0, "Tensor::min: empty tensor");
        T result = (*_data)[_offset];
        for (size_t i = 1; i < _numel; ++i) {
            result = std::min(result, (*_data)[_offset + i]);
        }
        return result;
    }

    /// Index of maximum value
    size_t argmax() const
    {
        OC_ASSERT(_numel > 0, "Tensor::argmax: empty tensor");
        size_t max_idx = 0;
        T max_val = (*_data)[_offset];
        for (size_t i = 1; i < _numel; ++i) {
            if ((*_data)[_offset + i] > max_val) {
                max_val = (*_data)[_offset + i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    /// Index of minimum value
    size_t argmin() const
    {
        OC_ASSERT(_numel > 0, "Tensor::argmin: empty tensor");
        size_t min_idx = 0;
        T min_val = (*_data)[_offset];
        for (size_t i = 1; i < _numel; ++i) {
            if ((*_data)[_offset + i] < min_val) {
                min_val = (*_data)[_offset + i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    /// L1 norm (sum of absolute values)
    T norm_l1() const
    {
        T result = T(0);
        for (size_t i = 0; i < _numel; ++i) {
            result += std::abs((*_data)[_offset + i]);
        }
        return result;
    }

    /// L2 norm (Euclidean norm)
    T norm() const
    {
        T result = T(0);
        for (size_t i = 0; i < _numel; ++i) {
            T val = (*_data)[_offset + i];
            result += val * val;
        }
        return std::sqrt(result);
    }

    /// Frobenius norm (same as L2 for vectors)
    T norm_frobenius() const { return norm(); }

    // ===================================================================
    // Linear algebra operations
    // ===================================================================

    /// Dot product (for 1D tensors)
    T dot(const Tensor& other) const
    {
        OC_ASSERT(_shape.size() == 1 && other._shape.size() == 1,
                  "Tensor::dot: both tensors must be 1D");
        OC_ASSERT(_numel == other._numel,
                  "Tensor::dot: tensors must have same length");

        T result = T(0);
        for (size_t i = 0; i < _numel; ++i) {
            result += (*_data)[_offset + i] * (*other._data)[other._offset + i];
        }
        return result;
    }

    /// Matrix multiplication
    Tensor matmul(const Tensor& other) const
    {
        OC_ASSERT(_shape.size() == 2 && other._shape.size() == 2,
                  "Tensor::matmul: both tensors must be 2D");
        OC_ASSERT(_shape[1] == other._shape[0],
                  "Tensor::matmul: inner dimensions must match");

        size_t m = _shape[0];
        size_t k = _shape[1];
        size_t n = other._shape[1];

        Tensor result = zeros({m, n});
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T(0);
                for (size_t l = 0; l < k; ++l) {
                    sum += at({i, l}) * other.at({l, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    /// Matrix-vector multiplication
    Tensor mv(const Tensor& vec) const
    {
        OC_ASSERT(_shape.size() == 2 && vec._shape.size() == 1,
                  "Tensor::mv: matrix must be 2D, vector must be 1D");
        OC_ASSERT(_shape[1] == vec._numel,
                  "Tensor::mv: matrix columns must match vector length");

        Tensor result = zeros({_shape[0]});
        for (size_t i = 0; i < _shape[0]; ++i) {
            T sum = T(0);
            for (size_t j = 0; j < _shape[1]; ++j) {
                sum += at({i, j}) * vec.at({j});
            }
            result.at({i}) = sum;
        }
        return result;
    }

    /// Outer product
    Tensor outer(const Tensor& other) const
    {
        OC_ASSERT(_shape.size() == 1 && other._shape.size() == 1,
                  "Tensor::outer: both tensors must be 1D");

        Tensor result(shape_type{_numel, other._numel});
        for (size_t i = 0; i < _numel; ++i) {
            for (size_t j = 0; j < other._numel; ++j) {
                result.at({i, j}) = (*_data)[_offset + i] * (*other._data)[other._offset + j];
            }
        }
        return result;
    }

    /// Trace (sum of diagonal elements)
    T trace() const
    {
        OC_ASSERT(_shape.size() == 2, "Tensor::trace: tensor must be 2D");
        size_t min_dim = std::min(_shape[0], _shape[1]);
        T result = T(0);
        for (size_t i = 0; i < min_dim; ++i) {
            result += at({i, i});
        }
        return result;
    }

    /// Diagonal elements as 1D tensor
    Tensor diag() const
    {
        OC_ASSERT(_shape.size() == 2, "Tensor::diag: tensor must be 2D");
        size_t min_dim = std::min(_shape[0], _shape[1]);
        Tensor result({min_dim});
        for (size_t i = 0; i < min_dim; ++i) {
            result.at({i}) = at({i, i});
        }
        return result;
    }

    // ===================================================================
    // Comparison operations
    // ===================================================================

    /// Check if all elements are equal
    bool equal(const Tensor& other) const
    {
        if (_shape != other._shape) return false;
        for (size_t i = 0; i < _numel; ++i) {
            if ((*_data)[_offset + i] != (*other._data)[other._offset + i]) {
                return false;
            }
        }
        return true;
    }

    /// Check approximate equality with tolerance
    bool allclose(const Tensor& other, T rtol = T(1e-5), T atol = T(1e-8)) const
    {
        if (_shape != other._shape) return false;
        for (size_t i = 0; i < _numel; ++i) {
            T a = (*_data)[_offset + i];
            T b = (*other._data)[other._offset + i];
            if (std::abs(a - b) > atol + rtol * std::abs(b)) {
                return false;
            }
        }
        return true;
    }

    // ===================================================================
    // String representation
    // ===================================================================

    /// Convert to string representation
    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < _shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << _shape[i];
        }
        oss << "], data=[";

        size_t max_display = 10;
        for (size_t i = 0; i < std::min(_numel, max_display); ++i) {
            if (i > 0) oss << ", ";
            oss << (*_data)[_offset + i];
        }
        if (_numel > max_display) {
            oss << ", ...";
        }
        oss << "])";
        return oss.str();
    }

    // ===================================================================
    // Iterators
    // ===================================================================

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin()
    {
        ensure_unique();
        return _data->begin() + _offset;
    }

    iterator end()
    {
        ensure_unique();
        return _data->begin() + _offset + _numel;
    }

    const_iterator begin() const { return _data->cbegin() + _offset; }
    const_iterator end() const { return _data->cbegin() + _offset + _numel; }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
};

// ===================================================================
// Free functions (ATen-style global operations)
// ===================================================================

/// Scalar-tensor addition
template<typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& tensor)
{
    return tensor + scalar;
}

/// Scalar-tensor subtraction
template<typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& tensor)
{
    Tensor<T> result(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); ++i) {
        result.data()[i] = scalar - tensor.data()[i];
    }
    return result;
}

/// Scalar-tensor multiplication
template<typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& tensor)
{
    return tensor * scalar;
}

/// Scalar-tensor division
template<typename T>
Tensor<T> operator/(T scalar, const Tensor<T>& tensor)
{
    Tensor<T> result(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); ++i) {
        result.data()[i] = scalar / tensor.data()[i];
    }
    return result;
}

/// Stream output
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    os << tensor.to_string();
    return os;
}

// ===================================================================
// ATen-style free functions
// ===================================================================

/// Create zeros tensor
template<typename T = double>
Tensor<T> zeros(const typename Tensor<T>::shape_type& shape)
{
    return Tensor<T>::zeros(shape);
}

/// Create ones tensor
template<typename T = double>
Tensor<T> ones(const typename Tensor<T>::shape_type& shape)
{
    return Tensor<T>::ones(shape);
}

/// Create random tensor
template<typename T = double>
Tensor<T> rand(const typename Tensor<T>::shape_type& shape)
{
    return Tensor<T>::rand(shape);
}

/// Create random normal tensor
template<typename T = double>
Tensor<T> randn(const typename Tensor<T>::shape_type& shape)
{
    return Tensor<T>::randn(shape);
}

/// Create identity matrix
template<typename T = double>
Tensor<T> eye(size_t n)
{
    return Tensor<T>::eye(n);
}

/// Concatenate tensors along a dimension
template<typename T>
Tensor<T> cat(const std::vector<Tensor<T>>& tensors, size_t dim = 0)
{
    OC_ASSERT(!tensors.empty(), "cat: tensor list cannot be empty");

    const auto& first_shape = tensors[0].shape();
    OC_ASSERT(dim < first_shape.size(), "cat: dimension out of range");

    // Calculate new shape
    typename Tensor<T>::shape_type new_shape = first_shape;
    for (size_t i = 1; i < tensors.size(); ++i) {
        OC_ASSERT(tensors[i].shape().size() == first_shape.size(),
                  "cat: all tensors must have same number of dimensions");
        for (size_t d = 0; d < first_shape.size(); ++d) {
            if (d == dim) {
                new_shape[d] += tensors[i].shape()[d];
            } else {
                OC_ASSERT(tensors[i].shape()[d] == first_shape[d],
                          "cat: all tensors must match in non-cat dimensions");
            }
        }
    }

    Tensor<T> result(new_shape);

    // Copy data
    size_t offset_in_dim = 0;
    for (const auto& t : tensors) {
        std::vector<size_t> src_idx(t.ndim(), 0);
        std::vector<size_t> dst_idx(t.ndim(), 0);

        for (size_t i = 0; i < t.numel(); ++i) {
            // Convert linear to multi-index
            size_t tmp = i;
            for (int d = static_cast<int>(t.ndim()) - 1; d >= 0; --d) {
                src_idx[d] = tmp % t.size(d);
                tmp /= t.size(d);
            }
            // Compute destination index
            dst_idx = src_idx;
            dst_idx[dim] += offset_in_dim;
            result.at(dst_idx) = t.at(src_idx);
        }
        offset_in_dim += t.size(dim);
    }

    return result;
}

/// Stack tensors along a new dimension
template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, size_t dim = 0)
{
    OC_ASSERT(!tensors.empty(), "stack: tensor list cannot be empty");

    std::vector<Tensor<T>> expanded;
    for (const auto& t : tensors) {
        expanded.push_back(t.unsqueeze(dim));
    }
    return cat(expanded, dim);
}

/// Matrix multiplication (free function)
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b)
{
    return a.matmul(b);
}

/// Softmax along dimension
template<typename T>
Tensor<T> softmax(const Tensor<T>& input, size_t dim = 0)
{
    // Simplified 1D softmax for now
    OC_ASSERT(input.ndim() == 1, "softmax: only 1D tensors supported currently");

    T max_val = input.max();
    Tensor<T> exp_vals = (input - max_val).exp();
    T sum_exp = exp_vals.sum();
    return exp_vals / sum_exp;
}

/// Cross-entropy loss (for 1D probability distributions)
template<typename T>
T cross_entropy(const Tensor<T>& input, const Tensor<T>& target)
{
    OC_ASSERT(input.shape() == target.shape(),
              "cross_entropy: shape mismatch");
    // -sum(target * log(input))
    return -(target * input.log()).sum();
}

/// Mean squared error
template<typename T>
T mse_loss(const Tensor<T>& input, const Tensor<T>& target)
{
    OC_ASSERT(input.shape() == target.shape(), "mse_loss: shape mismatch");
    return (input - target).square().mean();
}

// ===================================================================
// Type aliases for convenience
// ===================================================================

using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;
using IntTensor = Tensor<int>;
using LongTensor = Tensor<long>;

} // ~namespace opencog

/** @}*/

#endif // _OPENCOG_TENSOR_H
