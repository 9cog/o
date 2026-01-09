/*
 * TensorValue.h
 *
 * ATen Tensor Value type for AtomSpace integration
 * Copyright (c) 2025 OpenCog Foundation
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

#ifndef _OPENCOG_TENSOR_VALUE_H
#define _OPENCOG_TENSOR_VALUE_H

#ifdef HAVE_ATEN

#include <string>
#include <vector>
#include <memory>

#include <opencog/atoms/value/Value.h>
#include <opencog/atoms/value/FloatValue.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace opencog {

/**
 * TensorValue - An AtomSpace Value wrapping a PyTorch ATen tensor.
 *
 * This class bridges the gap between OpenCog's AtomSpace knowledge graph
 * and PyTorch's tensor computation framework. It allows tensors to be
 * stored as Values attached to Atoms, enabling integration of neural
 * network computations with symbolic AI reasoning.
 *
 * Key features:
 * - Wraps at::Tensor with AtomSpace Value semantics
 * - Supports GPU/CPU tensors transparently
 * - Serialization for distributed AtomSpaces
 * - Conversion to/from FloatValue for compatibility
 *
 * Usage:
 *   // Create a tensor value
 *   at::Tensor t = torch::zeros({3, 4});
 *   TensorValuePtr tv = createTensorValue(t);
 *
 *   // Attach to an atom
 *   atom->setValue(key, tv);
 *
 *   // Retrieve and use
 *   TensorValuePtr retrieved = TensorValueCast(atom->getValue(key));
 *   at::Tensor tensor = retrieved->tensor();
 */
class TensorValue : public Value
{
protected:
	at::Tensor _tensor;
	mutable std::string _cached_string;

	// Internal constructor for subclasses
	TensorValue(Type t) : Value(t) {}

	void update_string() const;

public:
	/**
	 * Construct a TensorValue from an existing ATen tensor.
	 * The tensor is stored by value (shared reference semantics).
	 */
	TensorValue(const at::Tensor& tensor);

	/**
	 * Construct a new tensor with given shape and data type.
	 * @param shape Vector of dimensions
	 * @param dtype Tensor data type (e.g., torch::kFloat32, torch::kFloat64)
	 * @param device Device to create tensor on (default: CPU)
	 */
	TensorValue(const std::vector<int64_t>& shape,
	            at::ScalarType dtype = at::kFloat,
	            at::Device device = at::kCPU);

	/**
	 * Construct from raw data.
	 * @param data Pointer to data
	 * @param shape Tensor dimensions
	 * @param dtype Data type
	 */
	TensorValue(const void* data,
	            const std::vector<int64_t>& shape,
	            at::ScalarType dtype = at::kFloat);

	virtual ~TensorValue();

	/** Get the underlying ATen tensor */
	const at::Tensor& tensor() const { return _tensor; }
	at::Tensor& tensor() { return _tensor; }

	/** Get tensor shape as vector */
	std::vector<int64_t> shape() const;

	/** Get number of dimensions */
	size_t ndim() const { return _tensor.dim(); }

	/** Get total number of elements */
	size_t numel() const { return _tensor.numel(); }

	/** Get data type */
	at::ScalarType dtype() const { return _tensor.scalar_type(); }

	/** Check if tensor is on GPU */
	bool is_cuda() const { return _tensor.is_cuda(); }

	/** Move tensor to CPU */
	TensorValue cpu() const;

	/** Move tensor to GPU (if available) */
	TensorValue cuda(int device_id = 0) const;

	/** Get device info as string */
	std::string device_string() const;

	// ---- Value interface implementation ----

	/** Compare for equality */
	virtual bool operator==(const Value&) const;

	/** Get string representation */
	virtual std::string to_string(const std::string& indent = "") const;

	/** Get short form string */
	virtual std::string to_short_string(const std::string& indent = "") const;

	// ---- Tensor operations ----

	/** Element-wise addition */
	TensorValue add(const TensorValue& other) const;

	/** Element-wise subtraction */
	TensorValue sub(const TensorValue& other) const;

	/** Element-wise multiplication */
	TensorValue mul(const TensorValue& other) const;

	/** Element-wise division */
	TensorValue div(const TensorValue& other) const;

	/** Matrix multiplication */
	TensorValue matmul(const TensorValue& other) const;

	/** Transpose */
	TensorValue transpose(int64_t dim0, int64_t dim1) const;

	/** Reshape tensor */
	TensorValue reshape(const std::vector<int64_t>& new_shape) const;

	/** Sum over all elements or specified dimension */
	TensorValue sum(int64_t dim = -1, bool keepdim = false) const;

	/** Mean over all elements or specified dimension */
	TensorValue mean(int64_t dim = -1, bool keepdim = false) const;

	/** Apply ReLU activation */
	TensorValue relu() const;

	/** Apply sigmoid activation */
	TensorValue sigmoid() const;

	/** Apply softmax over dimension */
	TensorValue softmax(int64_t dim = -1) const;

	/** Apply tanh activation */
	TensorValue tanh() const;

	// ---- Conversion utilities ----

	/**
	 * Convert to FloatValue (flattens tensor to 1D).
	 * Moves to CPU if necessary.
	 */
	FloatValuePtr to_float_value() const;

	/**
	 * Create TensorValue from FloatValue.
	 * @param fv Source FloatValue
	 * @param shape Desired tensor shape (product must equal fv size)
	 */
	static TensorValue from_float_value(const FloatValuePtr& fv,
	                                    const std::vector<int64_t>& shape);

	/**
	 * Get raw data as vector of doubles (for compatibility).
	 * Tensor is converted to double if needed.
	 */
	std::vector<double> to_vector() const;
};

typedef std::shared_ptr<TensorValue> TensorValuePtr;

/** Cast helper */
static inline TensorValuePtr TensorValueCast(const ValuePtr& v)
{
	return std::dynamic_pointer_cast<TensorValue>(v);
}

/** Factory function */
static inline TensorValuePtr createTensorValue(const at::Tensor& tensor)
{
	return std::make_shared<TensorValue>(tensor);
}

/** Factory function with shape */
static inline TensorValuePtr createTensorValue(
	const std::vector<int64_t>& shape,
	at::ScalarType dtype = at::kFloat,
	at::Device device = at::kCPU)
{
	return std::make_shared<TensorValue>(shape, dtype, device);
}

} // namespace opencog

#endif // HAVE_ATEN
#endif // _OPENCOG_TENSOR_VALUE_H
