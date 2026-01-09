/*
 * opencog/atoms/aten/ATenValue.h
 *
 * Copyright (C) 2024 OpenCog Foundation
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

#ifndef _OPENCOG_ATEN_VALUE_H
#define _OPENCOG_ATEN_VALUE_H

#include <vector>
#include <memory>
#include <opencog/atoms/value/Value.h>
#include <opencog/atoms/atom_types/atom_types.h>

#ifdef HAVE_ATEN
#include <ATen/ATen.h>
#endif

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * ATenValue wraps a PyTorch ATen tensor as an OpenCog Value.
 * This allows tensor data to be attached to Atoms and manipulated
 * through the AtomSpace framework.
 *
 * When ATen is not available, this class falls back to a simple
 * vector-based tensor representation that supports basic operations.
 */
class ATenValue : public Value
{
protected:
#ifdef HAVE_ATEN
	at::Tensor _tensor;
#else
	// Fallback: store tensor data as a flat vector with shape info
	std::vector<double> _data;
	std::vector<int64_t> _shape;
#endif

	std::string to_string(const std::string&, Type) const;

public:
	// Constructors
	ATenValue();
	ATenValue(const std::vector<double>& data, const std::vector<int64_t>& shape);
	ATenValue(const std::vector<int64_t>& shape); // Creates zeros tensor

#ifdef HAVE_ATEN
	ATenValue(const at::Tensor& tensor);
	ATenValue(at::Tensor&& tensor);
#endif

	virtual ~ATenValue() {}

	// Tensor properties
	std::vector<int64_t> shape() const;
	size_t ndim() const;
	size_t numel() const;
	virtual size_t size() const override { return numel(); }

	// Data access
	std::vector<double> to_vector() const;
	double item() const; // For scalar tensors

#ifdef HAVE_ATEN
	const at::Tensor& tensor() const { return _tensor; }
	at::Tensor& tensor() { return _tensor; }
#endif

	// Tensor operations - return new ATenValue
	ValuePtr add(const ATenValue&) const;
	ValuePtr sub(const ATenValue&) const;
	ValuePtr mul(const ATenValue&) const;
	ValuePtr div(const ATenValue&) const;
	ValuePtr matmul(const ATenValue&) const;
	ValuePtr transpose(int64_t dim0, int64_t dim1) const;
	ValuePtr reshape(const std::vector<int64_t>& new_shape) const;
	ValuePtr sum() const;
	ValuePtr mean() const;
	ValuePtr relu() const;
	ValuePtr sigmoid() const;
	ValuePtr tanh() const;
	ValuePtr softmax(int64_t dim) const;

	// Scalar operations
	ValuePtr add_scalar(double scalar) const;
	ValuePtr mul_scalar(double scalar) const;

	/** Returns a string representation of the tensor. */
	virtual std::string to_string(const std::string& indent = "") const override
	{ return to_string(indent, _type); }

	/** Returns true if two tensors are equal. */
	virtual bool operator==(const Value&) const override;

	/** Ordering for set insertion */
	virtual bool operator<(const Value& other) const override;
};

VALUE_PTR_DECL(ATenValue)
CREATE_VALUE_DECL(ATenValue)

// Convenience factory functions
ATenValuePtr createATenZeros(const std::vector<int64_t>& shape);
ATenValuePtr createATenOnes(const std::vector<int64_t>& shape);
ATenValuePtr createATenRandom(const std::vector<int64_t>& shape);
ATenValuePtr createATenFromVector(const std::vector<double>& data,
                                   const std::vector<int64_t>& shape);

/** @}*/
} // namespace opencog

#endif // _OPENCOG_ATEN_VALUE_H
