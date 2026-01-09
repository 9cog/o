/*
 * TensorValue.cc
 *
 * ATen Tensor Value implementation
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

#ifdef HAVE_ATEN

#include <sstream>
#include <iomanip>
#include <cstring>

#include "TensorValue.h"

using namespace opencog;

// ============================================================
// Constructors

TensorValue::TensorValue(const at::Tensor& tensor)
	: Value(TENSOR_VALUE), _tensor(tensor)
{
}

TensorValue::TensorValue(const std::vector<int64_t>& shape,
                         at::ScalarType dtype,
                         at::Device device)
	: Value(TENSOR_VALUE)
{
	auto options = torch::TensorOptions().dtype(dtype).device(device);
	_tensor = torch::zeros(shape, options);
}

TensorValue::TensorValue(const void* data,
                         const std::vector<int64_t>& shape,
                         at::ScalarType dtype)
	: Value(TENSOR_VALUE)
{
	// Create tensor from raw data
	size_t elem_size = torch::elementSize(dtype);
	size_t numel = 1;
	for (auto d : shape) numel *= d;

	auto options = torch::TensorOptions().dtype(dtype);
	_tensor = torch::empty(shape, options);
	std::memcpy(_tensor.data_ptr(), data, numel * elem_size);
}

TensorValue::~TensorValue()
{
}

// ============================================================
// Accessors

std::vector<int64_t> TensorValue::shape() const
{
	std::vector<int64_t> result;
	for (int64_t i = 0; i < _tensor.dim(); i++)
		result.push_back(_tensor.size(i));
	return result;
}

TensorValue TensorValue::cpu() const
{
	return TensorValue(_tensor.cpu());
}

TensorValue TensorValue::cuda(int device_id) const
{
	if (torch::cuda::is_available())
		return TensorValue(_tensor.cuda(device_id));
	return TensorValue(_tensor);  // Stay on CPU if CUDA unavailable
}

std::string TensorValue::device_string() const
{
	std::ostringstream ss;
	ss << _tensor.device();
	return ss.str();
}

// ============================================================
// Value interface

bool TensorValue::operator==(const Value& other) const
{
	if (get_type() != other.get_type()) return false;

	const TensorValue* tv = dynamic_cast<const TensorValue*>(&other);
	if (nullptr == tv) return false;

	// Compare shapes first
	if (_tensor.sizes() != tv->_tensor.sizes()) return false;
	if (_tensor.scalar_type() != tv->_tensor.scalar_type()) return false;

	// Element-wise comparison
	return torch::equal(_tensor, tv->_tensor);
}

void TensorValue::update_string() const
{
	std::ostringstream ss;
	ss << "(TensorValue ";

	// Shape
	ss << "shape=[";
	auto sh = shape();
	for (size_t i = 0; i < sh.size(); i++) {
		ss << sh[i];
		if (i < sh.size() - 1) ss << ",";
	}
	ss << "] ";

	// Dtype
	ss << "dtype=" << _tensor.dtype() << " ";

	// Device
	ss << "device=" << device_string() << " ";

	// Data preview (first few elements)
	ss << "data=[";
	at::Tensor flat = _tensor.contiguous().view({-1}).to(at::kDouble).cpu();
	auto* ptr = flat.data_ptr<double>();
	size_t n = std::min((size_t)flat.numel(), (size_t)8);
	for (size_t i = 0; i < n; i++) {
		ss << std::fixed << std::setprecision(4) << ptr[i];
		if (i < n - 1) ss << ", ";
	}
	if ((size_t)flat.numel() > 8) ss << ", ...";
	ss << "]";

	ss << ")";
	_cached_string = ss.str();
}

std::string TensorValue::to_string(const std::string& indent) const
{
	update_string();
	return indent + _cached_string;
}

std::string TensorValue::to_short_string(const std::string& indent) const
{
	std::ostringstream ss;
	ss << indent << "(TensorValue [";
	auto sh = shape();
	for (size_t i = 0; i < sh.size(); i++) {
		ss << sh[i];
		if (i < sh.size() - 1) ss << "x";
	}
	ss << "] " << _tensor.dtype() << ")";
	return ss.str();
}

// ============================================================
// Tensor operations

TensorValue TensorValue::add(const TensorValue& other) const
{
	return TensorValue(_tensor + other._tensor);
}

TensorValue TensorValue::sub(const TensorValue& other) const
{
	return TensorValue(_tensor - other._tensor);
}

TensorValue TensorValue::mul(const TensorValue& other) const
{
	return TensorValue(_tensor * other._tensor);
}

TensorValue TensorValue::div(const TensorValue& other) const
{
	return TensorValue(_tensor / other._tensor);
}

TensorValue TensorValue::matmul(const TensorValue& other) const
{
	return TensorValue(torch::matmul(_tensor, other._tensor));
}

TensorValue TensorValue::transpose(int64_t dim0, int64_t dim1) const
{
	return TensorValue(_tensor.transpose(dim0, dim1));
}

TensorValue TensorValue::reshape(const std::vector<int64_t>& new_shape) const
{
	return TensorValue(_tensor.reshape(new_shape));
}

TensorValue TensorValue::sum(int64_t dim, bool keepdim) const
{
	if (dim < 0)
		return TensorValue(torch::sum(_tensor));
	return TensorValue(torch::sum(_tensor, dim, keepdim));
}

TensorValue TensorValue::mean(int64_t dim, bool keepdim) const
{
	if (dim < 0)
		return TensorValue(torch::mean(_tensor));
	return TensorValue(torch::mean(_tensor, dim, keepdim));
}

TensorValue TensorValue::relu() const
{
	return TensorValue(torch::relu(_tensor));
}

TensorValue TensorValue::sigmoid() const
{
	return TensorValue(torch::sigmoid(_tensor));
}

TensorValue TensorValue::softmax(int64_t dim) const
{
	if (dim < 0) dim = _tensor.dim() - 1;
	return TensorValue(torch::softmax(_tensor, dim));
}

TensorValue TensorValue::tanh() const
{
	return TensorValue(torch::tanh(_tensor));
}

// ============================================================
// Conversion utilities

FloatValuePtr TensorValue::to_float_value() const
{
	// Flatten and convert to double on CPU
	at::Tensor flat = _tensor.contiguous().view({-1}).to(at::kDouble).cpu();
	auto* ptr = flat.data_ptr<double>();
	std::vector<double> data(ptr, ptr + flat.numel());
	return createFloatValue(data);
}

TensorValue TensorValue::from_float_value(const FloatValuePtr& fv,
                                          const std::vector<int64_t>& shape)
{
	const std::vector<double>& data = fv->value();

	// Verify shape matches data size
	size_t expected = 1;
	for (auto d : shape) expected *= d;
	if (expected != data.size()) {
		throw std::runtime_error(
			"Shape mismatch: expected " + std::to_string(expected) +
			" elements, got " + std::to_string(data.size()));
	}

	auto options = torch::TensorOptions().dtype(at::kDouble);
	at::Tensor t = torch::from_blob(
		const_cast<double*>(data.data()),
		{(int64_t)data.size()},
		options
	).clone().reshape(shape);

	return TensorValue(t);
}

std::vector<double> TensorValue::to_vector() const
{
	at::Tensor flat = _tensor.contiguous().view({-1}).to(at::kDouble).cpu();
	auto* ptr = flat.data_ptr<double>();
	return std::vector<double>(ptr, ptr + flat.numel());
}

#endif // HAVE_ATEN
