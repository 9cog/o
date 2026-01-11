/*
 * opencog/atoms/aten/ATenValue.cc
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

#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencog/util/exceptions.h>
#include <opencog/atoms/aten/ATenValue.h>
#include <opencog/atoms/value/ValueFactory.h>

using namespace opencog;

// ==============================================================
// Constructors

ATenValue::ATenValue()
	: Value(ATEN_VALUE)
{
#ifdef HAVE_ATEN
	_tensor = at::zeros({});
#else
	_data = {0.0};
	_shape = {};
#endif
}

ATenValue::ATenValue(const std::vector<double>& data,
                     const std::vector<int64_t>& shape)
	: Value(ATEN_VALUE)
{
#ifdef HAVE_ATEN
	// Create tensor from data with given shape
	auto options = at::TensorOptions().dtype(at::kDouble);
	_tensor = at::from_blob(
		const_cast<double*>(data.data()),
		shape,
		options
	).clone();
#else
	_data = data;
	_shape = shape;

	// Validate shape matches data size
	size_t expected_size = 1;
	for (auto dim : shape) expected_size *= dim;
	if (expected_size != data.size() && !shape.empty())
		throw RuntimeException(TRACE_INFO,
			"ATenValue: data size %zu doesn't match shape", data.size());
#endif
}

ATenValue::ATenValue(const std::vector<int64_t>& shape)
	: Value(ATEN_VALUE)
{
#ifdef HAVE_ATEN
	_tensor = at::zeros(shape, at::kDouble);
#else
	size_t total = 1;
	for (auto dim : shape) total *= dim;
	_data.resize(total, 0.0);
	_shape = shape;
#endif
}

#ifdef HAVE_ATEN
ATenValue::ATenValue(const at::Tensor& tensor)
	: Value(ATEN_VALUE), _tensor(tensor.to(at::kDouble))
{
}

ATenValue::ATenValue(at::Tensor&& tensor)
	: Value(ATEN_VALUE), _tensor(std::move(tensor))
{
	if (_tensor.dtype() != at::kDouble)
		_tensor = _tensor.to(at::kDouble);
}
#endif

// ==============================================================
// Tensor properties

std::vector<int64_t> ATenValue::shape() const
{
#ifdef HAVE_ATEN
	auto sizes = _tensor.sizes();
	return std::vector<int64_t>(sizes.begin(), sizes.end());
#else
	return _shape;
#endif
}

size_t ATenValue::ndim() const
{
#ifdef HAVE_ATEN
	return _tensor.dim();
#else
	return _shape.size();
#endif
}

size_t ATenValue::numel() const
{
#ifdef HAVE_ATEN
	return _tensor.numel();
#else
	return _data.size();
#endif
}

// ==============================================================
// Data access

std::vector<double> ATenValue::to_vector() const
{
#ifdef HAVE_ATEN
	auto flat = _tensor.flatten().contiguous();
	const double* data_ptr = flat.data_ptr<double>();
	return std::vector<double>(data_ptr, data_ptr + flat.numel());
#else
	return _data;
#endif
}

double ATenValue::item() const
{
#ifdef HAVE_ATEN
	if (_tensor.numel() != 1)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::item() requires scalar tensor");
	return _tensor.item<double>();
#else
	if (_data.size() != 1)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::item() requires scalar tensor");
	return _data[0];
#endif
}

// ==============================================================
// Tensor operations

ValuePtr ATenValue::add(const ATenValue& other) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor + other._tensor);
#else
	if (_shape != other._shape)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::add: shape mismatch");
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] + other._data[i];
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::sub(const ATenValue& other) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor - other._tensor);
#else
	if (_shape != other._shape)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::sub: shape mismatch");
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] - other._data[i];
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::mul(const ATenValue& other) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor * other._tensor);
#else
	if (_shape != other._shape)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::mul: shape mismatch");
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] * other._data[i];
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::div(const ATenValue& other) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor / other._tensor);
#else
	if (_shape != other._shape)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::div: shape mismatch");
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] / other._data[i];
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::matmul(const ATenValue& other) const
{
#ifdef HAVE_ATEN
	return createATenValue(at::matmul(_tensor, other._tensor));
#else
	// Simple 2D matrix multiplication fallback
	if (_shape.size() != 2 || other._shape.size() != 2)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::matmul: requires 2D tensors (without ATen)");

	int64_t m = _shape[0];
	int64_t k = _shape[1];
	int64_t n = other._shape[1];

	if (k != other._shape[0])
		throw RuntimeException(TRACE_INFO,
			"ATenValue::matmul: inner dimensions must match");

	std::vector<double> result(m * n, 0.0);
	for (int64_t i = 0; i < m; i++) {
		for (int64_t j = 0; j < n; j++) {
			for (int64_t kk = 0; kk < k; kk++) {
				result[i * n + j] += _data[i * k + kk] * other._data[kk * n + j];
			}
		}
	}
	return createATenFromVector(result, {m, n});
#endif
}

ValuePtr ATenValue::transpose(int64_t dim0, int64_t dim1) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor.transpose(dim0, dim1).contiguous());
#else
	if (_shape.size() != 2 || dim0 > 1 || dim1 > 1)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::transpose: only 2D supported without ATen");

	int64_t rows = _shape[0];
	int64_t cols = _shape[1];
	std::vector<double> result(rows * cols);
	for (int64_t i = 0; i < rows; i++) {
		for (int64_t j = 0; j < cols; j++) {
			result[j * rows + i] = _data[i * cols + j];
		}
	}
	return createATenFromVector(result, {cols, rows});
#endif
}

ValuePtr ATenValue::reshape(const std::vector<int64_t>& new_shape) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor.reshape(new_shape).contiguous());
#else
	size_t new_total = 1;
	for (auto dim : new_shape) new_total *= dim;
	if (new_total != _data.size())
		throw RuntimeException(TRACE_INFO,
			"ATenValue::reshape: total elements must match");
	return createATenFromVector(_data, new_shape);
#endif
}

ValuePtr ATenValue::sum() const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor.sum());
#else
	double s = std::accumulate(_data.begin(), _data.end(), 0.0);
	return createATenFromVector({s}, {});
#endif
}

ValuePtr ATenValue::mean() const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor.mean());
#else
	if (_data.empty()) return createATenFromVector({0.0}, {});
	double s = std::accumulate(_data.begin(), _data.end(), 0.0);
	return createATenFromVector({s / _data.size()}, {});
#endif
}

ValuePtr ATenValue::relu() const
{
#ifdef HAVE_ATEN
	return createATenValue(at::relu(_tensor));
#else
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = std::max(0.0, _data[i]);
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::sigmoid() const
{
#ifdef HAVE_ATEN
	return createATenValue(at::sigmoid(_tensor));
#else
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = 1.0 / (1.0 + std::exp(-_data[i]));
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::tanh() const
{
#ifdef HAVE_ATEN
	return createATenValue(at::tanh(_tensor));
#else
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = std::tanh(_data[i]);
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::softmax(int64_t dim) const
{
#ifdef HAVE_ATEN
	return createATenValue(at::softmax(_tensor, dim));
#else
	// Simple 1D softmax fallback
	if (_shape.size() != 1 && dim != -1)
		throw RuntimeException(TRACE_INFO,
			"ATenValue::softmax: only 1D supported without ATen");

	double max_val = *std::max_element(_data.begin(), _data.end());
	std::vector<double> exp_vals(_data.size());
	double sum_exp = 0.0;
	for (size_t i = 0; i < _data.size(); i++) {
		exp_vals[i] = std::exp(_data[i] - max_val);
		sum_exp += exp_vals[i];
	}
	for (size_t i = 0; i < _data.size(); i++)
		exp_vals[i] /= sum_exp;
	return createATenFromVector(exp_vals, _shape);
#endif
}

// ==============================================================
// Scalar operations

ValuePtr ATenValue::add_scalar(double scalar) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor + scalar);
#else
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] + scalar;
	return createATenFromVector(result, _shape);
#endif
}

ValuePtr ATenValue::mul_scalar(double scalar) const
{
#ifdef HAVE_ATEN
	return createATenValue(_tensor * scalar);
#else
	std::vector<double> result(_data.size());
	for (size_t i = 0; i < _data.size(); i++)
		result[i] = _data[i] * scalar;
	return createATenFromVector(result, _shape);
#endif
}

// ==============================================================
// String representation

std::string ATenValue::to_string(const std::string& indent, Type t) const
{
	std::string rv = indent + "(" + nameserver().getTypeName(t);

	// Add shape info
	rv += " shape=(";
	auto shp = shape();
	for (size_t i = 0; i < shp.size(); i++) {
		if (i > 0) rv += ",";
		rv += std::to_string(shp[i]);
	}
	rv += ")";

	// Add data preview (first few elements)
	auto vec = to_vector();
	rv += " data=[";
	size_t max_show = std::min((size_t)6, vec.size());
	for (size_t i = 0; i < max_show; i++) {
		if (i > 0) rv += ", ";
		char buf[32];
		snprintf(buf, sizeof(buf), "%.6g", vec[i]);
		rv += buf;
	}
	if (vec.size() > max_show)
		rv += ", ...";
	rv += "]";

	rv += ")";
	return rv;
}

// ==============================================================
// Comparison operators

bool ATenValue::operator==(const Value& other) const
{
	if (not other.is_type(ATEN_VALUE)) return false;

	const ATenValue* aov = static_cast<const ATenValue*>(&other);

	if (shape() != aov->shape()) return false;

	auto vec1 = to_vector();
	auto vec2 = aov->to_vector();

	if (vec1.size() != vec2.size()) return false;

	for (size_t i = 0; i < vec1.size(); i++) {
		// Use tolerance for floating point comparison
		if (std::abs(vec1[i] - vec2[i]) > 1e-10)
			return false;
	}
	return true;
}

bool ATenValue::operator<(const Value& other) const
{
	if (_type != other.get_type())
		return nameserver().getTypeName(_type) <
		       nameserver().getTypeName(other.get_type());

	const ATenValue* aov = static_cast<const ATenValue*>(&other);

	// Compare by shape first
	auto shp1 = shape();
	auto shp2 = aov->shape();
	if (shp1 != shp2)
		return shp1 < shp2;

	// Then by data
	return to_vector() < aov->to_vector();
}

// ==============================================================
// Convenience factory functions

ATenValuePtr opencog::createATenZeros(const std::vector<int64_t>& shape)
{
	return std::make_shared<ATenValue>(shape);
}

ATenValuePtr opencog::createATenOnes(const std::vector<int64_t>& shape)
{
#ifdef HAVE_ATEN
	return std::make_shared<ATenValue>(at::ones(shape, at::kDouble));
#else
	size_t total = 1;
	for (auto dim : shape) total *= dim;
	std::vector<double> data(total, 1.0);
	return std::make_shared<ATenValue>(data, shape);
#endif
}

ATenValuePtr opencog::createATenRandom(const std::vector<int64_t>& shape)
{
#ifdef HAVE_ATEN
	return std::make_shared<ATenValue>(at::rand(shape, at::kDouble));
#else
	size_t total = 1;
	for (auto dim : shape) total *= dim;
	std::vector<double> data(total);
	for (size_t i = 0; i < total; i++)
		data[i] = static_cast<double>(rand()) / RAND_MAX;
	return std::make_shared<ATenValue>(data, shape);
#endif
}

ATenValuePtr opencog::createATenFromVector(const std::vector<double>& data,
                                           const std::vector<int64_t>& shape)
{
	return std::make_shared<ATenValue>(data, shape);
}

// ==============================================================
// Factory registration

DEFINE_VALUE_FACTORY(ATEN_VALUE,
                     createATenValue, std::vector<double>, std::vector<int64_t>)
