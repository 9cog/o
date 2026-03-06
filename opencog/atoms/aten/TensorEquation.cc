/*
 * opencog/atoms/aten/TensorEquation.cc
 *
 * Copyright (C) 2025 OpenCog Foundation
 * All Rights Reserved
 *
 * Implementation of Tensor Logic as described by Pedro Domingos
 * (https://tensor-logic.org, arXiv:2510.12269)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#include <algorithm>
#include <cmath>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

#include <opencog/util/exceptions.h>
#include <opencog/util/oc_assert.h>
#include <opencog/atoms/atom_types/NameServer.h>
#include <opencog/atomspace/AtomSpace.h>

#include "TensorEquation.h"

using namespace opencog;

// ============================================================
// EinsumSpec Implementation
// ============================================================

EinsumSpec::EinsumSpec(const std::string& notation)
	: _notation(notation), _implicit_output(true)
{
	parse(notation);
}

void EinsumSpec::parse(const std::string& notation)
{
	// Parse Einstein summation notation
	// Format: "input1,input2,...->output" or "input1,input2,..."

	size_t arrow_pos = notation.find("->");
	std::string inputs_str;

	if (arrow_pos != std::string::npos)
	{
		inputs_str = notation.substr(0, arrow_pos);
		_output_spec = notation.substr(arrow_pos + 2);
		_implicit_output = false;
	}
	else
	{
		inputs_str = notation;
		_implicit_output = true;
	}

	// Parse input specifications (comma-separated)
	std::stringstream ss(inputs_str);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		// Trim whitespace
		item.erase(0, item.find_first_not_of(" \t"));
		item.erase(item.find_last_not_of(" \t") + 1);
		if (!item.empty())
			_input_specs.push_back(item);
	}

	// Collect all indices from inputs
	std::set<char> all_indices;
	std::map<char, int> index_count;

	for (const auto& spec : _input_specs)
	{
		for (char c : spec)
		{
			if (std::isalpha(c))
			{
				all_indices.insert(c);
				index_count[c]++;
			}
		}
	}

	// Determine output indices
	if (_implicit_output)
	{
		// Implicit: output contains indices that appear exactly once
		for (const auto& [idx, count] : index_count)
		{
			if (count == 1)
				_output_indices.push_back(idx);
		}
		// Sort for deterministic output
		std::sort(_output_indices.begin(), _output_indices.end());

		// Build output spec string
		_output_spec = std::string(_output_indices.begin(), _output_indices.end());
	}
	else
	{
		// Explicit output
		for (char c : _output_spec)
		{
			if (std::isalpha(c))
				_output_indices.push_back(c);
		}
	}

	// Indices to sum over are those not in output
	std::set<char> output_set(_output_indices.begin(), _output_indices.end());
	for (char idx : all_indices)
	{
		if (output_set.find(idx) == output_set.end())
			_sum_indices.push_back(idx);
	}
}

bool EinsumSpec::validate(const std::vector<ATenValuePtr>& tensors) const
{
	if (tensors.size() != _input_specs.size())
		return false;

	// Check that tensor dimensions match subscript counts
	for (size_t i = 0; i < tensors.size(); i++)
	{
		if (!tensors[i]) return false;

		size_t expected_dims = 0;
		for (char c : _input_specs[i])
		{
			if (std::isalpha(c)) expected_dims++;
		}

		if (tensors[i]->ndim() != expected_dims)
			return false;
	}

	return true;
}

std::vector<int64_t> EinsumSpec::output_shape(
	const std::vector<ATenValuePtr>& tensors) const
{
	// Build index -> dimension size mapping
	std::map<char, int64_t> index_sizes;

	for (size_t i = 0; i < tensors.size() && i < _input_specs.size(); i++)
	{
		const auto& shape = tensors[i]->shape();
		size_t dim_idx = 0;
		for (char c : _input_specs[i])
		{
			if (std::isalpha(c))
			{
				if (dim_idx < shape.size())
				{
					if (index_sizes.find(c) != index_sizes.end())
					{
						// Check consistency
						if (index_sizes[c] != shape[dim_idx])
							throw RuntimeException(TRACE_INFO,
								"Einsum dimension mismatch for index '%c'", c);
					}
					else
					{
						index_sizes[c] = shape[dim_idx];
					}
				}
				dim_idx++;
			}
		}
	}

	// Build output shape from output indices
	std::vector<int64_t> result;
	for (char c : _output_indices)
	{
		if (index_sizes.find(c) != index_sizes.end())
			result.push_back(index_sizes[c]);
		else
			throw RuntimeException(TRACE_INFO,
				"Output index '%c' not found in inputs", c);
	}

	return result;
}

std::string EinsumSpec::to_string() const
{
	return _notation;
}

// ============================================================
// Nonlinearity Functions
// ============================================================

ATenValuePtr opencog::apply_nonlinearity(const ATenValuePtr& tensor,
                                          Nonlinearity nl)
{
	if (!tensor) return nullptr;

	switch (nl)
	{
		case Nonlinearity::NONE:
			return tensor;

		case Nonlinearity::THRESHOLD:
		{
			// Boolean mode: threshold at 0.5
			auto data = tensor->to_vector();
			for (auto& x : data)
				x = (x > 0.5) ? 1.0 : 0.0;
			return createATenFromVector(data, tensor->shape());
		}

		case Nonlinearity::SIGMOID:
			return ATenValueCast(tensor->sigmoid());

		case Nonlinearity::RELU:
			return ATenValueCast(tensor->relu());

		case Nonlinearity::TANH:
			return ATenValueCast(tensor->tanh());

		case Nonlinearity::SOFTMAX:
			return ATenValueCast(tensor->softmax(-1)); // Over last dimension

		case Nonlinearity::LOG:
		{
			auto data = tensor->to_vector();
			for (auto& x : data)
				x = std::log(std::max(x, 1e-10));
			return createATenFromVector(data, tensor->shape());
		}

		case Nonlinearity::EXP:
		{
			auto data = tensor->to_vector();
			for (auto& x : data)
				x = std::exp(x);
			return createATenFromVector(data, tensor->shape());
		}

		case Nonlinearity::CLAMP01:
		{
			auto data = tensor->to_vector();
			for (auto& x : data)
				x = std::max(0.0, std::min(1.0, x));
			return createATenFromVector(data, tensor->shape());
		}

		default:
			return tensor;
	}
}

// ============================================================
// Einsum Implementation
// ============================================================

/**
 * Core einsum implementation.
 *
 * This implements Einstein summation which is the foundation of Tensor Logic:
 * - Matrix multiplication: einsum("ij,jk->ik", A, B)
 * - Trace: einsum("ii->", A)
 * - Outer product: einsum("i,j->ij", a, b)
 * - Batched operations: einsum("bij,bjk->bik", A, B)
 *
 * In Tensor Logic terms:
 * - Logical AND = element-wise multiplication
 * - Existential quantification = summation over index
 */
ATenValuePtr opencog::einsum(const EinsumSpec& spec,
                              const std::vector<ATenValuePtr>& tensors)
{
	if (!spec.validate(tensors))
	{
		throw InvalidParamException(TRACE_INFO,
			"Invalid tensors for einsum '%s'", spec.notation().c_str());
	}

	// Get output shape
	auto output_shape = spec.output_shape(tensors);

	// Special case: matrix multiplication (most common)
	if (tensors.size() == 2 &&
	    (spec.notation() == "ij,jk->ik" ||
	     spec.notation() == "ij,jk"))
	{
		return ATenValueCast(tensors[0]->matmul(*tensors[1]));
	}

	// Special case: matrix-vector multiplication
	if (tensors.size() == 2 &&
	    (spec.notation() == "ij,j->i" || spec.notation() == "ij,j"))
	{
		// Treat vector as column matrix, then squeeze
		ATenValuePtr v_expanded = ATenValueCast(tensors[1]->reshape({-1, 1}));
		ATenValuePtr result = ATenValueCast(tensors[0]->matmul(*v_expanded));
		return ATenValueCast(result->reshape({-1}));
	}

	// Special case: transpose
	if (tensors.size() == 1 && spec.notation() == "ij->ji")
	{
		return ATenValueCast(tensors[0]->transpose(0, 1));
	}

	// Special case: trace
	if (tensors.size() == 1 && spec.notation() == "ii->")
	{
		auto data = tensors[0]->to_vector();
		auto shape = tensors[0]->shape();
		if (shape.size() != 2 || shape[0] != shape[1])
			throw InvalidParamException(TRACE_INFO, "Trace requires square matrix");

		double trace = 0;
		for (int64_t i = 0; i < shape[0]; i++)
			trace += data[i * shape[1] + i];

		return createATenFromVector(std::vector<double>{trace}, {});
	}

	// Special case: outer product
	if (tensors.size() == 2 &&
	    (spec.notation() == "i,j->ij" || spec.notation() == "i,j"))
	{
		auto a = tensors[0]->to_vector();
		auto b = tensors[1]->to_vector();
		std::vector<double> result(a.size() * b.size());

		for (size_t i = 0; i < a.size(); i++)
		{
			for (size_t j = 0; j < b.size(); j++)
			{
				result[i * b.size() + j] = a[i] * b[j];
			}
		}

		return createATenFromVector(result, {(int64_t)a.size(), (int64_t)b.size()});
	}

	// Special case: dot product (i,i->)
	if (tensors.size() == 2 &&
	    spec.input_specs()[0] == spec.input_specs()[1] &&
	    spec.output_spec().empty())
	{
		// Element-wise multiply then sum
		ATenValuePtr prod = ATenValueCast(tensors[0]->mul(*tensors[1]));
		return ATenValueCast(prod->sum());
	}

	// Special case: element-wise multiplication with broadcasting
	if (tensors.size() == 2 && spec.input_specs()[0] == spec.input_specs()[1])
	{
		return ATenValueCast(tensors[0]->mul(*tensors[1]));
	}

	// Special case: sum reduction
	if (tensors.size() == 1 && spec.output_spec().empty())
	{
		return ATenValueCast(tensors[0]->sum());
	}

	// General case: implement full einsum
	// This is a simplified implementation; a production version would use
	// optimized contraction order and loop tiling

	// Build index -> dimension mapping
	std::map<char, int64_t> index_sizes;
	for (size_t t = 0; t < tensors.size(); t++)
	{
		const auto& shape = tensors[t]->shape();
		size_t dim = 0;
		for (char c : spec.input_specs()[t])
		{
			if (std::isalpha(c) && dim < shape.size())
			{
				index_sizes[c] = shape[dim++];
			}
		}
	}

	// Compute total size of output
	int64_t output_size = 1;
	for (int64_t s : output_shape)
		output_size *= s;

	if (output_shape.empty())
		output_size = 1;

	// Compute total size of sum indices
	int64_t sum_size = 1;
	for (char c : spec.sum_indices())
	{
		if (index_sizes.find(c) != index_sizes.end())
			sum_size *= index_sizes[c];
	}

	// Allocate output
	std::vector<double> result(output_size, 0.0);

	// Get input data
	std::vector<std::vector<double>> input_data;
	for (const auto& t : tensors)
		input_data.push_back(t->to_vector());

	// Iterate over all output indices
	std::map<char, int64_t> index_values;

	std::function<void(size_t)> iterate_output;
	iterate_output = [&](size_t dim) {
		if (dim >= spec.output_indices().size())
		{
			// Compute output index
			int64_t out_idx = 0;
			int64_t stride = 1;
			for (int i = spec.output_indices().size() - 1; i >= 0; i--)
			{
				out_idx += index_values[spec.output_indices()[i]] * stride;
				stride *= index_sizes[spec.output_indices()[i]];
			}

			// Sum over contracted indices
			std::function<void(size_t)> iterate_sum;
			iterate_sum = [&](size_t sum_dim) {
				if (sum_dim >= spec.sum_indices().size())
				{
					// Compute product of all input elements
					double product = 1.0;
					for (size_t t = 0; t < tensors.size(); t++)
					{
						int64_t in_idx = 0;
						int64_t in_stride = 1;
						const auto& shape = tensors[t]->shape();
						const auto& spec_str = spec.input_specs()[t];

						for (int i = spec_str.size() - 1; i >= 0; i--)
						{
							char c = spec_str[i];
							if (std::isalpha(c))
							{
								in_idx += index_values[c] * in_stride;
								// Find dimension for this index
								size_t dim_pos = 0;
								for (int j = 0; j < i; j++)
									if (std::isalpha(spec_str[j])) dim_pos++;
								if (dim_pos < shape.size())
									in_stride *= shape[shape.size() - 1 - (spec_str.size() - 1 - dim_pos)];
							}
						}

						if (in_idx < (int64_t)input_data[t].size())
							product *= input_data[t][in_idx];
					}
					result[out_idx] += product;
				}
				else
				{
					char c = spec.sum_indices()[sum_dim];
					for (int64_t i = 0; i < index_sizes[c]; i++)
					{
						index_values[c] = i;
						iterate_sum(sum_dim + 1);
					}
				}
			};

			iterate_sum(0);
		}
		else
		{
			char c = spec.output_indices()[dim];
			for (int64_t i = 0; i < index_sizes[c]; i++)
			{
				index_values[c] = i;
				iterate_output(dim + 1);
			}
		}
	};

	if (spec.output_indices().empty())
	{
		// Scalar output
		std::function<void(size_t)> iterate_all;
		iterate_all = [&](size_t idx) {
			if (idx >= spec.sum_indices().size())
			{
				double product = 1.0;
				for (size_t t = 0; t < tensors.size(); t++)
				{
					int64_t in_idx = 0;
					int64_t in_stride = 1;
					const auto& shape = tensors[t]->shape();

					for (int d = shape.size() - 1; d >= 0; d--)
					{
						char c = spec.input_specs()[t][d];
						in_idx += index_values[c] * in_stride;
						in_stride *= shape[d];
					}

					if (in_idx < (int64_t)input_data[t].size())
						product *= input_data[t][in_idx];
				}
				result[0] += product;
			}
			else
			{
				char c = spec.sum_indices()[idx];
				for (int64_t i = 0; i < index_sizes[c]; i++)
				{
					index_values[c] = i;
					iterate_all(idx + 1);
				}
			}
		};
		iterate_all(0);
	}
	else
	{
		iterate_output(0);
	}

	return createATenFromVector(result, output_shape);
}

ATenValuePtr opencog::einsum(const std::string& notation,
                              const std::vector<ATenValuePtr>& tensors)
{
	EinsumSpec spec(notation);
	return opencog::einsum(spec, tensors);
}

// ============================================================
// TensorEquation Implementation
// ============================================================

TensorEquation::TensorEquation(const std::string& name,
                               const std::string& lhs_name,
                               const std::vector<std::string>& rhs_names,
                               const std::string& einsum_notation,
                               Nonlinearity nl,
                               ReasoningMode mode)
	: _name(name),
	  _lhs_name(lhs_name),
	  _rhs_names(rhs_names),
	  _einsum(einsum_notation),
	  _nonlinearity(nl),
	  _mode(mode),
	  _learnable(false)
{
}

ATenValuePtr TensorEquation::execute(
	const std::map<std::string, ATenValuePtr>& inputs) const
{
	// Gather input tensors in order
	std::vector<ATenValuePtr> tensors;
	for (const auto& name : _rhs_names)
	{
		auto it = inputs.find(name);
		if (it == inputs.end())
		{
			throw RuntimeException(TRACE_INFO,
				"Missing input tensor '%s' for equation '%s'",
				name.c_str(), _name.c_str());
		}
		tensors.push_back(it->second);
	}

	return execute(tensors);
}

ATenValuePtr TensorEquation::execute(
	const std::vector<ATenValuePtr>& tensors) const
{
	// Execute einsum - use free function with spec notation
	ATenValuePtr result = opencog::einsum(_einsum.notation(), tensors);

	// Apply learnable weight if present
	if (_weight)
	{
		result = ATenValueCast(result->mul(*_weight));
	}

	// Apply learnable bias if present
	if (_bias)
	{
		result = ATenValueCast(result->add(*_bias));
	}

	// Apply nonlinearity
	result = apply_nonlinearity(result, _nonlinearity);

	// For Boolean mode, threshold the result
	if (_mode == ReasoningMode::BOOLEAN)
	{
		result = apply_nonlinearity(result, Nonlinearity::THRESHOLD);
	}

	return result;
}

std::vector<ATenValuePtr> TensorEquation::backward(
	const ATenValuePtr& grad_output,
	const std::vector<ATenValuePtr>& inputs)
{
	if (!grad_output || inputs.empty()) return {};

	// Step 1: Re-execute einsum forward pass to get intermediate values
	ATenValuePtr einsum_result = opencog::einsum(_einsum.notation(), inputs);

	// Step 2: Backpropagate through nonlinearity
	// Recompute pre-activation value
	ATenValuePtr pre_activation = einsum_result;
	if (_weight)
		pre_activation = ATenValueCast(pre_activation->mul(*_weight));
	if (_bias)
		pre_activation = ATenValueCast(pre_activation->add(*_bias));

	// Compute gradient through the activation function
	auto pre_act_vec = pre_activation->to_vector();
	auto grad_vec = grad_output->to_vector();
	std::vector<double> grad_pre_act(grad_vec.size());

	Nonlinearity effective_nl = (_mode == ReasoningMode::BOOLEAN)
		? Nonlinearity::THRESHOLD : _nonlinearity;

	for (size_t i = 0; i < grad_vec.size() && i < pre_act_vec.size(); i++)
	{
		double x = pre_act_vec[i];
		double g = grad_vec[i];

		switch (effective_nl)
		{
		case Nonlinearity::SIGMOID:
			// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
			{
				double s = 1.0 / (1.0 + std::exp(-x));
				grad_pre_act[i] = g * s * (1.0 - s);
			}
			break;
		case Nonlinearity::RELU:
			// d/dx relu(x) = 1 if x > 0, else 0
			grad_pre_act[i] = (x > 0.0) ? g : 0.0;
			break;
		case Nonlinearity::TANH:
			// d/dx tanh(x) = 1 - tanh(x)^2
			{
				double t = std::tanh(x);
				grad_pre_act[i] = g * (1.0 - t * t);
			}
			break;
		case Nonlinearity::CLAMP01:
			// d/dx clamp(x, 0, 1) = 1 if 0 < x < 1, else 0
			grad_pre_act[i] = (x > 0.0 && x < 1.0) ? g : 0.0;
			break;
		case Nonlinearity::THRESHOLD:
			// Non-differentiable — use straight-through estimator (pass-through)
			grad_pre_act[i] = g;
			break;
		default:
			// NONE, LOG, EXP, SOFTMAX: pass-through
			grad_pre_act[i] = g;
			break;
		}
	}

	ATenValuePtr grad_before_nl = createATenFromVector(grad_pre_act, grad_output->shape());

	// Step 3: Gradient for bias — sum over any broadcast/batch dimensions.
	// For forward: out[..., d] = einsum[..., d] + bias[d]
	// Grad bias[d] = sum of all grad_pre_act elements mapped to d.
	if (_learnable && _bias)
	{
		auto bias_shape = _bias->shape();
		size_t bias_numel = _bias->numel();
		std::vector<double> bias_grad_vec(bias_numel, 0.0);

		if (grad_pre_act.size() == bias_numel)
		{
			bias_grad_vec = grad_pre_act;
		}
		else if (bias_numel == 1)
		{
			double sum = 0.0;
			for (double g : grad_pre_act) sum += g;
			bias_grad_vec[0] = sum;
		}
		else
		{
			// Sum (not average) over broadcast dimensions.
			for (size_t i = 0; i < grad_pre_act.size(); i++)
				bias_grad_vec[i % bias_numel] += grad_pre_act[i];
		}

		// Accumulate into stored bias gradient
		if (_bias_grad)
		{
			auto prev = _bias_grad->to_vector();
			for (size_t i = 0; i < bias_grad_vec.size() && i < prev.size(); i++)
				bias_grad_vec[i] += prev[i];
		}
		_bias_grad = createATenFromVector(bias_grad_vec, bias_shape);
	}

	// Step 4: Gradient for weight and gradient flowing into einsum
	ATenValuePtr grad_einsum = grad_before_nl;
	if (_weight)
	{
		if (_learnable)
		{
			// grad_weight[d] = sum of einsum_result[..., d] * grad_pre_act[..., d]
			// (sum, not average, over any broadcast dimensions)
			auto einsum_vec = einsum_result->to_vector();
			auto w_shape = _weight->shape();
			size_t w_numel = _weight->numel();
			std::vector<double> weight_grad_vec(w_numel, 0.0);

			if (w_numel == 1)
			{
				double sum = 0.0;
				for (size_t i = 0; i < einsum_vec.size() && i < grad_pre_act.size(); i++)
					sum += einsum_vec[i] * grad_pre_act[i];
				weight_grad_vec[0] = sum;
			}
			else if (w_numel == einsum_vec.size())
			{
				for (size_t i = 0; i < einsum_vec.size() && i < grad_pre_act.size(); i++)
					weight_grad_vec[i] = einsum_vec[i] * grad_pre_act[i];
			}
			else
			{
				// Sum over broadcast dimensions.
				for (size_t i = 0; i < einsum_vec.size() && i < grad_pre_act.size(); i++)
					weight_grad_vec[i % w_numel] += einsum_vec[i] * grad_pre_act[i];
			}

			// Accumulate into stored weight gradient
			if (_weight_grad)
			{
				auto prev = _weight_grad->to_vector();
				for (size_t i = 0; i < weight_grad_vec.size() && i < prev.size(); i++)
					weight_grad_vec[i] += prev[i];
			}
			_weight_grad = createATenFromVector(weight_grad_vec, w_shape);
		}

		// Gradient flowing into einsum: grad_einsum = grad_before_nl * weight
		auto w_vec = _weight->to_vector();
		auto g_vec = grad_before_nl->to_vector();
		std::vector<double> grad_einsum_vec(g_vec.size());

		if (w_vec.size() == 1)
		{
			for (size_t i = 0; i < g_vec.size(); i++)
				grad_einsum_vec[i] = g_vec[i] * w_vec[0];
		}
		else if (w_vec.size() == g_vec.size())
		{
			for (size_t i = 0; i < g_vec.size(); i++)
				grad_einsum_vec[i] = g_vec[i] * w_vec[i];
		}
		else
		{
			for (size_t i = 0; i < g_vec.size(); i++)
				grad_einsum_vec[i] = g_vec[i] * w_vec[i % w_vec.size()];
		}

		grad_einsum = createATenFromVector(grad_einsum_vec, grad_before_nl->shape());
	}

	// Step 5: Backpropagate through einsum for each input
	// For input i with spec s_i, gradient = einsum(out_spec, s_j (j≠i), -> s_i, grad, inputs[j≠i])
	std::vector<ATenValuePtr> input_grads;
	const auto& specs = _einsum.input_specs();
	const std::string& out_spec = _einsum.output_spec();

	for (size_t i = 0; i < inputs.size(); i++)
	{
		// Build transposed einsum notation: out_spec, s_j (j≠i) -> s_i
		std::string transposed = out_spec;
		for (size_t j = 0; j < inputs.size(); j++)
		{
			if (j != i)
				transposed += "," + specs[j];
		}
		transposed += "->" + specs[i];

		// Gather tensors: [grad_einsum, inputs[j≠i], ...]
		std::vector<ATenValuePtr> grad_tensors;
		grad_tensors.push_back(grad_einsum);
		for (size_t j = 0; j < inputs.size(); j++)
		{
			if (j != i)
				grad_tensors.push_back(inputs[j]);
		}

		try
		{
			ATenValuePtr input_grad = opencog::einsum(transposed, grad_tensors);
			input_grads.push_back(input_grad);
		}
		catch (...)
		{
			// Fallback: broadcast mean gradient if transposed einsum fails
			auto shape = inputs[i]->shape();
			std::vector<double> fallback(inputs[i]->numel(), 0.0);
			ATenValuePtr mean_val = ATenValueCast(grad_einsum->mean());
			double mean_g = mean_val->item();
			for (auto& g : fallback) g = mean_g;
			input_grads.push_back(createATenFromVector(fallback, shape));
		}
	}

	return input_grads;
}

void TensorEquation::zero_grad()
{
	_weight_grad = nullptr;
	_bias_grad = nullptr;
}

std::string TensorEquation::to_string() const
{
	std::stringstream ss;
	ss << _name << ": " << _lhs_name << " = ";

	if (_nonlinearity != Nonlinearity::NONE)
		ss << "H(";

	ss << "einsum('" << _einsum.notation() << "', ";
	for (size_t i = 0; i < _rhs_names.size(); i++)
	{
		if (i > 0) ss << ", ";
		ss << _rhs_names[i];
	}
	ss << ")";

	if (_nonlinearity != Nonlinearity::NONE)
		ss << ")";

	return ss.str();
}

// ============================================================
// TensorProgram Implementation
// ============================================================

TensorProgram::TensorProgram(const std::string& name, ReasoningMode mode)
	: _name(name),
	  _mode(mode),
	  _max_iterations(100),
	  _convergence_threshold(1e-6),
	  _learning_rate(0.01),
	  _track_gradients(false),
	  _forward_count(0),
	  _backward_count(0)
{
}

void TensorProgram::add_fact(const std::string& name,
                              const ATenValuePtr& tensor)
{
	_facts[name] = tensor;
}

ATenValuePtr TensorProgram::get_fact(const std::string& name) const
{
	auto it = _facts.find(name);
	if (it != _facts.end())
		return it->second;
	return nullptr;
}

bool TensorProgram::has_fact(const std::string& name) const
{
	return _facts.find(name) != _facts.end();
}

void TensorProgram::remove_fact(const std::string& name)
{
	_facts.erase(name);
}

void TensorProgram::clear_facts()
{
	_facts.clear();
}

std::vector<std::string> TensorProgram::fact_names() const
{
	std::vector<std::string> names;
	for (const auto& [name, _] : _facts)
		names.push_back(name);
	return names;
}

void TensorProgram::add_equation(const TensorEquationPtr& eq)
{
	_equations.push_back(eq);
}

void TensorProgram::add_equation(const std::string& name,
                                  const std::string& lhs,
                                  const std::vector<std::string>& rhs,
                                  const std::string& einsum,
                                  Nonlinearity nl)
{
	auto eq = std::make_shared<TensorEquation>(name, lhs, rhs, einsum, nl, _mode);
	_equations.push_back(eq);
}

TensorEquationPtr TensorProgram::get_equation(const std::string& name) const
{
	for (const auto& eq : _equations)
	{
		if (eq->name() == name)
			return eq;
	}
	return nullptr;
}

void TensorProgram::remove_equation(const std::string& name)
{
	_equations.erase(
		std::remove_if(_equations.begin(), _equations.end(),
			[&name](const TensorEquationPtr& eq) {
				return eq->name() == name;
			}),
		_equations.end());
}

void TensorProgram::forward()
{
	_forward_count++;

	for (const auto& eq : _equations)
	{
		// Gather inputs from facts and derived
		std::map<std::string, ATenValuePtr> inputs;
		bool all_inputs_found = true;

		for (const auto& name : eq->rhs_names())
		{
			ATenValuePtr tensor = get_tensor(name);
			if (tensor)
				inputs[name] = tensor;
			else
				all_inputs_found = false;
		}

		// Execute equation if all inputs are available
		if (all_inputs_found)
		{
			ATenValuePtr result = eq->execute(inputs);
			_derived[eq->lhs_name()] = result;
		}
	}
}

void TensorProgram::forward_to_fixpoint()
{
	for (size_t iter = 0; iter < _max_iterations; iter++)
	{
		// Save current derived values
		std::map<std::string, std::vector<double>> prev_values;
		for (const auto& [name, tensor] : _derived)
		{
			if (tensor)
				prev_values[name] = tensor->to_vector();
		}

		// Execute one forward pass
		forward();

		// Check for convergence
		double max_diff = 0.0;
		for (const auto& [name, tensor] : _derived)
		{
			if (tensor && prev_values.find(name) != prev_values.end())
			{
				auto curr = tensor->to_vector();
				const auto& prev = prev_values[name];

				if (curr.size() == prev.size())
				{
					for (size_t i = 0; i < curr.size(); i++)
					{
						max_diff = std::max(max_diff, std::abs(curr[i] - prev[i]));
					}
				}
			}
		}

		if (max_diff < _convergence_threshold)
			break;
	}
}

ATenValuePtr TensorProgram::get_derived(const std::string& name) const
{
	auto it = _derived.find(name);
	if (it != _derived.end())
		return it->second;
	return nullptr;
}

ATenValuePtr TensorProgram::get_tensor(const std::string& name) const
{
	// First check derived
	auto derived = get_derived(name);
	if (derived)
		return derived;

	// Then check facts
	return get_fact(name);
}

void TensorProgram::clear_derived()
{
	_derived.clear();
}

ATenValuePtr TensorProgram::query(const std::string& query_name,
                                   const std::vector<int64_t>& indices)
{
	// Find equations that derive the query
	auto eqs = find_deriving_equations(query_name);

	if (eqs.empty())
	{
		// Query is a fact
		return get_fact(query_name);
	}

	// Execute equations recursively (backward chaining)
	for (const auto& eq : eqs)
	{
		// Ensure all inputs are available
		for (const auto& rhs_name : eq->rhs_names())
		{
			if (!get_tensor(rhs_name))
			{
				// Recursively query for this input
				query(rhs_name, {});
			}
		}

		// Now execute the equation
		std::map<std::string, ATenValuePtr> inputs;
		for (const auto& name : eq->rhs_names())
		{
			ATenValuePtr tensor = get_tensor(name);
			if (tensor)
				inputs[name] = tensor;
		}

		if (inputs.size() == eq->rhs_names().size())
		{
			ATenValuePtr result = eq->execute(inputs);
			_derived[eq->lhs_name()] = result;
		}
	}

	return get_tensor(query_name);
}

std::vector<TensorEquationPtr> TensorProgram::find_deriving_equations(
	const std::string& relation_name) const
{
	std::vector<TensorEquationPtr> result;
	for (const auto& eq : _equations)
	{
		if (eq->lhs_name() == relation_name)
			result.push_back(eq);
	}
	return result;
}

double TensorProgram::compute_loss(const std::string& name,
                                    const ATenValuePtr& target) const
{
	ATenValuePtr derived = get_derived(name);
	if (!derived || !target)
		return 0.0;

	// Mean squared error
	ATenValuePtr diff = ATenValueCast(derived->sub(*target));
	ATenValuePtr squared = ATenValueCast(diff->mul(*diff));
	ATenValuePtr mean_val = ATenValueCast(squared->mean());
	return mean_val->item();
}

void TensorProgram::backward(const std::string& output_name,
                              const ATenValuePtr& grad_output)
{
	_backward_count++;

	if (!grad_output) return;

	// Accumulate gradient for this tensor
	auto it = _tensor_grads.find(output_name);
	if (it != _tensor_grads.end())
	{
		auto prev = it->second->to_vector();
		auto cur = grad_output->to_vector();
		for (size_t i = 0; i < prev.size() && i < cur.size(); i++)
			cur[i] += prev[i];
		_tensor_grads[output_name] = createATenFromVector(cur, grad_output->shape());
	}
	else
	{
		_tensor_grads[output_name] = grad_output;
	}

	// Find equations that produce this output and backpropagate through them
	auto eqs = find_deriving_equations(output_name);

	for (const auto& eq : eqs)
	{
		// Collect the input tensors used in the forward pass
		std::vector<ATenValuePtr> input_tensors;
		bool all_found = true;

		for (const auto& name : eq->rhs_names())
		{
			ATenValuePtr tensor = get_tensor(name);
			if (!tensor)
			{
				all_found = false;
				break;
			}
			input_tensors.push_back(tensor);
		}

		if (!all_found) continue;

		// Compute gradients through this equation (accumulates weight/bias grads)
		std::vector<ATenValuePtr> input_grads =
			eq->backward(grad_output, input_tensors);

		// Recursively propagate gradients to upstream equations
		for (size_t i = 0; i < input_grads.size() && i < eq->rhs_names().size(); i++)
		{
			const std::string& input_name = eq->rhs_names()[i];
			if (!input_grads[i]) continue;

			// Propagate into derived tensors; facts receive gradient but
			// we don't recurse further (they have no producing equations)
			if (has_fact(input_name))
			{
				auto& fg = _tensor_grads[input_name];
				if (fg)
				{
					auto prev = fg->to_vector();
					auto cur = input_grads[i]->to_vector();
					for (size_t j = 0; j < prev.size() && j < cur.size(); j++)
						cur[j] += prev[j];
					_tensor_grads[input_name] = createATenFromVector(cur, input_grads[i]->shape());
				}
				else
				{
					_tensor_grads[input_name] = input_grads[i];
				}
			}
			else
			{
				backward(input_name, input_grads[i]);
			}
		}
	}
}

void TensorProgram::update_parameters()
{
	for (auto& eq : _equations)
	{
		if (!eq->is_learnable()) continue;

		// Update weight using accumulated gradient
		if (eq->weight() && eq->weight_grad())
		{
			auto w_vec = eq->weight()->to_vector();
			auto g_vec = eq->weight_grad()->to_vector();
			auto w_shape = eq->weight()->shape();

			for (size_t i = 0; i < w_vec.size() && i < g_vec.size(); i++)
				w_vec[i] -= _learning_rate * g_vec[i];

			eq->set_weight(createATenFromVector(w_vec, w_shape));
		}

		// Update bias using accumulated gradient
		if (eq->bias() && eq->bias_grad())
		{
			auto b_vec = eq->bias()->to_vector();
			auto g_vec = eq->bias_grad()->to_vector();
			auto b_shape = eq->bias()->shape();

			for (size_t i = 0; i < b_vec.size() && i < g_vec.size(); i++)
				b_vec[i] -= _learning_rate * g_vec[i];

			eq->set_bias(createATenFromVector(b_vec, b_shape));
		}

		// Reset accumulated gradients for next iteration
		eq->zero_grad();
	}

	// Clear tensor gradients
	_tensor_grads.clear();
}

double TensorProgram::train(
	const std::map<std::string, ATenValuePtr>& inputs,
	const std::map<std::string, ATenValuePtr>& targets,
	size_t epochs)
{
	double final_loss = 0.0;

	for (size_t epoch = 0; epoch < epochs; epoch++)
	{
		// Set inputs as facts
		for (const auto& [name, tensor] : inputs)
			add_fact(name, tensor);

		// Forward pass
		forward_to_fixpoint();

		// Compute loss
		final_loss = 0.0;
		for (const auto& [name, target] : targets)
		{
			final_loss += compute_loss(name, target);
		}

		// Backward pass (simplified)
		for (const auto& [name, target] : targets)
		{
			ATenValuePtr derived = get_derived(name);
			if (derived && target)
			{
				ATenValuePtr grad = ATenValueCast(derived->sub(*target));
				backward(name, grad);
			}
		}

		// Update parameters
		update_parameters();

		// Clear for next iteration
		clear_derived();
	}

	return final_loss;
}

std::string TensorProgram::to_string() const
{
	std::stringstream ss;
	ss << "TensorProgram: " << _name << "\n";
	ss << "Mode: " << (int)_mode << "\n";
	ss << "Facts: " << _facts.size() << "\n";
	for (const auto& [name, tensor] : _facts)
	{
		ss << "  " << name << ": shape=[";
		auto shape = tensor->shape();
		for (size_t i = 0; i < shape.size(); i++)
		{
			if (i > 0) ss << ", ";
			ss << shape[i];
		}
		ss << "]\n";
	}
	ss << "Equations: " << _equations.size() << "\n";
	for (const auto& eq : _equations)
	{
		ss << "  " << eq->to_string() << "\n";
	}
	return ss.str();
}

std::string TensorProgram::to_datalog() const
{
	std::stringstream ss;
	for (const auto& eq : _equations)
	{
		ss << eq->lhs_name() << "(";
		// Add index variables based on output shape
		for (size_t i = 0; i < eq->einsum().output_indices().size(); i++)
		{
			if (i > 0) ss << ", ";
			ss << (char)('X' + i);
		}
		ss << ") :- ";

		for (size_t j = 0; j < eq->rhs_names().size(); j++)
		{
			if (j > 0) ss << ", ";
			ss << eq->rhs_names()[j] << "(";
			// Add index variables based on input spec
			const auto& spec = eq->einsum().input_specs()[j];
			bool first = true;
			for (char c : spec)
			{
				if (std::isalpha(c))
				{
					if (!first) ss << ", ";
					ss << (char)(std::toupper(c));
					first = false;
				}
			}
			ss << ")";
		}
		ss << ".\n";
	}
	return ss.str();
}

std::vector<ATenValuePtr> TensorProgram::get_parameters() const
{
	std::vector<ATenValuePtr> params;
	for (const auto& eq : _equations)
	{
		if (eq->weight())
			params.push_back(eq->weight());
		if (eq->bias())
			params.push_back(eq->bias());
	}
	return params;
}

void TensorProgram::set_parameters(const std::vector<ATenValuePtr>& params)
{
	size_t idx = 0;
	for (auto& eq : _equations)
	{
		if (eq->weight() && idx < params.size())
		{
			eq->set_weight(params[idx++]);
		}
		if (eq->bias() && idx < params.size())
		{
			eq->set_bias(params[idx++]);
		}
	}
}

// ============================================================
// Helper Functions
// ============================================================

ATenValuePtr opencog::relation_to_tensor(
	AtomSpace* as,
	Type relation_type,
	const std::map<Handle, int64_t>& entity_index)
{
	if (!as) return nullptr;

	int64_t num_entities = entity_index.size();
	std::vector<double> data(num_entities * num_entities, 0.0);

	// Get all links of the given type
	HandleSeq links;
	as->get_handles_by_type(links, relation_type);

	for (const Handle& link : links)
	{
		if (link->get_arity() < 2) continue;

		Handle src = link->getOutgoingAtom(0);
		Handle dst = link->getOutgoingAtom(1);

		auto src_it = entity_index.find(src);
		auto dst_it = entity_index.find(dst);

		if (src_it != entity_index.end() && dst_it != entity_index.end())
		{
			int64_t src_idx = src_it->second;
			int64_t dst_idx = dst_it->second;
			data[src_idx * num_entities + dst_idx] = 1.0;
		}
	}

	return createATenFromVector(data, {num_entities, num_entities});
}

HandleSeq opencog::tensor_to_atoms(
	const ATenValuePtr& tensor,
	Type relation_type,
	const std::vector<Handle>& index_to_entity,
	AtomSpace* as,
	double threshold)
{
	HandleSeq result;
	if (!tensor || !as) return result;

	auto data = tensor->to_vector();
	auto shape = tensor->shape();

	if (shape.size() != 2) return result;

	int64_t rows = shape[0];
	int64_t cols = shape[1];

	for (int64_t i = 0; i < rows && i < (int64_t)index_to_entity.size(); i++)
	{
		for (int64_t j = 0; j < cols && j < (int64_t)index_to_entity.size(); j++)
		{
			double value = data[i * cols + j];
			if (value > threshold)
			{
				Handle link = as->add_link(relation_type,
					index_to_entity[i],
					index_to_entity[j]);
				result.push_back(link);
			}
		}
	}

	return result;
}

TensorEquationPtr opencog::parse_datalog_rule(const std::string& rule)
{
	// Parse Datalog rule like:
	// grandparent(X,Z) :- parent(X,Y), parent(Y,Z).

	std::regex head_regex(R"((\w+)\(([^)]+)\))");
	std::regex body_regex(R"((\w+)\(([^)]+)\))");

	size_t arrow_pos = rule.find(":-");
	if (arrow_pos == std::string::npos)
	{
		throw SyntaxException(TRACE_INFO,
			"Invalid Datalog rule: missing ':-'");
	}

	std::string head = rule.substr(0, arrow_pos);
	std::string body = rule.substr(arrow_pos + 2);

	// Remove trailing period
	if (!body.empty() && body.back() == '.')
		body.pop_back();

	// Parse head
	std::smatch head_match;
	if (!std::regex_search(head, head_match, head_regex))
	{
		throw SyntaxException(TRACE_INFO,
			"Invalid Datalog head: %s", head.c_str());
	}

	std::string lhs_name = head_match[1];
	std::string head_vars = head_match[2];

	// Parse body predicates
	std::vector<std::string> rhs_names;
	std::vector<std::string> body_specs;
	std::string output_spec;

	// Build einsum notation from variable usage
	std::map<char, char> var_to_index;
	char next_index = 'a';

	// Parse head variables for output spec
	std::stringstream head_ss(head_vars);
	std::string var;
	while (std::getline(head_ss, var, ','))
	{
		var.erase(0, var.find_first_not_of(" "));
		var.erase(var.find_last_not_of(" ") + 1);
		if (!var.empty())
		{
			char v = var[0];
			if (var_to_index.find(v) == var_to_index.end())
			{
				var_to_index[v] = next_index++;
			}
			output_spec += var_to_index[v];
		}
	}

	// Parse body predicates
	std::sregex_iterator it(body.begin(), body.end(), body_regex);
	std::sregex_iterator end;

	while (it != end)
	{
		std::smatch match = *it;
		rhs_names.push_back(match[1]);

		std::string pred_vars = match[2];
		std::string pred_spec;

		std::stringstream pred_ss(pred_vars);
		while (std::getline(pred_ss, var, ','))
		{
			var.erase(0, var.find_first_not_of(" "));
			var.erase(var.find_last_not_of(" ") + 1);
			if (!var.empty())
			{
				char v = var[0];
				if (var_to_index.find(v) == var_to_index.end())
				{
					var_to_index[v] = next_index++;
				}
				pred_spec += var_to_index[v];
			}
		}

		body_specs.push_back(pred_spec);
		++it;
	}

	// Build einsum notation
	std::string einsum_notation;
	for (size_t i = 0; i < body_specs.size(); i++)
	{
		if (i > 0) einsum_notation += ",";
		einsum_notation += body_specs[i];
	}
	einsum_notation += "->" + output_spec;

	return std::make_shared<TensorEquation>(
		lhs_name + "_rule",
		lhs_name,
		rhs_names,
		einsum_notation,
		Nonlinearity::THRESHOLD,  // Default to Boolean mode
		ReasoningMode::BOOLEAN);
}

// ====================== END OF FILE =======================
