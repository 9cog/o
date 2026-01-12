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
	const std::vector<ATenValuePtr>& inputs) const
{
	// Simplified gradient computation
	// A full implementation would compute proper einsum gradients
	std::vector<ATenValuePtr> gradients;

	// For each input, the gradient is roughly grad_output * (other inputs)
	// This is a simplification; proper einsum gradients are more complex

	for (size_t i = 0; i < inputs.size(); i++)
	{
		// Create a gradient tensor of the same shape as input
		auto shape = inputs[i]->shape();
		std::vector<double> grad_data(inputs[i]->numel(), 0.0);

		// Simplified: just propagate mean gradient
		ATenValuePtr mean_val = ATenValueCast(grad_output->mean());
		double mean_grad = mean_val->item();
		for (auto& g : grad_data)
			g = mean_grad;

		gradients.push_back(createATenFromVector(grad_data, shape));
	}

	return gradients;
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
	  _optimizer(std::make_shared<SGDOptimizer>(0.01)),
	  _tape(std::make_shared<GradientTape>()),
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
	// Simplified backpropagation - a full implementation would
	// track computation graph and compute proper gradients
}

void TensorProgram::update_parameters()
{
	for (auto& eq : _equations)
	{
		if (eq->is_learnable())
		{
			// Update weight and bias using stored gradients
			// This is a placeholder - actual implementation would use
			// computed gradients
		}
	}
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

// ============================================================
// Optimizer Implementation
// ============================================================

Optimizer::Optimizer(double lr, LRSchedule schedule)
	: _learning_rate(lr)
	, _initial_lr(lr)
	, _schedule(schedule)
	, _step_count(0)
	, _warmup_steps(1000)
	, _decay_rate(0.96)
{
}

void Optimizer::step_schedule()
{
	_step_count++;

	switch (_schedule)
	{
		case LRSchedule::CONSTANT:
			break;

		case LRSchedule::STEP:
			// Decay every 1000 steps
			if (_step_count % 1000 == 0)
				_learning_rate *= _decay_rate;
			break;

		case LRSchedule::EXPONENTIAL:
			_learning_rate = _initial_lr * std::pow(_decay_rate, _step_count / 1000.0);
			break;

		case LRSchedule::COSINE:
		{
			// Cosine annealing
			double progress = (double)_step_count / 10000.0;
			_learning_rate = _initial_lr * 0.5 * (1.0 + std::cos(M_PI * progress));
			break;
		}

		case LRSchedule::WARMUP:
			if (_step_count < _warmup_steps)
				_learning_rate = _initial_lr * ((double)_step_count / _warmup_steps);
			else
				_learning_rate = _initial_lr;
			break;
	}
}

void Optimizer::reset()
{
	_step_count = 0;
	_learning_rate = _initial_lr;
}

// ============================================================
// SGD Optimizer Implementation
// ============================================================

SGDOptimizer::SGDOptimizer(double lr, double momentum, double weight_decay)
	: Optimizer(lr)
	, _momentum(momentum)
	, _weight_decay(weight_decay)
{
}

ATenValuePtr SGDOptimizer::update(const ATenValuePtr& param,
                                   const ATenValuePtr& grad)
{
	if (!param || !grad) return param;

	auto param_vec = param->to_vector();
	auto grad_vec = grad->to_vector();
	auto shape = param->shape();

	if (param_vec.size() != grad_vec.size())
		return param;

	void* key = param.get();

	// Initialize velocity if needed
	if (_momentum > 0 && _velocity.find(key) == _velocity.end())
	{
		std::vector<double> zeros(param_vec.size(), 0.0);
		_velocity[key] = createATenFromVector(zeros, shape);
	}

	std::vector<double> updated(param_vec.size());

	if (_momentum > 0)
	{
		auto vel_vec = _velocity[key]->to_vector();
		for (size_t i = 0; i < param_vec.size(); i++)
		{
			// Apply weight decay
			double g = grad_vec[i] + _weight_decay * param_vec[i];

			// Update velocity: v = momentum * v + lr * g
			vel_vec[i] = _momentum * vel_vec[i] + _learning_rate * g;

			// Update param: p = p - v
			updated[i] = param_vec[i] - vel_vec[i];
		}
		_velocity[key] = createATenFromVector(vel_vec, shape);
	}
	else
	{
		for (size_t i = 0; i < param_vec.size(); i++)
		{
			double g = grad_vec[i] + _weight_decay * param_vec[i];
			updated[i] = param_vec[i] - _learning_rate * g;
		}
	}

	step_schedule();
	return createATenFromVector(updated, shape);
}

void SGDOptimizer::reset()
{
	Optimizer::reset();
	_velocity.clear();
}

// ============================================================
// Adam Optimizer Implementation
// ============================================================

AdamOptimizer::AdamOptimizer(double lr, double beta1, double beta2,
                             double epsilon, double weight_decay)
	: Optimizer(lr)
	, _beta1(beta1)
	, _beta2(beta2)
	, _epsilon(epsilon)
	, _weight_decay(weight_decay)
{
}

ATenValuePtr AdamOptimizer::update(const ATenValuePtr& param,
                                    const ATenValuePtr& grad)
{
	if (!param || !grad) return param;

	auto param_vec = param->to_vector();
	auto grad_vec = grad->to_vector();
	auto shape = param->shape();

	if (param_vec.size() != grad_vec.size())
		return param;

	void* key = param.get();

	// Initialize moment estimates if needed
	if (_m.find(key) == _m.end())
	{
		std::vector<double> zeros(param_vec.size(), 0.0);
		_m[key] = createATenFromVector(zeros, shape);
		_v[key] = createATenFromVector(zeros, shape);
	}

	auto m_vec = _m[key]->to_vector();
	auto v_vec = _v[key]->to_vector();

	// Bias correction terms
	double bias_correction1 = 1.0 - std::pow(_beta1, _step_count + 1);
	double bias_correction2 = 1.0 - std::pow(_beta2, _step_count + 1);

	std::vector<double> updated(param_vec.size());

	for (size_t i = 0; i < param_vec.size(); i++)
	{
		// Apply weight decay (decoupled as in AdamW)
		double g = grad_vec[i];
		if (_weight_decay > 0)
			param_vec[i] -= _learning_rate * _weight_decay * param_vec[i];

		// Update first moment: m = beta1 * m + (1 - beta1) * g
		m_vec[i] = _beta1 * m_vec[i] + (1.0 - _beta1) * g;

		// Update second moment: v = beta2 * v + (1 - beta2) * g^2
		v_vec[i] = _beta2 * v_vec[i] + (1.0 - _beta2) * g * g;

		// Bias-corrected estimates
		double m_hat = m_vec[i] / bias_correction1;
		double v_hat = v_vec[i] / bias_correction2;

		// Update parameter: p = p - lr * m_hat / (sqrt(v_hat) + epsilon)
		updated[i] = param_vec[i] - _learning_rate * m_hat / (std::sqrt(v_hat) + _epsilon);
	}

	_m[key] = createATenFromVector(m_vec, shape);
	_v[key] = createATenFromVector(v_vec, shape);

	step_schedule();
	return createATenFromVector(updated, shape);
}

void AdamOptimizer::reset()
{
	Optimizer::reset();
	_m.clear();
	_v.clear();
}

// ============================================================
// Sparse Tensor Implementation
// ============================================================

SparseTensor::SparseTensor(const std::vector<int64_t>& shape)
	: _shape(shape), _nnz(0)
{
}

double SparseTensor::get(const std::vector<int64_t>& indices) const
{
	if (indices.size() != _shape.size())
		return 0.0;

	// Linear search (could use hash map for better performance)
	for (size_t i = 0; i < _indices.size(); i++)
	{
		if (_indices[i] == indices)
			return _values[i];
	}
	return 0.0;
}

void SparseTensor::set(const std::vector<int64_t>& indices, double value)
{
	if (indices.size() != _shape.size())
		return;

	// Check if already exists
	for (size_t i = 0; i < _indices.size(); i++)
	{
		if (_indices[i] == indices)
		{
			if (std::abs(value) < 1e-10)
			{
				// Remove entry
				_indices.erase(_indices.begin() + i);
				_values.erase(_values.begin() + i);
				_nnz--;
			}
			else
			{
				_values[i] = value;
			}
			return;
		}
	}

	// Add new entry if non-zero
	if (std::abs(value) >= 1e-10)
	{
		add_entry(indices, value);
	}
}

void SparseTensor::add_entry(const std::vector<int64_t>& indices, double value)
{
	_indices.push_back(indices);
	_values.push_back(value);
	_nnz++;
}

ATenValuePtr SparseTensor::to_dense() const
{
	// Compute total size
	int64_t total = 1;
	for (int64_t dim : _shape)
		total *= dim;

	std::vector<double> data(total, 0.0);

	// Fill in non-zero values
	for (size_t i = 0; i < _nnz; i++)
	{
		int64_t linear_idx = 0;
		int64_t stride = 1;
		for (int j = _shape.size() - 1; j >= 0; j--)
		{
			linear_idx += _indices[i][j] * stride;
			stride *= _shape[j];
		}
		if (linear_idx < total)
			data[linear_idx] = _values[i];
	}

	return createATenFromVector(data, _shape);
}

SparseTensorPtr SparseTensor::from_dense(const ATenValuePtr& dense,
                                          double threshold)
{
	if (!dense) return nullptr;

	auto shape = dense->shape();
	auto data = dense->to_vector();

	auto sparse = std::make_shared<SparseTensor>(shape);

	// Find non-zero elements
	std::vector<int64_t> indices(shape.size());
	std::function<void(size_t, int64_t)> iterate;
	iterate = [&](size_t dim, int64_t base_idx) {
		if (dim >= shape.size())
		{
			double value = data[base_idx];
			if (std::abs(value) > threshold)
			{
				sparse->add_entry(indices, value);
			}
		}
		else
		{
			int64_t stride = 1;
			for (size_t d = dim + 1; d < shape.size(); d++)
				stride *= shape[d];

			for (int64_t i = 0; i < shape[dim]; i++)
			{
				indices[dim] = i;
				iterate(dim + 1, base_idx + i * stride);
			}
		}
	};

	iterate(0, 0);
	return sparse;
}

SparseTensorPtr SparseTensor::matmul(const SparseTensor& other) const
{
	// Sparse matrix multiplication for 2D tensors
	if (_shape.size() != 2 || other._shape.size() != 2)
		return nullptr;

	if (_shape[1] != other._shape[0])
		return nullptr;

	auto result = std::make_shared<SparseTensor>(
		std::vector<int64_t>{_shape[0], other._shape[1]});

	// For each non-zero in A and B where A[i,k] and B[k,j] share k
	std::map<std::pair<int64_t, int64_t>, double> accum;

	for (size_t a = 0; a < _nnz; a++)
	{
		int64_t i = _indices[a][0];
		int64_t k = _indices[a][1];
		double val_a = _values[a];

		for (size_t b = 0; b < other._nnz; b++)
		{
			if (other._indices[b][0] == k)
			{
				int64_t j = other._indices[b][1];
				double val_b = other._values[b];
				accum[{i, j}] += val_a * val_b;
			}
		}
	}

	for (const auto& [idx, val] : accum)
	{
		if (std::abs(val) > 1e-10)
			result->add_entry({idx.first, idx.second}, val);
	}

	return result;
}

SparseTensorPtr SparseTensor::add(const SparseTensor& other) const
{
	if (_shape != other._shape)
		return nullptr;

	auto result = std::make_shared<SparseTensor>(_shape);

	// Copy all entries from this
	for (size_t i = 0; i < _nnz; i++)
		result->add_entry(_indices[i], _values[i]);

	// Add entries from other
	for (size_t i = 0; i < other._nnz; i++)
	{
		double existing = result->get(other._indices[i]);
		result->set(other._indices[i], existing + other._values[i]);
	}

	return result;
}

SparseTensorPtr SparseTensor::mul(const SparseTensor& other) const
{
	if (_shape != other._shape)
		return nullptr;

	auto result = std::make_shared<SparseTensor>(_shape);

	// Element-wise: only non-zero where both are non-zero
	for (size_t i = 0; i < _nnz; i++)
	{
		double other_val = other.get(_indices[i]);
		if (std::abs(other_val) > 1e-10)
		{
			result->add_entry(_indices[i], _values[i] * other_val);
		}
	}

	return result;
}

void SparseTensor::threshold(double min_value)
{
	std::vector<std::vector<int64_t>> new_indices;
	std::vector<double> new_values;

	for (size_t i = 0; i < _nnz; i++)
	{
		if (std::abs(_values[i]) >= min_value)
		{
			new_indices.push_back(_indices[i]);
			new_values.push_back(_values[i]);
		}
	}

	_indices = std::move(new_indices);
	_values = std::move(new_values);
	_nnz = _values.size();
}

double SparseTensor::sparsity() const
{
	int64_t total = 1;
	for (int64_t dim : _shape)
		total *= dim;

	if (total == 0) return 0.0;
	return (double)_nnz / total;
}

std::string SparseTensor::to_string() const
{
	std::stringstream ss;
	ss << "SparseTensor(shape=[";
	for (size_t i = 0; i < _shape.size(); i++)
	{
		if (i > 0) ss << ", ";
		ss << _shape[i];
	}
	ss << "], nnz=" << _nnz << ", sparsity=" << sparsity() << ")";
	return ss.str();
}

// ============================================================
// Gradient Tape Implementation
// ============================================================

GradientTape::GradientTape()
	: _recording(false)
{
}

void GradientTape::start()
{
	_recording = true;
	_operations.clear();
	_gradients.clear();
}

void GradientTape::stop()
{
	_recording = false;
}

void GradientTape::record(
	const std::string& type,
	const std::vector<ATenValuePtr>& inputs,
	const ATenValuePtr& output,
	std::function<std::vector<ATenValuePtr>(const ATenValuePtr&)> backward_fn)
{
	if (!_recording) return;

	Operation op;
	op.type = type;
	op.inputs = inputs;
	op.output = output;
	op.backward_fn = backward_fn;
	_operations.push_back(std::move(op));
}

std::map<void*, ATenValuePtr> GradientTape::backward(
	const ATenValuePtr& output,
	const ATenValuePtr& grad_output)
{
	_gradients.clear();

	if (!output) return _gradients;

	// Initialize gradient for output
	_gradients[output.get()] = grad_output;

	// Traverse operations in reverse order
	for (auto it = _operations.rbegin(); it != _operations.rend(); ++it)
	{
		const Operation& op = *it;

		// Check if we have gradient for this operation's output
		auto grad_it = _gradients.find(op.output.get());
		if (grad_it == _gradients.end())
			continue;

		ATenValuePtr grad = grad_it->second;

		// Compute gradients for inputs
		if (op.backward_fn)
		{
			auto input_grads = op.backward_fn(grad);

			for (size_t i = 0; i < input_grads.size() && i < op.inputs.size(); i++)
			{
				void* key = op.inputs[i].get();
				if (_gradients.find(key) == _gradients.end())
				{
					_gradients[key] = input_grads[i];
				}
				else
				{
					// Accumulate gradients
					_gradients[key] = ATenValueCast(
						_gradients[key]->add(*input_grads[i]));
				}
			}
		}
	}

	return _gradients;
}

ATenValuePtr GradientTape::gradient(const ATenValuePtr& tensor) const
{
	if (!tensor) return nullptr;

	auto it = _gradients.find(tensor.get());
	if (it != _gradients.end())
		return it->second;
	return nullptr;
}

void GradientTape::clear()
{
	_operations.clear();
	_gradients.clear();
}

// ============================================================
// Semiring Implementation
// ============================================================

Semiring::Semiring(SemiringType type)
	: _type(type)
{
	switch (type)
	{
		case SemiringType::BOOLEAN:
			_zero = 0.0;  // false
			_one = 1.0;   // true
			break;
		case SemiringType::COUNTING:
		case SemiringType::PROBABILISTIC:
		case SemiringType::REAL:
			_zero = 0.0;
			_one = 1.0;
			break;
		case SemiringType::VITERBI:
			_zero = -std::numeric_limits<double>::infinity();
			_one = 0.0;
			break;
		case SemiringType::TROPICAL:
			_zero = std::numeric_limits<double>::infinity();
			_one = 0.0;
			break;
		case SemiringType::LUKASIEWICZ:
			_zero = 0.0;
			_one = 1.0;
			break;
	}
}

double Semiring::add(double a, double b) const
{
	switch (_type)
	{
		case SemiringType::BOOLEAN:
			return (a > 0.5 || b > 0.5) ? 1.0 : 0.0;  // OR
		case SemiringType::COUNTING:
		case SemiringType::PROBABILISTIC:
		case SemiringType::REAL:
			return a + b;
		case SemiringType::VITERBI:
			return std::max(a, b);
		case SemiringType::TROPICAL:
			return std::min(a, b);
		case SemiringType::LUKASIEWICZ:
			return std::max(a, b);  // max for fuzzy OR
	}
	return a + b;
}

double Semiring::mul(double a, double b) const
{
	switch (_type)
	{
		case SemiringType::BOOLEAN:
			return (a > 0.5 && b > 0.5) ? 1.0 : 0.0;  // AND
		case SemiringType::COUNTING:
		case SemiringType::PROBABILISTIC:
		case SemiringType::REAL:
			return a * b;
		case SemiringType::VITERBI:
		case SemiringType::TROPICAL:
			return a + b;  // log-domain multiplication
		case SemiringType::LUKASIEWICZ:
			return std::min(a, b);  // min for fuzzy AND
	}
	return a * b;
}

ATenValuePtr Semiring::tensor_add(const ATenValuePtr& a, const ATenValuePtr& b) const
{
	if (!a || !b) return nullptr;

	auto va = a->to_vector();
	auto vb = b->to_vector();
	auto shape = a->shape();

	if (va.size() != vb.size()) return nullptr;

	std::vector<double> result(va.size());
	for (size_t i = 0; i < va.size(); i++)
	{
		result[i] = add(va[i], vb[i]);
	}

	return createATenFromVector(result, shape);
}

ATenValuePtr Semiring::tensor_mul(const ATenValuePtr& a, const ATenValuePtr& b) const
{
	if (!a || !b) return nullptr;

	auto va = a->to_vector();
	auto vb = b->to_vector();
	auto shape = a->shape();

	if (va.size() != vb.size()) return nullptr;

	std::vector<double> result(va.size());
	for (size_t i = 0; i < va.size(); i++)
	{
		result[i] = mul(va[i], vb[i]);
	}

	return createATenFromVector(result, shape);
}

ATenValuePtr Semiring::tensor_sum(const ATenValuePtr& a) const
{
	if (!a) return nullptr;

	auto va = a->to_vector();
	double result = _zero;

	for (double v : va)
	{
		result = add(result, v);
	}

	return createATenFromVector({result}, {});
}

ATenValuePtr Semiring::einsum(const std::string& notation,
                               const std::vector<ATenValuePtr>& tensors) const
{
	// For most semirings, use standard einsum then apply normalization
	ATenValuePtr result = opencog::einsum(notation, tensors);

	// For probabilistic semiring, normalize
	if (_type == SemiringType::PROBABILISTIC && result)
	{
		auto vec = result->to_vector();
		double sum = 0.0;
		for (double v : vec) sum += v;
		if (sum > 1e-10)
		{
			for (double& v : vec) v /= sum;
		}
		return createATenFromVector(vec, result->shape());
	}

	// For boolean semiring, threshold
	if (_type == SemiringType::BOOLEAN && result)
	{
		auto vec = result->to_vector();
		for (double& v : vec) v = (v > 0.5) ? 1.0 : 0.0;
		return createATenFromVector(vec, result->shape());
	}

	return result;
}

std::string Semiring::to_string() const
{
	switch (_type)
	{
		case SemiringType::BOOLEAN: return "Boolean(OR,AND)";
		case SemiringType::COUNTING: return "Counting(+,×)";
		case SemiringType::VITERBI: return "Viterbi(max,+)";
		case SemiringType::PROBABILISTIC: return "Probabilistic(+,×)";
		case SemiringType::TROPICAL: return "Tropical(min,+)";
		case SemiringType::LUKASIEWICZ: return "Lukasiewicz(max,min)";
		case SemiringType::REAL: return "Real(+,×)";
	}
	return "Unknown";
}

// ============================================================
// PLN Truth Value Implementation
// ============================================================

PLNTruthValue PLNTruthValue::revise(const PLNTruthValue& other) const
{
	// PLN revision formula
	// Combined strength = weighted average by confidence
	// Combined confidence = increases with more evidence

	double w1 = confidence;
	double w2 = other.confidence;
	double total_w = w1 + w2;

	if (total_w < 1e-10)
		return PLNTruthValue(0.5, 0.0);

	double new_strength = (w1 * strength + w2 * other.strength) / total_w;

	// Confidence increases but asymptotes at 1
	// Using formula: c_new = c1 + c2 - c1*c2
	double new_confidence = w1 + w2 - w1 * w2;
	new_confidence = std::min(new_confidence, 0.9999);

	return PLNTruthValue(new_strength, new_confidence);
}

PLNTruthValue PLNTruthValue::deduction(const PLNTruthValue& ab,
                                        const PLNTruthValue& bc)
{
	// PLN deduction: P(A->C) from P(A->B) and P(B->C)
	// Simplified: s_ac = s_ab * s_bc
	// Confidence decreases with chain length

	double s_ac = ab.strength * bc.strength;
	double c_ac = ab.confidence * bc.confidence * 0.9;  // Decay factor

	return PLNTruthValue(s_ac, c_ac);
}

PLNTruthValue PLNTruthValue::induction(const PLNTruthValue& ab,
                                        const PLNTruthValue& ac)
{
	// PLN induction: P(B->C) from P(A->B) and P(A->C)
	// Less certain than deduction

	double s_bc = ab.strength > 0.5 ? ac.strength : 1.0 - ac.strength;
	double c_bc = ab.confidence * ac.confidence * 0.5;  // Lower confidence

	return PLNTruthValue(s_bc, c_bc);
}

PLNTruthValue PLNTruthValue::abduction(const PLNTruthValue& ab,
                                        const PLNTruthValue& cb)
{
	// PLN abduction: P(A->C) from P(A->B) and P(C->B)
	// Least certain inference type

	double s_ac = (ab.strength * cb.strength);
	double c_ac = ab.confidence * cb.confidence * 0.3;  // Lowest confidence

	return PLNTruthValue(s_ac, c_ac);
}

std::string PLNTruthValue::to_string() const
{
	std::ostringstream oss;
	oss << "<" << std::fixed << std::setprecision(3)
	    << strength << ", " << confidence << ">";
	return oss.str();
}

// ============================================================
// PLN Tensor Implementation
// ============================================================

PLNTensor::PLNTensor(const std::vector<int64_t>& shape)
	: _shape(shape)
{
	size_t size = 1;
	for (int64_t d : shape) size *= d;

	std::vector<double> zeros(size, 0.0);
	_strength = createATenFromVector(zeros, shape);
	_confidence = createATenFromVector(zeros, shape);
}

PLNTensor::PLNTensor(const ATenValuePtr& strength, const ATenValuePtr& confidence)
	: _strength(strength), _confidence(confidence)
{
	if (strength)
		_shape = strength->shape();
}

PLNTruthValue PLNTensor::get(const std::vector<int64_t>& indices) const
{
	if (!_strength || !_confidence) return PLNTruthValue();

	// Compute linear index
	int64_t idx = 0;
	int64_t stride = 1;
	for (int i = _shape.size() - 1; i >= 0; i--)
	{
		idx += indices[i] * stride;
		stride *= _shape[i];
	}

	auto s_vec = _strength->to_vector();
	auto c_vec = _confidence->to_vector();

	if (idx >= (int64_t)s_vec.size()) return PLNTruthValue();

	return PLNTruthValue(s_vec[idx], c_vec[idx]);
}

void PLNTensor::set(const std::vector<int64_t>& indices, const PLNTruthValue& tv)
{
	if (!_strength || !_confidence) return;

	int64_t idx = 0;
	int64_t stride = 1;
	for (int i = _shape.size() - 1; i >= 0; i--)
	{
		idx += indices[i] * stride;
		stride *= _shape[i];
	}

	auto s_vec = _strength->to_vector();
	auto c_vec = _confidence->to_vector();

	if (idx >= (int64_t)s_vec.size()) return;

	s_vec[idx] = tv.strength;
	c_vec[idx] = tv.confidence;

	_strength = createATenFromVector(s_vec, _shape);
	_confidence = createATenFromVector(c_vec, _shape);
}

PLNTensorPtr PLNTensor::revise(const PLNTensor& other) const
{
	if (_shape != other._shape) return nullptr;

	auto s1 = _strength->to_vector();
	auto c1 = _confidence->to_vector();
	auto s2 = other._strength->to_vector();
	auto c2 = other._confidence->to_vector();

	std::vector<double> new_s(s1.size());
	std::vector<double> new_c(c1.size());

	for (size_t i = 0; i < s1.size(); i++)
	{
		PLNTruthValue tv1(s1[i], c1[i]);
		PLNTruthValue tv2(s2[i], c2[i]);
		PLNTruthValue revised = tv1.revise(tv2);
		new_s[i] = revised.strength;
		new_c[i] = revised.confidence;
	}

	return std::make_shared<PLNTensor>(
		createATenFromVector(new_s, _shape),
		createATenFromVector(new_c, _shape));
}

PLNTensorPtr PLNTensor::deduction(const PLNTensor& other) const
{
	// Matrix multiplication with PLN uncertainty propagation
	if (_shape.size() != 2 || other._shape.size() != 2)
		return nullptr;

	if (_shape[1] != other._shape[0])
		return nullptr;

	int64_t m = _shape[0];
	int64_t k = _shape[1];
	int64_t n = other._shape[1];

	std::vector<double> new_s(m * n, 0.0);
	std::vector<double> new_c(m * n, 0.0);

	auto s1 = _strength->to_vector();
	auto c1 = _confidence->to_vector();
	auto s2 = other._strength->to_vector();
	auto c2 = other._confidence->to_vector();

	for (int64_t i = 0; i < m; i++)
	{
		for (int64_t j = 0; j < n; j++)
		{
			double sum_s = 0.0;
			double sum_c = 0.0;
			double total_weight = 0.0;

			for (int64_t l = 0; l < k; l++)
			{
				PLNTruthValue tv1(s1[i * k + l], c1[i * k + l]);
				PLNTruthValue tv2(s2[l * n + j], c2[l * n + j]);
				PLNTruthValue deduced = PLNTruthValue::deduction(tv1, tv2);

				// Weight by confidence
				double weight = deduced.confidence;
				sum_s += weight * deduced.strength;
				sum_c += weight * deduced.confidence;
				total_weight += weight;
			}

			if (total_weight > 1e-10)
			{
				new_s[i * n + j] = sum_s / total_weight;
				new_c[i * n + j] = sum_c / total_weight;
			}
		}
	}

	return std::make_shared<PLNTensor>(
		createATenFromVector(new_s, {m, n}),
		createATenFromVector(new_c, {m, n}));
}

PLNTensorPtr PLNTensor::from_tensor(const ATenValuePtr& tensor,
                                     double default_confidence)
{
	if (!tensor) return nullptr;

	auto shape = tensor->shape();
	auto data = tensor->to_vector();

	std::vector<double> conf(data.size(), default_confidence);

	return std::make_shared<PLNTensor>(
		tensor,
		createATenFromVector(conf, shape));
}

std::string PLNTensor::to_string() const
{
	std::ostringstream oss;
	oss << "PLNTensor(shape=[";
	for (size_t i = 0; i < _shape.size(); i++)
	{
		if (i > 0) oss << ", ";
		oss << _shape[i];
	}
	oss << "])";
	return oss.str();
}

// ============================================================
// Resource Metrics Implementation
// ============================================================

ResourceMetrics::ResourceMetrics()
	: memory_bytes(0)
	, flops(0)
	, tensor_elements(0)
	, sparsity(0.0)
	, cache_misses(0)
	, bandwidth_gb(0.0)
	, compute_time_ms(0.0)
{
}

void ResourceMetrics::reset()
{
	memory_bytes = 0;
	flops = 0;
	tensor_elements = 0;
	sparsity = 0.0;
	cache_misses = 0;
	bandwidth_gb = 0.0;
	compute_time_ms = 0.0;
}

void ResourceMetrics::add(const ResourceMetrics& other)
{
	memory_bytes += other.memory_bytes;
	flops += other.flops;
	tensor_elements += other.tensor_elements;
	// Weighted average for sparsity
	if (tensor_elements > 0)
	{
		sparsity = (sparsity * (tensor_elements - other.tensor_elements) +
		            other.sparsity * other.tensor_elements) / tensor_elements;
	}
	cache_misses += other.cache_misses;
	bandwidth_gb += other.bandwidth_gb;
	compute_time_ms += other.compute_time_ms;
}

ResourceMetrics ResourceMetrics::estimate_matmul(int64_t m, int64_t n, int64_t k)
{
	ResourceMetrics metrics;

	// Memory: input matrices + output matrix
	metrics.memory_bytes = (m * k + k * n + m * n) * sizeof(double);

	// FLOPs: 2 * m * n * k (multiply-add for each output element)
	metrics.flops = 2 * m * n * k;

	metrics.tensor_elements = m * k + k * n + m * n;

	// Estimate bandwidth (assumes all data read once)
	metrics.bandwidth_gb = metrics.memory_bytes / 1e9;

	// Rough compute time estimate (assuming 1 TFLOP/s)
	metrics.compute_time_ms = metrics.flops / 1e9;

	return metrics;
}

ResourceMetrics ResourceMetrics::estimate_einsum(const EinsumSpec& spec,
                                                   const std::vector<ATenValuePtr>& tensors)
{
	ResourceMetrics metrics;

	// Sum up input tensor elements
	for (const auto& t : tensors)
	{
		if (t)
		{
			metrics.memory_bytes += t->numel() * sizeof(double);
			metrics.tensor_elements += t->numel();
		}
	}

	// Estimate output size
	auto output_shape = spec.output_shape(tensors);
	size_t output_size = 1;
	for (int64_t d : output_shape) output_size *= d;
	metrics.memory_bytes += output_size * sizeof(double);
	metrics.tensor_elements += output_size;

	// FLOPs: roughly product of all dimensions
	metrics.flops = output_size;
	for (char c : spec.sum_indices())
	{
		// Each summed index multiplies the work
		for (size_t i = 0; i < tensors.size(); i++)
		{
			const auto& shape = tensors[i]->shape();
			const auto& spec_str = spec.input_specs()[i];
			for (size_t j = 0; j < spec_str.size(); j++)
			{
				if (spec_str[j] == c && j < shape.size())
				{
					metrics.flops *= shape[j];
					break;
				}
			}
		}
	}

	metrics.bandwidth_gb = metrics.memory_bytes / 1e9;
	metrics.compute_time_ms = metrics.flops / 1e9;

	return metrics;
}

std::string ResourceMetrics::to_string() const
{
	std::ostringstream oss;
	oss << "ResourceMetrics{memory=" << memory_bytes / 1024 << "KB"
	    << ", flops=" << flops
	    << ", elements=" << tensor_elements
	    << ", sparsity=" << std::fixed << std::setprecision(3) << sparsity
	    << ", time=" << compute_time_ms << "ms}";
	return oss.str();
}

// ============================================================
// Resource Tracker Implementation
// ============================================================

ResourceTracker::ResourceTracker()
	: _tracking(false)
	, _memory_limit(0)
	, _flops_limit(0)
{
}

void ResourceTracker::start_tracking()
{
	_tracking = true;
	_total.reset();
	_history.clear();
}

void ResourceTracker::stop_tracking()
{
	_tracking = false;
}

void ResourceTracker::record_operation(const ResourceMetrics& metrics)
{
	if (!_tracking) return;

	_total.add(metrics);
	_history.push_back(metrics);
}

void ResourceTracker::clear_history()
{
	_history.clear();
	_total.reset();
}

bool ResourceTracker::within_limits() const
{
	if (_memory_limit > 0 && _total.memory_bytes > _memory_limit)
		return false;
	if (_flops_limit > 0 && _total.flops > _flops_limit)
		return false;
	return true;
}

std::string ResourceTracker::to_string() const
{
	std::ostringstream oss;
	oss << "ResourceTracker{tracking=" << (_tracking ? "true" : "false")
	    << ", total=" << _total.to_string()
	    << ", ops=" << _history.size() << "}";
	return oss.str();
}

// ============================================================
// Managed Tensor Implementation
// ============================================================

ManagedTensor::ManagedTensor(const ATenValuePtr& tensor, LinearModality modality)
	: _tensor(tensor)
	, _modality(modality)
	, _ref_count(0)
	, _consumed(false)
{
}

ATenValuePtr ManagedTensor::acquire(bool consume)
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (_consumed) return nullptr;

	switch (_modality)
	{
		case LinearModality::LINEAR:
			// Must consume on first use
			if (!consume) return nullptr;
			_consumed = true;
			return _tensor;

		case LinearModality::AFFINE:
			// Can use at most once
			if (_ref_count > 0 && consume) return nullptr;
			if (consume) _consumed = true;
			_ref_count++;
			return _tensor;

		case LinearModality::BANG:
			// Read-only sharing
			if (consume) return nullptr;  // Can't consume shared
			_ref_count++;
			return _tensor;

		case LinearModality::WITH:
			// Choice - can acquire either way
			if (consume) _consumed = true;
			_ref_count++;
			return _tensor;
	}

	return nullptr;
}

void ManagedTensor::release()
{
	std::lock_guard<std::mutex> lock(_mtx);
	if (_ref_count > 0) _ref_count--;
}

bool ManagedTensor::available() const
{
	if (_consumed) return false;

	switch (_modality)
	{
		case LinearModality::LINEAR:
			return _ref_count == 0;
		case LinearModality::AFFINE:
			return _ref_count == 0;
		case LinearModality::BANG:
			return true;  // Always available for reading
		case LinearModality::WITH:
			return true;
	}
	return false;
}

std::string ManagedTensor::to_string() const
{
	std::ostringstream oss;
	oss << "ManagedTensor{modality=";
	switch (_modality)
	{
		case LinearModality::LINEAR: oss << "LINEAR"; break;
		case LinearModality::AFFINE: oss << "AFFINE"; break;
		case LinearModality::BANG: oss << "BANG"; break;
		case LinearModality::WITH: oss << "WITH"; break;
	}
	oss << ", refs=" << _ref_count
	    << ", consumed=" << (_consumed ? "true" : "false") << "}";
	return oss.str();
}

// ============================================================
// RAPTL Value Implementation
// ============================================================

RAPTLValue::RAPTLValue(const ATenValuePtr& tensor, SemiringType semiring)
	: _tensor(tensor)
	, _semiring(std::make_shared<Semiring>(semiring))
	, _modality(LinearModality::BANG)
{
	if (tensor)
	{
		_uncertainty = PLNTensor::from_tensor(tensor, 0.9);

		// Estimate resource usage
		_resources.memory_bytes = tensor->numel() * sizeof(double);
		_resources.tensor_elements = tensor->numel();
	}
}

RAPTLValuePtr RAPTLValue::multiply(const RAPTLValue& other) const
{
	if (!_tensor || !other._tensor) return nullptr;

	// Use semiring multiplication
	ATenValuePtr result_tensor = _semiring->tensor_mul(_tensor, other._tensor);

	auto result = std::make_shared<RAPTLValue>(result_tensor, _semiring->type());

	// Combine uncertainties (element-wise deduction-like operation)
	if (_uncertainty && other._uncertainty)
	{
		auto s1 = _uncertainty->strength()->to_vector();
		auto c1 = _uncertainty->confidence()->to_vector();
		auto s2 = other._uncertainty->strength()->to_vector();
		auto c2 = other._uncertainty->confidence()->to_vector();

		std::vector<double> new_s(s1.size());
		std::vector<double> new_c(c1.size());

		for (size_t i = 0; i < s1.size(); i++)
		{
			new_s[i] = s1[i] * s2[i];
			new_c[i] = c1[i] * c2[i] * 0.9;  // Confidence decay
		}

		result->_uncertainty = std::make_shared<PLNTensor>(
			createATenFromVector(new_s, _tensor->shape()),
			createATenFromVector(new_c, _tensor->shape()));
	}

	// Combine resources
	result->_resources.add(_resources);
	result->_resources.add(other._resources);
	result->_resources.flops += _tensor->numel();

	return result;
}

RAPTLValuePtr RAPTLValue::add(const RAPTLValue& other) const
{
	if (!_tensor || !other._tensor) return nullptr;

	ATenValuePtr result_tensor = _semiring->tensor_add(_tensor, other._tensor);

	auto result = std::make_shared<RAPTLValue>(result_tensor, _semiring->type());

	// Revise uncertainties
	if (_uncertainty && other._uncertainty)
	{
		result->_uncertainty = _uncertainty->revise(*other._uncertainty);
	}

	// Combine resources
	result->_resources.add(_resources);
	result->_resources.add(other._resources);

	return result;
}

RAPTLValuePtr RAPTLValue::einsum(
	const std::string& notation,
	const std::vector<RAPTLValuePtr>& inputs)
{
	if (inputs.empty()) return nullptr;

	// Gather tensors
	std::vector<ATenValuePtr> tensors;
	for (const auto& input : inputs)
	{
		if (input && input->tensor())
			tensors.push_back(input->tensor());
	}

	if (tensors.empty()) return nullptr;

	// Use first input's semiring
	auto semiring = inputs[0]->semiring();

	// Perform einsum
	ATenValuePtr result_tensor = semiring->einsum(notation, tensors);

	auto result = std::make_shared<RAPTLValue>(result_tensor, semiring->type());

	// Combine all uncertainties (simplified - just use first)
	if (inputs[0]->uncertainty())
	{
		result->_uncertainty = PLNTensor::from_tensor(result_tensor, 0.8);
	}

	// Sum up resources
	for (const auto& input : inputs)
	{
		if (input)
			result->_resources.add(input->resources());
	}

	// Add einsum operation resources
	EinsumSpec spec(notation);
	auto einsum_resources = ResourceMetrics::estimate_einsum(spec, tensors);
	result->_resources.add(einsum_resources);

	return result;
}

bool RAPTLValue::within_limits(size_t memory_limit, size_t flops_limit) const
{
	if (memory_limit > 0 && _resources.memory_bytes > memory_limit)
		return false;
	if (flops_limit > 0 && _resources.flops > flops_limit)
		return false;
	return true;
}

std::string RAPTLValue::to_string() const
{
	std::ostringstream oss;
	oss << "RAPTLValue{";
	if (_tensor)
	{
		oss << "shape=[";
		auto shape = _tensor->shape();
		for (size_t i = 0; i < shape.size(); i++)
		{
			if (i > 0) oss << ",";
			oss << shape[i];
		}
		oss << "]";
	}
	oss << ", semiring=" << _semiring->to_string()
	    << ", resources=" << _resources.to_string()
	    << "}";
	return oss.str();
}

// ============================================================
// RAPTL Program Implementation
// ============================================================

RAPTLProgram::RAPTLProgram(const std::string& name, SemiringType semiring)
	: _name(name)
	, _semiring(std::make_shared<Semiring>(semiring))
	, _tracker(std::make_shared<ResourceTracker>())
	, _mode(ReasoningMode::CONTINUOUS)
{
}

void RAPTLProgram::add_fact(const std::string& name, const RAPTLValuePtr& value)
{
	_facts[name] = value;
}

RAPTLValuePtr RAPTLProgram::get_fact(const std::string& name) const
{
	auto it = _facts.find(name);
	if (it != _facts.end())
		return it->second;
	return nullptr;
}

void RAPTLProgram::add_equation(const TensorEquationPtr& eq)
{
	_equations.push_back(eq);
}

void RAPTLProgram::add_equation(const std::string& name,
                                 const std::string& lhs,
                                 const std::vector<std::string>& rhs,
                                 const std::string& einsum,
                                 Nonlinearity nl)
{
	auto eq = std::make_shared<TensorEquation>(name, lhs, rhs, einsum, nl, _mode);
	_equations.push_back(eq);
}

void RAPTLProgram::forward()
{
	_tracker->start_tracking();

	for (const auto& eq : _equations)
	{
		// Gather inputs
		std::vector<RAPTLValuePtr> inputs;
		bool all_found = true;

		for (const auto& name : eq->rhs_names())
		{
			// Check derived first
			auto it = _derived.find(name);
			if (it != _derived.end())
			{
				inputs.push_back(it->second);
			}
			else
			{
				// Check facts
				auto fact_it = _facts.find(name);
				if (fact_it != _facts.end())
				{
					inputs.push_back(fact_it->second);
				}
				else
				{
					all_found = false;
					break;
				}
			}
		}

		if (all_found && !inputs.empty())
		{
			// Execute equation with RAPTL semantics
			RAPTLValuePtr result = RAPTLValue::einsum(
				eq->einsum().notation(), inputs);

			if (result)
			{
				// Apply nonlinearity
				ATenValuePtr tensor = result->tensor();
				tensor = apply_nonlinearity(tensor, eq->nonlinearity());

				// Create new RAPTL value with processed tensor
				auto final_result = std::make_shared<RAPTLValue>(
					tensor, _semiring->type());
				final_result->set_uncertainty(result->uncertainty());

				_derived[eq->lhs_name()] = final_result;

				// Track resources
				_tracker->record_operation(result->resources());
			}
		}
	}

	_tracker->stop_tracking();
}

RAPTLValuePtr RAPTLProgram::query(const std::string& name)
{
	// Check derived
	auto it = _derived.find(name);
	if (it != _derived.end())
		return it->second;

	// Check facts
	return get_fact(name);
}

void RAPTLProgram::set_resource_limits(size_t memory, size_t flops)
{
	_tracker->set_memory_limit(memory);
	_tracker->set_flops_limit(flops);
}

bool RAPTLProgram::within_limits() const
{
	return _tracker->within_limits();
}

void RAPTLProgram::set_semiring(SemiringType type)
{
	_semiring = std::make_shared<Semiring>(type);
}

std::string RAPTLProgram::to_string() const
{
	std::ostringstream oss;
	oss << "RAPTLProgram{name=" << _name
	    << ", facts=" << _facts.size()
	    << ", equations=" << _equations.size()
	    << ", derived=" << _derived.size()
	    << ", semiring=" << _semiring->to_string()
	    << ", " << _tracker->to_string()
	    << "}";
	return oss.str();
}

// ====================== END OF FILE =======================
