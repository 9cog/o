/*
 * opencog/atoms/aten/TensorLogicLinks.cc
 *
 * Copyright (C) 2025 OpenCog Foundation
 * All Rights Reserved
 *
 * Implementation of executable Link types for Tensor Logic.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#include <opencog/util/exceptions.h>
#include <opencog/atoms/atom_types/NameServer.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/value/LinkValue.h>
#include <opencog/atoms/value/StringValue.h>
#include <opencog/atomspace/AtomSpace.h>

#include "TensorLogicLinks.h"
#include "TensorLink.h"

using namespace opencog;

// ============================================================
// Helper functions
// ============================================================

static ATenValuePtr get_tensor_from_atom(AtomSpace* as, const Handle& h)
{
	// If it's a TensorOfLink, execute it
	if (h->is_executable())
	{
		ValuePtr vp = h->execute(as);
		if (vp && nameserver().isA(vp->get_type(), ATEN_VALUE))
			return ATenValueCast(vp);
	}

	// Try to get tensor from attached value
	Handle key = as->add_node(PREDICATE_NODE, "*-tensor-*");
	ValuePtr vp = h->getValue(key);
	if (vp && nameserver().isA(vp->get_type(), ATEN_VALUE))
		return ATenValueCast(vp);

	// Try to get from FloatValue
	vp = h->getValue(key);
	if (vp && nameserver().isA(vp->get_type(), FLOAT_VALUE))
	{
		FloatValuePtr fv = FloatValueCast(vp);
		return createATenFromVector(fv->value(), {(int64_t)fv->value().size()});
	}

	return nullptr;
}

static std::string get_string_from_atom(const Handle& h)
{
	if (h->is_node())
		return h->get_name();

	throw InvalidParamException(TRACE_INFO,
		"Expected a node with string value");
}

// ============================================================
// EinsumLink Implementation
// ============================================================

void EinsumLink::init()
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"EinsumLink requires notation and at least one tensor");

	// First argument is the einsum notation
	std::string notation = get_string_from_atom(_outgoing[0]);
	_spec = EinsumSpec(notation);

	if (_spec.num_inputs() != _outgoing.size() - 1)
		throw InvalidParamException(TRACE_INFO,
			"EinsumLink: notation expects %zu inputs, got %zu",
			_spec.num_inputs(), _outgoing.size() - 1);
}

EinsumLink::EinsumLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, EINSUM_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting an EinsumLink");
	init();
}

ValuePtr EinsumLink::execute(AtomSpace* as, bool silent)
{
	// Gather input tensors
	std::vector<ATenValuePtr> tensors;
	for (size_t i = 1; i < _outgoing.size(); i++)
	{
		ATenValuePtr tensor = get_tensor_from_atom(as, _outgoing[i]);
		if (!tensor)
		{
			if (silent) return nullptr;
			throw RuntimeException(TRACE_INFO,
				"EinsumLink: cannot get tensor from argument %zu", i);
		}
		tensors.push_back(tensor);
	}

	// Execute einsum
	return einsum(_spec, tensors);
}

Handle EinsumLink::factory(const Handle& h)
{
	return Handle(createEinsumLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorEquationLink Implementation
// ============================================================

void TensorEquationLink::init()
{
	if (_outgoing.size() < 4)
		throw InvalidParamException(TRACE_INFO,
			"TensorEquationLink requires LHS, RHS relations, and einsum notation");

	// Parse: LHS, RHS..., einsum, [nonlinearity]
	std::string lhs_name = get_string_from_atom(_outgoing[0]);

	std::vector<std::string> rhs_names;
	size_t einsum_idx = _outgoing.size() - 1;

	// Find the einsum notation (last or second-to-last argument)
	std::string last_str = get_string_from_atom(_outgoing[einsum_idx]);
	std::string einsum_notation;
	Nonlinearity nl = Nonlinearity::NONE;

	if (last_str.find("->") != std::string::npos ||
	    last_str.find(",") != std::string::npos)
	{
		// Last argument is einsum
		einsum_notation = last_str;
	}
	else
	{
		// Last is nonlinearity, second-to-last is einsum
		if (last_str == "threshold") nl = Nonlinearity::THRESHOLD;
		else if (last_str == "sigmoid") nl = Nonlinearity::SIGMOID;
		else if (last_str == "relu") nl = Nonlinearity::RELU;
		else if (last_str == "tanh") nl = Nonlinearity::TANH;
		else if (last_str == "softmax") nl = Nonlinearity::SOFTMAX;
		else if (last_str == "clamp01") nl = Nonlinearity::CLAMP01;

		einsum_idx--;
		einsum_notation = get_string_from_atom(_outgoing[einsum_idx]);
	}

	// Collect RHS names
	for (size_t i = 1; i < einsum_idx; i++)
	{
		rhs_names.push_back(get_string_from_atom(_outgoing[i]));
	}

	_equation = std::make_shared<TensorEquation>(
		lhs_name + "_eq",
		lhs_name,
		rhs_names,
		einsum_notation,
		nl,
		ReasoningMode::CONTINUOUS);
}

TensorEquationLink::TensorEquationLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_EQUATION_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorEquationLink");
	init();
}

ValuePtr TensorEquationLink::execute(AtomSpace* as, bool silent)
{
	// Gather input tensors by name
	std::map<std::string, ATenValuePtr> inputs;

	for (const auto& name : _equation->rhs_names())
	{
		// Find atom with this name
		Handle h = as->get_node(PREDICATE_NODE, std::string(name));
		if (!h)
			h = as->get_node(CONCEPT_NODE, std::string(name));

		if (h)
		{
			ATenValuePtr tensor = get_tensor_from_atom(as, h);
			if (tensor)
				inputs[name] = tensor;
		}
	}

	if (inputs.size() != _equation->rhs_names().size())
	{
		if (silent) return nullptr;
		throw RuntimeException(TRACE_INFO,
			"TensorEquationLink: missing input tensors");
	}

	return _equation->execute(inputs);
}

Handle TensorEquationLink::factory(const Handle& h)
{
	return Handle(createTensorEquationLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorProgramLink Implementation
// ============================================================

void TensorProgramLink::init()
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorProgramLink requires name and equations");

	std::string name = get_string_from_atom(_outgoing[0]);
	_program = std::make_shared<TensorProgram>(name);

	// Add equations from ListLink
	if (_outgoing[1]->get_type() == LIST_LINK)
	{
		for (const Handle& eq_h : _outgoing[1]->getOutgoingSet())
		{
			if (eq_h->get_type() == TENSOR_EQUATION_LINK)
			{
				TensorEquationLinkPtr eq_link = TensorEquationLinkCast(eq_h);
				if (eq_link && eq_link->equation())
				{
					_program->add_equation(eq_link->equation());
				}
			}
		}
	}
}

TensorProgramLink::TensorProgramLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_PROGRAM_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorProgramLink");
	init();
}

ValuePtr TensorProgramLink::execute(AtomSpace* as, bool silent)
{
	// Execute forward pass
	_program->forward_to_fixpoint();

	// Return summary as string value
	return createStringValue(_program->to_string());
}

Handle TensorProgramLink::factory(const Handle& h)
{
	return Handle(createTensorProgramLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorQueryLink Implementation
// ============================================================

void TensorQueryLink::init()
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorQueryLink requires program and query name");
}

TensorQueryLink::TensorQueryLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_QUERY_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorQueryLink");
	init();
}

ValuePtr TensorQueryLink::execute(AtomSpace* as, bool silent)
{
	// Get program
	if (_outgoing[0]->get_type() != TENSOR_PROGRAM_LINK)
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"TensorQueryLink: first argument must be TensorProgramLink");
	}

	TensorProgramLinkPtr prog_link = TensorProgramLinkCast(_outgoing[0]);
	TensorProgramPtr program = prog_link->program();

	// Get query name
	std::string query_name = get_string_from_atom(_outgoing[1]);

	// Get optional indices
	std::vector<int64_t> indices;
	if (_outgoing.size() > 2 && _outgoing[2]->get_type() == LIST_LINK)
	{
		for (const Handle& idx_h : _outgoing[2]->getOutgoingSet())
		{
			if (idx_h->get_type() == NUMBER_NODE)
			{
				indices.push_back((int64_t)std::stod(idx_h->get_name()));
			}
		}
	}

	return program->query(query_name, indices);
}

Handle TensorQueryLink::factory(const Handle& h)
{
	return Handle(createTensorQueryLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorFactLink Implementation
// ============================================================

void TensorFactLink::init()
{
	if (_outgoing.size() < 3)
		throw InvalidParamException(TRACE_INFO,
			"TensorFactLink requires program, name, and tensor");
}

TensorFactLink::TensorFactLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_FACT_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorFactLink");
	init();
}

ValuePtr TensorFactLink::execute(AtomSpace* as, bool silent)
{
	// Get program
	if (_outgoing[0]->get_type() != TENSOR_PROGRAM_LINK)
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"TensorFactLink: first argument must be TensorProgramLink");
	}

	TensorProgramLinkPtr prog_link = TensorProgramLinkCast(_outgoing[0]);
	TensorProgramPtr program = prog_link->program();

	// Get fact name
	std::string fact_name = get_string_from_atom(_outgoing[1]);

	// Get tensor
	ATenValuePtr tensor = get_tensor_from_atom(as, _outgoing[2]);
	if (!tensor)
	{
		if (silent) return nullptr;
		throw RuntimeException(TRACE_INFO,
			"TensorFactLink: cannot get tensor from argument");
	}

	program->add_fact(fact_name, tensor);

	return createStringValue("ok");
}

Handle TensorFactLink::factory(const Handle& h)
{
	return Handle(createTensorFactLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorForwardLink Implementation
// ============================================================

void TensorForwardLink::init()
{
	if (_outgoing.empty())
		throw InvalidParamException(TRACE_INFO,
			"TensorForwardLink requires at least a program");
}

TensorForwardLink::TensorForwardLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_FORWARD_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorForwardLink");
	init();
}

ValuePtr TensorForwardLink::execute(AtomSpace* as, bool silent)
{
	// Get program
	if (_outgoing[0]->get_type() != TENSOR_PROGRAM_LINK)
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"TensorForwardLink: first argument must be TensorProgramLink");
	}

	TensorProgramLinkPtr prog_link = TensorProgramLinkCast(_outgoing[0]);
	TensorProgramPtr program = prog_link->program();

	// Execute forward chaining
	program->forward_to_fixpoint();

	return createStringValue(program->to_string());
}

Handle TensorForwardLink::factory(const Handle& h)
{
	return Handle(createTensorForwardLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorTrainLink Implementation
// ============================================================

void TensorTrainLink::init()
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorTrainLink requires program and targets");
}

TensorTrainLink::TensorTrainLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_TRAIN_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorTrainLink");
	init();
}

ValuePtr TensorTrainLink::execute(AtomSpace* as, bool silent)
{
	// Get program
	if (_outgoing[0]->get_type() != TENSOR_PROGRAM_LINK)
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"TensorTrainLink: first argument must be TensorProgramLink");
	}

	TensorProgramLinkPtr prog_link = TensorProgramLinkCast(_outgoing[0]);
	TensorProgramPtr program = prog_link->program();

	// Parse targets
	std::map<std::string, ATenValuePtr> targets;
	if (_outgoing[1]->get_type() == LIST_LINK)
	{
		for (const Handle& pair : _outgoing[1]->getOutgoingSet())
		{
			if (pair->get_type() == LIST_LINK && pair->get_arity() >= 2)
			{
				std::string name = get_string_from_atom(pair->getOutgoingAtom(0));
				ATenValuePtr tensor = get_tensor_from_atom(as, pair->getOutgoingAtom(1));
				if (tensor)
					targets[name] = tensor;
			}
		}
	}

	// Get epochs and learning rate
	size_t epochs = 100;
	double lr = 0.01;

	if (_outgoing.size() > 2 && _outgoing[2]->get_type() == NUMBER_NODE)
		epochs = (size_t)std::stod(_outgoing[2]->get_name());

	if (_outgoing.size() > 3 && _outgoing[3]->get_type() == NUMBER_NODE)
		lr = std::stod(_outgoing[3]->get_name());

	program->set_learning_rate(lr);

	// Train
	std::map<std::string, ATenValuePtr> empty_inputs;
	double loss = program->train(empty_inputs, targets, epochs);

	return createFloatValue(loss);
}

Handle TensorTrainLink::factory(const Handle& h)
{
	return Handle(createTensorTrainLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// RelationToTensorLink Implementation
// ============================================================

void RelationToTensorLink::init()
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"RelationToTensorLink requires type and entity list");
}

RelationToTensorLink::RelationToTensorLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, RELATION_TO_TENSOR_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a RelationToTensorLink");
	init();
}

ValuePtr RelationToTensorLink::execute(AtomSpace* as, bool silent)
{
	// Get relation type
	Type rel_type = LINK;
	if (_outgoing[0]->get_type() == TYPE_NODE)
	{
		rel_type = nameserver().getType(_outgoing[0]->get_name());
	}

	// Get entity list
	std::vector<Handle> entities;
	std::map<Handle, int64_t> entity_index;

	if (_outgoing[1]->get_type() == LIST_LINK)
	{
		int64_t idx = 0;
		for (const Handle& e : _outgoing[1]->getOutgoingSet())
		{
			entities.push_back(e);
			entity_index[e] = idx++;
		}
	}

	return relation_to_tensor(as, rel_type, entity_index);
}

Handle RelationToTensorLink::factory(const Handle& h)
{
	return Handle(createRelationToTensorLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorToRelationLink Implementation
// ============================================================

void TensorToRelationLink::init()
{
	if (_outgoing.size() < 3)
		throw InvalidParamException(TRACE_INFO,
			"TensorToRelationLink requires tensor, type, and entity list");
}

TensorToRelationLink::TensorToRelationLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, TENSOR_TO_RELATION_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a TensorToRelationLink");
	init();
}

ValuePtr TensorToRelationLink::execute(AtomSpace* as, bool silent)
{
	// Get tensor
	ATenValuePtr tensor = get_tensor_from_atom(as, _outgoing[0]);
	if (!tensor)
	{
		if (silent) return nullptr;
		throw RuntimeException(TRACE_INFO,
			"TensorToRelationLink: cannot get tensor");
	}

	// Get relation type
	Type rel_type = LINK;
	if (_outgoing[1]->get_type() == TYPE_NODE)
	{
		rel_type = nameserver().getType(_outgoing[1]->get_name());
	}

	// Get entity list
	std::vector<Handle> entities;
	if (_outgoing[2]->get_type() == LIST_LINK)
	{
		for (const Handle& e : _outgoing[2]->getOutgoingSet())
		{
			entities.push_back(e);
		}
	}

	// Get threshold
	double threshold = 0.5;
	if (_outgoing.size() > 3 && _outgoing[3]->get_type() == NUMBER_NODE)
	{
		threshold = std::stod(_outgoing[3]->get_name());
	}

	HandleSeq atoms = tensor_to_atoms(tensor, rel_type, entities, as, threshold);

	return createLinkValue(atoms);
}

Handle TensorToRelationLink::factory(const Handle& h)
{
	return Handle(createTensorToRelationLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// DatalogToTensorLink Implementation
// ============================================================

void DatalogToTensorLink::init()
{
	if (_outgoing.empty())
		throw InvalidParamException(TRACE_INFO,
			"DatalogToTensorLink requires a rule string");
}

DatalogToTensorLink::DatalogToTensorLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, DATALOG_TO_TENSOR_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a DatalogToTensorLink");
	init();
}

ValuePtr DatalogToTensorLink::execute(AtomSpace* as, bool silent)
{
	std::string rule = get_string_from_atom(_outgoing[0]);

	try
	{
		TensorEquationPtr eq = parse_datalog_rule(rule);

		// Create TensorEquationLink
		HandleSeq oset;
		oset.push_back(as->add_node(PREDICATE_NODE, std::string(eq->lhs_name())));

		for (const auto& name : eq->rhs_names())
		{
			oset.push_back(as->add_node(PREDICATE_NODE, std::string(name)));
		}

		oset.push_back(as->add_node(CONCEPT_NODE, std::string(eq->einsum().notation())));

		// Add nonlinearity if present
		if (eq->nonlinearity() == Nonlinearity::THRESHOLD)
			oset.push_back(as->add_node(CONCEPT_NODE, std::string("threshold")));

		return as->add_link(TENSOR_EQUATION_LINK, std::move(oset));
	}
	catch (const std::exception& e)
	{
		if (silent) return nullptr;
		throw;
	}
}

Handle DatalogToTensorLink::factory(const Handle& h)
{
	return Handle(createDatalogToTensorLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// BooleanModeLink Implementation
// ============================================================

void BooleanModeLink::init()
{
	if (_outgoing.empty())
		throw InvalidParamException(TRACE_INFO,
			"BooleanModeLink requires an operation");
}

BooleanModeLink::BooleanModeLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, BOOLEAN_MODE_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a BooleanModeLink");
	init();
}

ValuePtr BooleanModeLink::execute(AtomSpace* as, bool silent)
{
	// Execute wrapped operation
	if (!_outgoing[0]->is_executable())
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"BooleanModeLink: wrapped operation not executable");
	}

	ValuePtr result = _outgoing[0]->execute(as, silent);

	// Apply threshold
	if (result && nameserver().isA(result->get_type(), ATEN_VALUE))
	{
		ATenValuePtr tensor = ATenValueCast(result);
		return apply_nonlinearity(tensor, Nonlinearity::THRESHOLD);
	}

	return result;
}

Handle BooleanModeLink::factory(const Handle& h)
{
	return Handle(createBooleanModeLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// ContinuousModeLink Implementation
// ============================================================

void ContinuousModeLink::init()
{
	if (_outgoing.empty())
		throw InvalidParamException(TRACE_INFO,
			"ContinuousModeLink requires an operation");
}

ContinuousModeLink::ContinuousModeLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	if (!nameserver().isA(t, CONTINUOUS_MODE_LINK))
		throw InvalidParamException(TRACE_INFO,
			"Expecting a ContinuousModeLink");
	init();
}

ValuePtr ContinuousModeLink::execute(AtomSpace* as, bool silent)
{
	// Execute wrapped operation
	if (!_outgoing[0]->is_executable())
	{
		if (silent) return nullptr;
		throw InvalidParamException(TRACE_INFO,
			"ContinuousModeLink: wrapped operation not executable");
	}

	ValuePtr result = _outgoing[0]->execute(as, silent);

	// Apply nonlinearity if specified
	if (result && nameserver().isA(result->get_type(), ATEN_VALUE) &&
	    _outgoing.size() > 1)
	{
		ATenValuePtr tensor = ATenValueCast(result);
		std::string nl_str = get_string_from_atom(_outgoing[1]);

		Nonlinearity nl = Nonlinearity::NONE;
		if (nl_str == "sigmoid") nl = Nonlinearity::SIGMOID;
		else if (nl_str == "relu") nl = Nonlinearity::RELU;
		else if (nl_str == "tanh") nl = Nonlinearity::TANH;
		else if (nl_str == "softmax") nl = Nonlinearity::SOFTMAX;
		else if (nl_str == "clamp01") nl = Nonlinearity::CLAMP01;

		return apply_nonlinearity(tensor, nl);
	}

	return result;
}

Handle ContinuousModeLink::factory(const Handle& h)
{
	return Handle(createContinuousModeLink(std::move(h->getOutgoingSet())));
}

// ====================== END OF FILE =======================
