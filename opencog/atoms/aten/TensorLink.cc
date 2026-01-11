/*
 * opencog/atoms/aten/TensorLink.cc
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

#include <opencog/util/exceptions.h>
#include <opencog/atoms/aten/TensorLink.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atomspace/AtomSpace.h>

using namespace opencog;

// ============================================================
// TensorOpLink - Base class

TensorOpLink::TensorOpLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	init();
}

void TensorOpLink::init()
{
	// Base initialization if needed
}

ATenValuePtr TensorOpLink::get_tensor(AtomSpace* as, const Handle& h) const
{
	// If the handle directly holds an ATenValue, get it
	// First, try to execute if it's executable
	if (h->is_executable()) {
		ValuePtr result = h->execute(as);
		ATenValuePtr avp = ATenValueCast(result);
		if (avp) return avp;

		// Check if it's a FloatValue we can convert
		FloatValuePtr fvp = FloatValueCast(result);
		if (fvp) {
			auto vec = fvp->value();
			return createATenFromVector(vec, {(int64_t)vec.size()});
		}
	}

	// Check for NumberNode
	if (h->is_type(NUMBER_NODE)) {
		NumberNodePtr nn = NumberNodeCast(h);
		auto vec = nn->value();
		return createATenFromVector(vec, {(int64_t)vec.size()});
	}

	// Check attached values with tensor key
	Handle tensor_key = as->get_node(PREDICATE_NODE, "*-TensorKey-*");
	if (tensor_key) {
		ValuePtr val = h->getValue(tensor_key);
		ATenValuePtr avp = ATenValueCast(val);
		if (avp) return avp;
	}

	throw RuntimeException(TRACE_INFO,
		"TensorOpLink: argument is not a tensor: %s",
		h->to_short_string().c_str());
}

ValuePtr TensorOpLink::execute(AtomSpace* as, bool silent)
{
	throw RuntimeException(TRACE_INFO,
		"TensorOpLink base class cannot be executed directly");
}

Handle TensorOpLink::factory(const Handle& h)
{
	return Handle(createTensorOpLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorAddLink

TensorAddLink::TensorAddLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorAddLink requires at least 2 arguments");
}

ValuePtr TensorAddLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr result = get_tensor(as, _outgoing[0]);

	for (size_t i = 1; i < _outgoing.size(); i++) {
		ATenValuePtr other = get_tensor(as, _outgoing[i]);
		result = ATenValueCast(result->add(*other));
	}

	return result;
}

Handle TensorAddLink::factory(const Handle& h)
{
	return Handle(createTensorAddLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorSubLink

TensorSubLink::TensorSubLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorSubLink requires exactly 2 arguments");
}

ValuePtr TensorSubLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr a = get_tensor(as, _outgoing[0]);
	ATenValuePtr b = get_tensor(as, _outgoing[1]);
	return a->sub(*b);
}

Handle TensorSubLink::factory(const Handle& h)
{
	return Handle(createTensorSubLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorMulLink

TensorMulLink::TensorMulLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorMulLink requires at least 2 arguments");
}

ValuePtr TensorMulLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr result = get_tensor(as, _outgoing[0]);

	for (size_t i = 1; i < _outgoing.size(); i++) {
		ATenValuePtr other = get_tensor(as, _outgoing[i]);
		result = ATenValueCast(result->mul(*other));
	}

	return result;
}

Handle TensorMulLink::factory(const Handle& h)
{
	return Handle(createTensorMulLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorDivLink

TensorDivLink::TensorDivLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorDivLink requires exactly 2 arguments");
}

ValuePtr TensorDivLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr a = get_tensor(as, _outgoing[0]);
	ATenValuePtr b = get_tensor(as, _outgoing[1]);
	return a->div(*b);
}

Handle TensorDivLink::factory(const Handle& h)
{
	return Handle(createTensorDivLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorMatmulLink

TensorMatmulLink::TensorMatmulLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorMatmulLink requires exactly 2 arguments");
}

ValuePtr TensorMatmulLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr a = get_tensor(as, _outgoing[0]);
	ATenValuePtr b = get_tensor(as, _outgoing[1]);
	return a->matmul(*b);
}

Handle TensorMatmulLink::factory(const Handle& h)
{
	return Handle(createTensorMatmulLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorTransposeLink

TensorTransposeLink::TensorTransposeLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 1 || _outgoing.size() > 3)
		throw InvalidParamException(TRACE_INFO,
			"TensorTransposeLink requires 1-3 arguments");
}

ValuePtr TensorTransposeLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);

	int64_t dim0 = 0, dim1 = 1;
	if (_outgoing.size() >= 2) {
		NumberNodePtr n0 = NumberNodeCast(_outgoing[1]);
		if (n0) dim0 = static_cast<int64_t>(n0->value()[0]);
	}
	if (_outgoing.size() >= 3) {
		NumberNodePtr n1 = NumberNodeCast(_outgoing[2]);
		if (n1) dim1 = static_cast<int64_t>(n1->value()[0]);
	}

	return tensor->transpose(dim0, dim1);
}

Handle TensorTransposeLink::factory(const Handle& h)
{
	return Handle(createTensorTransposeLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorReshapeLink

TensorReshapeLink::TensorReshapeLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorReshapeLink requires tensor and shape arguments");
}

ValuePtr TensorReshapeLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);

	// Get shape from remaining arguments
	std::vector<int64_t> new_shape;
	for (size_t i = 1; i < _outgoing.size(); i++) {
		NumberNodePtr nn = NumberNodeCast(_outgoing[i]);
		if (nn) {
			new_shape.push_back(static_cast<int64_t>(nn->value()[0]));
		}
	}

	return tensor->reshape(new_shape);
}

Handle TensorReshapeLink::factory(const Handle& h)
{
	return Handle(createTensorReshapeLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorReluLink

TensorReluLink::TensorReluLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"TensorReluLink requires exactly 1 argument");
}

ValuePtr TensorReluLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);
	return tensor->relu();
}

Handle TensorReluLink::factory(const Handle& h)
{
	return Handle(createTensorReluLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorSigmoidLink

TensorSigmoidLink::TensorSigmoidLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"TensorSigmoidLink requires exactly 1 argument");
}

ValuePtr TensorSigmoidLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);
	return tensor->sigmoid();
}

Handle TensorSigmoidLink::factory(const Handle& h)
{
	return Handle(createTensorSigmoidLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorTanhLink

TensorTanhLink::TensorTanhLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"TensorTanhLink requires exactly 1 argument");
}

ValuePtr TensorTanhLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);
	return tensor->tanh();
}

Handle TensorTanhLink::factory(const Handle& h)
{
	return Handle(createTensorTanhLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorSoftmaxLink

TensorSoftmaxLink::TensorSoftmaxLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 1 || _outgoing.size() > 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorSoftmaxLink requires 1-2 arguments");
}

ValuePtr TensorSoftmaxLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);

	int64_t dim = -1; // Default to last dimension
	if (_outgoing.size() >= 2) {
		NumberNodePtr nn = NumberNodeCast(_outgoing[1]);
		if (nn) dim = static_cast<int64_t>(nn->value()[0]);
	}

	return tensor->softmax(dim);
}

Handle TensorSoftmaxLink::factory(const Handle& h)
{
	return Handle(createTensorSoftmaxLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorSumLink

TensorSumLink::TensorSumLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"TensorSumLink requires exactly 1 argument");
}

ValuePtr TensorSumLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);
	return tensor->sum();
}

Handle TensorSumLink::factory(const Handle& h)
{
	return Handle(createTensorSumLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorMeanLink

TensorMeanLink::TensorMeanLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"TensorMeanLink requires exactly 1 argument");
}

ValuePtr TensorMeanLink::execute(AtomSpace* as, bool silent)
{
	ATenValuePtr tensor = get_tensor(as, _outgoing[0]);
	return tensor->mean();
}

Handle TensorMeanLink::factory(const Handle& h)
{
	return Handle(createTensorMeanLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// TensorOfLink

TensorOfLink::TensorOfLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 1 || _outgoing.size() > 2)
		throw InvalidParamException(TRACE_INFO,
			"TensorOfLink requires 1-2 arguments (atom [, key])");
}

ValuePtr TensorOfLink::execute(AtomSpace* as, bool silent)
{
	Handle atom = _outgoing[0];

	// Get the key (default or specified)
	Handle key;
	if (_outgoing.size() >= 2) {
		key = _outgoing[1];
	} else {
		key = as->add_node(PREDICATE_NODE, "*-TensorKey-*");
	}

	ValuePtr val = atom->getValue(key);
	if (not val) {
		if (silent) return ValuePtr();
		throw RuntimeException(TRACE_INFO,
			"TensorOfLink: no tensor value on atom %s",
			atom->to_short_string().c_str());
	}

	ATenValuePtr avp = ATenValueCast(val);
	if (not avp) {
		if (silent) return ValuePtr();
		throw RuntimeException(TRACE_INFO,
			"TensorOfLink: value is not a tensor");
	}

	return avp;
}

Handle TensorOfLink::factory(const Handle& h)
{
	return Handle(createTensorOfLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// SetTensorLink

SetTensorLink::SetTensorLink(const HandleSeq&& oset, Type t)
	: TensorOpLink(std::move(oset), t)
{
	if (_outgoing.size() < 2 || _outgoing.size() > 3)
		throw InvalidParamException(TRACE_INFO,
			"SetTensorLink requires 2-3 arguments (atom, tensor [, key])");
}

ValuePtr SetTensorLink::execute(AtomSpace* as, bool silent)
{
	Handle atom = _outgoing[0];
	ATenValuePtr tensor = get_tensor(as, _outgoing[1]);

	// Get the key (default or specified)
	Handle key;
	if (_outgoing.size() >= 3) {
		key = _outgoing[2];
	} else {
		key = as->add_node(PREDICATE_NODE, "*-TensorKey-*");
	}

	as->set_value(atom, key, tensor);
	return tensor;
}

Handle SetTensorLink::factory(const Handle& h)
{
	return Handle(createSetTensorLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// Factory registration
// Note: Factory methods are defined inline with each class above.
// The DEFINE_LINK_FACTORY macros are not needed here.
