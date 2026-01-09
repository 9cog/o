/*
 * opencog/atoms/aten/TensorLink.h
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

#ifndef _OPENCOG_TENSOR_LINK_H
#define _OPENCOG_TENSOR_LINK_H

#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/aten/ATenValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

// ============================================================
// TensorOpLink - Base class for tensor operations

/**
 * TensorOpLink is the base class for all tensor operation links.
 * When executed, it performs a tensor operation on its arguments.
 */
class TensorOpLink : public Link
{
protected:
	void init();

	// Helper to extract ATenValue from argument
	ATenValuePtr get_tensor(AtomSpace*, const Handle&) const;

public:
	TensorOpLink(const HandleSeq&&, Type = TENSOR_OP_LINK);
	TensorOpLink(const TensorOpLink&) = delete;
	TensorOpLink& operator=(const TensorOpLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorOpLink)
#define createTensorOpLink CREATE_DECL(TensorOpLink)

// ============================================================
// TensorAddLink - Element-wise addition

class TensorAddLink : public TensorOpLink
{
public:
	TensorAddLink(const HandleSeq&&, Type = TENSOR_ADD_LINK);
	TensorAddLink(const TensorAddLink&) = delete;
	TensorAddLink& operator=(const TensorAddLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorAddLink)
#define createTensorAddLink CREATE_DECL(TensorAddLink)

// ============================================================
// TensorSubLink - Element-wise subtraction

class TensorSubLink : public TensorOpLink
{
public:
	TensorSubLink(const HandleSeq&&, Type = TENSOR_SUB_LINK);
	TensorSubLink(const TensorSubLink&) = delete;
	TensorSubLink& operator=(const TensorSubLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorSubLink)
#define createTensorSubLink CREATE_DECL(TensorSubLink)

// ============================================================
// TensorMulLink - Element-wise multiplication

class TensorMulLink : public TensorOpLink
{
public:
	TensorMulLink(const HandleSeq&&, Type = TENSOR_MUL_LINK);
	TensorMulLink(const TensorMulLink&) = delete;
	TensorMulLink& operator=(const TensorMulLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorMulLink)
#define createTensorMulLink CREATE_DECL(TensorMulLink)

// ============================================================
// TensorDivLink - Element-wise division

class TensorDivLink : public TensorOpLink
{
public:
	TensorDivLink(const HandleSeq&&, Type = TENSOR_DIV_LINK);
	TensorDivLink(const TensorDivLink&) = delete;
	TensorDivLink& operator=(const TensorDivLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorDivLink)
#define createTensorDivLink CREATE_DECL(TensorDivLink)

// ============================================================
// TensorMatmulLink - Matrix multiplication

class TensorMatmulLink : public TensorOpLink
{
public:
	TensorMatmulLink(const HandleSeq&&, Type = TENSOR_MATMUL_LINK);
	TensorMatmulLink(const TensorMatmulLink&) = delete;
	TensorMatmulLink& operator=(const TensorMatmulLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorMatmulLink)
#define createTensorMatmulLink CREATE_DECL(TensorMatmulLink)

// ============================================================
// TensorTransposeLink - Transpose tensor dimensions

class TensorTransposeLink : public TensorOpLink
{
public:
	TensorTransposeLink(const HandleSeq&&, Type = TENSOR_TRANSPOSE_LINK);
	TensorTransposeLink(const TensorTransposeLink&) = delete;
	TensorTransposeLink& operator=(const TensorTransposeLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorTransposeLink)
#define createTensorTransposeLink CREATE_DECL(TensorTransposeLink)

// ============================================================
// TensorReshapeLink - Reshape tensor

class TensorReshapeLink : public TensorOpLink
{
public:
	TensorReshapeLink(const HandleSeq&&, Type = TENSOR_RESHAPE_LINK);
	TensorReshapeLink(const TensorReshapeLink&) = delete;
	TensorReshapeLink& operator=(const TensorReshapeLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorReshapeLink)
#define createTensorReshapeLink CREATE_DECL(TensorReshapeLink)

// ============================================================
// TensorReluLink - ReLU activation

class TensorReluLink : public TensorOpLink
{
public:
	TensorReluLink(const HandleSeq&&, Type = TENSOR_RELU_LINK);
	TensorReluLink(const TensorReluLink&) = delete;
	TensorReluLink& operator=(const TensorReluLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorReluLink)
#define createTensorReluLink CREATE_DECL(TensorReluLink)

// ============================================================
// TensorSigmoidLink - Sigmoid activation

class TensorSigmoidLink : public TensorOpLink
{
public:
	TensorSigmoidLink(const HandleSeq&&, Type = TENSOR_SIGMOID_LINK);
	TensorSigmoidLink(const TensorSigmoidLink&) = delete;
	TensorSigmoidLink& operator=(const TensorSigmoidLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorSigmoidLink)
#define createTensorSigmoidLink CREATE_DECL(TensorSigmoidLink)

// ============================================================
// TensorTanhLink - Tanh activation

class TensorTanhLink : public TensorOpLink
{
public:
	TensorTanhLink(const HandleSeq&&, Type = TENSOR_TANH_LINK);
	TensorTanhLink(const TensorTanhLink&) = delete;
	TensorTanhLink& operator=(const TensorTanhLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorTanhLink)
#define createTensorTanhLink CREATE_DECL(TensorTanhLink)

// ============================================================
// TensorSoftmaxLink - Softmax activation

class TensorSoftmaxLink : public TensorOpLink
{
public:
	TensorSoftmaxLink(const HandleSeq&&, Type = TENSOR_SOFTMAX_LINK);
	TensorSoftmaxLink(const TensorSoftmaxLink&) = delete;
	TensorSoftmaxLink& operator=(const TensorSoftmaxLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorSoftmaxLink)
#define createTensorSoftmaxLink CREATE_DECL(TensorSoftmaxLink)

// ============================================================
// TensorSumLink - Sum all elements

class TensorSumLink : public TensorOpLink
{
public:
	TensorSumLink(const HandleSeq&&, Type = TENSOR_SUM_LINK);
	TensorSumLink(const TensorSumLink&) = delete;
	TensorSumLink& operator=(const TensorSumLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorSumLink)
#define createTensorSumLink CREATE_DECL(TensorSumLink)

// ============================================================
// TensorMeanLink - Mean of all elements

class TensorMeanLink : public TensorOpLink
{
public:
	TensorMeanLink(const HandleSeq&&, Type = TENSOR_MEAN_LINK);
	TensorMeanLink(const TensorMeanLink&) = delete;
	TensorMeanLink& operator=(const TensorMeanLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorMeanLink)
#define createTensorMeanLink CREATE_DECL(TensorMeanLink)

// ============================================================
// TensorOfLink - Get tensor value from an atom

class TensorOfLink : public TensorOpLink
{
public:
	TensorOfLink(const HandleSeq&&, Type = TENSOR_OF_LINK);
	TensorOfLink(const TensorOfLink&) = delete;
	TensorOfLink& operator=(const TensorOfLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorOfLink)
#define createTensorOfLink CREATE_DECL(TensorOfLink)

// ============================================================
// SetTensorLink - Attach tensor to an atom

class SetTensorLink : public TensorOpLink
{
public:
	SetTensorLink(const HandleSeq&&, Type = SET_TENSOR_LINK);
	SetTensorLink(const SetTensorLink&) = delete;
	SetTensorLink& operator=(const SetTensorLink&) = delete;

	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(SetTensorLink)
#define createSetTensorLink CREATE_DECL(SetTensorLink)

/** @}*/
} // namespace opencog

#endif // _OPENCOG_TENSOR_LINK_H
