/*
 * opencog/atoms/aten/TensorLogicLinks.h
 *
 * Copyright (C) 2025 OpenCog Foundation
 * All Rights Reserved
 *
 * Executable Link types for Tensor Logic operations.
 * These enable tensor logic to be expressed directly as atoms.
 *
 * Based on Tensor Logic by Pedro Domingos (https://tensor-logic.org)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#ifndef _OPENCOG_TENSOR_LOGIC_LINKS_H
#define _OPENCOG_TENSOR_LOGIC_LINKS_H

#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/aten/ATenValue.h>
#include <opencog/atoms/aten/TensorEquation.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

// ============================================================
// EinsumLink - Einstein summation operation
// ============================================================

/**
 * EinsumLink performs Einstein summation (tensor contraction).
 * This is the core operation of Tensor Logic.
 *
 * Usage:
 *   (EinsumLink
 *     (Concept "ij,jk->ik")  ; Einsum notation
 *     (TensorOfLink (Concept "A"))  ; First tensor
 *     (TensorOfLink (Concept "B"))) ; Second tensor
 *
 * This computes: C[i,k] = sum_j A[i,j] * B[j,k]
 */
class EinsumLink : public Link
{
protected:
	void init();
	EinsumSpec _spec;

public:
	EinsumLink(const HandleSeq&&, Type = EINSUM_LINK);
	EinsumLink(const EinsumLink&) = delete;
	EinsumLink& operator=(const EinsumLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	const EinsumSpec& spec() const { return _spec; }

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(EinsumLink)
#define createEinsumLink CREATE_DECL(EinsumLink)

// ============================================================
// TensorEquationLink - Define a tensor equation
// ============================================================

/**
 * TensorEquationLink defines a tensor equation (rule).
 *
 * Usage:
 *   (TensorEquationLink
 *     (Predicate "grandparent")           ; LHS (output)
 *     (Predicate "parent")                ; RHS input 1
 *     (Predicate "parent")                ; RHS input 2
 *     (Concept "ij,jk->ik")               ; Einsum notation
 *     (Concept "threshold"))              ; Nonlinearity (optional)
 *
 * This defines: grandparent[i,k] = H(sum_j parent[i,j] * parent[j,k])
 */
class TensorEquationLink : public Link
{
protected:
	void init();
	TensorEquationPtr _equation;

public:
	TensorEquationLink(const HandleSeq&&, Type = TENSOR_EQUATION_LINK);
	TensorEquationLink(const TensorEquationLink&) = delete;
	TensorEquationLink& operator=(const TensorEquationLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	TensorEquationPtr equation() const { return _equation; }

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorEquationLink)
#define createTensorEquationLink CREATE_DECL(TensorEquationLink)

// ============================================================
// TensorProgramLink - Define a tensor program
// ============================================================

/**
 * TensorProgramLink defines a complete tensor logic program.
 *
 * Usage:
 *   (TensorProgramLink
 *     (Concept "family_reasoning")  ; Program name
 *     (ListLink                     ; Equations
 *       (TensorEquationLink ...)
 *       (TensorEquationLink ...)))
 */
class TensorProgramLink : public Link
{
protected:
	void init();
	TensorProgramPtr _program;

public:
	TensorProgramLink(const HandleSeq&&, Type = TENSOR_PROGRAM_LINK);
	TensorProgramLink(const TensorProgramLink&) = delete;
	TensorProgramLink& operator=(const TensorProgramLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	TensorProgramPtr program() const { return _program; }

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorProgramLink)
#define createTensorProgramLink CREATE_DECL(TensorProgramLink)

// ============================================================
// TensorQueryLink - Query a tensor program
// ============================================================

/**
 * TensorQueryLink queries a tensor program for a relation.
 *
 * Usage:
 *   (TensorQueryLink
 *     (TensorProgramLink ...)        ; The program
 *     (Predicate "grandparent")      ; Relation to query
 *     (ListLink                      ; Optional: indices to query
 *       (Number 0)
 *       (Number 2)))
 */
class TensorQueryLink : public Link
{
protected:
	void init();

public:
	TensorQueryLink(const HandleSeq&&, Type = TENSOR_QUERY_LINK);
	TensorQueryLink(const TensorQueryLink&) = delete;
	TensorQueryLink& operator=(const TensorQueryLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorQueryLink)
#define createTensorQueryLink CREATE_DECL(TensorQueryLink)

// ============================================================
// TensorFactLink - Add a fact to a tensor program
// ============================================================

/**
 * TensorFactLink adds a fact tensor to a program.
 *
 * Usage:
 *   (TensorFactLink
 *     (TensorProgramLink ...)        ; The program
 *     (Predicate "parent")           ; Relation name
 *     (TensorOfLink ...))            ; The tensor data
 */
class TensorFactLink : public Link
{
protected:
	void init();

public:
	TensorFactLink(const HandleSeq&&, Type = TENSOR_FACT_LINK);
	TensorFactLink(const TensorFactLink&) = delete;
	TensorFactLink& operator=(const TensorFactLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorFactLink)
#define createTensorFactLink CREATE_DECL(TensorFactLink)

// ============================================================
// TensorForwardLink - Execute forward chaining
// ============================================================

/**
 * TensorForwardLink executes forward chaining on a program.
 *
 * Usage:
 *   (TensorForwardLink
 *     (TensorProgramLink ...)        ; The program
 *     (Number 10))                   ; Max iterations (optional)
 */
class TensorForwardLink : public Link
{
protected:
	void init();

public:
	TensorForwardLink(const HandleSeq&&, Type = TENSOR_FORWARD_LINK);
	TensorForwardLink(const TensorForwardLink&) = delete;
	TensorForwardLink& operator=(const TensorForwardLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorForwardLink)
#define createTensorForwardLink CREATE_DECL(TensorForwardLink)

// ============================================================
// TensorTrainLink - Train a tensor program
// ============================================================

/**
 * TensorTrainLink trains learnable parameters in a program.
 *
 * Usage:
 *   (TensorTrainLink
 *     (TensorProgramLink ...)        ; The program
 *     (ListLink                      ; Target relation-tensor pairs
 *       (ListLink (Predicate "target") (TensorOfLink ...)))
 *     (Number 100)                   ; Epochs
 *     (Number 0.01))                 ; Learning rate
 */
class TensorTrainLink : public Link
{
protected:
	void init();

public:
	TensorTrainLink(const HandleSeq&&, Type = TENSOR_TRAIN_LINK);
	TensorTrainLink(const TensorTrainLink&) = delete;
	TensorTrainLink& operator=(const TensorTrainLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorTrainLink)
#define createTensorTrainLink CREATE_DECL(TensorTrainLink)

// ============================================================
// RelationToTensorLink - Convert AtomSpace relation to tensor
// ============================================================

/**
 * RelationToTensorLink converts an AtomSpace relation to a tensor.
 *
 * Usage:
 *   (RelationToTensorLink
 *     (Type "InheritanceLink")       ; Relation type
 *     (ListLink                      ; Entity list
 *       (Concept "A")
 *       (Concept "B")
 *       (Concept "C")))
 *
 * Returns a tensor where entry [i,j] is 1 if (InheritanceLink entities[i] entities[j]) exists.
 */
class RelationToTensorLink : public Link
{
protected:
	void init();

public:
	RelationToTensorLink(const HandleSeq&&, Type = RELATION_TO_TENSOR_LINK);
	RelationToTensorLink(const RelationToTensorLink&) = delete;
	RelationToTensorLink& operator=(const RelationToTensorLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(RelationToTensorLink)
#define createRelationToTensorLink CREATE_DECL(RelationToTensorLink)

// ============================================================
// TensorToRelationLink - Convert tensor to AtomSpace atoms
// ============================================================

/**
 * TensorToRelationLink converts a tensor back to AtomSpace atoms.
 *
 * Usage:
 *   (TensorToRelationLink
 *     (TensorOfLink ...)             ; The tensor
 *     (Type "InheritanceLink")       ; Relation type to create
 *     (ListLink                      ; Entity list
 *       (Concept "A")
 *       (Concept "B")
 *       (Concept "C"))
 *     (Number 0.5))                  ; Threshold (optional, default 0.5)
 *
 * Creates atoms for entries above threshold.
 */
class TensorToRelationLink : public Link
{
protected:
	void init();

public:
	TensorToRelationLink(const HandleSeq&&, Type = TENSOR_TO_RELATION_LINK);
	TensorToRelationLink(const TensorToRelationLink&) = delete;
	TensorToRelationLink& operator=(const TensorToRelationLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorToRelationLink)
#define createTensorToRelationLink CREATE_DECL(TensorToRelationLink)

// ============================================================
// DatalogToTensorLink - Parse Datalog rule to tensor equation
// ============================================================

/**
 * DatalogToTensorLink parses a Datalog-style rule into a tensor equation.
 *
 * Usage:
 *   (DatalogToTensorLink
 *     (Concept "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)."))
 *
 * Returns a TensorEquationLink.
 */
class DatalogToTensorLink : public Link
{
protected:
	void init();

public:
	DatalogToTensorLink(const HandleSeq&&, Type = DATALOG_TO_TENSOR_LINK);
	DatalogToTensorLink(const DatalogToTensorLink&) = delete;
	DatalogToTensorLink& operator=(const DatalogToTensorLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(DatalogToTensorLink)
#define createDatalogToTensorLink CREATE_DECL(DatalogToTensorLink)

// ============================================================
// BooleanModeLink - Set boolean reasoning mode
// ============================================================

/**
 * BooleanModeLink wraps a tensor operation for strict Boolean reasoning.
 *
 * Usage:
 *   (BooleanModeLink
 *     (EinsumLink ...))
 *
 * All results are thresholded to {0, 1}.
 */
class BooleanModeLink : public Link
{
protected:
	void init();

public:
	BooleanModeLink(const HandleSeq&&, Type = BOOLEAN_MODE_LINK);
	BooleanModeLink(const BooleanModeLink&) = delete;
	BooleanModeLink& operator=(const BooleanModeLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(BooleanModeLink)
#define createBooleanModeLink CREATE_DECL(BooleanModeLink)

// ============================================================
// ContinuousModeLink - Set continuous reasoning mode
// ============================================================

/**
 * ContinuousModeLink wraps a tensor operation for differentiable reasoning.
 *
 * Usage:
 *   (ContinuousModeLink
 *     (EinsumLink ...)
 *     (Concept "sigmoid"))  ; Optional nonlinearity
 *
 * Results are smooth values in [0, 1].
 */
class ContinuousModeLink : public Link
{
protected:
	void init();

public:
	ContinuousModeLink(const HandleSeq&&, Type = CONTINUOUS_MODE_LINK);
	ContinuousModeLink(const ContinuousModeLink&) = delete;
	ContinuousModeLink& operator=(const ContinuousModeLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(ContinuousModeLink)
#define createContinuousModeLink CREATE_DECL(ContinuousModeLink)

/** @}*/
} // namespace opencog

#endif // _OPENCOG_TENSOR_LOGIC_LINKS_H
