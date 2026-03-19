/*
 * opencog/atoms/aten/TensorEquation.h
 *
 * Copyright (C) 2025 OpenCog Foundation
 * All Rights Reserved
 *
 * Implementation of Tensor Logic as described by Pedro Domingos
 * (https://tensor-logic.org, arXiv:2510.12269)
 *
 * The core insight: logical rules and Einstein summation are essentially
 * the same operation. A tensor equation is a numeric Datalog rule (Einsum).
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#ifndef _OPENCOG_TENSOR_EQUATION_H
#define _OPENCOG_TENSOR_EQUATION_H

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencog/atoms/base/Handle.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/aten/ATenValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

class AtomSpace;
class TensorEquation;
class TensorProgram;

typedef std::shared_ptr<TensorEquation> TensorEquationPtr;
typedef std::shared_ptr<TensorProgram> TensorProgramPtr;

// ============================================================
// Reasoning Modes
// ============================================================

/**
 * ReasoningMode determines how tensor operations are performed:
 * - BOOLEAN: Strict logical deduction (0 or 1), no hallucinations
 * - CONTINUOUS: Probabilistic reasoning with learnable parameters
 * - HYBRID: Combines both modes with a confidence threshold
 */
enum class ReasoningMode {
	BOOLEAN,     // Strict logic: {0, 1} values, threshold at 0.5
	CONTINUOUS,  // Probabilistic: [0, 1] values, differentiable
	HYBRID       // Combined: boolean output with continuous internal
};

// ============================================================
// Einsum Specification
// ============================================================

/**
 * EinsumSpec parses and represents Einstein summation notation.
 *
 * Examples:
 *   "ij,jk->ik"     Matrix multiplication
 *   "ij,j->i"       Matrix-vector product
 *   "ijk,jkl->il"   Tensor contraction
 *   "ii->"          Trace
 *   "ij->ji"        Transpose
 *   "i,j->ij"       Outer product
 *
 * In Tensor Logic, this is the foundation of all operations:
 * - Logical AND = element-wise multiplication
 * - Logical OR = element-wise maximum (or addition with clipping)
 * - Existential quantification = sum over index
 * - Universal quantification = product over index
 */
class EinsumSpec
{
private:
	std::string _notation;                  // Original notation string
	std::vector<std::string> _input_specs;  // Input subscript specs
	std::string _output_spec;               // Output subscript spec
	std::vector<char> _sum_indices;         // Indices to sum over
	std::vector<char> _output_indices;      // Indices in output
	bool _implicit_output;                  // Output computed implicitly

	void parse(const std::string& notation);

public:
	EinsumSpec() : _implicit_output(true) {}
	EinsumSpec(const std::string& notation);

	const std::string& notation() const { return _notation; }
	const std::vector<std::string>& input_specs() const { return _input_specs; }
	const std::string& output_spec() const { return _output_spec; }
	const std::vector<char>& sum_indices() const { return _sum_indices; }
	const std::vector<char>& output_indices() const { return _output_indices; }
	size_t num_inputs() const { return _input_specs.size(); }
	bool has_explicit_output() const { return !_implicit_output; }

	// Validate that tensors match the spec
	bool validate(const std::vector<ATenValuePtr>& tensors) const;

	// Get expected output shape
	std::vector<int64_t> output_shape(
		const std::vector<ATenValuePtr>& tensors) const;

	std::string to_string() const;
};

// ============================================================
// Nonlinearity Functions
// ============================================================

/**
 * Nonlinearity functions applied element-wise after tensor operations.
 * These transform continuous values for different reasoning modes.
 */
enum class Nonlinearity {
	NONE,       // Identity function (linear)
	THRESHOLD,  // H(x) = 1 if x > 0.5, else 0 (Boolean mode)
	SIGMOID,    // σ(x) = 1/(1+exp(-x)) (smooth approximation)
	RELU,       // max(0, x)
	TANH,       // tanh(x)
	SOFTMAX,    // exp(x)/sum(exp(x)) (for distributions)
	LOG,        // log(x) (for log-space operations)
	EXP,        // exp(x) (for probabilities)
	CLAMP01     // clamp(x, 0, 1) (for probabilities)
};

/**
 * Apply nonlinearity to tensor.
 */
ATenValuePtr apply_nonlinearity(const ATenValuePtr& tensor,
                                 Nonlinearity nl);

// ============================================================
// Tensor Equation
// ============================================================

/**
 * TensorEquation represents a single tensor logic equation:
 *
 *   LHS = H(Einsum(RHS_tensors))
 *
 * where:
 *   - LHS is the output tensor being computed
 *   - Einsum performs tensor joins and projections
 *   - H is an optional nonlinearity applied element-wise
 *
 * In Datalog terms:
 *   head(X,Z) :- body1(X,Y), body2(Y,Z).
 *
 * becomes:
 *   head[x,z] = H(sum_y body1[x,y] * body2[y,z])
 *
 * This unifies logical rules with neural network layers.
 */
class TensorEquation
{
private:
	std::string _name;              // Equation identifier
	std::string _lhs_name;          // Name of output tensor/relation
	std::vector<std::string> _rhs_names; // Names of input tensors/relations
	EinsumSpec _einsum;             // Einstein summation specification
	Nonlinearity _nonlinearity;     // Element-wise nonlinearity
	ReasoningMode _mode;            // Reasoning mode

	// For learnable parameters
	bool _learnable;                // Whether this equation has learnable params
	ATenValuePtr _weight;           // Optional learnable weight tensor
	ATenValuePtr _bias;             // Optional learnable bias tensor

	// Accumulated gradients (reset by zero_grad())
	ATenValuePtr _weight_grad;      // Gradient for weight
	ATenValuePtr _bias_grad;        // Gradient for bias

public:
	/**
	 * Construct a tensor equation.
	 *
	 * @param name Unique identifier for this equation
	 * @param lhs_name Name of the output tensor
	 * @param rhs_names Names of input tensors
	 * @param einsum_notation Einstein summation notation
	 * @param nl Nonlinearity to apply
	 * @param mode Reasoning mode
	 */
	TensorEquation(const std::string& name,
	               const std::string& lhs_name,
	               const std::vector<std::string>& rhs_names,
	               const std::string& einsum_notation,
	               Nonlinearity nl = Nonlinearity::NONE,
	               ReasoningMode mode = ReasoningMode::CONTINUOUS);

	// Accessors
	const std::string& name() const { return _name; }
	const std::string& lhs_name() const { return _lhs_name; }
	const std::vector<std::string>& rhs_names() const { return _rhs_names; }
	const EinsumSpec& einsum() const { return _einsum; }
	Nonlinearity nonlinearity() const { return _nonlinearity; }
	ReasoningMode mode() const { return _mode; }

	void set_nonlinearity(Nonlinearity nl) { _nonlinearity = nl; }
	void set_mode(ReasoningMode m) { _mode = m; }

	// Learnable parameters
	bool is_learnable() const { return _learnable; }
	void set_learnable(bool l) { _learnable = l; }

	ATenValuePtr weight() const { return _weight; }
	ATenValuePtr bias() const { return _bias; }
	void set_weight(const ATenValuePtr& w) { _weight = w; }
	void set_bias(const ATenValuePtr& b) { _bias = b; }

	// Accumulated gradient accessors
	ATenValuePtr weight_grad() const { return _weight_grad; }
	ATenValuePtr bias_grad() const { return _bias_grad; }

	/**
	 * Reset accumulated gradients to zero.
	 */
	void zero_grad();

	/**
	 * Execute the equation with given input tensors.
	 *
	 * @param inputs Map from tensor names to tensor values
	 * @return The computed output tensor
	 */
	ATenValuePtr execute(
		const std::map<std::string, ATenValuePtr>& inputs) const;

	/**
	 * Execute the equation on specific tensor values.
	 *
	 * @param tensors Input tensors in order
	 * @return The computed output tensor
	 */
	ATenValuePtr execute(const std::vector<ATenValuePtr>& tensors) const;

	/**
	 * Compute gradient of output with respect to inputs.
	 * Accumulates gradients into _weight_grad and _bias_grad when
	 * is_learnable() is true.
	 *
	 * @param grad_output gradient of loss with respect to this equation's output
	 * @param inputs the input tensors used in the forward pass
	 * @return gradients with respect to each input tensor (same order as inputs)
	 */
	std::vector<ATenValuePtr> backward(
		const ATenValuePtr& grad_output,
		const std::vector<ATenValuePtr>& inputs);

	std::string to_string() const;
};

// ============================================================
// Tensor Program
// ============================================================

/**
 * TensorProgram is a collection of tensor equations that together
 * define a reasoning system. It combines:
 *
 * - Facts: Tensors representing known relations (sparse or dense)
 * - Equations: Rules for deriving new relations
 * - Weights: Learnable parameters for continuous reasoning
 *
 * Execution modes:
 * - Forward chaining: Start from facts, derive all conclusions
 * - Backward chaining: Start from query, find supporting evidence
 * - Gradient descent: Learn weights from examples
 *
 * This is the complete representation of a Tensor Logic program.
 */
class TensorProgram
{
private:
	std::string _name;

	// Facts: Named tensors representing ground truth
	std::map<std::string, ATenValuePtr> _facts;

	// Equations: Rules for deriving new facts
	std::vector<TensorEquationPtr> _equations;

	// Derived tensors: Computed results
	std::map<std::string, ATenValuePtr> _derived;

	// Execution parameters
	ReasoningMode _mode;
	size_t _max_iterations;   // For fixed-point computation
	double _convergence_threshold;

	// For learning
	double _learning_rate;
	bool _track_gradients;
	double _grad_clip;        // Gradient clipping threshold (0 = disabled)

	// Accumulated tensor gradients (keyed by tensor name)
	std::map<std::string, ATenValuePtr> _tensor_grads;

	// Training history
	std::vector<double> _loss_history;

	// Statistics
	size_t _forward_count;
	size_t _backward_count;

public:
	/**
	 * Construct a tensor program.
	 *
	 * @param name Program identifier
	 * @param mode Default reasoning mode
	 */
	TensorProgram(const std::string& name = "unnamed",
	              ReasoningMode mode = ReasoningMode::CONTINUOUS);

	// Accessors
	const std::string& name() const { return _name; }
	ReasoningMode mode() const { return _mode; }
	void set_mode(ReasoningMode m) { _mode = m; }

	/**
	 * Set maximum number of iterations for forward_to_fixpoint().
	 */
	void set_max_iterations(size_t n) { _max_iterations = n; }
	size_t max_iterations() const { return _max_iterations; }

	/**
	 * Set convergence threshold for forward_to_fixpoint().
	 * Iteration stops when the maximum absolute change in any
	 * derived tensor value falls below this threshold.
	 */
	void set_convergence_threshold(double t) { _convergence_threshold = t; }
	double convergence_threshold() const { return _convergence_threshold; }

	// ========================================
	// Fact Management

	/**
	 * Add a fact tensor to the program.
	 */
	void add_fact(const std::string& name, const ATenValuePtr& tensor);

	/**
	 * Get a fact tensor by name.
	 */
	ATenValuePtr get_fact(const std::string& name) const;

	/**
	 * Check if a fact exists.
	 */
	bool has_fact(const std::string& name) const;

	/**
	 * Remove a fact.
	 */
	void remove_fact(const std::string& name);

	/**
	 * Clear all facts.
	 */
	void clear_facts();

	/**
	 * Get all fact names.
	 */
	std::vector<std::string> fact_names() const;

	// ========================================
	// Equation Management

	/**
	 * Add an equation to the program.
	 */
	void add_equation(const TensorEquationPtr& eq);

	/**
	 * Add an equation from specification.
	 */
	void add_equation(const std::string& name,
	                  const std::string& lhs,
	                  const std::vector<std::string>& rhs,
	                  const std::string& einsum,
	                  Nonlinearity nl = Nonlinearity::NONE);

	/**
	 * Get equation by name.
	 */
	TensorEquationPtr get_equation(const std::string& name) const;

	/**
	 * Get all equations.
	 */
	const std::vector<TensorEquationPtr>& equations() const { return _equations; }

	/**
	 * Remove equation by name.
	 */
	void remove_equation(const std::string& name);

	// ========================================
	// Execution

	/**
	 * Execute all equations once (single forward pass).
	 * Updates derived tensors.
	 */
	void forward();

	/**
	 * Execute until fixed point or max iterations.
	 * For recursive rules, iterates until values converge.
	 */
	void forward_to_fixpoint();

	/**
	 * Get a derived tensor by name.
	 */
	ATenValuePtr get_derived(const std::string& name) const;

	/**
	 * Get a tensor (fact or derived) by name.
	 */
	ATenValuePtr get_tensor(const std::string& name) const;

	/**
	 * Clear all derived tensors.
	 */
	void clear_derived();

	// ========================================
	// Backward Chaining (Query-Driven)

	/**
	 * Query for a specific relation.
	 * Uses backward chaining to find supporting evidence.
	 *
	 * @param query_name Name of the relation to query
	 * @param indices Specific indices to query (or empty for all)
	 * @return The computed tensor for the query
	 */
	ATenValuePtr query(const std::string& query_name,
	                   const std::vector<int64_t>& indices = {});

	/**
	 * Find equations that can derive a given relation.
	 */
	std::vector<TensorEquationPtr> find_deriving_equations(
		const std::string& relation_name) const;

	// ========================================
	// Learning

	/**
	 * Set learning rate.
	 */
	void set_learning_rate(double lr) { _learning_rate = lr; }
	double learning_rate() const { return _learning_rate; }

	/**
	 * Enable/disable gradient tracking.
	 */
	void set_track_gradients(bool t) { _track_gradients = t; }
	bool track_gradients() const { return _track_gradients; }

	/**
	 * Set gradient clipping threshold.
	 * Gradients with L2-norm exceeding this value are scaled down.
	 * Set to 0 (default) to disable clipping.
	 */
	void set_grad_clip(double clip) { _grad_clip = clip; }
	double grad_clip() const { return _grad_clip; }

	/**
	 * Return per-epoch loss values recorded by train().
	 */
	const std::vector<double>& loss_history() const { return _loss_history; }

	/**
	 * Compute loss between derived and target tensors.
	 */
	double compute_loss(const std::string& name,
	                    const ATenValuePtr& target) const;

	/**
	 * Backpropagate gradients through the program.
	 */
	void backward(const std::string& output_name,
	              const ATenValuePtr& grad_output);

	/**
	 * Update learnable parameters using computed gradients.
	 */
	void update_parameters();

	/**
	 * Train on a batch of examples.
	 *
	 * @param inputs Map of input tensor names to values
	 * @param targets Map of target tensor names to values
	 * @param epochs Number of training epochs
	 * @return Final loss value
	 */
	double train(const std::map<std::string, ATenValuePtr>& inputs,
	             const std::map<std::string, ATenValuePtr>& targets,
	             size_t epochs = 100);

	// ========================================
	// Conversion

	/**
	 * Export program to human-readable format.
	 */
	std::string to_string() const;

	/**
	 * Export program to Datalog format.
	 */
	std::string to_datalog() const;

	/**
	 * Get all learnable parameters.
	 */
	std::vector<ATenValuePtr> get_parameters() const;

	/**
	 * Set all learnable parameters.
	 */
	void set_parameters(const std::vector<ATenValuePtr>& params);

	// ========================================
	// Statistics

	size_t num_facts() const { return _facts.size(); }
	size_t num_equations() const { return _equations.size(); }
	size_t num_derived() const { return _derived.size(); }
	size_t forward_count() const { return _forward_count; }
	size_t backward_count() const { return _backward_count; }
};

// ============================================================
// Einsum Implementation
// ============================================================

/**
 * Execute Einstein summation on tensors.
 * This is the core operation of Tensor Logic.
 *
 * @param notation Einsum notation string (e.g., "ij,jk->ik")
 * @param tensors Input tensors
 * @return Result tensor
 */
ATenValuePtr einsum(const std::string& notation,
                     const std::vector<ATenValuePtr>& tensors);

/**
 * Execute Einstein summation with specification object.
 */
ATenValuePtr einsum(const EinsumSpec& spec,
                     const std::vector<ATenValuePtr>& tensors);

// ============================================================
// Helper Functions
// ============================================================

/**
 * Create a fact tensor from an AtomSpace relation.
 * Converts atoms to sparse tensor representation.
 *
 * @param as AtomSpace to query
 * @param relation_type Type of relation to extract
 * @param entity_index Mapping from atoms to indices
 * @return Sparse tensor representing the relation
 */
ATenValuePtr relation_to_tensor(
	AtomSpace* as,
	Type relation_type,
	const std::map<Handle, int64_t>& entity_index);

/**
 * Convert tensor back to AtomSpace atoms.
 * Creates atoms for non-zero entries.
 *
 * @param tensor The tensor to convert
 * @param relation_type Type of relation to create
 * @param index_to_entity Mapping from indices to atoms
 * @param as AtomSpace to add atoms to
 * @param threshold Minimum value to create atom (for continuous mode)
 * @return Created atoms
 */
HandleSeq tensor_to_atoms(
	const ATenValuePtr& tensor,
	Type relation_type,
	const std::vector<Handle>& index_to_entity,
	AtomSpace* as,
	double threshold = 0.5);

/**
 * Parse a Datalog-style rule into a TensorEquation.
 *
 * @param rule String like "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)."
 * @return Parsed tensor equation
 */
TensorEquationPtr parse_datalog_rule(const std::string& rule);

/** @}*/
} // namespace opencog

#endif // _OPENCOG_TENSOR_EQUATION_H
