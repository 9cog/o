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
class Optimizer;
class SparseTensor;

typedef std::shared_ptr<TensorEquation> TensorEquationPtr;
typedef std::shared_ptr<TensorProgram> TensorProgramPtr;
typedef std::shared_ptr<Optimizer> OptimizerPtr;
typedef std::shared_ptr<SparseTensor> SparseTensorPtr;

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
// Optimizer Classes
// ============================================================

/**
 * Learning rate schedule types.
 */
enum class LRSchedule {
	CONSTANT,     // No decay
	STEP,         // Decay by factor at intervals
	EXPONENTIAL,  // Continuous exponential decay
	COSINE,       // Cosine annealing
	WARMUP        // Linear warmup then constant
};

/**
 * Base Optimizer class for gradient-based learning.
 */
class Optimizer
{
protected:
	double _learning_rate;
	double _initial_lr;
	LRSchedule _schedule;
	size_t _step_count;
	size_t _warmup_steps;
	double _decay_rate;

public:
	Optimizer(double lr = 0.01, LRSchedule schedule = LRSchedule::CONSTANT);
	virtual ~Optimizer() = default;

	double learning_rate() const { return _learning_rate; }
	void set_learning_rate(double lr) { _learning_rate = lr; _initial_lr = lr; }
	void set_schedule(LRSchedule s) { _schedule = s; }
	void set_warmup_steps(size_t steps) { _warmup_steps = steps; }
	void set_decay_rate(double rate) { _decay_rate = rate; }

	/**
	 * Update learning rate based on schedule.
	 */
	virtual void step_schedule();

	/**
	 * Apply gradient update to parameter.
	 * @param param Current parameter tensor
	 * @param grad Gradient tensor
	 * @return Updated parameter tensor
	 */
	virtual ATenValuePtr update(const ATenValuePtr& param,
	                            const ATenValuePtr& grad) = 0;

	/**
	 * Reset optimizer state (momentum, etc.)
	 */
	virtual void reset();

	size_t step_count() const { return _step_count; }
};

/**
 * Stochastic Gradient Descent with optional momentum.
 */
class SGDOptimizer : public Optimizer
{
private:
	double _momentum;
	double _weight_decay;
	std::map<void*, ATenValuePtr> _velocity;  // Momentum buffer

public:
	SGDOptimizer(double lr = 0.01, double momentum = 0.0,
	             double weight_decay = 0.0);

	void set_momentum(double m) { _momentum = m; }
	double momentum() const { return _momentum; }

	ATenValuePtr update(const ATenValuePtr& param,
	                    const ATenValuePtr& grad) override;
	void reset() override;
};

typedef std::shared_ptr<SGDOptimizer> SGDOptimizerPtr;

/**
 * Adam optimizer with adaptive learning rates.
 * Combines momentum (first moment) with RMSprop (second moment).
 */
class AdamOptimizer : public Optimizer
{
private:
	double _beta1;        // First moment decay (default 0.9)
	double _beta2;        // Second moment decay (default 0.999)
	double _epsilon;      // Numerical stability (default 1e-8)
	double _weight_decay; // L2 regularization

	// Per-parameter state
	std::map<void*, ATenValuePtr> _m;  // First moment estimates
	std::map<void*, ATenValuePtr> _v;  // Second moment estimates

public:
	AdamOptimizer(double lr = 0.001, double beta1 = 0.9,
	              double beta2 = 0.999, double epsilon = 1e-8,
	              double weight_decay = 0.0);

	void set_betas(double beta1, double beta2) {
		_beta1 = beta1;
		_beta2 = beta2;
	}

	ATenValuePtr update(const ATenValuePtr& param,
	                    const ATenValuePtr& grad) override;
	void reset() override;
};

typedef std::shared_ptr<AdamOptimizer> AdamOptimizerPtr;

// ============================================================
// Sparse Tensor
// ============================================================

/**
 * SparseTensor represents a tensor with mostly zero values.
 * Uses COO (Coordinate) format: lists of (indices, value) pairs.
 *
 * Efficient for:
 * - Large relation tensors with few true facts
 * - Knowledge graph representations
 * - Symbolic reasoning with discrete facts
 */
class SparseTensor
{
private:
	std::vector<std::vector<int64_t>> _indices; // List of index tuples
	std::vector<double> _values;                 // Corresponding values
	std::vector<int64_t> _shape;                 // Dense shape
	size_t _nnz;                                 // Number of non-zeros

public:
	SparseTensor(const std::vector<int64_t>& shape);

	// Access
	const std::vector<int64_t>& shape() const { return _shape; }
	size_t nnz() const { return _nnz; }
	size_t ndim() const { return _shape.size(); }

	/**
	 * Get value at indices (returns 0 if not set).
	 */
	double get(const std::vector<int64_t>& indices) const;

	/**
	 * Set value at indices.
	 */
	void set(const std::vector<int64_t>& indices, double value);

	/**
	 * Add a non-zero entry.
	 */
	void add_entry(const std::vector<int64_t>& indices, double value);

	/**
	 * Get all indices.
	 */
	const std::vector<std::vector<int64_t>>& indices() const { return _indices; }

	/**
	 * Get all values.
	 */
	const std::vector<double>& values() const { return _values; }

	/**
	 * Convert to dense ATenValue.
	 */
	ATenValuePtr to_dense() const;

	/**
	 * Create from dense ATenValue (with threshold for sparsity).
	 */
	static SparseTensorPtr from_dense(const ATenValuePtr& dense,
	                                   double threshold = 1e-10);

	/**
	 * Sparse matrix multiplication: C = A * B
	 * Works with sparse A or B, produces sparse result.
	 */
	SparseTensorPtr matmul(const SparseTensor& other) const;

	/**
	 * Element-wise operations with sparse tensors.
	 */
	SparseTensorPtr add(const SparseTensor& other) const;
	SparseTensorPtr mul(const SparseTensor& other) const;

	/**
	 * Threshold to remove small values.
	 */
	void threshold(double min_value);

	/**
	 * Get sparsity ratio (nnz / total elements).
	 */
	double sparsity() const;

	std::string to_string() const;
};

// ============================================================
// Gradient Tape for Automatic Differentiation
// ============================================================

/**
 * GradientTape records operations for automatic differentiation.
 * Enable gradient tracking during forward pass, then call
 * backward() to compute gradients.
 */
class GradientTape
{
private:
	struct Operation {
		std::string type;
		std::vector<ATenValuePtr> inputs;
		ATenValuePtr output;
		std::function<std::vector<ATenValuePtr>(const ATenValuePtr&)> backward_fn;
	};

	std::vector<Operation> _operations;
	std::map<void*, ATenValuePtr> _gradients;
	bool _recording;

public:
	GradientTape();

	/**
	 * Start recording operations.
	 */
	void start();

	/**
	 * Stop recording.
	 */
	void stop();

	/**
	 * Check if recording.
	 */
	bool is_recording() const { return _recording; }

	/**
	 * Record an operation.
	 */
	void record(const std::string& type,
	            const std::vector<ATenValuePtr>& inputs,
	            const ATenValuePtr& output,
	            std::function<std::vector<ATenValuePtr>(const ATenValuePtr&)> backward_fn);

	/**
	 * Compute gradients via backpropagation.
	 * @param output The output tensor to differentiate
	 * @param grad_output Gradient of loss with respect to output
	 * @return Map from input tensors to their gradients
	 */
	std::map<void*, ATenValuePtr> backward(const ATenValuePtr& output,
	                                        const ATenValuePtr& grad_output);

	/**
	 * Get gradient for a specific tensor.
	 */
	ATenValuePtr gradient(const ATenValuePtr& tensor) const;

	/**
	 * Clear recorded operations and gradients.
	 */
	void clear();
};

typedef std::shared_ptr<GradientTape> GradientTapePtr;

// ============================================================
// Semiring Abstraction for Flexible Algebraic Reasoning
// ============================================================

/**
 * Semiring types for different reasoning tasks.
 * Based on Ben Goertzel's RAPTL (Resource-Aware Probabilistic Tensor Logic).
 *
 * Different semirings enable different kinds of inference:
 * - BOOLEAN: Reachability queries (OR/AND)
 * - COUNTING: Path enumeration (+/×)
 * - VITERBI: Optimal path finding (max/+)
 * - PROBABILISTIC: Expected value computation
 * - TROPICAL: Shortest path (min/+)
 * - LUKASIEWICZ: Fuzzy logic
 */
enum class SemiringType {
	BOOLEAN,       // (OR, AND, false, true) - reachability
	COUNTING,      // (+, ×, 0, 1) - path counting
	VITERBI,       // (max, +, -∞, 0) - most probable path
	PROBABILISTIC, // (+, ×, 0, 1) with normalization
	TROPICAL,      // (min, +, ∞, 0) - shortest path
	LUKASIEWICZ,   // (max, min, 0, 1) - fuzzy logic
	REAL           // Standard real arithmetic
};

/**
 * Semiring provides pluggable algebraic structures for tensor operations.
 * This enables the same tensor contraction code to perform different
 * kinds of reasoning depending on the semiring.
 */
class Semiring
{
private:
	SemiringType _type;
	double _zero;  // Additive identity
	double _one;   // Multiplicative identity

public:
	Semiring(SemiringType type = SemiringType::REAL);

	SemiringType type() const { return _type; }
	double zero() const { return _zero; }
	double one() const { return _one; }

	/**
	 * Semiring addition (⊕).
	 */
	double add(double a, double b) const;

	/**
	 * Semiring multiplication (⊗).
	 */
	double mul(double a, double b) const;

	/**
	 * Apply semiring operations to tensor element-wise.
	 */
	ATenValuePtr tensor_add(const ATenValuePtr& a, const ATenValuePtr& b) const;
	ATenValuePtr tensor_mul(const ATenValuePtr& a, const ATenValuePtr& b) const;

	/**
	 * Semiring sum reduction.
	 */
	ATenValuePtr tensor_sum(const ATenValuePtr& a) const;

	/**
	 * Execute einsum using this semiring's operations.
	 */
	ATenValuePtr einsum(const std::string& notation,
	                    const std::vector<ATenValuePtr>& tensors) const;

	std::string to_string() const;
};

typedef std::shared_ptr<Semiring> SemiringPtr;

// ============================================================
// PLN Truth Value Tensors
// ============================================================

/**
 * PLNTruthValue represents OpenCog's strength-confidence pairs.
 *
 * - Strength (s): Probability estimate [0, 1]
 * - Confidence (c): Evidence weight [0, 1)
 *
 * Confidence relates to count via: c = n / (n + k)
 * where n is observation count and k is a constant (typically 1-10).
 */
struct PLNTruthValue
{
	double strength;
	double confidence;

	PLNTruthValue(double s = 0.0, double c = 0.0)
		: strength(s), confidence(c) {}

	// PLN revision: combine two truth values
	PLNTruthValue revise(const PLNTruthValue& other) const;

	// PLN deduction: A->B, B->C => A->C
	static PLNTruthValue deduction(const PLNTruthValue& ab,
	                                const PLNTruthValue& bc);

	// PLN induction: A->B, A->C => B->C
	static PLNTruthValue induction(const PLNTruthValue& ab,
	                                const PLNTruthValue& ac);

	// PLN abduction: A->B, C->B => A->C
	static PLNTruthValue abduction(const PLNTruthValue& ab,
	                                const PLNTruthValue& cb);

	std::string to_string() const;
};

/**
 * PLNTensor stores strength-confidence pairs for each tensor element.
 * This enables proper uncertainty propagation through tensor operations.
 */
class PLNTensor
{
private:
	ATenValuePtr _strength;    // Strength values
	ATenValuePtr _confidence;  // Confidence values
	std::vector<int64_t> _shape;

public:
	PLNTensor(const std::vector<int64_t>& shape);
	PLNTensor(const ATenValuePtr& strength, const ATenValuePtr& confidence);

	const std::vector<int64_t>& shape() const { return _shape; }
	ATenValuePtr strength() const { return _strength; }
	ATenValuePtr confidence() const { return _confidence; }

	/**
	 * Get truth value at indices.
	 */
	PLNTruthValue get(const std::vector<int64_t>& indices) const;

	/**
	 * Set truth value at indices.
	 */
	void set(const std::vector<int64_t>& indices, const PLNTruthValue& tv);

	/**
	 * PLN revision: combine two PLN tensors element-wise.
	 */
	std::shared_ptr<PLNTensor> revise(const PLNTensor& other) const;

	/**
	 * PLN deduction via matrix multiplication with uncertainty propagation.
	 */
	std::shared_ptr<PLNTensor> deduction(const PLNTensor& other) const;

	/**
	 * Convert to standard tensor (using strength values).
	 */
	ATenValuePtr to_tensor() const { return _strength; }

	/**
	 * Create from standard tensor (with default confidence).
	 */
	static std::shared_ptr<PLNTensor> from_tensor(const ATenValuePtr& tensor,
	                                               double default_confidence = 0.9);

	std::string to_string() const;
};

typedef std::shared_ptr<PLNTensor> PLNTensorPtr;

// ============================================================
// Resource Tracking
// ============================================================

/**
 * ResourceMetrics tracks computational resource usage.
 * Based on RAPTL's resource-aware computation model.
 */
struct ResourceMetrics
{
	size_t memory_bytes;      // Memory consumption
	size_t flops;             // Floating point operations
	size_t tensor_elements;   // Total tensor elements processed
	double sparsity;          // Average sparsity of tensors
	size_t cache_misses;      // Estimated cache misses
	double bandwidth_gb;      // Memory bandwidth used (GB)
	double compute_time_ms;   // Estimated compute time

	ResourceMetrics();

	void reset();
	void add(const ResourceMetrics& other);

	/**
	 * Estimate resources for a tensor operation.
	 */
	static ResourceMetrics estimate_matmul(int64_t m, int64_t n, int64_t k);
	static ResourceMetrics estimate_einsum(const EinsumSpec& spec,
	                                        const std::vector<ATenValuePtr>& tensors);

	std::string to_string() const;
};

/**
 * ResourceTracker accumulates resource usage across operations.
 */
class ResourceTracker
{
private:
	ResourceMetrics _total;
	ResourceMetrics _current_op;
	std::vector<ResourceMetrics> _history;
	bool _tracking;
	size_t _memory_limit;
	size_t _flops_limit;

public:
	ResourceTracker();

	void start_tracking();
	void stop_tracking();
	bool is_tracking() const { return _tracking; }

	void record_operation(const ResourceMetrics& metrics);
	void clear_history();

	const ResourceMetrics& total() const { return _total; }
	const std::vector<ResourceMetrics>& history() const { return _history; }

	void set_memory_limit(size_t bytes) { _memory_limit = bytes; }
	void set_flops_limit(size_t flops) { _flops_limit = flops; }
	bool within_limits() const;

	std::string to_string() const;
};

typedef std::shared_ptr<ResourceTracker> ResourceTrackerPtr;

// ============================================================
// Linear Logic Modalities for Memory Safety
// ============================================================

/**
 * LinearModality represents linear logic resource semantics.
 * Used for safe tensor memory management.
 *
 * From Ben Goertzel's RAPTL:
 * - LINEAR: Use exactly once, then delete (move semantics)
 * - AFFINE: Use at most once (optional consumption)
 * - BANG: Read-only sharing (reference counting)
 * - WITH: Choice between alternatives
 */
enum class LinearModality {
	LINEAR,   // A ⊸ B : exactly once, move semantics
	AFFINE,   // A → B : at most once, optional
	BANG,     // !A : read-only, shared reference
	WITH      // A & B : choice/branching
};

/**
 * ManagedTensor wraps ATenValue with linear logic semantics.
 */
class ManagedTensor
{
private:
	ATenValuePtr _tensor;
	LinearModality _modality;
	size_t _ref_count;
	bool _consumed;
	mutable std::mutex _mtx;

public:
	ManagedTensor(const ATenValuePtr& tensor,
	              LinearModality modality = LinearModality::BANG);

	LinearModality modality() const { return _modality; }
	bool is_consumed() const { return _consumed; }
	size_t ref_count() const { return _ref_count; }

	/**
	 * Acquire tensor for use. Respects linear modality.
	 * @param consume Whether to consume (move) the tensor
	 * @return The tensor, or nullptr if unavailable
	 */
	ATenValuePtr acquire(bool consume = false);

	/**
	 * Release reference (for BANG modality).
	 */
	void release();

	/**
	 * Check if tensor can be acquired.
	 */
	bool available() const;

	std::string to_string() const;
};

typedef std::shared_ptr<ManagedTensor> ManagedTensorPtr;

// ============================================================
// RAPTL: Resource-Aware Probabilistic Tensor Logic
// ============================================================

/**
 * RAPTLValue represents the "triple product quantale" from Ben Goertzel's
 * RAPTL framework: Q = Q-logic × Q-uncertainty × Q-resource
 *
 * Each piece of information carries:
 * - Logical content (the tensor data)
 * - Uncertainty (PLN truth values)
 * - Resource requirements (memory, FLOPs)
 */
class RAPTLValue
{
private:
	ATenValuePtr _tensor;       // Logical content
	PLNTensorPtr _uncertainty;  // Uncertainty (strength-confidence)
	ResourceMetrics _resources; // Resource tracking
	SemiringPtr _semiring;      // Algebraic structure
	LinearModality _modality;   // Memory semantics

public:
	RAPTLValue(const ATenValuePtr& tensor,
	           SemiringType semiring = SemiringType::REAL);

	// Accessors
	ATenValuePtr tensor() const { return _tensor; }
	PLNTensorPtr uncertainty() const { return _uncertainty; }
	const ResourceMetrics& resources() const { return _resources; }
	SemiringPtr semiring() const { return _semiring; }
	LinearModality modality() const { return _modality; }

	// Setters
	void set_uncertainty(const PLNTensorPtr& u) { _uncertainty = u; }
	void set_modality(LinearModality m) { _modality = m; }

	/**
	 * RAPTL multiplication: combines logical, uncertainty, and resource.
	 */
	std::shared_ptr<RAPTLValue> multiply(const RAPTLValue& other) const;

	/**
	 * RAPTL addition: combines using semiring operations.
	 */
	std::shared_ptr<RAPTLValue> add(const RAPTLValue& other) const;

	/**
	 * RAPTL einsum: tensor contraction with full tracking.
	 */
	static std::shared_ptr<RAPTLValue> einsum(
		const std::string& notation,
		const std::vector<std::shared_ptr<RAPTLValue>>& inputs);

	/**
	 * Check if operation is within resource limits.
	 */
	bool within_limits(size_t memory_limit, size_t flops_limit) const;

	std::string to_string() const;
};

typedef std::shared_ptr<RAPTLValue> RAPTLValuePtr;

/**
 * RAPTLProgram extends TensorProgram with full RAPTL semantics.
 */
class RAPTLProgram
{
private:
	std::string _name;
	std::map<std::string, RAPTLValuePtr> _facts;
	std::vector<TensorEquationPtr> _equations;
	std::map<std::string, RAPTLValuePtr> _derived;

	SemiringPtr _semiring;
	ResourceTrackerPtr _tracker;
	ReasoningMode _mode;

public:
	RAPTLProgram(const std::string& name = "raptl",
	             SemiringType semiring = SemiringType::REAL);

	// Fact management
	void add_fact(const std::string& name, const RAPTLValuePtr& value);
	RAPTLValuePtr get_fact(const std::string& name) const;

	// Equation management
	void add_equation(const TensorEquationPtr& eq);
	void add_equation(const std::string& name,
	                  const std::string& lhs,
	                  const std::vector<std::string>& rhs,
	                  const std::string& einsum,
	                  Nonlinearity nl = Nonlinearity::NONE);

	// Inference with full RAPTL tracking
	void forward();
	RAPTLValuePtr query(const std::string& name);

	// Resource management
	ResourceTrackerPtr tracker() const { return _tracker; }
	void set_resource_limits(size_t memory, size_t flops);
	bool within_limits() const;

	// Semiring selection
	void set_semiring(SemiringType type);
	SemiringPtr semiring() const { return _semiring; }

	std::string to_string() const;
};

typedef std::shared_ptr<RAPTLProgram> RAPTLProgramPtr;

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
	 * Used for backpropagation in continuous mode.
	 */
	std::vector<ATenValuePtr> backward(
		const ATenValuePtr& grad_output,
		const std::vector<ATenValuePtr>& inputs) const;

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
	OptimizerPtr _optimizer;
	GradientTapePtr _tape;

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
	 * Set optimizer for gradient updates.
	 */
	void set_optimizer(const OptimizerPtr& opt) { _optimizer = opt; }
	OptimizerPtr optimizer() const { return _optimizer; }

	/**
	 * Get gradient tape for automatic differentiation.
	 */
	GradientTapePtr tape() const { return _tape; }

	/**
	 * Enable/disable gradient tracking.
	 */
	void set_track_gradients(bool t) { _track_gradients = t; }
	bool track_gradients() const { return _track_gradients; }

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
