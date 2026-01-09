/*
 * TensorLogic.h
 *
 * Multi-Entity & Multi-Scale Network-Aware Tensor Logic for AtomSpace
 *
 * This module implements "Tensor Logic" - a framework that bridges symbolic
 * AI reasoning with neural tensor computations, following the ATenSpace
 * architecture patterns.
 *
 * Key concepts:
 * - Multi-Entity: Tensors representing collections of entities with shared structure
 * - Multi-Scale: Hierarchical embeddings from local to global context
 * - Network-Aware: Embeddings that encode graph structure and relationships
 *
 * Copyright (c) 2025 OpenCog Foundation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation.
 */

#ifndef _OPENCOG_TENSOR_LOGIC_H
#define _OPENCOG_TENSOR_LOGIC_H

#ifdef HAVE_ATEN

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <mutex>

#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/truthvalue/TruthValue.h>

#include "TensorValue.h"
#include "ATenSpace.h"

namespace opencog {

// Forward declarations
class MultiScaleTensor;
class NetworkAwareTensor;
class TensorAttentionBank;
class TensorPatternMatcher;

typedef std::shared_ptr<MultiScaleTensor> MultiScaleTensorPtr;
typedef std::shared_ptr<NetworkAwareTensor> NetworkAwareTensorPtr;
typedef std::shared_ptr<TensorAttentionBank> TensorAttentionBankPtr;
typedef std::shared_ptr<TensorPatternMatcher> TensorPatternMatcherPtr;

/**
 * ScaleLevel - Defines the hierarchical scale for embeddings
 */
enum class ScaleLevel {
	LOCAL,      // Node-level features
	NEIGHBOR,   // 1-hop neighborhood aggregation
	COMMUNITY,  // Local cluster/community level
	GLOBAL      // Whole-graph context
};

/**
 * AttentionType - Types of attention mechanisms
 */
enum class AttentionType {
	SELF,       // Self-attention within entity set
	CROSS,      // Cross-attention between entity sets
	GRAPH,      // Graph attention (GAT-style)
	HEBBIAN     // Hebbian learning-based attention
};

// ============================================================
// MultiScaleTensor
// ============================================================

/**
 * MultiScaleTensor - Hierarchical tensor embeddings at multiple scales.
 *
 * This class implements multi-scale representations where each atom has
 * embeddings at different granularities:
 * - Local: Direct features of the atom itself
 * - Neighbor: Aggregated features from immediate connections
 * - Community: Cluster-level context
 * - Global: Whole-graph summary
 *
 * The multi-scale representation enables reasoning at different levels
 * of abstraction simultaneously.
 */
class MultiScaleTensor
{
private:
	std::unordered_map<ScaleLevel, TensorValuePtr> _scale_embeddings;
	std::vector<int64_t> _base_shape;
	int64_t _embedding_dim;
	mutable std::mutex _mtx;

public:
	MultiScaleTensor(int64_t embedding_dim = 128);
	~MultiScaleTensor();

	/** Set embedding at a specific scale */
	void set_scale(ScaleLevel level, const TensorValuePtr& embedding);

	/** Get embedding at a specific scale */
	TensorValuePtr get_scale(ScaleLevel level) const;

	/** Check if scale exists */
	bool has_scale(ScaleLevel level) const;

	/** Get concatenated multi-scale embedding */
	TensorValuePtr get_concatenated() const;

	/** Get weighted combination of scales */
	TensorValuePtr get_weighted(const std::vector<double>& weights) const;

	/** Compute hierarchical attention over scales */
	TensorValuePtr get_attended(const TensorValuePtr& query) const;

	/** Get embedding dimension */
	int64_t embedding_dim() const { return _embedding_dim; }

	/** Number of scales present */
	size_t num_scales() const;

	/** Convert to string representation */
	std::string to_string() const;
};

// ============================================================
// NetworkAwareTensor
// ============================================================

/**
 * NetworkAwareTensor - Graph-structure-aware tensor embeddings.
 *
 * This class produces embeddings that encode not just node features
 * but also structural information from the knowledge graph:
 * - Degree centrality encoding
 * - Positional encodings (graph Laplacian eigenvectors)
 * - Type-aware embeddings for different atom types
 * - Relationship-aware aggregation
 *
 * Follows message-passing neural network patterns for GNN integration.
 */
class NetworkAwareTensor
{
private:
	AtomSpace* _as;
	ATenSpace* _aten;
	int64_t _embedding_dim;
	int64_t _num_heads;
	bool _use_edge_features;

	// Learned parameters (can be initialized or loaded)
	TensorValuePtr _type_embeddings;      // Embeddings for atom types
	TensorValuePtr _position_encodings;   // Positional encodings
	TensorValuePtr _attention_weights;    // For graph attention

	mutable std::mutex _mtx;

	// Internal helpers
	TensorValuePtr compute_degree_encoding(const Handle& atom) const;
	TensorValuePtr compute_type_embedding(Type atom_type) const;
	TensorValuePtr aggregate_neighbors(const Handle& atom,
	                                   const Handle& embedding_key,
	                                   const std::string& aggregation) const;

public:
	NetworkAwareTensor(AtomSpace* as, ATenSpace* aten,
	                   int64_t embedding_dim = 128,
	                   int64_t num_heads = 4);
	~NetworkAwareTensor();

	/**
	 * Initialize type embeddings for known atom types.
	 */
	void init_type_embeddings(const std::vector<Type>& types);

	/**
	 * Compute network-aware embedding for an atom.
	 * Combines node features with structural information.
	 */
	TensorValuePtr compute_embedding(const Handle& atom,
	                                 const Handle& feature_key) const;

	/**
	 * Compute embeddings for multiple atoms with message passing.
	 * @param atoms List of atoms to embed
	 * @param feature_key Key for base features
	 * @param num_layers Number of message passing layers
	 * @param aggregation Aggregation method: "mean", "sum", "max", "attention"
	 */
	std::unordered_map<Handle, TensorValuePtr> compute_embeddings(
		const HandleSeq& atoms,
		const Handle& feature_key,
		int num_layers = 2,
		const std::string& aggregation = "attention") const;

	/**
	 * Perform one layer of graph attention message passing.
	 */
	std::unordered_map<Handle, TensorValuePtr> graph_attention_layer(
		const HandleSeq& atoms,
		const std::unordered_map<Handle, TensorValuePtr>& current_embeddings,
		const Handle& edge_key = Handle::UNDEFINED) const;

	/**
	 * Compute pairwise relationship-aware embeddings.
	 * For a pair (source, target), computes embedding that captures
	 * the relationship context.
	 */
	TensorValuePtr compute_pair_embedding(const Handle& source,
	                                      const Handle& target,
	                                      const Handle& relation_key) const;

	/** Get embedding dimension */
	int64_t embedding_dim() const { return _embedding_dim; }
};

// ============================================================
// TensorAttentionBank
// ============================================================

/**
 * TensorAttentionBank - ECAN-style attention allocation with tensors.
 *
 * Implements Economic Attention Networks (ECAN) patterns using tensors:
 * - Short-Term Importance (STI): Current focus/salience
 * - Long-Term Importance (LTI): Persistent relevance
 * - Attention spreading via Hebbian links
 * - Forgetting mechanism based on attention decay
 *
 * This enables cognitive-inspired attention allocation for guiding
 * reasoning and retrieval in the tensor-enhanced AtomSpace.
 */
class TensorAttentionBank
{
public:
	struct AttentionValue {
		double sti;         // Short-term importance
		double lti;         // Long-term importance
		double vlti;        // Very long-term importance
		TensorValuePtr attention_embedding;  // Neural attention vector
	};

private:
	AtomSpace* _as;
	ATenSpace* _aten;

	// Attention storage
	std::unordered_map<Handle, AttentionValue> _attention_values;
	mutable std::mutex _mtx;

	// Configuration
	double _sti_decay_rate;
	double _spreading_rate;
	double _hebbian_learning_rate;
	int64_t _attention_dim;

	// Attention focus set
	HandleSeq _attentional_focus;
	size_t _focus_size;

	// Hebbian links for attention spreading
	std::unordered_map<Handle, std::vector<std::pair<Handle, double>>> _hebbian_links;

	// Internal
	void decay_sti();
	void update_focus();
	double compute_activation(const Handle& atom) const;

public:
	TensorAttentionBank(AtomSpace* as, ATenSpace* aten,
	                    size_t focus_size = 100,
	                    int64_t attention_dim = 64);
	~TensorAttentionBank();

	/**
	 * Stimulate an atom, increasing its attention.
	 * @param atom The atom to stimulate
	 * @param stimulus Amount to increase STI
	 * @param context_embedding Optional context for neural attention
	 */
	void stimulate(const Handle& atom, double stimulus,
	               const TensorValuePtr& context_embedding = nullptr);

	/**
	 * Get attention value for an atom.
	 */
	AttentionValue get_attention(const Handle& atom) const;

	/**
	 * Set attention values directly.
	 */
	void set_attention(const Handle& atom, double sti, double lti);

	/**
	 * Get the attentional focus - atoms with highest attention.
	 */
	HandleSeq get_attentional_focus() const;

	/**
	 * Spread attention from source to connected atoms.
	 * Uses both symbolic links and tensor similarity.
	 */
	void spread_attention(const Handle& source);

	/**
	 * Update Hebbian link strength based on co-activation.
	 */
	void hebbian_update(const Handle& a, const Handle& b);

	/**
	 * Run one attention allocation cycle.
	 * Performs decay, spreading, and focus update.
	 */
	void attention_cycle();

	/**
	 * Compute attention-weighted retrieval.
	 * Returns atoms weighted by their attention values.
	 */
	std::vector<std::pair<Handle, double>> attention_weighted_query(
		const TensorValuePtr& query_embedding,
		const Handle& embedding_key,
		size_t k = 10) const;

	/**
	 * Get atoms similar to query, filtered by attention threshold.
	 */
	HandleSeq query_with_attention_filter(
		const TensorValuePtr& query_embedding,
		const Handle& embedding_key,
		double min_sti = 0.0,
		size_t max_results = 100) const;

	/** Configuration setters */
	void set_decay_rate(double rate) { _sti_decay_rate = rate; }
	void set_spreading_rate(double rate) { _spreading_rate = rate; }
	void set_hebbian_rate(double rate) { _hebbian_learning_rate = rate; }
};

// ============================================================
// TensorTruthValue
// ============================================================

/**
 * TensorTruthValue - Truth values backed by tensor representations.
 *
 * Extends traditional truth values with tensor-based uncertainty:
 * - Distributional representation of confidence
 * - Higher-order truth value features
 * - Support for fuzzy/probabilistic reasoning
 */
class TensorTruthValue
{
private:
	double _strength;
	double _confidence;
	TensorValuePtr _distribution;  // Full probability distribution
	TensorValuePtr _features;      // Higher-order features

public:
	TensorTruthValue(double strength = 1.0, double confidence = 1.0);
	TensorTruthValue(const TensorValuePtr& distribution);

	double strength() const { return _strength; }
	double confidence() const { return _confidence; }
	TensorValuePtr distribution() const { return _distribution; }
	TensorValuePtr features() const { return _features; }

	/** Set distributional truth value */
	void set_distribution(const TensorValuePtr& dist);

	/** Merge two truth values using tensor operations */
	static TensorTruthValue merge(const TensorTruthValue& a,
	                              const TensorTruthValue& b);

	/** Revision (Bayesian update) */
	static TensorTruthValue revision(const TensorTruthValue& a,
	                                 const TensorTruthValue& b);

	/** Deduction truth value formula */
	static TensorTruthValue deduction(const TensorTruthValue& premise1,
	                                  const TensorTruthValue& premise2);

	/** Induction truth value formula */
	static TensorTruthValue induction(const TensorTruthValue& premise1,
	                                  const TensorTruthValue& premise2);

	/** Convert to standard TruthValue */
	TruthValuePtr to_truth_value() const;

	/** Create from standard TruthValue */
	static TensorTruthValue from_truth_value(const TruthValuePtr& tv);
};

// ============================================================
// TensorLogic - Main Interface
// ============================================================

/**
 * TensorLogic - Unified interface for tensor-enhanced reasoning.
 *
 * This class provides the main API for Multi-Entity & Multi-Scale
 * Network-Aware Tensor Logic operations, combining:
 * - Multi-scale embeddings
 * - Network-aware representations
 * - Attention-guided reasoning
 * - Tensor-enhanced inference
 */
class TensorLogic
{
private:
	AtomSpace* _as;
	ATenSpace* _aten;
	TensorAttentionBankPtr _attention_bank;
	std::unique_ptr<NetworkAwareTensor> _network_tensor;

	int64_t _embedding_dim;
	mutable std::mutex _mtx;

	// Predefined keys
	Handle _embedding_key;
	Handle _multiscale_key;
	Handle _attention_key;
	Handle _truth_key;

public:
	TensorLogic(AtomSpace* as, int64_t embedding_dim = 128);
	~TensorLogic();

	/** Get underlying ATenSpace */
	ATenSpace* get_aten_space() { return _aten; }

	/** Get attention bank */
	TensorAttentionBankPtr get_attention_bank() { return _attention_bank; }

	// ---- Multi-Entity Operations ----

	/**
	 * Create entity batch embedding.
	 * Creates a tensor representing multiple entities simultaneously.
	 * @param entities List of entity atoms
	 * @param key Key for individual embeddings
	 * @return Batched tensor [N, D]
	 */
	TensorValuePtr create_entity_batch(const HandleSeq& entities,
	                                   const Handle& key) const;

	/**
	 * Compute entity-set embedding.
	 * Aggregates multiple entity embeddings into a single set representation.
	 * @param aggregation "mean", "max", "attention", "deepsets"
	 */
	TensorValuePtr aggregate_entities(const HandleSeq& entities,
	                                  const Handle& key,
	                                  const std::string& aggregation = "attention") const;

	/**
	 * Multi-entity similarity matrix.
	 * Computes pairwise similarities for a set of entities.
	 */
	TensorValuePtr entity_similarity_matrix(const HandleSeq& entities,
	                                        const Handle& key) const;

	/**
	 * Cluster entities by embedding similarity.
	 * @param k Number of clusters
	 * @return Map from cluster ID to entity handles
	 */
	std::unordered_map<int, HandleSeq> cluster_entities(
		const HandleSeq& entities,
		const Handle& key,
		int k) const;

	// ---- Multi-Scale Operations ----

	/**
	 * Compute multi-scale embedding for an atom.
	 * Generates embeddings at LOCAL, NEIGHBOR, COMMUNITY, GLOBAL scales.
	 */
	MultiScaleTensorPtr compute_multiscale_embedding(
		const Handle& atom,
		const Handle& feature_key) const;

	/**
	 * Store multi-scale embedding on an atom.
	 */
	void set_multiscale_embedding(const Handle& atom,
	                              const MultiScaleTensorPtr& mst);

	/**
	 * Retrieve multi-scale embedding from an atom.
	 */
	MultiScaleTensorPtr get_multiscale_embedding(const Handle& atom) const;

	/**
	 * Query at specific scale level.
	 * Finds atoms similar to query at the specified scale.
	 */
	std::vector<std::pair<Handle, double>> query_at_scale(
		const TensorValuePtr& query,
		ScaleLevel scale,
		size_t k = 10) const;

	/**
	 * Cross-scale similarity.
	 * Computes similarity between atoms at different scales.
	 */
	double cross_scale_similarity(const Handle& a, ScaleLevel scale_a,
	                              const Handle& b, ScaleLevel scale_b) const;

	// ---- Network-Aware Operations ----

	/**
	 * Compute network-aware embedding for atom.
	 * Incorporates graph structure into the embedding.
	 */
	TensorValuePtr compute_network_embedding(const Handle& atom,
	                                         const Handle& feature_key,
	                                         int num_layers = 2) const;

	/**
	 * Propagate embeddings through the graph.
	 * Message-passing style update for all atoms in set.
	 */
	void propagate_embeddings(const HandleSeq& atoms,
	                          const Handle& key,
	                          int num_iterations = 3);

	/**
	 * Compute relationship-aware embedding.
	 * For links, computes embedding that captures relationship semantics.
	 */
	TensorValuePtr compute_relation_embedding(const Handle& link) const;

	// ---- Tensor Logic Inference ----

	/**
	 * Tensor-enhanced deduction.
	 * Combines symbolic deduction with embedding-based inference.
	 * Given (A -> B) and (B -> C), infers (A -> C) with tensor TV.
	 */
	std::pair<Handle, TensorTruthValue> tensor_deduction(
		const Handle& premise1,
		const Handle& premise2) const;

	/**
	 * Tensor-enhanced induction.
	 * Infers general rule from specific instances using embeddings.
	 */
	std::pair<Handle, TensorTruthValue> tensor_induction(
		const HandleSeq& instances) const;

	/**
	 * Analogy by embedding similarity.
	 * Given (A : B :: C : ?), finds D such that rel(C,D) ≈ rel(A,B).
	 */
	Handle tensor_analogy(const Handle& A, const Handle& B,
	                      const Handle& C,
	                      const HandleSeq& candidates) const;

	/**
	 * Tensor-guided unification.
	 * Pattern matching guided by embedding similarity.
	 */
	std::vector<std::unordered_map<Handle, Handle>> tensor_unify(
		const Handle& pattern,
		const HandleSeq& candidates,
		double similarity_threshold = 0.5) const;

	// ---- Attention-Guided Reasoning ----

	/**
	 * Stimulate atoms relevant to query.
	 * Increases attention on atoms similar to query embedding.
	 */
	void stimulate_by_query(const TensorValuePtr& query,
	                        const Handle& key,
	                        double base_stimulus = 1.0);

	/**
	 * Attention-weighted forward chaining step.
	 * Applies rules prioritized by attention.
	 */
	HandleSeq attention_forward_step(const Handle& rule_base,
	                                 size_t max_applications = 10);

	/**
	 * Goal-directed reasoning with attention.
	 * Backward chaining guided by attention allocation.
	 */
	bool attention_backward_chain(const Handle& goal,
	                              int max_depth = 5);

	// ---- Utility ----

	/** Initialize default embeddings for all atoms of given types */
	void initialize_embeddings(const std::vector<Type>& types,
	                           const std::string& init = "randn");

	/** Run attention allocation cycle */
	void attention_cycle() { _attention_bank->attention_cycle(); }

	/** Get statistics */
	std::string stats() const;
};

// ============================================================
// Factory functions
// ============================================================

/** Create TensorLogic instance */
inline std::shared_ptr<TensorLogic> createTensorLogic(
	AtomSpace* as, int64_t embedding_dim = 128)
{
	return std::make_shared<TensorLogic>(as, embedding_dim);
}

/** Create MultiScaleTensor */
inline MultiScaleTensorPtr createMultiScaleTensor(int64_t embedding_dim = 128)
{
	return std::make_shared<MultiScaleTensor>(embedding_dim);
}

} // namespace opencog

#endif // HAVE_ATEN
#endif // _OPENCOG_TENSOR_LOGIC_H
