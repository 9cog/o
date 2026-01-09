/*
 * opencog/atoms/aten/TensorLogic.h
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * Tensor Logic for Multi-Entity & Multi-Scale Network-Aware
 * Tensor-Enhanced AtomSpace
 *
 * This module bridges symbolic AI (AtomSpace) with neural tensor embeddings,
 * implementing the integration patterns from o9nn/ATen and o9nn/ATenSpace.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#ifndef _OPENCOG_TENSOR_LOGIC_H
#define _OPENCOG_TENSOR_LOGIC_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <functional>

#include <opencog/atoms/base/Handle.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/aten/ATenValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

// Forward declarations
class AtomSpace;
class TensorLogic;

typedef std::shared_ptr<TensorLogic> TensorLogicPtr;

// ============================================================
// Entity Embedding - Maps atoms to dense vector representations

/**
 * EntityEmbedding provides dense vector representations for atoms,
 * enabling neural network style operations on symbolic structures.
 */
class EntityEmbedding
{
private:
	Handle _entity;
	ATenValuePtr _embedding;
	int64_t _scale_level;  // Multi-scale hierarchy level
	double _importance;    // Attention/importance weight

public:
	EntityEmbedding(const Handle& entity, const ATenValuePtr& embedding,
	                int64_t scale = 0, double importance = 1.0);

	const Handle& entity() const { return _entity; }
	const ATenValuePtr& embedding() const { return _embedding; }
	int64_t scale_level() const { return _scale_level; }
	double importance() const { return _importance; }

	void set_embedding(const ATenValuePtr& emb) { _embedding = emb; }
	void set_importance(double imp) { _importance = imp; }

	// Similarity computation
	double cosine_similarity(const EntityEmbedding& other) const;
	double euclidean_distance(const EntityEmbedding& other) const;
};

typedef std::shared_ptr<EntityEmbedding> EntityEmbeddingPtr;

// ============================================================
// Multi-Scale Tensor - Hierarchical tensor representations

/**
 * MultiScaleTensor manages tensor representations at different
 * scales of abstraction, enabling hierarchical reasoning.
 */
class MultiScaleTensor
{
private:
	std::vector<ATenValuePtr> _scales;  // Tensors at different scales
	std::vector<int64_t> _scale_dims;   // Dimension at each scale
	int64_t _num_scales;

public:
	MultiScaleTensor(int64_t num_scales, const std::vector<int64_t>& scale_dims);

	// Access specific scale
	ATenValuePtr at_scale(int64_t scale) const;
	void set_scale(int64_t scale, const ATenValuePtr& tensor);

	// Cross-scale operations
	ATenValuePtr upsample(int64_t from_scale, int64_t to_scale) const;
	ATenValuePtr downsample(int64_t from_scale, int64_t to_scale) const;
	ATenValuePtr aggregate_scales() const;

	int64_t num_scales() const { return _num_scales; }
	const std::vector<int64_t>& scale_dims() const { return _scale_dims; }
};

typedef std::shared_ptr<MultiScaleTensor> MultiScaleTensorPtr;

// ============================================================
// Network-Aware Tensor - Graph structure-aware operations

/**
 * NetworkAwareTensor incorporates graph topology into tensor operations,
 * enabling message passing and attention across the hypergraph.
 */
class NetworkAwareTensor
{
private:
	ATenValuePtr _node_features;      // Node feature matrix [N x D]
	ATenValuePtr _edge_features;      // Edge feature matrix [E x D]
	ATenValuePtr _adjacency;          // Adjacency matrix [N x N]
	ATenValuePtr _attention_weights;  // Attention weights [N x N]

	int64_t _num_nodes;
	int64_t _num_edges;
	int64_t _feature_dim;

public:
	NetworkAwareTensor(int64_t num_nodes, int64_t feature_dim);

	// Feature access
	ATenValuePtr node_features() const { return _node_features; }
	ATenValuePtr edge_features() const { return _edge_features; }
	ATenValuePtr adjacency() const { return _adjacency; }
	ATenValuePtr attention_weights() const { return _attention_weights; }

	void set_node_features(const ATenValuePtr& features);
	void set_edge_features(const ATenValuePtr& features);
	void set_adjacency(const ATenValuePtr& adj);

	// Graph operations
	ATenValuePtr message_passing(int64_t num_hops = 1) const;
	ATenValuePtr graph_attention() const;
	ATenValuePtr aggregate_neighbors(int64_t node_idx) const;

	// Update mechanisms
	void update_attention(const ATenValuePtr& query, const ATenValuePtr& key);
	void apply_hebbian_update(double learning_rate);

	int64_t num_nodes() const { return _num_nodes; }
	int64_t feature_dim() const { return _feature_dim; }
};

typedef std::shared_ptr<NetworkAwareTensor> NetworkAwareTensorPtr;

// ============================================================
// TruthValue Tensor - PLN-based probabilistic tensors

/**
 * TruthValueTensor combines PLN truth values with tensor representations,
 * enabling probabilistic reasoning over embeddings.
 */
class TruthValueTensor
{
private:
	ATenValuePtr _strength;    // Tensor of strength values
	ATenValuePtr _confidence;  // Tensor of confidence values
	ATenValuePtr _count;       // Evidence count tensor

public:
	TruthValueTensor(const std::vector<int64_t>& shape);
	TruthValueTensor(const ATenValuePtr& strength, const ATenValuePtr& confidence);

	ATenValuePtr strength() const { return _strength; }
	ATenValuePtr confidence() const { return _confidence; }
	ATenValuePtr count() const { return _count; }

	// PLN operations on tensors
	TruthValueTensor revision(const TruthValueTensor& other) const;
	TruthValueTensor deduction(const TruthValueTensor& other) const;
	TruthValueTensor induction(const TruthValueTensor& other) const;
	TruthValueTensor abduction(const TruthValueTensor& other) const;

	// Convert to/from scalars
	void set_uniform(double strength, double confidence);
	double mean_strength() const;
	double mean_confidence() const;
};

typedef std::shared_ptr<TruthValueTensor> TruthValueTensorPtr;

// ============================================================
// Tensor Logic - Main integration class

/**
 * TensorLogic is the main class that bridges AtomSpace symbolic reasoning
 * with neural tensor operations. It provides:
 *
 * - Entity embedding management
 * - Multi-scale tensor operations
 * - Network-aware message passing
 * - PLN-tensor integration
 * - Hebbian learning and attention
 *
 * Based on the integration patterns from o9nn/ATen and o9nn/ATenSpace.
 */
class TensorLogic
{
private:
	mutable std::mutex _mtx;

	// Associated AtomSpace
	AtomSpace* _atomspace;

	// Embedding storage
	std::map<Handle, EntityEmbeddingPtr> _embeddings;

	// Configuration
	int64_t _embedding_dim;
	int64_t _num_scales;
	std::string _device;  // "cpu" or "cuda:N"

	// Network representation
	NetworkAwareTensorPtr _network;

	// Multi-scale manager
	MultiScaleTensorPtr _multi_scale;

	// Keys for atom-tensor association
	Handle _embedding_key;
	Handle _attention_key;
	Handle _truth_tensor_key;

	// Statistics
	size_t _num_embeddings;
	size_t _update_count;

public:
	/**
	 * Construct TensorLogic with configuration.
	 *
	 * @param as Associated AtomSpace
	 * @param embedding_dim Dimension of entity embeddings
	 * @param num_scales Number of scale levels
	 * @param device Computation device ("cpu" or "cuda:N")
	 */
	TensorLogic(AtomSpace* as = nullptr,
	            int64_t embedding_dim = 128,
	            int64_t num_scales = 3,
	            const std::string& device = "cpu");

	~TensorLogic();

	// Prevent copying
	TensorLogic(const TensorLogic&) = delete;
	TensorLogic& operator=(const TensorLogic&) = delete;

	// ========================================
	// Configuration

	void set_atomspace(AtomSpace* as);
	AtomSpace* get_atomspace() const { return _atomspace; }

	void set_device(const std::string& device);
	std::string get_device() const { return _device; }

	int64_t embedding_dim() const { return _embedding_dim; }
	int64_t num_scales() const { return _num_scales; }

	// ========================================
	// Entity Embedding Management

	/**
	 * Get or create embedding for an atom.
	 * If no embedding exists, creates a random initialization.
	 */
	EntityEmbeddingPtr get_embedding(const Handle& atom);

	/**
	 * Set embedding for an atom.
	 */
	void set_embedding(const Handle& atom, const ATenValuePtr& embedding,
	                   int64_t scale = 0);

	/**
	 * Check if atom has an embedding.
	 */
	bool has_embedding(const Handle& atom) const;

	/**
	 * Remove embedding for an atom.
	 */
	void remove_embedding(const Handle& atom);

	/**
	 * Initialize embeddings for a set of atoms.
	 */
	void initialize_embeddings(const HandleSeq& atoms,
	                          const std::string& init_method = "random");

	// ========================================
	// Similarity and Distance

	/**
	 * Compute cosine similarity between two atoms.
	 */
	double similarity(const Handle& a, const Handle& b);

	/**
	 * Find k most similar atoms to the query.
	 */
	HandleSeq query_similar(const Handle& query, size_t k);

	/**
	 * Find k most similar atoms to a tensor query.
	 */
	HandleSeq query_similar_tensor(const ATenValuePtr& query, size_t k);

	// ========================================
	// Link Embedding

	/**
	 * Compute embedding for a link based on its outgoing set.
	 * Uses aggregation function (mean, sum, attention, etc.)
	 */
	ATenValuePtr compute_link_embedding(const Handle& link,
	                                     const std::string& aggregation = "mean");

	/**
	 * Update link embedding based on outgoing atoms.
	 */
	void update_link_embedding(const Handle& link);

	// ========================================
	// Multi-Scale Operations

	/**
	 * Get multi-scale representation for an atom.
	 */
	MultiScaleTensorPtr get_multi_scale(const Handle& atom);

	/**
	 * Compute hierarchical abstraction.
	 * Aggregates embeddings at different scales.
	 */
	ATenValuePtr hierarchical_abstraction(const HandleSeq& atoms,
	                                       int64_t target_scale);

	// ========================================
	// Network Operations

	/**
	 * Build network tensor from AtomSpace structure.
	 */
	void build_network_tensor(Type link_type = LINK);

	/**
	 * Perform message passing over the network.
	 */
	void message_passing(int64_t num_iterations = 1);

	/**
	 * Update attention weights based on importance.
	 */
	void update_attention();

	/**
	 * Apply Hebbian learning update.
	 */
	void hebbian_update(double learning_rate = 0.01);

	// ========================================
	// PLN Integration

	/**
	 * Get truth value tensor for an atom.
	 */
	TruthValueTensorPtr get_truth_tensor(const Handle& atom);

	/**
	 * Perform tensor-based PLN deduction.
	 */
	TruthValueTensorPtr tensor_deduction(const Handle& premise1,
	                                      const Handle& premise2);

	/**
	 * Perform tensor-based PLN revision.
	 */
	TruthValueTensorPtr tensor_revision(const Handle& atom,
	                                     const TruthValueTensorPtr& evidence);

	// ========================================
	// Inference Operations

	/**
	 * Neural symbolic inference step.
	 * Combines embedding similarity with logical rules.
	 */
	HandleSeq neural_inference(const Handle& query,
	                           const std::string& rule_type = "deduction");

	/**
	 * Attention-based pattern matching.
	 * Uses embeddings to guide pattern search.
	 */
	HandleSeq attention_query(const Handle& pattern, size_t max_results = 10);

	// ========================================
	// Training and Learning

	/**
	 * Train embeddings using contrastive learning.
	 */
	void contrastive_update(const Handle& anchor,
	                        const Handle& positive,
	                        const Handle& negative,
	                        double learning_rate = 0.01);

	/**
	 * Propagate gradients through the network.
	 */
	void backpropagate(const ATenValuePtr& loss);

	// ========================================
	// Persistence

	/**
	 * Save all embeddings to a file.
	 */
	void save_embeddings(const std::string& path);

	/**
	 * Load embeddings from a file.
	 */
	void load_embeddings(const std::string& path);

	// ========================================
	// Statistics

	size_t num_embeddings() const { return _num_embeddings; }
	size_t update_count() const { return _update_count; }

	std::string to_string() const;
};

// ============================================================
// TensorLogicLink - Executable links for tensor logic operations

/**
 * Base class for tensor logic operation links.
 */
class TensorLogicLink : public Link
{
protected:
	void init();

public:
	TensorLogicLink(const HandleSeq&&, Type = TENSOR_LOGIC_LINK);
	TensorLogicLink(const TensorLogicLink&) = delete;
	TensorLogicLink& operator=(const TensorLogicLink&) = delete;

	virtual bool is_executable() const { return true; }
	virtual ValuePtr execute(AtomSpace*, bool silent = false);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(TensorLogicLink)
#define createTensorLogicLink CREATE_DECL(TensorLogicLink)

// ============================================================
// EmbeddingLink - Get/set entity embeddings

class EmbeddingOfLink : public TensorLogicLink
{
public:
	EmbeddingOfLink(const HandleSeq&&, Type = EMBEDDING_OF_LINK);
	virtual ValuePtr execute(AtomSpace*, bool silent = false);
	static Handle factory(const Handle&);
};

LINK_PTR_DECL(EmbeddingOfLink)
#define createEmbeddingOfLink CREATE_DECL(EmbeddingOfLink)

class SetEmbeddingLink : public TensorLogicLink
{
public:
	SetEmbeddingLink(const HandleSeq&&, Type = SET_EMBEDDING_LINK);
	virtual ValuePtr execute(AtomSpace*, bool silent = false);
	static Handle factory(const Handle&);
};

LINK_PTR_DECL(SetEmbeddingLink)
#define createSetEmbeddingLink CREATE_DECL(SetEmbeddingLink)

// ============================================================
// SimilarityQueryLink - Find similar atoms

class SimilarityQueryLink : public TensorLogicLink
{
public:
	SimilarityQueryLink(const HandleSeq&&, Type = SIMILARITY_QUERY_LINK);
	virtual ValuePtr execute(AtomSpace*, bool silent = false);
	static Handle factory(const Handle&);
};

LINK_PTR_DECL(SimilarityQueryLink)
#define createSimilarityQueryLink CREATE_DECL(SimilarityQueryLink)

// ============================================================
// HebbianLink - Hebbian learning connection

class HebbianLink : public Link
{
protected:
	double _weight;
	ATenValuePtr _weight_tensor;

public:
	HebbianLink(const HandleSeq&&, Type = HEBBIAN_LINK);
	HebbianLink(const HebbianLink&) = delete;
	HebbianLink& operator=(const HebbianLink&) = delete;

	double weight() const { return _weight; }
	void set_weight(double w) { _weight = w; }

	ATenValuePtr weight_tensor() const { return _weight_tensor; }
	void set_weight_tensor(const ATenValuePtr& wt) { _weight_tensor = wt; }

	// Hebbian update: weight += lr * pre_activation * post_activation
	void hebbian_update(double pre, double post, double lr = 0.01);

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(HebbianLink)
#define createHebbianLink CREATE_DECL(HebbianLink)

// ============================================================
// AttentionValueTensor - ECAN-style attention with tensors

class AttentionValueTensor
{
private:
	ATenValuePtr _sti;  // Short-term importance
	ATenValuePtr _lti;  // Long-term importance
	ATenValuePtr _vlti; // Very long-term importance

public:
	AttentionValueTensor(int64_t dim = 1);
	AttentionValueTensor(const ATenValuePtr& sti,
	                     const ATenValuePtr& lti = nullptr,
	                     const ATenValuePtr& vlti = nullptr);

	ATenValuePtr sti() const { return _sti; }
	ATenValuePtr lti() const { return _lti; }
	ATenValuePtr vlti() const { return _vlti; }

	void set_sti(const ATenValuePtr& s) { _sti = s; }
	void set_lti(const ATenValuePtr& l) { _lti = l; }
	void set_vlti(const ATenValuePtr& v) { _vlti = v; }

	// ECAN operations
	void stimulate(double amount);
	void decay(double factor);
	void spread(const AttentionValueTensor& other, double fraction);

	double mean_sti() const;
	double mean_lti() const;
};

typedef std::shared_ptr<AttentionValueTensor> AttentionValueTensorPtr;

/** @}*/
} // namespace opencog

#endif // _OPENCOG_TENSOR_LOGIC_H
