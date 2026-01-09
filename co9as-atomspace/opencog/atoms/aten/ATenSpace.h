/*
 * opencog/atoms/aten/ATenSpace.h
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * ATenSpace bridges symbolic AI (AtomSpace) with neural tensor embeddings.
 * Based on the integration patterns from o9nn/ATen and o9nn/ATenSpace.
 *
 * This creates a hypergraph knowledge representation system with
 * efficient tensor operations, enabling hybrid symbolic-neural AI.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#ifndef _OPENCOG_ATEN_SPACE_H
#define _OPENCOG_ATEN_SPACE_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <functional>

#include <opencog/atoms/base/Handle.h>
#include <opencog/atoms/aten/ATenValue.h>
#include <opencog/atoms/aten/TensorLogic.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

// Forward declarations
class AtomSpace;
class TensorLogic;

/**
 * ATenSpace provides a centralized manager for tensor-enhanced AtomSpace.
 * It bridges symbolic knowledge representation with neural tensor embeddings.
 *
 * Based on o9nn/ATenSpace, this provides:
 * - Entity embedding management (tensor representations for atoms)
 * - Semantic similarity operations via tensor mathematics
 * - GPU acceleration through ATen tensor library
 * - Hebbian learning and attention allocation
 * - PLN-tensor integration for probabilistic reasoning
 *
 * Key differences from original AtomSpace:
 * - Replaces pure pattern matching with embedding similarity
 * - Enables neural network style operations on symbolic structures
 * - Supports multi-scale hierarchical representations
 * - Provides network-aware message passing
 */
class ATenSpace
{
private:
	// Thread safety
	mutable std::mutex _mtx;

	// Named tensor registry
	std::map<std::string, ATenValuePtr> _named_tensors;

	// Tensor key for attaching to atoms
	Handle _tensor_key;
	Handle _embedding_key;
	Handle _attention_key;

	// Device configuration
	std::string _device; // "cpu" or "cuda:N"
	bool _requires_grad;

	// Reference to the AtomSpace (non-owning)
	AtomSpace* _atomspace;

	// TensorLogic for advanced operations
	std::shared_ptr<TensorLogic> _tensor_logic;

	// Configuration
	int64_t _embedding_dim;
	int64_t _num_scales;

	// Statistics
	size_t _tensor_count;
	size_t _total_elements;

public:
	/**
	 * Construct an ATenSpace.
	 *
	 * @param as Optional AtomSpace to associate with
	 * @param embedding_dim Dimension of entity embeddings (default 128)
	 * @param device Device for tensor allocation ("cpu" or "cuda:N")
	 */
	ATenSpace(AtomSpace* as = nullptr,
	          int64_t embedding_dim = 128,
	          const std::string& device = "cpu");
	~ATenSpace();

	// Prevent copying
	ATenSpace(const ATenSpace&) = delete;
	ATenSpace& operator=(const ATenSpace&) = delete;

	// ========================================
	// Configuration

	/**
	 * Set the associated AtomSpace.
	 */
	void set_atomspace(AtomSpace* as);
	AtomSpace* get_atomspace() const { return _atomspace; }

	/**
	 * Set the default device for tensor allocation.
	 * @param device "cpu" or "cuda:N"
	 */
	void set_device(const std::string& device);
	std::string get_device() const { return _device; }

	/**
	 * Enable/disable gradient tracking for new tensors.
	 */
	void set_requires_grad(bool req) { _requires_grad = req; }
	bool get_requires_grad() const { return _requires_grad; }

	/**
	 * Get embedding dimension.
	 */
	int64_t embedding_dim() const { return _embedding_dim; }

	/**
	 * Get TensorLogic instance for advanced operations.
	 */
	TensorLogic* tensor_logic() { return _tensor_logic.get(); }

	// ========================================
	// Named tensor management

	/**
	 * Register a tensor with a name.
	 * @param name Unique identifier for the tensor
	 * @param tensor The tensor value to register
	 */
	void register_tensor(const std::string& name, const ATenValuePtr& tensor);

	/**
	 * Get a tensor by name.
	 * @param name The registered name
	 * @return The tensor, or nullptr if not found
	 */
	ATenValuePtr get_tensor(const std::string& name) const;

	/**
	 * Check if a tensor exists with the given name.
	 */
	bool has_tensor(const std::string& name) const;

	/**
	 * Remove a tensor by name.
	 */
	void remove_tensor(const std::string& name);

	/**
	 * Get all registered tensor names.
	 */
	std::vector<std::string> get_tensor_names() const;

	/**
	 * Clear all registered tensors.
	 */
	void clear();

	// ========================================
	// Tensor creation utilities

	/**
	 * Create a zeros tensor and optionally register it.
	 */
	ATenValuePtr zeros(const std::vector<int64_t>& shape,
	                   const std::string& name = "");

	/**
	 * Create a ones tensor and optionally register it.
	 */
	ATenValuePtr ones(const std::vector<int64_t>& shape,
	                  const std::string& name = "");

	/**
	 * Create a random tensor and optionally register it.
	 */
	ATenValuePtr random(const std::vector<int64_t>& shape,
	                    const std::string& name = "");

	/**
	 * Create a tensor from data and optionally register it.
	 */
	ATenValuePtr from_vector(const std::vector<double>& data,
	                         const std::vector<int64_t>& shape,
	                         const std::string& name = "");

	// ========================================
	// Atom-Tensor integration (o9nn/ATenSpace style)

	/**
	 * Create a ConceptNode with an embedding.
	 * Similar to o9nn/ATenSpace createConceptNode.
	 */
	Handle create_concept_node(const std::string& name,
	                           const ATenValuePtr& embedding = nullptr);

	/**
	 * Attach a tensor embedding to an atom.
	 */
	void attach_embedding(const Handle& atom, const ATenValuePtr& tensor);

	/**
	 * Get embedding attached to an atom.
	 */
	ATenValuePtr get_embedding(const Handle& atom) const;

	/**
	 * Check if an atom has an attached embedding.
	 */
	bool has_embedding(const Handle& atom) const;

	/**
	 * Attach a tensor to an atom (alias for compatibility).
	 */
	void attach_tensor(const Handle& atom, const ATenValuePtr& tensor);

	/**
	 * Get a tensor attached to an atom.
	 */
	ATenValuePtr get_attached_tensor(const Handle& atom) const;

	/**
	 * Check if an atom has an attached tensor.
	 */
	bool has_attached_tensor(const Handle& atom) const;

	/**
	 * Get the default tensor key predicate.
	 */
	Handle get_tensor_key() const { return _tensor_key; }

	// ========================================
	// Semantic similarity operations

	/**
	 * Query for k atoms most similar to the given tensor.
	 * Uses cosine similarity on embeddings.
	 */
	HandleSeq query_similar(const ATenValuePtr& query_tensor, size_t k = 10);

	/**
	 * Query for k atoms most similar to the given atom.
	 */
	HandleSeq query_similar(const Handle& query_atom, size_t k = 10);

	/**
	 * Compute cosine similarity between two atoms.
	 */
	double similarity(const Handle& a, const Handle& b);

	/**
	 * Compute Euclidean distance between two atoms.
	 */
	double distance(const Handle& a, const Handle& b);

	// ========================================
	// Link operations

	/**
	 * Create a link with aggregated embedding from outgoing atoms.
	 */
	Handle create_link(Type link_type, const HandleSeq& outgoing,
	                   const std::string& aggregation = "mean");

	/**
	 * Compute embedding for a link based on its structure.
	 */
	ATenValuePtr compute_link_embedding(const Handle& link,
	                                     const std::string& aggregation = "mean");

	// ========================================
	// Hebbian learning (ECAN-style)

	/**
	 * Create a HebbianLink between atoms.
	 */
	Handle create_hebbian_link(const Handle& source, const Handle& target,
	                           double weight = 1.0);

	/**
	 * Apply Hebbian update to all HebbianLinks.
	 */
	void hebbian_update(double learning_rate = 0.01);

	/**
	 * Spread activation through the network.
	 */
	void spread_activation(const Handle& source, double amount, int hops = 1);

	// ========================================
	// Attention allocation

	/**
	 * Stimulate attention for an atom.
	 */
	void stimulate(const Handle& atom, double amount);

	/**
	 * Decay attention across all atoms.
	 */
	void decay_attention(double factor = 0.9);

	/**
	 * Get attention value (STI) for an atom.
	 */
	double get_attention(const Handle& atom) const;

	// ========================================
	// Pattern matching with embeddings

	/**
	 * Find atoms matching pattern using embedding similarity.
	 */
	HandleSeq pattern_match_embedding(const Handle& pattern,
	                                   double threshold = 0.7,
	                                   size_t max_results = 100);

	// ========================================
	// PLN integration

	/**
	 * Perform tensor-based deduction inference.
	 */
	Handle tensor_deduction(const Handle& premise1, const Handle& premise2);

	/**
	 * Perform tensor-based revision.
	 */
	void tensor_revision(const Handle& atom, double strength, double confidence);

	// ========================================
	// Network operations

	/**
	 * Build tensor network representation from AtomSpace structure.
	 */
	void build_network();

	/**
	 * Perform message passing over the network.
	 */
	void message_passing(int num_iterations = 1);

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

	/**
	 * Get the number of registered tensors.
	 */
	size_t tensor_count() const { return _tensor_count; }

	/**
	 * Get the total number of elements across all tensors.
	 */
	size_t total_elements() const { return _total_elements; }

	/**
	 * Get a summary string of the ATenSpace state.
	 */
	std::string to_string() const;
};

typedef std::shared_ptr<ATenSpace> ATenSpacePtr;

/**
 * Create a new ATenSpace.
 */
template<typename ... Args>
ATenSpacePtr createATenSpace(Args&& ... args)
{
	return std::make_shared<ATenSpace>(std::forward<Args>(args)...);
}

/** @}*/
} // namespace opencog

#endif // _OPENCOG_ATEN_SPACE_H
