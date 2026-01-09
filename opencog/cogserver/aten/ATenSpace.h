/*
 * ATenSpace.h
 *
 * ATenSpace - Bridge between AtomSpace and ATen tensors
 * Copyright (c) 2025 OpenCog Foundation
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

#ifndef _OPENCOG_ATEN_SPACE_H
#define _OPENCOG_ATEN_SPACE_H

#ifdef HAVE_ATEN

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include <opencog/atomspace/AtomSpace.h>
#include "TensorValue.h"

namespace opencog {

/**
 * ATenSpace - Extension of AtomSpace for tensor operations.
 *
 * This class provides a bridge between OpenCog's symbolic knowledge
 * representation (AtomSpace) and PyTorch's numerical tensor operations
 * (ATen). It enables:
 *
 * 1. **Tensor-aware knowledge graphs**: Store neural embeddings,
 *    weight matrices, and activations as Values on Atoms.
 *
 * 2. **Hybrid reasoning**: Combine symbolic pattern matching with
 *    neural computations.
 *
 * 3. **Neuro-symbolic integration**: Execute tensor operations as
 *    part of AtomSpace queries and pattern matching.
 *
 * Key concepts:
 * - TensorValue: AtomSpace Value wrapping at::Tensor
 * - Tensor keys: Atoms used as keys for tensor storage
 * - Computation graphs: Lazy evaluation of tensor expressions
 *
 * Example:
 *   ATenSpace ats(&atomspace);
 *
 *   // Create embedding
 *   Handle word = atomspace.add_node(CONCEPT_NODE, "cat");
 *   Handle key = atomspace.add_node(PREDICATE_NODE, "embedding");
 *   ats.set_tensor(word, key, {128}, torch::kFloat32);
 *
 *   // Compute similarity
 *   TensorValuePtr emb1 = ats.get_tensor(word1, key);
 *   TensorValuePtr emb2 = ats.get_tensor(word2, key);
 *   float sim = ats.cosine_similarity(emb1, emb2);
 */
class ATenSpace
{
private:
	AtomSpace* _as;
	mutable std::mutex _mtx;

	// Cache for frequently accessed tensors
	mutable std::unordered_map<std::string, TensorValuePtr> _cache;
	bool _use_cache;
	size_t _cache_max_size;

	std::string make_cache_key(const Handle& atom, const Handle& key) const;
	void cache_tensor(const std::string& cache_key, const TensorValuePtr& tv) const;

public:
	/**
	 * Construct ATenSpace attached to an AtomSpace.
	 * @param as Pointer to AtomSpace (must outlive this object)
	 * @param use_cache Enable tensor caching (default: true)
	 */
	ATenSpace(AtomSpace* as, bool use_cache = true);
	~ATenSpace();

	/** Get the associated AtomSpace */
	AtomSpace* get_atomspace() { return _as; }

	// ============================================================
	// Tensor storage operations

	/**
	 * Create and store a new tensor on an atom.
	 * @param atom The atom to attach tensor to
	 * @param key The key for this tensor
	 * @param shape Tensor dimensions
	 * @param dtype Data type (default: float32)
	 * @param init Initialization: "zeros", "ones", "randn", "xavier"
	 * @return The created TensorValue
	 */
	TensorValuePtr set_tensor(const Handle& atom,
	                          const Handle& key,
	                          const std::vector<int64_t>& shape,
	                          at::ScalarType dtype = at::kFloat,
	                          const std::string& init = "zeros");

	/**
	 * Store an existing tensor on an atom.
	 */
	void set_tensor(const Handle& atom,
	                const Handle& key,
	                const TensorValuePtr& tensor);

	/**
	 * Store tensor from at::Tensor directly.
	 */
	void set_tensor(const Handle& atom,
	                const Handle& key,
	                const at::Tensor& tensor);

	/**
	 * Get tensor stored on an atom.
	 * @return TensorValue or nullptr if not found/not a tensor
	 */
	TensorValuePtr get_tensor(const Handle& atom, const Handle& key) const;

	/**
	 * Check if atom has a tensor at key.
	 */
	bool has_tensor(const Handle& atom, const Handle& key) const;

	/**
	 * Remove tensor from atom.
	 */
	void remove_tensor(const Handle& atom, const Handle& key);

	/**
	 * Get all atoms that have tensors stored at a given key.
	 */
	HandleSeq get_atoms_with_tensor(const Handle& key) const;

	// ============================================================
	// Tensor computations

	/**
	 * Matrix multiply two tensors.
	 */
	TensorValuePtr matmul(const TensorValuePtr& a, const TensorValuePtr& b) const;

	/**
	 * Element-wise operation.
	 * @param op "add", "sub", "mul", "div"
	 */
	TensorValuePtr elementwise(const TensorValuePtr& a,
	                           const TensorValuePtr& b,
	                           const std::string& op) const;

	/**
	 * Apply activation function.
	 * @param activation "relu", "sigmoid", "tanh", "softmax", "gelu"
	 */
	TensorValuePtr activate(const TensorValuePtr& tensor,
	                        const std::string& activation) const;

	/**
	 * Compute cosine similarity between two tensors.
	 */
	double cosine_similarity(const TensorValuePtr& a, const TensorValuePtr& b) const;

	/**
	 * Compute dot product.
	 */
	double dot(const TensorValuePtr& a, const TensorValuePtr& b) const;

	/**
	 * Compute L2 norm.
	 */
	double norm(const TensorValuePtr& tensor, int p = 2) const;

	/**
	 * Normalize tensor (L2 normalization).
	 */
	TensorValuePtr normalize(const TensorValuePtr& tensor) const;

	// ============================================================
	// Batch operations on AtomSpace

	/**
	 * Get all tensors for a set of atoms at a given key.
	 * Returns a single batched tensor [N, ...].
	 */
	TensorValuePtr batch_get(const HandleSeq& atoms, const Handle& key) const;

	/**
	 * Set tensors for multiple atoms from a batched tensor.
	 * Batch dimension must match atoms.size().
	 */
	void batch_set(const HandleSeq& atoms,
	               const Handle& key,
	               const TensorValuePtr& batch_tensor);

	/**
	 * Compute pairwise similarities between all pairs of atoms.
	 * Returns a similarity matrix [N, N].
	 */
	TensorValuePtr pairwise_similarity(const HandleSeq& atoms,
	                                   const Handle& key) const;

	/**
	 * Find k nearest neighbors for an atom based on tensor similarity.
	 * @return Pairs of (atom, similarity_score)
	 */
	std::vector<std::pair<Handle, double>> nearest_neighbors(
		const Handle& query_atom,
		const Handle& key,
		const HandleSeq& candidates,
		size_t k = 10) const;

	// ============================================================
	// Neural network layer operations

	/**
	 * Apply linear transformation: y = Wx + b
	 */
	TensorValuePtr linear(const TensorValuePtr& input,
	                      const TensorValuePtr& weight,
	                      const TensorValuePtr& bias = nullptr) const;

	/**
	 * Apply multi-head attention.
	 */
	TensorValuePtr attention(const TensorValuePtr& query,
	                         const TensorValuePtr& key,
	                         const TensorValuePtr& value,
	                         int num_heads = 1) const;

	/**
	 * Apply layer normalization.
	 */
	TensorValuePtr layer_norm(const TensorValuePtr& input,
	                          const std::vector<int64_t>& normalized_shape) const;

	// ============================================================
	// Utility functions

	/** Clear the tensor cache */
	void clear_cache();

	/** Get cache statistics */
	size_t cache_size() const;

	/** Report tensor memory usage */
	std::string memory_stats() const;

	/** Check CUDA availability */
	static bool cuda_available();

	/** Get number of CUDA devices */
	static int cuda_device_count();

	/** Parse scalar type from string */
	static at::ScalarType parse_dtype(const std::string& dtype_str);

	/** Convert scalar type to string */
	static std::string dtype_to_string(at::ScalarType dtype);
};

} // namespace opencog

#endif // HAVE_ATEN
#endif // _OPENCOG_ATEN_SPACE_H
