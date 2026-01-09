/*
 * opencog/util/tensor_logic.h
 *
 * Copyright (C) 2025 OpenCog Foundation
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

#ifndef _OPENCOG_TENSOR_LOGIC_H
#define _OPENCOG_TENSOR_LOGIC_H

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include <opencog/util/tensor_space.h>

/** \addtogroup grp_cogutil
 *  @{
 */

namespace opencog
{

// ===================================================================
// Multi-Scale Network Representation
// ===================================================================

/**
 * @brief Scale level in a multi-scale network hierarchy.
 *
 * Represents different levels of abstraction/granularity in
 * a multi-scale network representation.
 */
enum class NetworkScale : uint8_t
{
    MICRO,      // Individual entities/neurons
    MESO,       // Small clusters/communities
    MACRO,      // Large-scale structures
    GLOBAL      // Entire network
};

/**
 * @brief Multi-scale network node aggregation.
 *
 * Represents an aggregated view of multiple atoms at a coarser scale.
 */
struct ScaleAggregate
{
    NetworkScale scale;
    std::vector<TensorAtomPtr> members;
    Tensor<double> embedding;  // Aggregated embedding
    TruthValue tv;             // Aggregated truth value
    AttentionValue av;         // Aggregated attention

    ScaleAggregate(NetworkScale s = NetworkScale::MICRO)
        : scale(s), tv(), av()
    {}

    /// Compute aggregated embedding from members
    void compute_embedding(const std::string& method = "mean")
    {
        std::vector<Tensor<double>> embeddings;
        for (const auto& m : members) {
            if (m && m->has_embedding()) {
                embeddings.push_back(m->embedding());
            }
        }

        if (embeddings.empty()) {
            embedding = Tensor<double>();
            return;
        }

        const auto& shape = embeddings[0].shape();
        embedding = Tensor<double>::zeros(shape);

        if (method == "mean") {
            for (const auto& e : embeddings) {
                if (e.shape() == shape) embedding += e;
            }
            embedding /= static_cast<double>(embeddings.size());
        } else if (method == "max") {
            embedding = embeddings[0].clone();
            for (size_t i = 1; i < embeddings.size(); ++i) {
                if (embeddings[i].shape() == shape) {
                    for (size_t j = 0; j < embedding.numel(); ++j) {
                        embedding.data()[j] = std::max(
                            embedding.data()[j], embeddings[i].data()[j]);
                    }
                }
            }
        }
    }

    /// Compute aggregated truth value
    void compute_tv(const std::string& method = "mean")
    {
        if (members.empty()) {
            tv = TruthValue();
            return;
        }

        double sum_s = 0.0, sum_c = 0.0;
        for (const auto& m : members) {
            sum_s += m->tv().strength;
            sum_c += m->tv().confidence;
        }

        if (method == "mean") {
            tv = TruthValue(sum_s / members.size(), sum_c / members.size());
        } else if (method == "min") {
            double min_s = 1.0, min_c = 1.0;
            for (const auto& m : members) {
                min_s = std::min(min_s, m->tv().strength);
                min_c = std::min(min_c, m->tv().confidence);
            }
            tv = TruthValue(min_s, min_c);
        }
    }
};

// ===================================================================
// Entity Representation for Multi-Entity Systems
// ===================================================================

/**
 * @brief Entity in a multi-entity tensor system.
 *
 * Represents a distinct entity with its own internal state,
 * relationships, and embedding space.
 */
class TensorEntity
{
public:
    using EntityId = uint64_t;

private:
    EntityId _id;
    std::string _name;
    TensorNodePtr _node;                    // Corresponding atom
    Tensor<double> _state;                  // Internal state vector
    std::map<EntityId, double> _relations;  // Relations to other entities
    std::map<std::string, Tensor<double>> _properties;

    static std::atomic<EntityId> _id_counter;

public:
    TensorEntity(const std::string& name, size_t state_dim = 64)
        : _id(_id_counter.fetch_add(1)),
          _name(name),
          _state(Tensor<double>::zeros({state_dim}))
    {}

    // Accessors
    EntityId id() const { return _id; }
    const std::string& name() const { return _name; }
    TensorNodePtr node() const { return _node; }
    const Tensor<double>& state() const { return _state; }

    // State operations
    void set_state(const Tensor<double>& state) { _state = state; }
    void set_state(Tensor<double>&& state) { _state = std::move(state); }

    /// Update state with a delta
    void update_state(const Tensor<double>& delta, double learning_rate = 0.1)
    {
        _state += delta * learning_rate;
    }

    // Relations
    void set_relation(EntityId other, double strength)
    {
        _relations[other] = strength;
    }

    double get_relation(EntityId other) const
    {
        auto it = _relations.find(other);
        return it != _relations.end() ? it->second : 0.0;
    }

    const std::map<EntityId, double>& relations() const { return _relations; }

    // Properties (named tensor attributes)
    void set_property(const std::string& name, const Tensor<double>& value)
    {
        _properties[name] = value;
    }

    std::optional<Tensor<double>> get_property(const std::string& name) const
    {
        auto it = _properties.find(name);
        if (it != _properties.end()) return it->second;
        return std::nullopt;
    }

    // Link to TensorSpace
    void link_to_space(TensorSpace& space)
    {
        _node = space.add_entity(_name);
        if (_state.numel() > 0) {
            _node->set_embedding(_state);
        }
    }
};

inline std::atomic<TensorEntity::EntityId> TensorEntity::_id_counter{1};

// ===================================================================
// TensorLogic - Multi-Entity, Multi-Scale Operations
// ===================================================================

/**
 * @brief Tensor logic operations for multi-entity, multi-scale networks.
 *
 * TensorLogic provides operations that bridge symbolic reasoning with
 * tensor computations across multiple entities and network scales:
 *
 * - Fuzzy logic operations on tensor truth values
 * - Multi-entity interaction computations
 * - Multi-scale aggregation and propagation
 * - Network-aware tensor operations
 */
class TensorLogic
{
public:
    // ===================================================================
    // Fuzzy Tensor Logic Operations
    // ===================================================================

    /**
     * @brief Fuzzy AND operation on tensors.
     *
     * Computes element-wise minimum (Gödel t-norm).
     */
    static Tensor<double> fuzzy_and(const Tensor<double>& a,
                                    const Tensor<double>& b)
    {
        OC_ASSERT(a.shape() == b.shape(), "TensorLogic: shape mismatch");
        Tensor<double> result(a.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            result.data()[i] = std::min(a.data()[i], b.data()[i]);
        }
        return result;
    }

    /**
     * @brief Fuzzy OR operation on tensors.
     *
     * Computes element-wise maximum (Gödel t-conorm).
     */
    static Tensor<double> fuzzy_or(const Tensor<double>& a,
                                   const Tensor<double>& b)
    {
        OC_ASSERT(a.shape() == b.shape(), "TensorLogic: shape mismatch");
        Tensor<double> result(a.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            result.data()[i] = std::max(a.data()[i], b.data()[i]);
        }
        return result;
    }

    /**
     * @brief Fuzzy NOT operation on tensors.
     *
     * Computes element-wise complement (1 - x).
     */
    static Tensor<double> fuzzy_not(const Tensor<double>& a)
    {
        Tensor<double> result(a.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            result.data()[i] = 1.0 - a.data()[i];
        }
        return result;
    }

    /**
     * @brief Fuzzy implication (Lukasiewicz).
     *
     * Computes: min(1, 1 - a + b)
     */
    static Tensor<double> fuzzy_implication(const Tensor<double>& a,
                                            const Tensor<double>& b)
    {
        OC_ASSERT(a.shape() == b.shape(), "TensorLogic: shape mismatch");
        Tensor<double> result(a.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            result.data()[i] = std::min(1.0, 1.0 - a.data()[i] + b.data()[i]);
        }
        return result;
    }

    /**
     * @brief Product t-norm (fuzzy AND variant).
     *
     * Computes element-wise product.
     */
    static Tensor<double> product_tnorm(const Tensor<double>& a,
                                        const Tensor<double>& b)
    {
        return a * b;
    }

    /**
     * @brief Lukasiewicz t-norm.
     *
     * Computes: max(0, a + b - 1)
     */
    static Tensor<double> lukasiewicz_tnorm(const Tensor<double>& a,
                                            const Tensor<double>& b)
    {
        OC_ASSERT(a.shape() == b.shape(), "TensorLogic: shape mismatch");
        Tensor<double> result(a.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            result.data()[i] = std::max(0.0, a.data()[i] + b.data()[i] - 1.0);
        }
        return result;
    }

    // ===================================================================
    // Tensor Truth Value Operations
    // ===================================================================

    /**
     * @brief Convert tensor to truth value (scalar summary).
     */
    static TruthValue tensor_to_tv(const Tensor<double>& t)
    {
        if (t.numel() == 0) return TruthValue(0.0, 0.0);

        // Strength = mean of tensor values
        double strength = t.mean();
        strength = std::max(0.0, std::min(1.0, strength));

        // Confidence based on variance (lower variance = higher confidence)
        double variance = t.var(false);
        double confidence = 1.0 / (1.0 + variance);

        return TruthValue(strength, confidence);
    }

    /**
     * @brief Convert truth value to tensor (broadcast).
     */
    static Tensor<double> tv_to_tensor(const TruthValue& tv,
                                       const std::vector<size_t>& shape)
    {
        return Tensor<double>::full(shape, tv.strength);
    }

    /**
     * @brief PLN-style deduction on tensor embeddings.
     *
     * Given A->B (with embedding e_ab) and B->C (with embedding e_bc),
     * compute A->C embedding.
     */
    static Tensor<double> tensor_deduction(const Tensor<double>& ab,
                                           const Tensor<double>& bc,
                                           double decay = 0.9)
    {
        OC_ASSERT(ab.shape() == bc.shape(), "TensorLogic: shape mismatch");
        // Simple composition: element-wise product with decay
        return product_tnorm(ab, bc) * decay;
    }

    /**
     * @brief PLN-style induction on tensor embeddings.
     */
    static Tensor<double> tensor_induction(const Tensor<double>& ab,
                                           const Tensor<double>& cb,
                                           double decay = 0.8)
    {
        OC_ASSERT(ab.shape() == cb.shape(), "TensorLogic: shape mismatch");
        // Similarity-based induction
        return product_tnorm(ab, cb) * decay;
    }

    // ===================================================================
    // Multi-Entity Operations
    // ===================================================================

    /**
     * @brief Compute interaction tensor between two entities.
     *
     * Produces a tensor capturing the relationship between two entities
     * based on their state vectors.
     */
    static Tensor<double> entity_interaction(const TensorEntity& e1,
                                             const TensorEntity& e2)
    {
        const auto& s1 = e1.state();
        const auto& s2 = e2.state();

        if (s1.numel() == 0 || s2.numel() == 0) {
            return Tensor<double>();
        }

        // Outer product for full interaction matrix
        if (s1.ndim() == 1 && s2.ndim() == 1) {
            return s1.outer(s2);
        }

        // Element-wise interaction for matching shapes
        if (s1.shape() == s2.shape()) {
            return s1 * s2;
        }

        return Tensor<double>();
    }

    /**
     * @brief Compute multi-entity attention distribution.
     *
     * Given a set of entities, compute attention weights based on
     * their state similarities to a query entity.
     */
    static Tensor<double> multi_entity_attention(
        const TensorEntity& query,
        const std::vector<TensorEntity*>& entities,
        double temperature = 1.0)
    {
        size_t n = entities.size();
        if (n == 0) return Tensor<double>();

        Tensor<double> scores(Tensor<double>::shape_type{n});
        const auto& q = query.state();

        for (size_t i = 0; i < n; ++i) {
            if (!entities[i]) {
                scores(i) = 0.0;
                continue;
            }

            const auto& k = entities[i]->state();
            if (q.shape() != k.shape()) {
                scores(i) = 0.0;
                continue;
            }

            // Dot product attention
            double dot = 0.0;
            for (size_t j = 0; j < q.numel(); ++j) {
                dot += q.data()[j] * k.data()[j];
            }
            scores(i) = dot / temperature;
        }

        // Softmax
        return softmax(scores);
    }

    /**
     * @brief Update entity states based on interactions.
     *
     * Applies message passing between entities.
     */
    static void message_passing(std::vector<TensorEntity>& entities,
                                double learning_rate = 0.1)
    {
        std::vector<Tensor<double>> messages(entities.size());

        // Compute messages
        for (size_t i = 0; i < entities.size(); ++i) {
            auto& e = entities[i];
            Tensor<double> msg = Tensor<double>::zeros(e.state().shape());

            for (const auto& [other_id, strength] : e.relations()) {
                // Find the other entity
                for (const auto& other : entities) {
                    if (other.id() == other_id && e.state().shape() == other.state().shape()) {
                        msg += other.state() * strength;
                        break;
                    }
                }
            }

            messages[i] = msg;
        }

        // Apply updates
        for (size_t i = 0; i < entities.size(); ++i) {
            entities[i].update_state(messages[i], learning_rate);
        }
    }

    // ===================================================================
    // Multi-Scale Operations
    // ===================================================================

    /**
     * @brief Aggregate atoms into scale aggregates using clustering.
     *
     * Simple k-means style clustering based on embeddings.
     */
    static std::vector<ScaleAggregate> scale_aggregate(
        const std::vector<TensorAtomPtr>& atoms,
        size_t num_clusters,
        NetworkScale target_scale = NetworkScale::MESO)
    {
        std::vector<ScaleAggregate> aggregates(num_clusters);
        for (auto& agg : aggregates) {
            agg.scale = target_scale;
        }

        // Filter atoms with embeddings
        std::vector<TensorAtomPtr> embedded;
        for (const auto& atom : atoms) {
            if (atom && atom->has_embedding()) {
                embedded.push_back(atom);
            }
        }

        if (embedded.empty() || num_clusters == 0) return aggregates;

        size_t dim = embedded[0]->embedding().numel();

        // Initialize centroids randomly from data
        std::vector<Tensor<double>> centroids;
        std::vector<size_t> indices;
        for (size_t i = 0; i < embedded.size(); ++i) indices.push_back(i);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t i = 0; i < std::min(num_clusters, embedded.size()); ++i) {
            centroids.push_back(embedded[indices[i]]->embedding().clone());
        }

        // Fill remaining centroids if needed
        while (centroids.size() < num_clusters) {
            centroids.push_back(Tensor<double>::randn({dim}));
        }

        // K-means iterations
        const int max_iter = 20;
        for (int iter = 0; iter < max_iter; ++iter) {
            // Clear assignments
            for (auto& agg : aggregates) {
                agg.members.clear();
            }

            // Assign atoms to nearest centroid
            for (const auto& atom : embedded) {
                const auto& emb = atom->embedding();
                double min_dist = std::numeric_limits<double>::max();
                size_t best_cluster = 0;

                for (size_t c = 0; c < num_clusters; ++c) {
                    double dist = 0.0;
                    for (size_t j = 0; j < dim; ++j) {
                        double diff = emb.data()[j] - centroids[c].data()[j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }

                aggregates[best_cluster].members.push_back(atom);
            }

            // Update centroids
            for (size_t c = 0; c < num_clusters; ++c) {
                if (aggregates[c].members.empty()) continue;

                centroids[c].zero_();
                for (const auto& atom : aggregates[c].members) {
                    centroids[c] += atom->embedding();
                }
                centroids[c] /= static_cast<double>(aggregates[c].members.size());
            }
        }

        // Compute aggregate properties
        for (auto& agg : aggregates) {
            agg.compute_embedding();
            agg.compute_tv();
        }

        return aggregates;
    }

    /**
     * @brief Propagate information across scales (coarse to fine).
     */
    static void propagate_coarse_to_fine(
        const ScaleAggregate& coarse,
        std::vector<TensorAtomPtr>& fine_atoms,
        double influence = 0.3)
    {
        if (!coarse.embedding.numel()) return;

        for (auto& atom : fine_atoms) {
            if (!atom || !atom->has_embedding()) continue;
            if (atom->embedding().shape() != coarse.embedding.shape()) continue;

            // Blend with coarse embedding
            Tensor<double> new_emb = atom->embedding() * (1.0 - influence) +
                                     coarse.embedding * influence;
            atom->set_embedding(std::move(new_emb));
        }
    }

    /**
     * @brief Propagate information across scales (fine to coarse).
     */
    static void propagate_fine_to_coarse(
        const std::vector<TensorAtomPtr>& fine_atoms,
        ScaleAggregate& coarse)
    {
        coarse.members.clear();
        for (const auto& atom : fine_atoms) {
            if (atom) coarse.members.push_back(atom);
        }
        coarse.compute_embedding();
        coarse.compute_tv();
    }

    // ===================================================================
    // Network-Aware Operations
    // ===================================================================

    /**
     * @brief Compute network centrality tensor.
     *
     * Returns a tensor of centrality scores for atoms in the space.
     */
    static Tensor<double> network_centrality(const TensorSpace& space)
    {
        auto atoms = space.get_all_atoms();
        size_t n = atoms.size();
        if (n == 0) return Tensor<double>();

        // Map atoms to indices
        std::map<TensorAtom::UUID, size_t> atom_idx;
        std::vector<TensorAtomPtr> atom_list(atoms.begin(), atoms.end());
        for (size_t i = 0; i < n; ++i) {
            atom_idx[atom_list[i]->uuid()] = i;
        }

        // Compute degree centrality
        Tensor<double> centrality(Tensor<double>::shape_type{n});
        for (size_t i = 0; i < n; ++i) {
            auto incoming = space.get_incoming(atom_list[i]);
            centrality(i) = static_cast<double>(incoming.size());

            if (atom_list[i]->is_link()) {
                auto link = std::static_pointer_cast<TensorLink>(atom_list[i]);
                centrality(i) += static_cast<double>(link->arity());
            }
        }

        // Normalize
        double max_cent = centrality.max();
        if (max_cent > 0) {
            centrality /= max_cent;
        }

        return centrality;
    }

    /**
     * @brief Compute network PageRank-style tensor.
     */
    static Tensor<double> network_pagerank(const TensorSpace& space,
                                           double damping = 0.85,
                                           int iterations = 20)
    {
        auto atoms = space.get_all_atoms();
        size_t n = atoms.size();
        if (n == 0) return Tensor<double>();

        std::map<TensorAtom::UUID, size_t> atom_idx;
        std::vector<TensorAtomPtr> atom_list(atoms.begin(), atoms.end());
        for (size_t i = 0; i < n; ++i) {
            atom_idx[atom_list[i]->uuid()] = i;
        }

        Tensor<double> rank(Tensor<double>::shape_type{n});
        rank.fill_(1.0 / static_cast<double>(n));

        Tensor<double> new_rank(Tensor<double>::shape_type{n});

        for (int iter = 0; iter < iterations; ++iter) {
            new_rank.fill_((1.0 - damping) / static_cast<double>(n));

            for (size_t i = 0; i < n; ++i) {
                auto incoming = space.get_incoming(atom_list[i]);
                for (const auto& link_atom : incoming) {
                    auto link = std::static_pointer_cast<TensorLink>(link_atom);
                    size_t link_idx = atom_idx[link->uuid()];
                    double out_degree = static_cast<double>(link->arity());
                    if (out_degree > 0) {
                        new_rank(i) += damping * rank(link_idx) / out_degree;
                    }
                }
            }

            // Swap
            std::swap(rank, new_rank);
        }

        return rank;
    }

    /**
     * @brief Graph Neural Network-style message passing on the hypergraph.
     *
     * Updates atom embeddings based on neighborhood aggregation.
     */
    static void gnn_update(TensorSpace& space,
                           size_t embedding_dim,
                           double learning_rate = 0.1)
    {
        auto atoms = space.get_all_atoms();

        // Initialize embeddings for atoms without them
        for (const auto& atom : atoms) {
            if (!atom->has_embedding()) {
                atom->set_embedding(Tensor<double>::randn({embedding_dim}));
            }
        }

        // Message passing
        std::map<TensorAtom::UUID, Tensor<double>> messages;

        for (const auto& atom : atoms) {
            auto incoming = space.get_incoming(atom);
            if (incoming.empty()) continue;

            Tensor<double> msg = Tensor<double>::zeros({embedding_dim});
            size_t count = 0;

            for (const auto& link : incoming) {
                if (link->has_embedding() &&
                    link->embedding().numel() == embedding_dim) {
                    msg += link->embedding();
                    count++;
                }
            }

            if (count > 0) {
                msg /= static_cast<double>(count);
                messages[atom->uuid()] = msg;
            }
        }

        // Update embeddings
        for (const auto& [uuid, msg] : messages) {
            auto atom = space.get_by_uuid(uuid);
            if (atom && atom->has_embedding()) {
                Tensor<double> new_emb = atom->embedding() * (1.0 - learning_rate) +
                                         msg * learning_rate;
                // Apply non-linearity
                new_emb = new_emb.relu();
                atom->set_embedding(std::move(new_emb));
            }
        }
    }

    // ===================================================================
    // Utility Functions
    // ===================================================================

    /**
     * @brief Normalize a tensor to unit norm.
     */
    static Tensor<double> normalize(const Tensor<double>& t)
    {
        double n = t.norm();
        return n > 1e-10 ? t / n : t;
    }

    /**
     * @brief Project tensor onto simplex (for probability distributions).
     */
    static Tensor<double> simplex_projection(const Tensor<double>& t)
    {
        // Sort in descending order
        std::vector<double> sorted(t.data(), t.data() + t.numel());
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());

        // Find threshold
        double cumsum = 0.0;
        double threshold = 0.0;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i];
            double tmp = (cumsum - 1.0) / static_cast<double>(i + 1);
            if (sorted[i] - tmp > 0) {
                threshold = tmp;
            }
        }

        // Project
        Tensor<double> result(t.shape());
        for (size_t i = 0; i < t.numel(); ++i) {
            result.data()[i] = std::max(0.0, t.data()[i] - threshold);
        }
        return result;
    }

    /**
     * @brief Compute tensor entropy.
     */
    static double tensor_entropy(const Tensor<double>& t)
    {
        double sum = t.sum();
        if (sum <= 0) return 0.0;

        double entropy = 0.0;
        for (size_t i = 0; i < t.numel(); ++i) {
            double p = t.data()[i] / sum;
            if (p > 1e-10) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    }
};

// ===================================================================
// Multi-Entity Network System
// ===================================================================

/**
 * @brief Multi-entity network system with tensor logic.
 *
 * Manages a collection of entities that interact through a shared
 * tensor space, supporting multi-scale reasoning and network-aware
 * operations.
 */
class MultiEntityNetwork
{
private:
    TensorSpace _space;
    std::vector<TensorEntity> _entities;
    std::map<NetworkScale, std::vector<ScaleAggregate>> _scale_hierarchy;
    size_t _embedding_dim;

public:
    explicit MultiEntityNetwork(const std::string& name = "MEN",
                                size_t embedding_dim = 64)
        : _space(name), _embedding_dim(embedding_dim)
    {}

    // Access
    TensorSpace& space() { return _space; }
    const TensorSpace& space() const { return _space; }
    std::vector<TensorEntity>& entities() { return _entities; }
    const std::vector<TensorEntity>& entities() const { return _entities; }

    /// Add an entity to the network
    TensorEntity& add_entity(const std::string& name)
    {
        _entities.emplace_back(name, _embedding_dim);
        _entities.back().link_to_space(_space);
        return _entities.back();
    }

    /// Get entity by name
    TensorEntity* get_entity(const std::string& name)
    {
        for (auto& e : _entities) {
            if (e.name() == name) return &e;
        }
        return nullptr;
    }

    /// Connect two entities
    void connect(const std::string& name1, const std::string& name2,
                 double strength = 1.0)
    {
        auto* e1 = get_entity(name1);
        auto* e2 = get_entity(name2);
        if (e1 && e2) {
            e1->set_relation(e2->id(), strength);
            e2->set_relation(e1->id(), strength);

            // Create Hebbian link in space
            if (e1->node() && e2->node()) {
                _space.add(TensorLink::create_hebbian(
                    e1->node(), e2->node(), TruthValue(strength, 0.9)));
            }
        }
    }

    /// Build multi-scale hierarchy
    void build_hierarchy(size_t meso_clusters = 10, size_t macro_clusters = 3)
    {
        // Get all embedded atoms
        std::vector<TensorAtomPtr> atoms;
        for (const auto& atom : _space.get_all_atoms()) {
            if (atom->has_embedding()) {
                atoms.push_back(atom);
            }
        }

        // Build meso scale
        _scale_hierarchy[NetworkScale::MESO] =
            TensorLogic::scale_aggregate(atoms, meso_clusters, NetworkScale::MESO);

        // Build macro scale from meso
        std::vector<TensorAtomPtr> meso_reps;
        for (auto& agg : _scale_hierarchy[NetworkScale::MESO]) {
            if (agg.embedding.numel() > 0) {
                auto rep = TensorNode::create_tensor("meso_" +
                    std::to_string(meso_reps.size()), agg.embedding);
                meso_reps.push_back(rep);
            }
        }

        _scale_hierarchy[NetworkScale::MACRO] =
            TensorLogic::scale_aggregate(meso_reps, macro_clusters, NetworkScale::MACRO);
    }

    /// Run one step of network dynamics
    void step(double learning_rate = 0.1)
    {
        // Entity message passing
        TensorLogic::message_passing(_entities, learning_rate);

        // Sync entity states to atom embeddings
        for (auto& e : _entities) {
            if (e.node()) {
                e.node()->set_embedding(e.state());
            }
        }

        // GNN update on space
        TensorLogic::gnn_update(_space, _embedding_dim, learning_rate * 0.5);

        // Attention decay
        _space.decay_sti(0.95);
    }

    /// Run multiple steps
    void run(size_t steps, double learning_rate = 0.1)
    {
        for (size_t i = 0; i < steps; ++i) {
            step(learning_rate);
        }
    }

    /// Query similar entities to a given entity
    std::vector<std::pair<TensorEntity*, double>>
    find_similar_entities(const std::string& name, size_t k = 5)
    {
        auto* query_entity = get_entity(name);
        if (!query_entity || !query_entity->node()) return {};

        auto similar = _space.find_similar(query_entity->node(), k);

        std::vector<std::pair<TensorEntity*, double>> results;
        for (const auto& [atom, score] : similar) {
            // Find corresponding entity
            for (auto& e : _entities) {
                if (e.node() && e.node()->uuid() == atom->uuid()) {
                    results.emplace_back(&e, score);
                    break;
                }
            }
        }
        return results;
    }

    /// Get network statistics
    std::string statistics() const
    {
        std::ostringstream oss;
        oss << "MultiEntityNetwork \"" << _space.name() << "\":\n";
        oss << "  Entities: " << _entities.size() << "\n";
        oss << "  Embedding dim: " << _embedding_dim << "\n";
        oss << _space.statistics();
        return oss.str();
    }
};

} // ~namespace opencog

/** @}*/

#endif // _OPENCOG_TENSOR_LOGIC_H
