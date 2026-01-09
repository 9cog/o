/*
 * opencog/util/tensor_space.h
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

#ifndef _OPENCOG_TENSOR_SPACE_H
#define _OPENCOG_TENSOR_SPACE_H

#include <algorithm>
#include <functional>
#include <map>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <opencog/util/tensor_atom.h>

/** \addtogroup grp_cogutil
 *  @{
 */

namespace opencog
{

// ===================================================================
// Hash and equality for atom pointers
// ===================================================================

struct AtomPtrHash
{
    size_t operator()(const TensorAtomPtr& atom) const
    {
        return atom ? atom->hash() : 0;
    }
};

struct AtomPtrEqual
{
    bool operator()(const TensorAtomPtr& a, const TensorAtomPtr& b) const
    {
        if (!a || !b) return a == b;
        return a->equals(*b);
    }
};

// ===================================================================
// TensorSpace - Hypergraph container with tensor operations
// ===================================================================

/**
 * @brief Container for tensor-enhanced atoms forming a hypergraph.
 *
 * TensorSpace is the main container for symbolic knowledge representation
 * with neural tensor embeddings. It provides:
 * - Thread-safe storage of atoms (nodes and links)
 * - Index by type, name, and UUID for fast retrieval
 * - Tensor similarity search
 * - Attention bank for ECAN-style focus management
 * - Multi-scale network operations
 *
 * Based on OpenCog's AtomSpace, enhanced with ATen-style tensor support.
 */
class TensorSpace
{
public:
    using AtomSet = std::unordered_set<TensorAtomPtr, AtomPtrHash, AtomPtrEqual>;
    using AtomIndex = std::unordered_map<TensorAtom::UUID, TensorAtomPtr>;
    using TypeIndex = std::unordered_map<AtomType, AtomSet>;
    using NameIndex = std::unordered_map<std::string, AtomSet>;
    using IncomingIndex = std::unordered_map<TensorAtom::UUID, AtomSet>;

private:
    mutable std::shared_mutex _mutex;

    AtomSet _atoms;              // All atoms
    AtomIndex _uuid_index;       // UUID -> Atom
    TypeIndex _type_index;       // Type -> Set of atoms
    NameIndex _name_index;       // Name -> Set of nodes
    IncomingIndex _incoming;     // Atom UUID -> Links containing it

    std::string _name;           // Space name

    // Embedding index for tensor similarity search
    std::vector<TensorAtomPtr> _embedded_atoms;
    Tensor<double> _embedding_matrix;  // Stacked embeddings
    bool _embedding_index_dirty = true;

public:
    explicit TensorSpace(const std::string& name = "default")
        : _name(name)
    {}

    ~TensorSpace() = default;

    // Disable copy
    TensorSpace(const TensorSpace&) = delete;
    TensorSpace& operator=(const TensorSpace&) = delete;

    // ===================================================================
    // Basic operations
    // ===================================================================

    /// Get the name of this space
    const std::string& name() const { return _name; }

    /// Get total number of atoms
    size_t size() const
    {
        std::shared_lock lock(_mutex);
        return _atoms.size();
    }

    /// Check if empty
    bool empty() const { return size() == 0; }

    /// Clear all atoms
    void clear()
    {
        std::unique_lock lock(_mutex);
        _atoms.clear();
        _uuid_index.clear();
        _type_index.clear();
        _name_index.clear();
        _incoming.clear();
        _embedded_atoms.clear();
        _embedding_index_dirty = true;
    }

    // ===================================================================
    // Adding atoms
    // ===================================================================

    /// Add an atom to the space (returns existing if duplicate)
    TensorAtomPtr add(const TensorAtomPtr& atom)
    {
        if (!atom) return nullptr;

        std::unique_lock lock(_mutex);

        // Check if equivalent atom exists
        auto it = _atoms.find(atom);
        if (it != _atoms.end()) {
            return *it;
        }

        // Add to main set and indices
        _atoms.insert(atom);
        _uuid_index[atom->uuid()] = atom;
        _type_index[atom->type()].insert(atom);

        // Index by name if node
        if (atom->is_node()) {
            auto node = std::static_pointer_cast<TensorNode>(atom);
            _name_index[node->name()].insert(atom);
        }

        // Index incoming for links
        if (atom->is_link()) {
            auto link = std::static_pointer_cast<TensorLink>(atom);
            for (const auto& target : link->outgoing()) {
                if (target) {
                    _incoming[target->uuid()].insert(atom);
                }
            }
        }

        // Mark embedding index as dirty
        if (atom->has_embedding()) {
            _embedding_index_dirty = true;
        }

        return atom;
    }

    /// Add a concept node
    TensorNodePtr add_concept(const std::string& name,
                              const TruthValue& tv = TruthValue())
    {
        auto node = TensorNode::create_concept(name, tv);
        return std::static_pointer_cast<TensorNode>(add(node));
    }

    /// Add a predicate node
    TensorNodePtr add_predicate(const std::string& name,
                                const TruthValue& tv = TruthValue())
    {
        auto node = TensorNode::create_predicate(name, tv);
        return std::static_pointer_cast<TensorNode>(add(node));
    }

    /// Add an entity node
    TensorNodePtr add_entity(const std::string& name,
                             const TruthValue& tv = TruthValue())
    {
        auto node = TensorNode::create_entity(name, tv);
        return std::static_pointer_cast<TensorNode>(add(node));
    }

    /// Add an inheritance link
    TensorLinkPtr add_inheritance(TensorAtomPtr a, TensorAtomPtr b,
                                  const TruthValue& tv = TruthValue())
    {
        auto link = TensorLink::create_inheritance(a, b, tv);
        return std::static_pointer_cast<TensorLink>(add(link));
    }

    /// Add a similarity link
    TensorLinkPtr add_similarity(TensorAtomPtr a, TensorAtomPtr b,
                                 const TruthValue& tv = TruthValue())
    {
        auto link = TensorLink::create_similarity(a, b, tv);
        return std::static_pointer_cast<TensorLink>(add(link));
    }

    /// Add an evaluation link
    TensorLinkPtr add_evaluation(TensorAtomPtr predicate,
                                 const std::vector<TensorAtomPtr>& args,
                                 const TruthValue& tv = TruthValue())
    {
        auto link = TensorLink::create_evaluation(predicate, args, tv);
        return std::static_pointer_cast<TensorLink>(add(link));
    }

    // ===================================================================
    // Removing atoms
    // ===================================================================

    /// Remove an atom from the space
    bool remove(const TensorAtomPtr& atom)
    {
        if (!atom) return false;

        std::unique_lock lock(_mutex);

        auto it = _atoms.find(atom);
        if (it == _atoms.end()) return false;

        // Remove from indices
        _uuid_index.erase(atom->uuid());
        _type_index[atom->type()].erase(atom);

        if (atom->is_node()) {
            auto node = std::static_pointer_cast<TensorNode>(atom);
            _name_index[node->name()].erase(atom);
        }

        if (atom->is_link()) {
            auto link = std::static_pointer_cast<TensorLink>(atom);
            for (const auto& target : link->outgoing()) {
                if (target) {
                    _incoming[target->uuid()].erase(atom);
                }
            }
        }

        _atoms.erase(it);

        if (atom->has_embedding()) {
            _embedding_index_dirty = true;
        }

        return true;
    }

    /// Remove atom by UUID
    bool remove(TensorAtom::UUID uuid)
    {
        auto atom = get_by_uuid(uuid);
        return atom ? remove(atom) : false;
    }

    // ===================================================================
    // Retrieval operations
    // ===================================================================

    /// Get atom by UUID
    TensorAtomPtr get_by_uuid(TensorAtom::UUID uuid) const
    {
        std::shared_lock lock(_mutex);
        auto it = _uuid_index.find(uuid);
        return it != _uuid_index.end() ? it->second : nullptr;
    }

    /// Get all atoms of a specific type
    AtomSet get_by_type(AtomType type) const
    {
        std::shared_lock lock(_mutex);
        auto it = _type_index.find(type);
        return it != _type_index.end() ? it->second : AtomSet{};
    }

    /// Get all nodes with a specific name
    AtomSet get_by_name(const std::string& name) const
    {
        std::shared_lock lock(_mutex);
        auto it = _name_index.find(name);
        return it != _name_index.end() ? it->second : AtomSet{};
    }

    /// Get node by type and name
    TensorNodePtr get_node(AtomType type, const std::string& name) const
    {
        std::shared_lock lock(_mutex);
        auto name_it = _name_index.find(name);
        if (name_it == _name_index.end()) return nullptr;

        for (const auto& atom : name_it->second) {
            if (atom->type() == type) {
                return std::static_pointer_cast<TensorNode>(atom);
            }
        }
        return nullptr;
    }

    /// Get incoming links for an atom
    AtomSet get_incoming(const TensorAtomPtr& atom) const
    {
        if (!atom) return {};
        std::shared_lock lock(_mutex);
        auto it = _incoming.find(atom->uuid());
        return it != _incoming.end() ? it->second : AtomSet{};
    }

    /// Get incoming links of a specific type
    AtomSet get_incoming(const TensorAtomPtr& atom, AtomType type) const
    {
        AtomSet result;
        for (const auto& link : get_incoming(atom)) {
            if (link->type() == type) {
                result.insert(link);
            }
        }
        return result;
    }

    /// Check if atom exists in space
    bool contains(const TensorAtomPtr& atom) const
    {
        if (!atom) return false;
        std::shared_lock lock(_mutex);
        return _atoms.find(atom) != _atoms.end();
    }

    /// Get all atoms
    AtomSet get_all_atoms() const
    {
        std::shared_lock lock(_mutex);
        return _atoms;
    }

    // ===================================================================
    // Tensor operations
    // ===================================================================

    /// Rebuild the embedding index for similarity search
    void rebuild_embedding_index()
    {
        std::unique_lock lock(_mutex);

        _embedded_atoms.clear();
        std::vector<std::vector<double>> embeddings;

        size_t embedding_dim = 0;
        for (const auto& atom : _atoms) {
            if (atom->has_embedding()) {
                const auto& emb = atom->embedding();
                if (embedding_dim == 0) {
                    embedding_dim = emb.numel();
                } else if (emb.numel() != embedding_dim) {
                    continue; // Skip mismatched dimensions
                }
                _embedded_atoms.push_back(atom);
                embeddings.push_back(
                    std::vector<double>(emb.data(), emb.data() + emb.numel()));
            }
        }

        if (!embeddings.empty()) {
            size_t n = embeddings.size();
            std::vector<double> flat_data;
            flat_data.reserve(n * embedding_dim);
            for (const auto& emb : embeddings) {
                flat_data.insert(flat_data.end(), emb.begin(), emb.end());
            }
            _embedding_matrix = Tensor<double>(flat_data, {n, embedding_dim});
        } else {
            _embedding_matrix = Tensor<double>();
        }

        _embedding_index_dirty = false;
    }

    /// Find k most similar atoms to a query embedding
    std::vector<std::pair<TensorAtomPtr, double>>
    find_similar(const Tensor<double>& query, size_t k) const
    {
        // Ensure index is up to date
        if (_embedding_index_dirty) {
            const_cast<TensorSpace*>(this)->rebuild_embedding_index();
        }

        std::shared_lock lock(_mutex);

        if (_embedded_atoms.empty() || query.numel() == 0) {
            return {};
        }

        // Compute similarities
        std::vector<std::pair<TensorAtomPtr, double>> results;

        double query_norm = query.norm();
        if (query_norm < 1e-10) return {};

        for (size_t i = 0; i < _embedded_atoms.size(); ++i) {
            const auto& atom = _embedded_atoms[i];
            // Compute directly
            const auto& emb = atom->embedding();
            if (emb.numel() != query.numel()) continue;

            double dot = 0.0, emb_norm = 0.0;
            for (size_t j = 0; j < query.numel(); ++j) {
                dot += query.data()[j] * emb.data()[j];
                emb_norm += emb.data()[j] * emb.data()[j];
            }
            emb_norm = std::sqrt(emb_norm);

            double similarity = (emb_norm > 1e-10) ?
                dot / (query_norm * emb_norm) : 0.0;

            results.emplace_back(atom, similarity);
        }

        // Sort by similarity (descending)
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;
                  });

        // Return top k
        if (results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    /// Find similar atoms to an existing atom
    std::vector<std::pair<TensorAtomPtr, double>>
    find_similar(const TensorAtomPtr& atom, size_t k) const
    {
        if (!atom || !atom->has_embedding()) return {};
        auto results = find_similar(atom->embedding(), k + 1);

        // Remove the query atom itself from results
        results.erase(
            std::remove_if(results.begin(), results.end(),
                           [&atom](const auto& p) {
                               return p.first->uuid() == atom->uuid();
                           }),
            results.end());

        if (results.size() > k) results.resize(k);
        return results;
    }

    /// Compute embedding for an atom based on its neighborhood
    Tensor<double> compute_neighborhood_embedding(
        const TensorAtomPtr& atom, size_t hops = 1) const
    {
        if (!atom) return Tensor<double>();

        std::shared_lock lock(_mutex);

        std::unordered_set<TensorAtom::UUID> visited;
        std::vector<TensorAtomPtr> neighbors;

        // BFS to find neighbors
        std::queue<std::pair<TensorAtomPtr, size_t>> queue;
        queue.push({atom, 0});
        visited.insert(atom->uuid());

        while (!queue.empty()) {
            auto [current, depth] = queue.front();
            queue.pop();

            if (current->has_embedding()) {
                neighbors.push_back(current);
            }

            if (depth < hops) {
                // Add outgoing (for links)
                if (current->is_link()) {
                    auto link = std::static_pointer_cast<TensorLink>(current);
                    for (const auto& target : link->outgoing()) {
                        if (target && visited.find(target->uuid()) == visited.end()) {
                            visited.insert(target->uuid());
                            queue.push({target, depth + 1});
                        }
                    }
                }

                // Add incoming
                auto inc_it = _incoming.find(current->uuid());
                if (inc_it != _incoming.end()) {
                    for (const auto& incoming_link : inc_it->second) {
                        if (visited.find(incoming_link->uuid()) == visited.end()) {
                            visited.insert(incoming_link->uuid());
                            queue.push({incoming_link, depth + 1});
                        }
                    }
                }
            }
        }

        if (neighbors.empty()) return Tensor<double>();

        // Average embeddings
        size_t dim = neighbors[0]->embedding().numel();
        Tensor<double> result = Tensor<double>::zeros({dim});

        size_t count = 0;
        for (const auto& neighbor : neighbors) {
            if (neighbor->embedding().numel() == dim) {
                result += neighbor->embedding();
                count++;
            }
        }

        return count > 0 ? result / static_cast<double>(count) : result;
    }

    // ===================================================================
    // Attention Bank operations (ECAN)
    // ===================================================================

    /// Get atoms in the attentional focus (STI > threshold)
    AtomSet get_attentional_focus(double threshold = 0.0) const
    {
        std::shared_lock lock(_mutex);
        AtomSet result;
        for (const auto& atom : _atoms) {
            if (atom->av().in_focus(threshold)) {
                result.insert(atom);
            }
        }
        return result;
    }

    /// Get top-k atoms by STI
    std::vector<TensorAtomPtr> get_top_by_sti(size_t k) const
    {
        std::shared_lock lock(_mutex);

        std::vector<TensorAtomPtr> atoms(_atoms.begin(), _atoms.end());
        std::sort(atoms.begin(), atoms.end(),
                  [](const auto& a, const auto& b) {
                      return a->av().sti > b->av().sti;
                  });

        if (atoms.size() > k) atoms.resize(k);
        return atoms;
    }

    /// Decay all STI values (for forgetting)
    void decay_sti(double factor = 0.95)
    {
        std::shared_lock lock(_mutex);
        for (const auto& atom : _atoms) {
            atom->set_av(atom->av().decay(factor));
        }
    }

    /// Spread attention from source to targets via Hebbian links
    void spread_attention(const TensorAtomPtr& source, double amount)
    {
        if (!source) return;

        auto hebbian_links = get_incoming(source, AtomType::SYMMETRIC_HEBBIAN_LINK);
        if (hebbian_links.empty()) return;

        // Distribute attention proportionally to link strengths
        double total_strength = 0.0;
        for (const auto& link : hebbian_links) {
            total_strength += link->tv().strength;
        }

        if (total_strength < 1e-10) return;

        for (const auto& link_atom : hebbian_links) {
            auto link = std::static_pointer_cast<TensorLink>(link_atom);
            double proportion = link->tv().strength / total_strength;
            double spread = amount * proportion;

            for (const auto& target : link->outgoing()) {
                if (target && target->uuid() != source->uuid()) {
                    target->set_av(target->av().stimulate(spread));
                }
            }
        }
    }

    // ===================================================================
    // Query/Pattern matching (simplified)
    // ===================================================================

    /// Find atoms matching a predicate
    template<typename Predicate>
    AtomSet filter(Predicate pred) const
    {
        std::shared_lock lock(_mutex);
        AtomSet result;
        for (const auto& atom : _atoms) {
            if (pred(atom)) {
                result.insert(atom);
            }
        }
        return result;
    }

    /// Find atoms by truth value threshold
    AtomSet filter_by_tv(double min_strength, double min_confidence = 0.0) const
    {
        return filter([=](const TensorAtomPtr& atom) {
            return atom->tv().strength >= min_strength &&
                   atom->tv().confidence >= min_confidence;
        });
    }

    /// Find links with a specific target
    AtomSet find_links_with_target(const TensorAtomPtr& target) const
    {
        return get_incoming(target);
    }

    // ===================================================================
    // Statistics and debugging
    // ===================================================================

    /// Get count of atoms by type
    std::map<AtomType, size_t> type_counts() const
    {
        std::shared_lock lock(_mutex);
        std::map<AtomType, size_t> counts;
        for (const auto& [type, atoms] : _type_index) {
            counts[type] = atoms.size();
        }
        return counts;
    }

    /// Get count of atoms with embeddings
    size_t embedding_count() const
    {
        std::shared_lock lock(_mutex);
        size_t count = 0;
        for (const auto& atom : _atoms) {
            if (atom->has_embedding()) count++;
        }
        return count;
    }

    /// Print statistics
    std::string statistics() const
    {
        std::ostringstream oss;
        oss << "TensorSpace \"" << _name << "\" statistics:\n";
        oss << "  Total atoms: " << size() << "\n";
        oss << "  With embeddings: " << embedding_count() << "\n";
        oss << "  By type:\n";
        for (const auto& [type, count] : type_counts()) {
            oss << "    " << atom_type_to_string(type) << ": " << count << "\n";
        }
        return oss.str();
    }
};

} // ~namespace opencog

/** @}*/

#endif // _OPENCOG_TENSOR_SPACE_H
