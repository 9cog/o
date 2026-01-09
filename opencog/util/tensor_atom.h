/*
 * opencog/util/tensor_atom.h
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

#ifndef _OPENCOG_TENSOR_ATOM_H
#define _OPENCOG_TENSOR_ATOM_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencog/util/tensor.h>
#include <opencog/util/oc_assert.h>

/** \addtogroup grp_cogutil
 *  @{
 */

namespace opencog
{

// Forward declarations
class TensorAtom;
class TensorNode;
class TensorLink;
class TensorSpace;

using TensorAtomPtr = std::shared_ptr<TensorAtom>;
using TensorNodePtr = std::shared_ptr<TensorNode>;
using TensorLinkPtr = std::shared_ptr<TensorLink>;
using TensorAtomWeakPtr = std::weak_ptr<TensorAtom>;

// ===================================================================
// Truth Value - Probabilistic Logic Networks (PLN) support
// ===================================================================

/**
 * @brief Truth value for probabilistic reasoning.
 *
 * Implements simple truth values with strength and confidence,
 * supporting PLN-style uncertain reasoning.
 */
struct TruthValue
{
    double strength;     // Probability estimate [0, 1]
    double confidence;   // Confidence in the estimate [0, 1]

    TruthValue(double s = 1.0, double c = 1.0)
        : strength(std::max(0.0, std::min(1.0, s))),
          confidence(std::max(0.0, std::min(1.0, c)))
    {}

    /// Compute mean (strength weighted by confidence)
    double mean() const { return strength; }

    /// Convert confidence to count (for PLN formulas)
    double count() const
    {
        // Using standard PLN conversion: c = n / (n + k), so n = c*k / (1-c)
        const double k = 800.0; // Default PLN lookahead
        if (confidence >= 1.0) return 1e10;
        return (confidence * k) / (1.0 - confidence);
    }

    /// Create from count
    static TruthValue from_count(double strength, double count, double k = 800.0)
    {
        double conf = count / (count + k);
        return TruthValue(strength, conf);
    }

    /// PLN revision: combine two truth values
    TruthValue revision(const TruthValue& other) const
    {
        double n1 = count(), n2 = other.count();
        double new_count = n1 + n2;
        double new_strength = (n1 * strength + n2 * other.strength) / new_count;
        return from_count(new_strength, new_count);
    }

    /// PLN conjunction (AND)
    TruthValue conjunction(const TruthValue& other) const
    {
        return TruthValue(strength * other.strength,
                          std::min(confidence, other.confidence));
    }

    /// PLN disjunction (OR)
    TruthValue disjunction(const TruthValue& other) const
    {
        double s = strength + other.strength - strength * other.strength;
        return TruthValue(s, std::min(confidence, other.confidence));
    }

    /// PLN negation (NOT)
    TruthValue negation() const
    {
        return TruthValue(1.0 - strength, confidence);
    }

    bool operator==(const TruthValue& other) const
    {
        return std::abs(strength - other.strength) < 1e-10 &&
               std::abs(confidence - other.confidence) < 1e-10;
    }

    std::string to_string() const
    {
        return "TruthValue(s=" + std::to_string(strength) +
               ", c=" + std::to_string(confidence) + ")";
    }
};

// ===================================================================
// Attention Value - ECAN (Economic Attention Networks) support
// ===================================================================

/**
 * @brief Attention value for cognitive focus management.
 *
 * Implements Short-Term Importance (STI), Long-Term Importance (LTI),
 * and Very-Long-Term Importance (VLTI) for attention allocation.
 */
struct AttentionValue
{
    double sti;   // Short-term importance [-1, 1]
    double lti;   // Long-term importance [0, 1]
    double vlti;  // Very-long-term importance (persistence flag)

    AttentionValue(double s = 0.0, double l = 0.0, double v = 0.0)
        : sti(std::max(-1.0, std::min(1.0, s))),
          lti(std::max(0.0, std::min(1.0, l))),
          vlti(std::max(0.0, std::min(1.0, v)))
    {}

    /// Total importance score
    double importance() const { return sti + lti; }

    /// Check if in attentional focus
    bool in_focus(double threshold = 0.0) const { return sti > threshold; }

    /// Decay STI over time
    AttentionValue decay(double factor = 0.95) const
    {
        return AttentionValue(sti * factor, lti, vlti);
    }

    /// Stimulate (increase STI)
    AttentionValue stimulate(double amount) const
    {
        return AttentionValue(sti + amount, lti, vlti);
    }

    std::string to_string() const
    {
        return "AttentionValue(sti=" + std::to_string(sti) +
               ", lti=" + std::to_string(lti) +
               ", vlti=" + std::to_string(vlti) + ")";
    }
};

// ===================================================================
// Atom Types - Enumeration of symbolic atom types
// ===================================================================

enum class AtomType : uint16_t
{
    // Base types
    ATOM = 0,
    NODE,
    LINK,

    // Node types
    CONCEPT_NODE,
    PREDICATE_NODE,
    VARIABLE_NODE,
    NUMBER_NODE,
    SCHEMA_NODE,
    GROUNDED_SCHEMA_NODE,
    ANCHOR_NODE,
    ENTITY_NODE,

    // Link types - Logical
    AND_LINK,
    OR_LINK,
    NOT_LINK,
    IMPLICATION_LINK,
    EQUIVALENCE_LINK,

    // Link types - Set/List
    SET_LINK,
    LIST_LINK,
    MEMBER_LINK,
    SUBSET_LINK,

    // Link types - Relational
    INHERITANCE_LINK,
    SIMILARITY_LINK,
    EVALUATION_LINK,
    EXECUTION_LINK,

    // Link types - Contextual
    CONTEXT_LINK,
    DEFINE_LINK,

    // Link types - Temporal
    AT_TIME_LINK,
    TIME_INTERVAL_LINK,
    SEQUENTIAL_AND_LINK,
    SEQUENTIAL_OR_LINK,

    // Link types - Attention (ECAN)
    HEBBIAN_LINK,
    ASYMMETRIC_HEBBIAN_LINK,
    SYMMETRIC_HEBBIAN_LINK,

    // Tensor-specific types
    TENSOR_NODE,
    TENSOR_LINK,
    EMBEDDING_LINK,
    SIMILARITY_TENSOR_LINK,

    // Custom/User types start here
    USER_TYPE_BEGIN = 1000
};

/// Convert atom type to string
inline std::string atom_type_to_string(AtomType type)
{
    switch (type) {
        case AtomType::ATOM: return "Atom";
        case AtomType::NODE: return "Node";
        case AtomType::LINK: return "Link";
        case AtomType::CONCEPT_NODE: return "ConceptNode";
        case AtomType::PREDICATE_NODE: return "PredicateNode";
        case AtomType::VARIABLE_NODE: return "VariableNode";
        case AtomType::NUMBER_NODE: return "NumberNode";
        case AtomType::SCHEMA_NODE: return "SchemaNode";
        case AtomType::GROUNDED_SCHEMA_NODE: return "GroundedSchemaNode";
        case AtomType::ANCHOR_NODE: return "AnchorNode";
        case AtomType::ENTITY_NODE: return "EntityNode";
        case AtomType::AND_LINK: return "AndLink";
        case AtomType::OR_LINK: return "OrLink";
        case AtomType::NOT_LINK: return "NotLink";
        case AtomType::IMPLICATION_LINK: return "ImplicationLink";
        case AtomType::EQUIVALENCE_LINK: return "EquivalenceLink";
        case AtomType::SET_LINK: return "SetLink";
        case AtomType::LIST_LINK: return "ListLink";
        case AtomType::MEMBER_LINK: return "MemberLink";
        case AtomType::SUBSET_LINK: return "SubsetLink";
        case AtomType::INHERITANCE_LINK: return "InheritanceLink";
        case AtomType::SIMILARITY_LINK: return "SimilarityLink";
        case AtomType::EVALUATION_LINK: return "EvaluationLink";
        case AtomType::EXECUTION_LINK: return "ExecutionLink";
        case AtomType::CONTEXT_LINK: return "ContextLink";
        case AtomType::DEFINE_LINK: return "DefineLink";
        case AtomType::AT_TIME_LINK: return "AtTimeLink";
        case AtomType::TIME_INTERVAL_LINK: return "TimeIntervalLink";
        case AtomType::SEQUENTIAL_AND_LINK: return "SequentialAndLink";
        case AtomType::SEQUENTIAL_OR_LINK: return "SequentialOrLink";
        case AtomType::HEBBIAN_LINK: return "HebbianLink";
        case AtomType::ASYMMETRIC_HEBBIAN_LINK: return "AsymmetricHebbianLink";
        case AtomType::SYMMETRIC_HEBBIAN_LINK: return "SymmetricHebbianLink";
        case AtomType::TENSOR_NODE: return "TensorNode";
        case AtomType::TENSOR_LINK: return "TensorLink";
        case AtomType::EMBEDDING_LINK: return "EmbeddingLink";
        case AtomType::SIMILARITY_TENSOR_LINK: return "SimilarityTensorLink";
        default: return "UnknownType(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

// ===================================================================
// TensorAtom - Base class for all atoms with tensor embeddings
// ===================================================================

/**
 * @brief Base class for atoms in a tensor-enhanced AtomSpace.
 *
 * TensorAtom provides the foundation for symbolic knowledge representation
 * enhanced with neural tensor embeddings. Each atom can have:
 * - A unique identifier (UUID)
 * - A type (from AtomType enum)
 * - A truth value (for probabilistic reasoning)
 * - An attention value (for cognitive focus)
 * - An optional tensor embedding (for neural representations)
 *
 * Atoms are immutable once created; their identity is fixed.
 */
class TensorAtom : public std::enable_shared_from_this<TensorAtom>
{
public:
    using UUID = uint64_t;
    using EmbeddingType = Tensor<double>;

protected:
    UUID _uuid;
    AtomType _type;
    TruthValue _tv;
    AttentionValue _av;
    std::optional<EmbeddingType> _embedding;
    mutable std::mutex _mutex;

    // UUID generator
    static std::atomic<UUID> _uuid_counter;

    // Protected constructor (use factory methods)
    TensorAtom(AtomType type, const TruthValue& tv = TruthValue())
        : _uuid(_uuid_counter.fetch_add(1)),
          _type(type),
          _tv(tv),
          _av()
    {}

public:
    virtual ~TensorAtom() = default;

    // Disable copy, enable move
    TensorAtom(const TensorAtom&) = delete;
    TensorAtom& operator=(const TensorAtom&) = delete;
    TensorAtom(TensorAtom&&) = default;
    TensorAtom& operator=(TensorAtom&&) = default;

    // ===================================================================
    // Accessors
    // ===================================================================

    UUID uuid() const { return _uuid; }
    AtomType type() const { return _type; }
    std::string type_name() const { return atom_type_to_string(_type); }

    const TruthValue& tv() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _tv;
    }

    void set_tv(const TruthValue& tv)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _tv = tv;
    }

    const AttentionValue& av() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _av;
    }

    void set_av(const AttentionValue& av)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _av = av;
    }

    // ===================================================================
    // Embedding operations
    // ===================================================================

    bool has_embedding() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _embedding.has_value();
    }

    const EmbeddingType& embedding() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        OC_ASSERT(_embedding.has_value(), "TensorAtom: no embedding set");
        return *_embedding;
    }

    void set_embedding(const EmbeddingType& emb)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _embedding = emb;
    }

    void set_embedding(EmbeddingType&& emb)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _embedding = std::move(emb);
    }

    void clear_embedding()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _embedding.reset();
    }

    /// Get embedding dimension (0 if no embedding)
    size_t embedding_dim() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _embedding.has_value() ? _embedding->numel() : 0;
    }

    // ===================================================================
    // Virtual methods
    // ===================================================================

    /// Check if this is a node
    virtual bool is_node() const { return false; }

    /// Check if this is a link
    virtual bool is_link() const { return false; }

    /// Get string representation (for debugging)
    virtual std::string to_string() const
    {
        return type_name() + "[" + std::to_string(_uuid) + "]";
    }

    /// Get short string representation
    virtual std::string short_string() const { return to_string(); }

    /// Compute hash for atom (for use in hash tables)
    virtual size_t hash() const { return std::hash<UUID>{}(_uuid); }

    /// Check equality with another atom
    virtual bool equals(const TensorAtom& other) const
    {
        return _uuid == other._uuid;
    }
};

// Initialize static UUID counter
inline std::atomic<TensorAtom::UUID> TensorAtom::_uuid_counter{1};

// ===================================================================
// TensorNode - Node with name and tensor embedding
// ===================================================================

/**
 * @brief Node representing an entity or concept.
 *
 * TensorNode extends TensorAtom with a name and represents entities,
 * concepts, predicates, variables, and other named objects in the
 * knowledge hypergraph.
 */
class TensorNode : public TensorAtom
{
protected:
    std::string _name;

public:
    TensorNode(AtomType type, const std::string& name,
               const TruthValue& tv = TruthValue())
        : TensorAtom(type, tv),
          _name(name)
    {}

    /// Create a concept node
    static TensorNodePtr create_concept(const std::string& name,
                                        const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorNode>(AtomType::CONCEPT_NODE, name, tv);
    }

    /// Create a predicate node
    static TensorNodePtr create_predicate(const std::string& name,
                                          const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorNode>(AtomType::PREDICATE_NODE, name, tv);
    }

    /// Create a variable node
    static TensorNodePtr create_variable(const std::string& name)
    {
        return std::make_shared<TensorNode>(AtomType::VARIABLE_NODE, name);
    }

    /// Create an entity node (for multi-entity systems)
    static TensorNodePtr create_entity(const std::string& name,
                                       const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorNode>(AtomType::ENTITY_NODE, name, tv);
    }

    /// Create a tensor node (node that primarily holds tensor data)
    static TensorNodePtr create_tensor(const std::string& name,
                                       const Tensor<double>& tensor)
    {
        auto node = std::make_shared<TensorNode>(AtomType::TENSOR_NODE, name);
        node->set_embedding(tensor);
        return node;
    }

    // Accessors
    const std::string& name() const { return _name; }

    // Virtual overrides
    bool is_node() const override { return true; }

    std::string to_string() const override
    {
        std::string result = type_name() + "(\"" + _name + "\")";
        if (has_embedding()) {
            result += "[emb:" + std::to_string(embedding_dim()) + "d]";
        }
        return result;
    }

    std::string short_string() const override
    {
        return "(" + type_name() + " \"" + _name + "\")";
    }

    size_t hash() const override
    {
        size_t h = std::hash<uint16_t>{}(static_cast<uint16_t>(_type));
        h ^= std::hash<std::string>{}(_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }

    bool equals(const TensorAtom& other) const override
    {
        if (!other.is_node()) return false;
        const auto& node = static_cast<const TensorNode&>(other);
        return _type == node._type && _name == node._name;
    }
};

// ===================================================================
// TensorLink - Link connecting atoms with tensor operations
// ===================================================================

/**
 * @brief Link representing relationships between atoms.
 *
 * TensorLink extends TensorAtom to represent hypergraph edges connecting
 * multiple atoms. Links can represent logical relationships, set
 * membership, temporal relations, and tensor-based similarities.
 */
class TensorLink : public TensorAtom
{
public:
    using OutgoingSet = std::vector<TensorAtomPtr>;

protected:
    OutgoingSet _outgoing;

public:
    TensorLink(AtomType type, const OutgoingSet& outgoing,
               const TruthValue& tv = TruthValue())
        : TensorAtom(type, tv),
          _outgoing(outgoing)
    {}

    TensorLink(AtomType type, std::initializer_list<TensorAtomPtr> outgoing,
               const TruthValue& tv = TruthValue())
        : TensorAtom(type, tv),
          _outgoing(outgoing)
    {}

    // ===================================================================
    // Factory methods for common link types
    // ===================================================================

    /// Create an inheritance link: A inherits from B
    static TensorLinkPtr create_inheritance(TensorAtomPtr a, TensorAtomPtr b,
                                            const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::INHERITANCE_LINK,
                                            OutgoingSet{a, b}, tv);
    }

    /// Create a similarity link: A is similar to B
    static TensorLinkPtr create_similarity(TensorAtomPtr a, TensorAtomPtr b,
                                           const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::SIMILARITY_LINK,
                                            OutgoingSet{a, b}, tv);
    }

    /// Create an evaluation link: Predicate(arguments)
    static TensorLinkPtr create_evaluation(TensorAtomPtr predicate,
                                           const OutgoingSet& args,
                                           const TruthValue& tv = TruthValue())
    {
        OutgoingSet outgoing = {predicate};
        outgoing.insert(outgoing.end(), args.begin(), args.end());
        return std::make_shared<TensorLink>(AtomType::EVALUATION_LINK, outgoing, tv);
    }

    /// Create an AND link
    static TensorLinkPtr create_and(const OutgoingSet& atoms,
                                    const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::AND_LINK, atoms, tv);
    }

    /// Create an OR link
    static TensorLinkPtr create_or(const OutgoingSet& atoms,
                                   const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::OR_LINK, atoms, tv);
    }

    /// Create a NOT link
    static TensorLinkPtr create_not(TensorAtomPtr atom,
                                    const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::NOT_LINK,
                                            OutgoingSet{atom}, tv);
    }

    /// Create an implication link: A implies B
    static TensorLinkPtr create_implication(TensorAtomPtr a, TensorAtomPtr b,
                                            const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::IMPLICATION_LINK,
                                            OutgoingSet{a, b}, tv);
    }

    /// Create a set link
    static TensorLinkPtr create_set(const OutgoingSet& atoms)
    {
        return std::make_shared<TensorLink>(AtomType::SET_LINK, atoms);
    }

    /// Create a list link (ordered)
    static TensorLinkPtr create_list(const OutgoingSet& atoms)
    {
        return std::make_shared<TensorLink>(AtomType::LIST_LINK, atoms);
    }

    /// Create a member link: element is member of set
    static TensorLinkPtr create_member(TensorAtomPtr element, TensorAtomPtr set,
                                       const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::MEMBER_LINK,
                                            OutgoingSet{element, set}, tv);
    }

    /// Create a Hebbian link (for ECAN attention spreading)
    static TensorLinkPtr create_hebbian(TensorAtomPtr a, TensorAtomPtr b,
                                        const TruthValue& tv = TruthValue())
    {
        return std::make_shared<TensorLink>(AtomType::SYMMETRIC_HEBBIAN_LINK,
                                            OutgoingSet{a, b}, tv);
    }

    /// Create an embedding link (associates atom with tensor)
    static TensorLinkPtr create_embedding(TensorAtomPtr atom,
                                          TensorAtomPtr tensor_node)
    {
        return std::make_shared<TensorLink>(AtomType::EMBEDDING_LINK,
                                            OutgoingSet{atom, tensor_node});
    }

    /// Create a tensor similarity link (computed from embeddings)
    static TensorLinkPtr create_tensor_similarity(TensorAtomPtr a, TensorAtomPtr b,
                                                  double similarity)
    {
        return std::make_shared<TensorLink>(AtomType::SIMILARITY_TENSOR_LINK,
                                            OutgoingSet{a, b},
                                            TruthValue(similarity, 0.9));
    }

    // ===================================================================
    // Accessors
    // ===================================================================

    const OutgoingSet& outgoing() const { return _outgoing; }

    size_t arity() const { return _outgoing.size(); }

    TensorAtomPtr outgoing_atom(size_t idx) const
    {
        OC_ASSERT(idx < _outgoing.size(), "TensorLink: index out of bounds");
        return _outgoing[idx];
    }

    // ===================================================================
    // Virtual overrides
    // ===================================================================

    bool is_link() const override { return true; }

    std::string to_string() const override
    {
        std::string result = type_name() + "(";
        for (size_t i = 0; i < _outgoing.size(); ++i) {
            if (i > 0) result += ", ";
            result += _outgoing[i] ? _outgoing[i]->short_string() : "null";
        }
        result += ")";
        return result;
    }

    std::string short_string() const override
    {
        std::string result = "(" + type_name();
        for (const auto& atom : _outgoing) {
            result += " " + (atom ? atom->short_string() : "null");
        }
        result += ")";
        return result;
    }

    size_t hash() const override
    {
        size_t h = std::hash<uint16_t>{}(static_cast<uint16_t>(_type));
        for (const auto& atom : _outgoing) {
            h ^= (atom ? atom->hash() : 0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    bool equals(const TensorAtom& other) const override
    {
        if (!other.is_link()) return false;
        const auto& link = static_cast<const TensorLink&>(other);
        if (_type != link._type || _outgoing.size() != link._outgoing.size()) {
            return false;
        }
        for (size_t i = 0; i < _outgoing.size(); ++i) {
            if (!_outgoing[i] || !link._outgoing[i]) {
                if (_outgoing[i] != link._outgoing[i]) return false;
            } else if (!_outgoing[i]->equals(*link._outgoing[i])) {
                return false;
            }
        }
        return true;
    }

    // ===================================================================
    // Tensor operations on linked atoms
    // ===================================================================

    /// Compute combined embedding from all atoms with embeddings
    std::optional<Tensor<double>> combined_embedding(
        const std::string& method = "mean") const
    {
        std::vector<Tensor<double>> embeddings;
        for (const auto& atom : _outgoing) {
            if (atom && atom->has_embedding()) {
                embeddings.push_back(atom->embedding());
            }
        }

        if (embeddings.empty()) return std::nullopt;

        // Ensure all embeddings have same shape
        const auto& shape = embeddings[0].shape();
        for (size_t i = 1; i < embeddings.size(); ++i) {
            if (embeddings[i].shape() != shape) {
                return std::nullopt; // Shape mismatch
            }
        }

        if (method == "mean") {
            Tensor<double> result = embeddings[0].clone();
            for (size_t i = 1; i < embeddings.size(); ++i) {
                result += embeddings[i];
            }
            return result / static_cast<double>(embeddings.size());
        } else if (method == "sum") {
            Tensor<double> result = embeddings[0].clone();
            for (size_t i = 1; i < embeddings.size(); ++i) {
                result += embeddings[i];
            }
            return result;
        } else if (method == "max") {
            Tensor<double> result = embeddings[0].clone();
            for (size_t i = 1; i < embeddings.size(); ++i) {
                for (size_t j = 0; j < result.numel(); ++j) {
                    result.data()[j] = std::max(result.data()[j],
                                                embeddings[i].data()[j]);
                }
            }
            return result;
        }

        return std::nullopt;
    }
};

// ===================================================================
// Helper functions for atom operations
// ===================================================================

/// Compute cosine similarity between two atoms with embeddings
inline double tensor_similarity(const TensorAtom& a, const TensorAtom& b)
{
    if (!a.has_embedding() || !b.has_embedding()) {
        return 0.0;
    }

    const auto& emb_a = a.embedding();
    const auto& emb_b = b.embedding();

    if (emb_a.shape() != emb_b.shape()) {
        return 0.0;
    }

    // Cosine similarity
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < emb_a.numel(); ++i) {
        dot += emb_a.data()[i] * emb_b.data()[i];
        norm_a += emb_a.data()[i] * emb_a.data()[i];
        norm_b += emb_b.data()[i] * emb_b.data()[i];
    }

    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 1e-10 ? dot / denom : 0.0;
}

/// Compute Euclidean distance between embeddings
inline double tensor_distance(const TensorAtom& a, const TensorAtom& b)
{
    if (!a.has_embedding() || !b.has_embedding()) {
        return std::numeric_limits<double>::infinity();
    }

    const auto& emb_a = a.embedding();
    const auto& emb_b = b.embedding();

    if (emb_a.shape() != emb_b.shape()) {
        return std::numeric_limits<double>::infinity();
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < emb_a.numel(); ++i) {
        double diff = emb_a.data()[i] - emb_b.data()[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

} // ~namespace opencog

/** @}*/

#endif // _OPENCOG_TENSOR_ATOM_H
