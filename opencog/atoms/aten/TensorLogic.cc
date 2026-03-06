/*
 * opencog/atoms/aten/TensorLogic.cc
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * Tensor Logic for Multi-Entity & Multi-Scale Network-Aware
 * Tensor-Enhanced AtomSpace
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

#include <opencog/util/exceptions.h>
#include <opencog/atoms/aten/TensorLogic.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/value/LinkValue.h>
#include <opencog/atomspace/AtomSpace.h>

using namespace opencog;

// ============================================================
// EntityEmbedding Implementation

EntityEmbedding::EntityEmbedding(const Handle& entity,
                                 const ATenValuePtr& embedding,
                                 int64_t scale,
                                 double importance)
	: _entity(entity)
	, _embedding(embedding)
	, _scale_level(scale)
	, _importance(importance)
{
}

double EntityEmbedding::cosine_similarity(const EntityEmbedding& other) const
{
	if (!_embedding || !other._embedding)
		return 0.0;

	auto vec1 = _embedding->to_vector();
	auto vec2 = other._embedding->to_vector();

	if (vec1.size() != vec2.size())
		return 0.0;

	double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
	for (size_t i = 0; i < vec1.size(); i++) {
		dot += vec1[i] * vec2[i];
		norm1 += vec1[i] * vec1[i];
		norm2 += vec2[i] * vec2[i];
	}

	if (norm1 == 0.0 || norm2 == 0.0)
		return 0.0;

	return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

double EntityEmbedding::euclidean_distance(const EntityEmbedding& other) const
{
	if (!_embedding || !other._embedding)
		return std::numeric_limits<double>::max();

	auto vec1 = _embedding->to_vector();
	auto vec2 = other._embedding->to_vector();

	if (vec1.size() != vec2.size())
		return std::numeric_limits<double>::max();

	double sum_sq = 0.0;
	for (size_t i = 0; i < vec1.size(); i++) {
		double diff = vec1[i] - vec2[i];
		sum_sq += diff * diff;
	}

	return std::sqrt(sum_sq);
}

// ============================================================
// MultiScaleTensor Implementation

MultiScaleTensor::MultiScaleTensor(int64_t num_scales,
                                   const std::vector<int64_t>& scale_dims)
	: _num_scales(num_scales)
	, _scale_dims(scale_dims)
{
	_scales.resize(num_scales);
	for (int64_t i = 0; i < num_scales; i++) {
		int64_t dim = (i < (int64_t)scale_dims.size()) ? scale_dims[i] : 64;
		_scales[i] = createATenZeros({dim});
	}
}

ATenValuePtr MultiScaleTensor::at_scale(int64_t scale) const
{
	if (scale < 0 || scale >= _num_scales)
		throw RuntimeException(TRACE_INFO,
			"MultiScaleTensor: invalid scale %ld", scale);
	return _scales[scale];
}

void MultiScaleTensor::set_scale(int64_t scale, const ATenValuePtr& tensor)
{
	if (scale < 0 || scale >= _num_scales)
		throw RuntimeException(TRACE_INFO,
			"MultiScaleTensor: invalid scale %ld", scale);
	_scales[scale] = tensor;
}

ATenValuePtr MultiScaleTensor::upsample(int64_t from_scale,
                                         int64_t to_scale) const
{
	if (from_scale >= to_scale)
		return at_scale(from_scale);

	ATenValuePtr current = at_scale(from_scale);
	auto vec = current->to_vector();

	// Simple linear interpolation upsample
	int64_t target_dim = _scale_dims[to_scale];
	std::vector<double> result(target_dim);

	double scale = (double)vec.size() / target_dim;
	for (int64_t i = 0; i < target_dim; i++) {
		double pos = i * scale;
		int64_t idx = (int64_t)pos;
		double frac = pos - idx;

		if (idx + 1 < (int64_t)vec.size())
			result[i] = vec[idx] * (1 - frac) + vec[idx + 1] * frac;
		else
			result[i] = vec[idx];
	}

	return createATenFromVector(result, {target_dim});
}

ATenValuePtr MultiScaleTensor::downsample(int64_t from_scale,
                                           int64_t to_scale) const
{
	if (from_scale <= to_scale)
		return at_scale(from_scale);

	ATenValuePtr current = at_scale(from_scale);
	auto vec = current->to_vector();

	// Average pooling downsample
	int64_t target_dim = _scale_dims[to_scale];
	std::vector<double> result(target_dim, 0.0);

	int64_t pool_size = vec.size() / target_dim;
	for (int64_t i = 0; i < target_dim; i++) {
		double sum = 0.0;
		for (int64_t j = 0; j < pool_size; j++) {
			int64_t idx = i * pool_size + j;
			if (idx < (int64_t)vec.size())
				sum += vec[idx];
		}
		result[i] = sum / pool_size;
	}

	return createATenFromVector(result, {target_dim});
}

ATenValuePtr MultiScaleTensor::aggregate_scales() const
{
	// Concatenate all scale representations
	std::vector<double> result;
	for (const auto& scale_tensor : _scales) {
		if (scale_tensor) {
			auto vec = scale_tensor->to_vector();
			result.insert(result.end(), vec.begin(), vec.end());
		}
	}
	return createATenFromVector(result, {(int64_t)result.size()});
}

// ============================================================
// NetworkAwareTensor Implementation

NetworkAwareTensor::NetworkAwareTensor(int64_t num_nodes, int64_t feature_dim)
	: _num_nodes(num_nodes)
	, _num_edges(0)
	, _feature_dim(feature_dim)
{
	_node_features = createATenZeros({num_nodes, feature_dim});
	_adjacency = createATenZeros({num_nodes, num_nodes});
	_attention_weights = createATenOnes({num_nodes, num_nodes});

	// Normalize attention weights
	auto att_vec = _attention_weights->to_vector();
	for (auto& v : att_vec) v = 1.0 / num_nodes;
	_attention_weights = createATenFromVector(att_vec, {num_nodes, num_nodes});
}

void NetworkAwareTensor::set_node_features(const ATenValuePtr& features)
{
	_node_features = features;
}

void NetworkAwareTensor::set_edge_features(const ATenValuePtr& features)
{
	_edge_features = features;
}

void NetworkAwareTensor::set_adjacency(const ATenValuePtr& adj)
{
	_adjacency = adj;
}

ATenValuePtr NetworkAwareTensor::message_passing(int64_t num_hops) const
{
	if (!_node_features || !_adjacency)
		return nullptr;

	ATenValuePtr current = _node_features;

	for (int64_t hop = 0; hop < num_hops; hop++) {
		// Message = A * H (adjacency times node features)
		current = ATenValueCast(current->matmul(*_adjacency));
	}

	return current;
}

ATenValuePtr NetworkAwareTensor::graph_attention() const
{
	if (!_node_features || !_attention_weights)
		return nullptr;

	// Attention-weighted message passing: H' = softmax(A) * H
	return ATenValueCast(_node_features->matmul(*_attention_weights));
}

ATenValuePtr NetworkAwareTensor::aggregate_neighbors(int64_t node_idx) const
{
	if (!_node_features || !_adjacency || node_idx >= _num_nodes)
		return nullptr;

	auto adj_vec = _adjacency->to_vector();
	auto feat_vec = _node_features->to_vector();

	std::vector<double> result(_feature_dim, 0.0);
	double weight_sum = 0.0;

	for (int64_t j = 0; j < _num_nodes; j++) {
		double weight = adj_vec[node_idx * _num_nodes + j];
		if (weight > 0) {
			weight_sum += weight;
			for (int64_t d = 0; d < _feature_dim; d++) {
				result[d] += weight * feat_vec[j * _feature_dim + d];
			}
		}
	}

	if (weight_sum > 0) {
		for (auto& v : result) v /= weight_sum;
	}

	return createATenFromVector(result, {_feature_dim});
}

void NetworkAwareTensor::update_attention(const ATenValuePtr& query,
                                          const ATenValuePtr& key)
{
	// Compute attention: softmax(Q * K^T / sqrt(d))
	auto q_vec = query->to_vector();
	auto k_vec = key->to_vector();

	std::vector<double> attention(_num_nodes * _num_nodes);
	double scale = 1.0 / std::sqrt(_feature_dim);

	for (int64_t i = 0; i < _num_nodes; i++) {
		double max_score = -std::numeric_limits<double>::max();
		for (int64_t j = 0; j < _num_nodes; j++) {
			double score = 0.0;
			for (int64_t d = 0; d < _feature_dim; d++) {
				score += q_vec[i * _feature_dim + d] * k_vec[j * _feature_dim + d];
			}
			score *= scale;
			attention[i * _num_nodes + j] = score;
			max_score = std::max(max_score, score);
		}

		// Softmax normalization
		double sum_exp = 0.0;
		for (int64_t j = 0; j < _num_nodes; j++) {
			attention[i * _num_nodes + j] =
				std::exp(attention[i * _num_nodes + j] - max_score);
			sum_exp += attention[i * _num_nodes + j];
		}
		for (int64_t j = 0; j < _num_nodes; j++) {
			attention[i * _num_nodes + j] /= sum_exp;
		}
	}

	_attention_weights = createATenFromVector(attention, {_num_nodes, _num_nodes});
}

void NetworkAwareTensor::apply_hebbian_update(double learning_rate)
{
	if (!_node_features || !_adjacency)
		return;

	auto feat_vec = _node_features->to_vector();
	auto adj_vec = _adjacency->to_vector();

	// Hebbian: delta_w_ij = lr * h_i * h_j
	for (int64_t i = 0; i < _num_nodes; i++) {
		for (int64_t j = 0; j < _num_nodes; j++) {
			double h_i = 0.0, h_j = 0.0;
			for (int64_t d = 0; d < _feature_dim; d++) {
				h_i += feat_vec[i * _feature_dim + d];
				h_j += feat_vec[j * _feature_dim + d];
			}
			h_i /= _feature_dim;
			h_j /= _feature_dim;

			adj_vec[i * _num_nodes + j] += learning_rate * h_i * h_j;
		}
	}

	_adjacency = createATenFromVector(adj_vec, {_num_nodes, _num_nodes});
}

// ============================================================
// TruthValueTensor Implementation

TruthValueTensor::TruthValueTensor(const std::vector<int64_t>& shape)
{
	_strength = createATenFromVector(std::vector<double>(1, 0.5), shape);
	_confidence = createATenFromVector(std::vector<double>(1, 0.0), shape);
	_count = createATenZeros(shape);
}

TruthValueTensor::TruthValueTensor(const ATenValuePtr& strength,
                                   const ATenValuePtr& confidence)
	: _strength(strength)
	, _confidence(confidence)
{
	_count = createATenOnes(strength->shape());
}

TruthValueTensor TruthValueTensor::revision(const TruthValueTensor& other) const
{
	// PLN revision formula: weighted average by confidence
	auto s1 = _strength->to_vector();
	auto c1 = _confidence->to_vector();
	auto s2 = other._strength->to_vector();
	auto c2 = other._confidence->to_vector();

	std::vector<double> new_strength(s1.size());
	std::vector<double> new_confidence(c1.size());

	for (size_t i = 0; i < s1.size(); i++) {
		double w1 = c1[i], w2 = c2[i];
		double w_total = w1 + w2 - w1 * w2;

		if (w_total > 0) {
			new_strength[i] = (w1 * s1[i] + w2 * s2[i] - w1 * w2 * s1[i] * s2[i]) / w_total;
			new_confidence[i] = w_total;
		} else {
			new_strength[i] = 0.5;
			new_confidence[i] = 0.0;
		}
	}

	return TruthValueTensor(
		createATenFromVector(new_strength, _strength->shape()),
		createATenFromVector(new_confidence, _confidence->shape())
	);
}

TruthValueTensor TruthValueTensor::deduction(const TruthValueTensor& other) const
{
	// PLN deduction: s_c = s_a * s_b + (1 - s_a) * (s_c - s_a * s_b) / (1 - s_a)
	// Simplified: s_c ~= s_a * s_b
	auto s1 = _strength->to_vector();
	auto c1 = _confidence->to_vector();
	auto s2 = other._strength->to_vector();
	auto c2 = other._confidence->to_vector();

	std::vector<double> new_strength(s1.size());
	std::vector<double> new_confidence(c1.size());

	for (size_t i = 0; i < s1.size(); i++) {
		new_strength[i] = s1[i] * s2[i];
		new_confidence[i] = c1[i] * c2[i] * 0.9; // Confidence decay
	}

	return TruthValueTensor(
		createATenFromVector(new_strength, _strength->shape()),
		createATenFromVector(new_confidence, _confidence->shape())
	);
}

TruthValueTensor TruthValueTensor::induction(const TruthValueTensor& other) const
{
	// PLN induction
	auto s1 = _strength->to_vector();
	auto c1 = _confidence->to_vector();
	auto s2 = other._strength->to_vector();
	auto c2 = other._confidence->to_vector();

	std::vector<double> new_strength(s1.size());
	std::vector<double> new_confidence(c1.size());

	for (size_t i = 0; i < s1.size(); i++) {
		if (s2[i] > 0) {
			new_strength[i] = s1[i] / s2[i];
			new_strength[i] = std::min(1.0, std::max(0.0, new_strength[i]));
		} else {
			new_strength[i] = 0.5;
		}
		new_confidence[i] = c1[i] * c2[i] * 0.8;
	}

	return TruthValueTensor(
		createATenFromVector(new_strength, _strength->shape()),
		createATenFromVector(new_confidence, _confidence->shape())
	);
}

TruthValueTensor TruthValueTensor::abduction(const TruthValueTensor& other) const
{
	// PLN abduction (similar to induction with different semantics)
	return induction(other);
}

void TruthValueTensor::set_uniform(double strength, double confidence)
{
	auto s_vec = _strength->to_vector();
	auto c_vec = _confidence->to_vector();

	std::fill(s_vec.begin(), s_vec.end(), strength);
	std::fill(c_vec.begin(), c_vec.end(), confidence);

	_strength = createATenFromVector(s_vec, _strength->shape());
	_confidence = createATenFromVector(c_vec, _confidence->shape());
}

double TruthValueTensor::mean_strength() const
{
	auto vec = _strength->to_vector();
	double sum = 0.0;
	for (double v : vec) sum += v;
	return sum / vec.size();
}

double TruthValueTensor::mean_confidence() const
{
	auto vec = _confidence->to_vector();
	double sum = 0.0;
	for (double v : vec) sum += v;
	return sum / vec.size();
}

// ============================================================
// TensorLogic Implementation

TensorLogic::TensorLogic(AtomSpace* as,
                         int64_t embedding_dim,
                         int64_t num_scales,
                         const std::string& device)
	: _atomspace(as)
	, _embedding_dim(embedding_dim)
	, _num_scales(num_scales)
	, _device(device)
	, _num_embeddings(0)
	, _update_count(0)
{
	if (_atomspace) {
		_embedding_key = _atomspace->add_node(PREDICATE_NODE, "*-EmbeddingKey-*");
		_attention_key = _atomspace->add_node(PREDICATE_NODE, "*-AttentionKey-*");
		_truth_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TruthTensorKey-*");
	}

	// Initialize multi-scale with decreasing dimensions
	std::vector<int64_t> scale_dims;
	for (int64_t i = 0; i < num_scales; i++) {
		scale_dims.push_back(embedding_dim / (1 << i));
	}
	_multi_scale = std::make_shared<MultiScaleTensor>(num_scales, scale_dims);
}

TensorLogic::~TensorLogic()
{
	_embeddings.clear();
}

void TensorLogic::set_atomspace(AtomSpace* as)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_atomspace = as;
	if (_atomspace) {
		_embedding_key = _atomspace->add_node(PREDICATE_NODE, "*-EmbeddingKey-*");
		_attention_key = _atomspace->add_node(PREDICATE_NODE, "*-AttentionKey-*");
		_truth_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TruthTensorKey-*");
	}
}

void TensorLogic::set_device(const std::string& device)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_device = device;
}

EntityEmbeddingPtr TensorLogic::get_embedding(const Handle& atom)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _embeddings.find(atom);
	if (it != _embeddings.end())
		return it->second;

	// Create random embedding
	ATenValuePtr emb = createATenRandom({_embedding_dim});
	auto entity_emb = std::make_shared<EntityEmbedding>(atom, emb);
	_embeddings[atom] = entity_emb;
	_num_embeddings++;

	// Also attach to atom
	if (_atomspace && _embedding_key)
		_atomspace->set_value(atom, _embedding_key, emb);

	return entity_emb;
}

void TensorLogic::set_embedding(const Handle& atom, const ATenValuePtr& embedding,
                                int64_t scale)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto entity_emb = std::make_shared<EntityEmbedding>(atom, embedding, scale);
	_embeddings[atom] = entity_emb;

	if (_embeddings.find(atom) == _embeddings.end())
		_num_embeddings++;

	if (_atomspace && _embedding_key)
		_atomspace->set_value(atom, _embedding_key, embedding);

	_update_count++;
}

bool TensorLogic::has_embedding(const Handle& atom) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _embeddings.find(atom) != _embeddings.end();
}

void TensorLogic::remove_embedding(const Handle& atom)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _embeddings.find(atom);
	if (it != _embeddings.end()) {
		_embeddings.erase(it);
		_num_embeddings--;
	}
}

void TensorLogic::initialize_embeddings(const HandleSeq& atoms,
                                        const std::string& init_method)
{
	for (const Handle& atom : atoms) {
		if (!has_embedding(atom)) {
			ATenValuePtr emb;
			if (init_method == "zeros") {
				emb = createATenZeros({_embedding_dim});
			} else if (init_method == "ones") {
				emb = createATenOnes({_embedding_dim});
			} else {
				emb = createATenRandom({_embedding_dim});
			}
			set_embedding(atom, emb);
		}
	}
}

double TensorLogic::similarity(const Handle& a, const Handle& b)
{
	EntityEmbeddingPtr emb_a = get_embedding(a);
	EntityEmbeddingPtr emb_b = get_embedding(b);

	return emb_a->cosine_similarity(*emb_b);
}

HandleSeq TensorLogic::query_similar(const Handle& query, size_t k)
{
	std::lock_guard<std::mutex> lock(_mtx);

	EntityEmbeddingPtr query_emb = get_embedding(query);

	std::vector<std::pair<double, Handle>> scores;
	for (const auto& pair : _embeddings) {
		if (pair.first != query) {
			double sim = query_emb->cosine_similarity(*pair.second);
			scores.emplace_back(sim, pair.first);
		}
	}

	std::sort(scores.begin(), scores.end(),
		[](const auto& a, const auto& b) { return a.first > b.first; });

	HandleSeq results;
	for (size_t i = 0; i < std::min(k, scores.size()); i++) {
		results.push_back(scores[i].second);
	}

	return results;
}

HandleSeq TensorLogic::query_similar_tensor(const ATenValuePtr& query, size_t k)
{
	std::lock_guard<std::mutex> lock(_mtx);

	EntityEmbedding query_emb(Handle::UNDEFINED, query);

	std::vector<std::pair<double, Handle>> scores;
	for (const auto& pair : _embeddings) {
		double sim = query_emb.cosine_similarity(*pair.second);
		scores.emplace_back(sim, pair.first);
	}

	std::sort(scores.begin(), scores.end(),
		[](const auto& a, const auto& b) { return a.first > b.first; });

	HandleSeq results;
	for (size_t i = 0; i < std::min(k, scores.size()); i++) {
		results.push_back(scores[i].second);
	}

	return results;
}

ATenValuePtr TensorLogic::compute_link_embedding(const Handle& link,
                                                  const std::string& aggregation)
{
	if (!link->is_link())
		return get_embedding(link)->embedding();

	const HandleSeq& outgoing = link->getOutgoingSet();
	if (outgoing.empty())
		return createATenZeros({_embedding_dim});

	std::vector<ATenValuePtr> embeddings;
	for (const Handle& h : outgoing) {
		embeddings.push_back(get_embedding(h)->embedding());
	}

	// Aggregate embeddings
	std::vector<double> result(_embedding_dim, 0.0);

	if (aggregation == "sum" || aggregation == "mean") {
		for (const auto& emb : embeddings) {
			auto vec = emb->to_vector();
			for (size_t i = 0; i < vec.size() && i < (size_t)_embedding_dim; i++) {
				result[i] += vec[i];
			}
		}
		if (aggregation == "mean") {
			for (auto& v : result) v /= embeddings.size();
		}
	} else if (aggregation == "max") {
		for (size_t i = 0; i < (size_t)_embedding_dim; i++) {
			double max_val = -std::numeric_limits<double>::max();
			for (const auto& emb : embeddings) {
				auto vec = emb->to_vector();
				if (i < vec.size()) max_val = std::max(max_val, vec[i]);
			}
			result[i] = max_val;
		}
	} else if (aggregation == "attention") {
		// Attention-weighted aggregation
		double total_weight = 0.0;
		for (size_t j = 0; j < embeddings.size(); j++) {
			double weight = get_embedding(outgoing[j])->importance();
			total_weight += weight;
			auto vec = embeddings[j]->to_vector();
			for (size_t i = 0; i < vec.size() && i < (size_t)_embedding_dim; i++) {
				result[i] += weight * vec[i];
			}
		}
		if (total_weight > 0) {
			for (auto& v : result) v /= total_weight;
		}
	}

	return createATenFromVector(result, {_embedding_dim});
}

void TensorLogic::update_link_embedding(const Handle& link)
{
	ATenValuePtr emb = compute_link_embedding(link);
	set_embedding(link, emb);
}

MultiScaleTensorPtr TensorLogic::get_multi_scale(const Handle& atom)
{
	EntityEmbeddingPtr emb = get_embedding(atom);

	auto ms = std::make_shared<MultiScaleTensor>(
		_num_scales, _multi_scale->scale_dims());

	// Set base scale
	ms->set_scale(0, emb->embedding());

	// Compute other scales by pooling
	for (int64_t s = 1; s < _num_scales; s++) {
		ms->set_scale(s, ms->downsample(0, s));
	}

	return ms;
}

ATenValuePtr TensorLogic::hierarchical_abstraction(const HandleSeq& atoms,
                                                    int64_t target_scale)
{
	if (atoms.empty())
		return createATenZeros({_multi_scale->scale_dims()[target_scale]});

	std::vector<double> result(_multi_scale->scale_dims()[target_scale], 0.0);

	for (const Handle& atom : atoms) {
		auto ms = get_multi_scale(atom);
		auto vec = ms->at_scale(target_scale)->to_vector();

		for (size_t i = 0; i < vec.size() && i < result.size(); i++) {
			result[i] += vec[i];
		}
	}

	for (auto& v : result) v /= atoms.size();

	return createATenFromVector(result, {(int64_t)result.size()});
}

void TensorLogic::build_network_tensor(Type link_type)
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (!_atomspace) return;

	// Collect all nodes
	HandleSeq nodes;
	_atomspace->get_handles_by_type(nodes, NODE);

	int64_t num_nodes = nodes.size();
	if (num_nodes == 0) return;

	_network = std::make_shared<NetworkAwareTensor>(num_nodes, _embedding_dim);

	// Build node index
	std::map<Handle, int64_t> node_index;
	for (size_t i = 0; i < nodes.size(); i++) {
		node_index[nodes[i]] = i;
	}

	// Set node features
	std::vector<double> features(num_nodes * _embedding_dim, 0.0);
	for (size_t i = 0; i < nodes.size(); i++) {
		auto emb = get_embedding(nodes[i])->embedding()->to_vector();
		for (size_t j = 0; j < emb.size() && j < (size_t)_embedding_dim; j++) {
			features[i * _embedding_dim + j] = emb[j];
		}
	}
	_network->set_node_features(
		createATenFromVector(features, {num_nodes, _embedding_dim}));

	// Build adjacency from links
	std::vector<double> adjacency(num_nodes * num_nodes, 0.0);
	HandleSeq links;
	_atomspace->get_handles_by_type(links, link_type, true);

	for (const Handle& link : links) {
		const HandleSeq& outgoing = link->getOutgoingSet();
		for (size_t i = 0; i < outgoing.size(); i++) {
			for (size_t j = i + 1; j < outgoing.size(); j++) {
				auto it_i = node_index.find(outgoing[i]);
				auto it_j = node_index.find(outgoing[j]);

				if (it_i != node_index.end() && it_j != node_index.end()) {
					int64_t idx_i = it_i->second;
					int64_t idx_j = it_j->second;
					adjacency[idx_i * num_nodes + idx_j] = 1.0;
					adjacency[idx_j * num_nodes + idx_i] = 1.0;
				}
			}
		}
	}
	_network->set_adjacency(
		createATenFromVector(adjacency, {num_nodes, num_nodes}));
}

void TensorLogic::message_passing(int64_t num_iterations)
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (!_network) return;

	ATenValuePtr updated = _network->message_passing(num_iterations);
	_network->set_node_features(updated);
	_update_count++;
}

void TensorLogic::update_attention()
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (!_network) return;

	_network->update_attention(
		_network->node_features(),
		_network->node_features()
	);
	_update_count++;
}

void TensorLogic::hebbian_update(double learning_rate)
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (!_network) return;

	_network->apply_hebbian_update(learning_rate);
	_update_count++;
}

TruthValueTensorPtr TensorLogic::get_truth_tensor(const Handle& atom)
{
	// Check if already has truth tensor
	if (_atomspace && _truth_tensor_key) {
		ValuePtr val = atom->getValue(_truth_tensor_key);
		if (val) {
			// Try to deserialize as ATenValue (strength, confidence pair)
			ATenValuePtr tensor = ATenValueCast(val);
			if (tensor && tensor->numel() >= 2) {
				auto vec = tensor->to_vector();
				ATenValuePtr strength = createATenFromVector({vec[0]}, {1});
				ATenValuePtr confidence = createATenFromVector({vec[1]}, {1});
				return std::make_shared<TruthValueTensor>(strength, confidence);
			}
		}
	}

	// Create from atom's truth value using truth_key()
	auto tvt = std::make_shared<TruthValueTensor>(std::vector<int64_t>{1});

	// Get truth value from the atom's truth key
	ValuePtr truth_val = atom->getValue(truth_key());
	if (truth_val) {
		// FloatValue format: [strength, confidence, count, ...]
		FloatValuePtr fv = FloatValueCast(truth_val);
		if (fv) {
			auto values = fv->value();
			double strength = (values.size() > 0) ? values[0] : 0.5;
			double confidence = (values.size() > 1) ? values[1] : 0.0;
			tvt->set_uniform(strength, confidence);
		}
	}

	return tvt;
}

TruthValueTensorPtr TensorLogic::tensor_deduction(const Handle& premise1,
                                                   const Handle& premise2)
{
	auto tv1 = get_truth_tensor(premise1);
	auto tv2 = get_truth_tensor(premise2);

	return std::make_shared<TruthValueTensor>(tv1->deduction(*tv2));
}

TruthValueTensorPtr TensorLogic::tensor_revision(const Handle& atom,
                                                  const TruthValueTensorPtr& evidence)
{
	auto tv = get_truth_tensor(atom);
	return std::make_shared<TruthValueTensor>(tv->revision(*evidence));
}

HandleSeq TensorLogic::neural_inference(const Handle& query,
                                         const std::string& rule_type)
{
	// Find similar atoms as potential inference partners
	HandleSeq similar = query_similar(query, 10);

	HandleSeq results;
	for (const Handle& h : similar) {
		double sim = similarity(query, h);
		if (sim > 0.7) { // Threshold for inference
			results.push_back(h);
		}
	}

	return results;
}

HandleSeq TensorLogic::attention_query(const Handle& pattern, size_t max_results)
{
	return query_similar(pattern, max_results);
}

void TensorLogic::contrastive_update(const Handle& anchor,
                                     const Handle& positive,
                                     const Handle& negative,
                                     double learning_rate)
{
	auto emb_anchor = get_embedding(anchor)->embedding();
	auto emb_pos = get_embedding(positive)->embedding();
	auto emb_neg = get_embedding(negative)->embedding();

	auto anc_vec = emb_anchor->to_vector();
	auto pos_vec = emb_pos->to_vector();
	auto neg_vec = emb_neg->to_vector();

	// Triplet loss gradient update
	std::vector<double> updated(_embedding_dim);
	for (int64_t i = 0; i < _embedding_dim; i++) {
		double grad = 2 * (pos_vec[i] - anc_vec[i]) - 2 * (neg_vec[i] - anc_vec[i]);
		updated[i] = anc_vec[i] + learning_rate * grad;
	}

	set_embedding(anchor, createATenFromVector(updated, {_embedding_dim}));
}

void TensorLogic::backpropagate(const ATenValuePtr& loss)
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (!loss || !_network) {
		_update_count++;
		return;
	}

	// Basic backpropagation through embeddings
	// Uses simplified gradient descent with loss as scaling factor
	auto loss_vec = loss->to_vector();
	double loss_scalar = (loss_vec.size() > 0) ? loss_vec[0] : 1.0;

	// Learning rate for backprop (could be made configurable)
	double learning_rate = 0.01;

	// Update all embeddings with gradient-like scaling
	for (auto& pair : _embeddings) {
		auto emb = pair.second->embedding();
		if (!emb) continue;

		auto emb_vec = emb->to_vector();
		std::vector<double> updated(emb_vec.size());

		// Simple gradient descent step: reduce embedding values proportional to loss
		// NOTE: This is a simplified backprop implementation for basic gradient updates.
		// For full neural network backpropagation, a proper autograd system would be needed.

		// Gradient scaling factor - controls how strongly the loss affects updates
		constexpr double GRADIENT_SCALE = 0.1;

		for (size_t i = 0; i < emb_vec.size(); i++) {
			// Gradient approximation: scale by loss and gradient factor
			double grad = loss_scalar * emb_vec[i] * GRADIENT_SCALE;
			updated[i] = emb_vec[i] - learning_rate * grad;
		}

		pair.second->set_embedding(
			createATenFromVector(updated, {static_cast<int64_t>(updated.size())}));
	}

	_update_count++;
}

void TensorLogic::save_embeddings(const std::string& path)
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::ofstream file(path, std::ios::binary);
	if (!file.is_open())
		throw RuntimeException(TRACE_INFO, "Cannot open file: %s", path.c_str());

	// Write header
	int64_t num = _embeddings.size();
	file.write(reinterpret_cast<char*>(&num), sizeof(num));
	file.write(reinterpret_cast<char*>(&_embedding_dim), sizeof(_embedding_dim));

	// Write embeddings
	for (const auto& pair : _embeddings) {
		// Write atom ID (simplified - use string representation)
		std::string atom_str = pair.first->to_short_string();
		int64_t str_len = atom_str.size();
		file.write(reinterpret_cast<char*>(&str_len), sizeof(str_len));
		file.write(atom_str.c_str(), str_len);

		// Write embedding
		auto vec = pair.second->embedding()->to_vector();
		file.write(reinterpret_cast<const char*>(vec.data()),
		           vec.size() * sizeof(double));
	}

	file.close();
}

void TensorLogic::load_embeddings(const std::string& path)
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw RuntimeException(TRACE_INFO, "Cannot open file: %s", path.c_str());

	// Read header
	int64_t num = 0;
	int64_t emb_dim = 0;
	file.read(reinterpret_cast<char*>(&num), sizeof(num));
	file.read(reinterpret_cast<char*>(&emb_dim), sizeof(emb_dim));

	if (emb_dim != _embedding_dim) {
		throw RuntimeException(TRACE_INFO,
			"Embedding dimension mismatch: expected %ld, got %ld",
			_embedding_dim, emb_dim);
	}

	// Read embeddings
	for (int64_t i = 0; i < num; i++) {
		// Read atom string
		int64_t str_len = 0;
		file.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));

		std::string atom_str(str_len, '\0');
		file.read(&atom_str[0], str_len);

		// Read embedding vector
		std::vector<double> vec(emb_dim);
		file.read(reinterpret_cast<char*>(vec.data()), emb_dim * sizeof(double));

		// Lookup atom in atomspace (if available) and set embedding
		if (_atomspace) {
			// Try to find the atom by its string representation
			// NOTE: This is a simplified lookup that matches by to_short_string().
			// For production use, consider proper atom string parsing.
			HandleSeq atoms;
			_atomspace->get_handles_by_type(atoms, NODE, true);

			for (const Handle& h : atoms) {
				if (h->to_short_string() == atom_str) {
					ATenValuePtr emb = createATenFromVector(vec, {emb_dim});
					auto entity_emb = std::make_shared<EntityEmbedding>(h, emb, 0);

					// Only increment count if this is a new embedding
					bool is_new = (_embeddings.find(h) == _embeddings.end());
					_embeddings[h] = entity_emb;
					if (is_new)
						_num_embeddings++;

					if (_embedding_key)
						_atomspace->set_value(h, _embedding_key, emb);

					break;
				}
			}
		}
	}

	file.close();
}

std::string TensorLogic::to_string() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::ostringstream oss;
	oss << "TensorLogic {\n";
	oss << "  device: " << _device << "\n";
	oss << "  embedding_dim: " << _embedding_dim << "\n";
	oss << "  num_scales: " << _num_scales << "\n";
	oss << "  num_embeddings: " << _num_embeddings << "\n";
	oss << "  update_count: " << _update_count << "\n";
	oss << "}";
	return oss.str();
}

// ============================================================
// TensorLogicLink Implementation

TensorLogicLink::TensorLogicLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
{
	init();
}

void TensorLogicLink::init()
{
	// Base initialization
}

ValuePtr TensorLogicLink::execute(AtomSpace* as, bool silent)
{
	throw RuntimeException(TRACE_INFO,
		"TensorLogicLink base class cannot be executed directly");
}

Handle TensorLogicLink::factory(const Handle& h)
{
	return Handle(createTensorLogicLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// EmbeddingOfLink Implementation

EmbeddingOfLink::EmbeddingOfLink(const HandleSeq&& oset, Type t)
	: TensorLogicLink(std::move(oset), t)
{
	if (_outgoing.size() != 1)
		throw InvalidParamException(TRACE_INFO,
			"EmbeddingOfLink requires exactly 1 argument");
}

ValuePtr EmbeddingOfLink::execute(AtomSpace* as, bool silent)
{
	// Get TensorLogic instance (simplified - would need proper registry)
	TensorLogic logic(as);
	auto emb = logic.get_embedding(_outgoing[0]);
	return emb->embedding();
}

Handle EmbeddingOfLink::factory(const Handle& h)
{
	return Handle(createEmbeddingOfLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// SetEmbeddingLink Implementation

SetEmbeddingLink::SetEmbeddingLink(const HandleSeq&& oset, Type t)
	: TensorLogicLink(std::move(oset), t)
{
	if (_outgoing.size() != 2)
		throw InvalidParamException(TRACE_INFO,
			"SetEmbeddingLink requires exactly 2 arguments");
}

ValuePtr SetEmbeddingLink::execute(AtomSpace* as, bool silent)
{
	Handle target_atom = _outgoing[0];
	Handle tensor_source = _outgoing[1];

	// Get tensor from second argument
	ATenValuePtr tensor;

	// Try to execute if it's executable
	if (tensor_source->is_executable()) {
		ValuePtr result = tensor_source->execute(as);
		tensor = ATenValueCast(result);
		if (!tensor) {
			// Try FloatValue conversion
			FloatValuePtr fvp = FloatValueCast(result);
			if (fvp) {
				auto vec = fvp->value();
				tensor = createATenFromVector(vec, {static_cast<int64_t>(vec.size())});
			}
		}
	}

	// Check for NumberNode
	if (!tensor && tensor_source->is_type(NUMBER_NODE)) {
		NumberNodePtr nn = NumberNodeCast(tensor_source);
		auto vec = nn->value();
		tensor = createATenFromVector(vec, {static_cast<int64_t>(vec.size())});
	}

	// Check if source is already an ATenValue holder
	if (!tensor) {
		Handle tensor_key = as->get_node(PREDICATE_NODE, "*-TensorKey-*");
		if (tensor_key) {
			ValuePtr val = tensor_source->getValue(tensor_key);
			tensor = ATenValueCast(val);
		}
	}

	if (!tensor) {
		throw RuntimeException(TRACE_INFO,
			"SetEmbeddingLink: second argument is not a tensor: %s",
			tensor_source->to_short_string().c_str());
	}

	// Set the embedding using TensorLogic
	TensorLogic logic(as);
	logic.set_embedding(target_atom, tensor);

	// Return the tensor that was set
	return tensor;
}

Handle SetEmbeddingLink::factory(const Handle& h)
{
	return Handle(createSetEmbeddingLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// SimilarityQueryLink Implementation

SimilarityQueryLink::SimilarityQueryLink(const HandleSeq&& oset, Type t)
	: TensorLogicLink(std::move(oset), t)
{
	if (_outgoing.size() < 1)
		throw InvalidParamException(TRACE_INFO,
			"SimilarityQueryLink requires at least 1 argument");
}

ValuePtr SimilarityQueryLink::execute(AtomSpace* as, bool silent)
{
	size_t k = 10;
	if (_outgoing.size() >= 2) {
		// Get k from second argument
	}

	TensorLogic logic(as);
	HandleSeq similar = logic.query_similar(_outgoing[0], k);

	std::vector<ValuePtr> values;
	for (const Handle& h : similar)
		values.push_back(ValueCast(h));

	return createLinkValue(values);
}

Handle SimilarityQueryLink::factory(const Handle& h)
{
	return Handle(createSimilarityQueryLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// HebbianLink Implementation

HebbianLink::HebbianLink(const HandleSeq&& oset, Type t)
	: Link(std::move(oset), t)
	, _weight(1.0)
{
	if (_outgoing.size() < 2)
		throw InvalidParamException(TRACE_INFO,
			"HebbianLink requires at least 2 arguments");

	_weight_tensor = createATenOnes({1});
}

void HebbianLink::hebbian_update(double pre, double post, double lr)
{
	_weight += lr * pre * post;

	auto wt_vec = _weight_tensor->to_vector();
	wt_vec[0] = _weight;
	_weight_tensor = createATenFromVector(wt_vec, {1});
}

Handle HebbianLink::factory(const Handle& h)
{
	return Handle(createHebbianLink(std::move(h->getOutgoingSet())));
}

// ============================================================
// AttentionValueTensor Implementation

AttentionValueTensor::AttentionValueTensor(int64_t dim)
{
	_sti = createATenZeros({dim});
	_lti = createATenZeros({dim});
	_vlti = createATenZeros({dim});
}

AttentionValueTensor::AttentionValueTensor(const ATenValuePtr& sti,
                                           const ATenValuePtr& lti,
                                           const ATenValuePtr& vlti)
	: _sti(sti), _lti(lti), _vlti(vlti)
{
	if (!_lti) _lti = createATenZeros(_sti->shape());
	if (!_vlti) _vlti = createATenZeros(_sti->shape());
}

void AttentionValueTensor::stimulate(double amount)
{
	auto vec = _sti->to_vector();
	for (auto& v : vec) v += amount;
	_sti = createATenFromVector(vec, _sti->shape());
}

void AttentionValueTensor::decay(double factor)
{
	_sti = ATenValueCast(_sti->mul_scalar(factor));
}

void AttentionValueTensor::spread(const AttentionValueTensor& other,
                                   double fraction)
{
	auto sti_vec = _sti->to_vector();
	auto other_vec = other._sti->to_vector();

	double transfer = 0.0;
	for (double v : sti_vec) transfer += v * fraction;
	transfer /= other_vec.size();

	for (auto& v : other_vec) v += transfer;
	for (auto& v : sti_vec) v *= (1 - fraction);

	_sti = createATenFromVector(sti_vec, _sti->shape());
}

double AttentionValueTensor::mean_sti() const
{
	auto vec = _sti->to_vector();
	double sum = 0.0;
	for (double v : vec) sum += v;
	return sum / vec.size();
}

double AttentionValueTensor::mean_lti() const
{
	auto vec = _lti->to_vector();
	double sum = 0.0;
	for (double v : vec) sum += v;
	return sum / vec.size();
}

// ============================================================
// Factory registration
// Note: Factory methods are defined inline with each class above.
// The DEFINE_LINK_FACTORY macros are not needed here.
