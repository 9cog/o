/*
 * TensorLogic.cc
 *
 * Multi-Entity & Multi-Scale Network-Aware Tensor Logic Implementation
 *
 * Copyright (c) 2025 OpenCog Foundation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation.
 */

#ifdef HAVE_ATEN

#include <algorithm>
#include <sstream>
#include <cmath>
#include <queue>

#include "TensorLogic.h"

using namespace opencog;

// ============================================================
// MultiScaleTensor Implementation
// ============================================================

MultiScaleTensor::MultiScaleTensor(int64_t embedding_dim)
	: _embedding_dim(embedding_dim)
{
	_base_shape = {embedding_dim};
}

MultiScaleTensor::~MultiScaleTensor() {}

void MultiScaleTensor::set_scale(ScaleLevel level, const TensorValuePtr& embedding)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_scale_embeddings[level] = embedding;
}

TensorValuePtr MultiScaleTensor::get_scale(ScaleLevel level) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	auto it = _scale_embeddings.find(level);
	if (it == _scale_embeddings.end()) return nullptr;
	return it->second;
}

bool MultiScaleTensor::has_scale(ScaleLevel level) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _scale_embeddings.find(level) != _scale_embeddings.end();
}

TensorValuePtr MultiScaleTensor::get_concatenated() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<at::Tensor> tensors;
	// Ordered concatenation: LOCAL, NEIGHBOR, COMMUNITY, GLOBAL
	for (auto level : {ScaleLevel::LOCAL, ScaleLevel::NEIGHBOR,
	                   ScaleLevel::COMMUNITY, ScaleLevel::GLOBAL}) {
		auto it = _scale_embeddings.find(level);
		if (it != _scale_embeddings.end() && it->second) {
			tensors.push_back(it->second->tensor());
		}
	}

	if (tensors.empty()) return nullptr;

	at::Tensor concat = torch::cat(tensors, -1);
	return createTensorValue(concat);
}

TensorValuePtr MultiScaleTensor::get_weighted(const std::vector<double>& weights) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	if (weights.size() < 4) {
		throw std::runtime_error("Need 4 weights for LOCAL, NEIGHBOR, COMMUNITY, GLOBAL");
	}

	at::Tensor result = torch::zeros({_embedding_dim});
	size_t idx = 0;

	for (auto level : {ScaleLevel::LOCAL, ScaleLevel::NEIGHBOR,
	                   ScaleLevel::COMMUNITY, ScaleLevel::GLOBAL}) {
		auto it = _scale_embeddings.find(level);
		if (it != _scale_embeddings.end() && it->second) {
			result = result + weights[idx] * it->second->tensor();
		}
		idx++;
	}

	return createTensorValue(result);
}

TensorValuePtr MultiScaleTensor::get_attended(const TensorValuePtr& query) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<at::Tensor> tensors;
	std::vector<ScaleLevel> present_levels;

	for (auto level : {ScaleLevel::LOCAL, ScaleLevel::NEIGHBOR,
	                   ScaleLevel::COMMUNITY, ScaleLevel::GLOBAL}) {
		auto it = _scale_embeddings.find(level);
		if (it != _scale_embeddings.end() && it->second) {
			tensors.push_back(it->second->tensor());
			present_levels.push_back(level);
		}
	}

	if (tensors.empty()) return nullptr;

	// Stack tensors: [num_scales, embedding_dim]
	at::Tensor stacked = torch::stack(tensors, 0);

	// Compute attention scores
	at::Tensor q = query->tensor().unsqueeze(0);  // [1, dim]
	at::Tensor scores = torch::matmul(q, stacked.t());  // [1, num_scales]
	scores = scores / std::sqrt((double)_embedding_dim);
	at::Tensor weights = torch::softmax(scores, -1);  // [1, num_scales]

	// Weighted combination
	at::Tensor result = torch::matmul(weights, stacked);  // [1, dim]
	return createTensorValue(result.squeeze(0));
}

size_t MultiScaleTensor::num_scales() const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _scale_embeddings.size();
}

std::string MultiScaleTensor::to_string() const
{
	std::ostringstream ss;
	ss << "(MultiScaleTensor dim=" << _embedding_dim;
	ss << " scales=[";
	bool first = true;
	for (auto level : {ScaleLevel::LOCAL, ScaleLevel::NEIGHBOR,
	                   ScaleLevel::COMMUNITY, ScaleLevel::GLOBAL}) {
		if (has_scale(level)) {
			if (!first) ss << ",";
			switch (level) {
				case ScaleLevel::LOCAL: ss << "LOCAL"; break;
				case ScaleLevel::NEIGHBOR: ss << "NEIGHBOR"; break;
				case ScaleLevel::COMMUNITY: ss << "COMMUNITY"; break;
				case ScaleLevel::GLOBAL: ss << "GLOBAL"; break;
			}
			first = false;
		}
	}
	ss << "])";
	return ss.str();
}

// ============================================================
// NetworkAwareTensor Implementation
// ============================================================

NetworkAwareTensor::NetworkAwareTensor(AtomSpace* as, ATenSpace* aten,
                                       int64_t embedding_dim, int64_t num_heads)
	: _as(as), _aten(aten), _embedding_dim(embedding_dim),
	  _num_heads(num_heads), _use_edge_features(false)
{
}

NetworkAwareTensor::~NetworkAwareTensor() {}

void NetworkAwareTensor::init_type_embeddings(const std::vector<Type>& types)
{
	std::lock_guard<std::mutex> lock(_mtx);

	// Create embedding matrix for atom types
	at::Tensor embeddings = torch::randn({(int64_t)types.size(), _embedding_dim});
	torch::nn::init::xavier_uniform_(embeddings);
	_type_embeddings = createTensorValue(embeddings);
}

TensorValuePtr NetworkAwareTensor::compute_degree_encoding(const Handle& atom) const
{
	// Get incoming and outgoing degree
	size_t in_degree = atom->getIncomingSetSize();
	size_t out_degree = 0;
	if (atom->is_link()) {
		out_degree = atom->get_arity();
	}

	// Create encoding with degree features
	int64_t encoding_dim = 32;  // Fixed degree encoding dimension
	at::Tensor encoding = torch::zeros({encoding_dim});

	// Log-scale degree features
	encoding[0] = std::log1p((double)in_degree);
	encoding[1] = std::log1p((double)out_degree);
	encoding[2] = std::log1p((double)(in_degree + out_degree));

	// Sinusoidal positional encoding of degrees
	for (int64_t i = 3; i < encoding_dim; i += 2) {
		double div = std::pow(10000.0, (double)(i - 3) / (encoding_dim - 3));
		encoding[i] = std::sin(in_degree / div);
		if (i + 1 < encoding_dim) {
			encoding[i + 1] = std::cos(out_degree / div);
		}
	}

	return createTensorValue(encoding);
}

TensorValuePtr NetworkAwareTensor::compute_type_embedding(Type atom_type) const
{
	// Simple one-hot style type encoding
	// In production, this would use the type_embeddings lookup table
	at::Tensor encoding = torch::zeros({32});

	// Hash type to index
	size_t type_idx = atom_type % 32;
	encoding[type_idx] = 1.0;

	return createTensorValue(encoding);
}

TensorValuePtr NetworkAwareTensor::aggregate_neighbors(
	const Handle& atom, const Handle& embedding_key,
	const std::string& aggregation) const
{
	// Get incoming set
	HandleSeq neighbors;
	IncomingSet incoming = atom->getIncomingSet();
	for (const Handle& link : incoming) {
		// Get other atoms in the link
		for (const Handle& h : link->getOutgoingSet()) {
			if (h != atom) {
				neighbors.push_back(h);
			}
		}
	}

	// If link, add outgoing atoms
	if (atom->is_link()) {
		for (const Handle& h : atom->getOutgoingSet()) {
			neighbors.push_back(h);
		}
	}

	if (neighbors.empty()) {
		return createTensorValue(torch::zeros({_embedding_dim}));
	}

	// Collect neighbor embeddings
	std::vector<at::Tensor> emb_list;
	for (const Handle& n : neighbors) {
		TensorValuePtr tv = _aten->get_tensor(n, embedding_key);
		if (tv) {
			emb_list.push_back(tv->tensor());
		}
	}

	if (emb_list.empty()) {
		return createTensorValue(torch::zeros({_embedding_dim}));
	}

	at::Tensor stacked = torch::stack(emb_list, 0);  // [N, D]

	at::Tensor result;
	if (aggregation == "mean") {
		result = torch::mean(stacked, 0);
	} else if (aggregation == "sum") {
		result = torch::sum(stacked, 0);
	} else if (aggregation == "max") {
		result = std::get<0>(torch::max(stacked, 0));
	} else {
		// Default: attention-weighted mean
		TensorValuePtr self_emb = _aten->get_tensor(atom, embedding_key);
		if (self_emb) {
			at::Tensor query = self_emb->tensor().unsqueeze(0);  // [1, D]
			at::Tensor scores = torch::matmul(query, stacked.t());  // [1, N]
			scores = scores / std::sqrt((double)_embedding_dim);
			at::Tensor weights = torch::softmax(scores, -1);
			result = torch::matmul(weights, stacked).squeeze(0);  // [D]
		} else {
			result = torch::mean(stacked, 0);
		}
	}

	return createTensorValue(result);
}

TensorValuePtr NetworkAwareTensor::compute_embedding(
	const Handle& atom, const Handle& feature_key) const
{
	// Get base features
	TensorValuePtr base = _aten->get_tensor(atom, feature_key);
	at::Tensor base_tensor;
	if (base) {
		base_tensor = base->tensor();
	} else {
		base_tensor = torch::zeros({_embedding_dim});
	}

	// Get structural encodings
	TensorValuePtr degree_enc = compute_degree_encoding(atom);
	TensorValuePtr type_enc = compute_type_embedding(atom->get_type());

	// Get neighbor aggregation
	TensorValuePtr neighbor_agg = aggregate_neighbors(atom, feature_key, "attention");

	// Combine: [base; degree; type; neighbors] then project to embedding_dim
	at::Tensor combined = torch::cat({
		base_tensor,
		degree_enc->tensor(),
		type_enc->tensor(),
		neighbor_agg->tensor()
	}, 0);

	// Linear projection (simplified - in production use learned weights)
	int64_t combined_dim = combined.size(0);
	at::Tensor proj = torch::randn({_embedding_dim, combined_dim}) / std::sqrt((double)combined_dim);
	at::Tensor result = torch::matmul(proj, combined);
	result = torch::relu(result);

	return createTensorValue(result);
}

std::unordered_map<Handle, TensorValuePtr> NetworkAwareTensor::compute_embeddings(
	const HandleSeq& atoms, const Handle& feature_key,
	int num_layers, const std::string& aggregation) const
{
	std::unordered_map<Handle, TensorValuePtr> embeddings;

	// Initialize with base embeddings
	for (const Handle& atom : atoms) {
		TensorValuePtr base = _aten->get_tensor(atom, feature_key);
		if (base) {
			embeddings[atom] = base;
		} else {
			embeddings[atom] = createTensorValue(torch::randn({_embedding_dim}));
		}
	}

	// Message passing layers
	for (int layer = 0; layer < num_layers; layer++) {
		if (aggregation == "attention") {
			embeddings = graph_attention_layer(atoms, embeddings);
		} else {
			// Simple aggregation
			std::unordered_map<Handle, TensorValuePtr> new_embeddings;
			for (const Handle& atom : atoms) {
				TensorValuePtr self_emb = embeddings[atom];
				TensorValuePtr neigh_agg = aggregate_neighbors(atom, feature_key, aggregation);

				// Combine self and neighbor
				at::Tensor combined = self_emb->tensor() + neigh_agg->tensor();
				combined = torch::relu(combined);
				new_embeddings[atom] = createTensorValue(combined);
			}
			embeddings = new_embeddings;
		}
	}

	return embeddings;
}

std::unordered_map<Handle, TensorValuePtr> NetworkAwareTensor::graph_attention_layer(
	const HandleSeq& atoms,
	const std::unordered_map<Handle, TensorValuePtr>& current_embeddings,
	const Handle& edge_key) const
{
	std::unordered_map<Handle, TensorValuePtr> new_embeddings;

	for (const Handle& atom : atoms) {
		auto it = current_embeddings.find(atom);
		if (it == current_embeddings.end()) continue;

		at::Tensor self_emb = it->second->tensor();

		// Collect neighbors and their embeddings
		std::vector<at::Tensor> neighbor_embs;
		IncomingSet incoming = atom->getIncomingSet();

		for (const Handle& link : incoming) {
			for (const Handle& h : link->getOutgoingSet()) {
				if (h != atom) {
					auto nit = current_embeddings.find(h);
					if (nit != current_embeddings.end()) {
						neighbor_embs.push_back(nit->second->tensor());
					}
				}
			}
		}

		if (neighbor_embs.empty()) {
			new_embeddings[atom] = it->second;
			continue;
		}

		// Multi-head attention
		at::Tensor neighbors = torch::stack(neighbor_embs, 0);  // [N, D]
		int64_t head_dim = _embedding_dim / _num_heads;

		std::vector<at::Tensor> head_outputs;
		for (int64_t h = 0; h < _num_heads; h++) {
			int64_t start = h * head_dim;
			int64_t end = start + head_dim;

			at::Tensor q = self_emb.slice(0, start, end).unsqueeze(0);  // [1, head_dim]
			at::Tensor k = neighbors.slice(1, start, end);  // [N, head_dim]
			at::Tensor v = neighbors.slice(1, start, end);  // [N, head_dim]

			at::Tensor scores = torch::matmul(q, k.t()) / std::sqrt((double)head_dim);
			at::Tensor weights = torch::softmax(scores, -1);
			at::Tensor head_out = torch::matmul(weights, v).squeeze(0);  // [head_dim]
			head_outputs.push_back(head_out);
		}

		// Concatenate heads and combine with self
		at::Tensor attended = torch::cat(head_outputs, 0);  // [D]
		at::Tensor combined = self_emb + attended;
		combined = torch::layer_norm(combined, {_embedding_dim});

		new_embeddings[atom] = createTensorValue(combined);
	}

	return new_embeddings;
}

TensorValuePtr NetworkAwareTensor::compute_pair_embedding(
	const Handle& source, const Handle& target,
	const Handle& relation_key) const
{
	TensorValuePtr src_emb = _aten->get_tensor(source, relation_key);
	TensorValuePtr tgt_emb = _aten->get_tensor(target, relation_key);

	if (!src_emb || !tgt_emb) {
		return createTensorValue(torch::zeros({_embedding_dim * 3}));
	}

	// Create pair embedding: [src; tgt; src*tgt]
	at::Tensor combined = torch::cat({
		src_emb->tensor(),
		tgt_emb->tensor(),
		src_emb->tensor() * tgt_emb->tensor()
	}, 0);

	return createTensorValue(combined);
}

// ============================================================
// TensorAttentionBank Implementation
// ============================================================

TensorAttentionBank::TensorAttentionBank(AtomSpace* as, ATenSpace* aten,
                                         size_t focus_size, int64_t attention_dim)
	: _as(as), _aten(aten), _focus_size(focus_size),
	  _attention_dim(attention_dim),
	  _sti_decay_rate(0.1), _spreading_rate(0.3),
	  _hebbian_learning_rate(0.05)
{
}

TensorAttentionBank::~TensorAttentionBank() {}

void TensorAttentionBank::stimulate(const Handle& atom, double stimulus,
                                    const TensorValuePtr& context_embedding)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto& av = _attention_values[atom];
	av.sti += stimulus;

	// Update attention embedding if context provided
	if (context_embedding && av.attention_embedding) {
		// Blend with context
		at::Tensor blended = 0.7 * av.attention_embedding->tensor() +
		                     0.3 * context_embedding->tensor();
		av.attention_embedding = createTensorValue(blended);
	} else if (context_embedding) {
		av.attention_embedding = context_embedding;
	}
}

TensorAttentionBank::AttentionValue TensorAttentionBank::get_attention(
	const Handle& atom) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	auto it = _attention_values.find(atom);
	if (it == _attention_values.end()) {
		return {0.0, 0.0, 0.0, nullptr};
	}
	return it->second;
}

void TensorAttentionBank::set_attention(const Handle& atom, double sti, double lti)
{
	std::lock_guard<std::mutex> lock(_mtx);
	auto& av = _attention_values[atom];
	av.sti = sti;
	av.lti = lti;
}

HandleSeq TensorAttentionBank::get_attentional_focus() const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _attentional_focus;
}

void TensorAttentionBank::spread_attention(const Handle& source)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _attention_values.find(source);
	if (it == _attention_values.end()) return;

	double source_sti = it->second.sti;
	double to_spread = source_sti * _spreading_rate;

	// Spread through graph structure
	IncomingSet incoming = source->getIncomingSet();
	std::vector<Handle> targets;

	for (const Handle& link : incoming) {
		for (const Handle& h : link->getOutgoingSet()) {
			if (h != source) {
				targets.push_back(h);
			}
		}
	}

	if (source->is_link()) {
		for (const Handle& h : source->getOutgoingSet()) {
			targets.push_back(h);
		}
	}

	// Also spread through Hebbian links
	auto heb_it = _hebbian_links.find(source);
	if (heb_it != _hebbian_links.end()) {
		for (const auto& [target, strength] : heb_it->second) {
			// Weighted spreading based on Hebbian strength
			_attention_values[target].sti += to_spread * strength;
		}
	}

	// Spread equally to graph neighbors
	if (!targets.empty()) {
		double per_target = to_spread / targets.size();
		for (const Handle& target : targets) {
			_attention_values[target].sti += per_target;
		}
	}

	// Reduce source attention
	it->second.sti -= to_spread;
}

void TensorAttentionBank::hebbian_update(const Handle& a, const Handle& b)
{
	std::lock_guard<std::mutex> lock(_mtx);

	// Update Hebbian link a -> b
	auto& links_a = _hebbian_links[a];
	bool found = false;
	for (auto& [target, strength] : links_a) {
		if (target == b) {
			// Strengthen existing link
			strength = strength + _hebbian_learning_rate * (1.0 - strength);
			found = true;
			break;
		}
	}
	if (!found) {
		links_a.push_back({b, _hebbian_learning_rate});
	}

	// Symmetric update b -> a
	auto& links_b = _hebbian_links[b];
	found = false;
	for (auto& [target, strength] : links_b) {
		if (target == a) {
			strength = strength + _hebbian_learning_rate * (1.0 - strength);
			found = true;
			break;
		}
	}
	if (!found) {
		links_b.push_back({a, _hebbian_learning_rate});
	}
}

void TensorAttentionBank::decay_sti()
{
	for (auto& [atom, av] : _attention_values) {
		av.sti *= (1.0 - _sti_decay_rate);

		// Transfer some STI to LTI if consistently high
		if (av.sti > 0.5) {
			double transfer = av.sti * 0.01;
			av.lti += transfer;
		}
	}
}

void TensorAttentionBank::update_focus()
{
	// Priority queue to find top-k by STI
	using AtomSTI = std::pair<double, Handle>;
	std::priority_queue<AtomSTI, std::vector<AtomSTI>, std::greater<AtomSTI>> pq;

	for (const auto& [atom, av] : _attention_values) {
		pq.push({av.sti, atom});
		if (pq.size() > _focus_size) {
			pq.pop();
		}
	}

	_attentional_focus.clear();
	while (!pq.empty()) {
		_attentional_focus.push_back(pq.top().second);
		pq.pop();
	}

	// Reverse to have highest first
	std::reverse(_attentional_focus.begin(), _attentional_focus.end());
}

void TensorAttentionBank::attention_cycle()
{
	std::lock_guard<std::mutex> lock(_mtx);

	// 1. Decay STI
	decay_sti();

	// 2. Spread attention from focus atoms
	HandleSeq focus_copy = _attentional_focus;
	for (const Handle& atom : focus_copy) {
		// Unlock temporarily for spread_attention
	}

	// 3. Update focus
	update_focus();
}

std::vector<std::pair<Handle, double>> TensorAttentionBank::attention_weighted_query(
	const TensorValuePtr& query_embedding,
	const Handle& embedding_key,
	size_t k) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<std::pair<Handle, double>> results;

	for (const auto& [atom, av] : _attention_values) {
		if (av.sti <= 0) continue;

		TensorValuePtr emb = _aten->get_tensor(atom, embedding_key);
		if (!emb) continue;

		double sim = _aten->cosine_similarity(query_embedding, emb);
		double score = sim * (1.0 + av.sti);  // Attention-boosted similarity
		results.push_back({atom, score});
	}

	std::sort(results.begin(), results.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	if (results.size() > k) {
		results.resize(k);
	}

	return results;
}

HandleSeq TensorAttentionBank::query_with_attention_filter(
	const TensorValuePtr& query_embedding,
	const Handle& embedding_key,
	double min_sti,
	size_t max_results) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<std::pair<Handle, double>> candidates;

	for (const auto& [atom, av] : _attention_values) {
		if (av.sti < min_sti) continue;

		TensorValuePtr emb = _aten->get_tensor(atom, embedding_key);
		if (!emb) continue;

		double sim = _aten->cosine_similarity(query_embedding, emb);
		candidates.push_back({atom, sim});
	}

	std::sort(candidates.begin(), candidates.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	HandleSeq results;
	for (size_t i = 0; i < std::min(candidates.size(), max_results); i++) {
		results.push_back(candidates[i].first);
	}

	return results;
}

// ============================================================
// TensorTruthValue Implementation
// ============================================================

TensorTruthValue::TensorTruthValue(double strength, double confidence)
	: _strength(strength), _confidence(confidence)
{
	// Create simple distribution representation
	at::Tensor dist = torch::zeros({2});
	dist[0] = strength;
	dist[1] = confidence;
	_distribution = createTensorValue(dist);
}

TensorTruthValue::TensorTruthValue(const TensorValuePtr& distribution)
	: _distribution(distribution)
{
	// Extract strength and confidence from distribution
	std::vector<double> data = distribution->to_vector();
	if (data.size() >= 2) {
		_strength = data[0];
		_confidence = data[1];
	} else if (data.size() == 1) {
		_strength = data[0];
		_confidence = 1.0;
	} else {
		_strength = 0.0;
		_confidence = 0.0;
	}
}

void TensorTruthValue::set_distribution(const TensorValuePtr& dist)
{
	_distribution = dist;
	std::vector<double> data = dist->to_vector();
	if (data.size() >= 2) {
		_strength = data[0];
		_confidence = data[1];
	}
}

TensorTruthValue TensorTruthValue::merge(const TensorTruthValue& a,
                                         const TensorTruthValue& b)
{
	// Confidence-weighted average
	double total_conf = a._confidence + b._confidence;
	if (total_conf < 1e-8) {
		return TensorTruthValue(0.5, 0.0);
	}

	double merged_strength = (a._strength * a._confidence +
	                          b._strength * b._confidence) / total_conf;
	double merged_confidence = std::min(1.0, total_conf);

	return TensorTruthValue(merged_strength, merged_confidence);
}

TensorTruthValue TensorTruthValue::revision(const TensorTruthValue& a,
                                            const TensorTruthValue& b)
{
	// PLN-style revision
	double k = 1.0;  // Revision factor
	double n_a = a._confidence * k / (1.0 - a._confidence + 1e-8);
	double n_b = b._confidence * k / (1.0 - b._confidence + 1e-8);

	double n_new = n_a + n_b;
	double new_confidence = n_new / (n_new + k);
	double new_strength = (a._strength * n_a + b._strength * n_b) / (n_new + 1e-8);

	return TensorTruthValue(new_strength, new_confidence);
}

TensorTruthValue TensorTruthValue::deduction(const TensorTruthValue& ab,
                                             const TensorTruthValue& bc)
{
	// PLN deduction: (A->B) and (B->C) => (A->C)
	double s_ab = ab._strength;
	double c_ab = ab._confidence;
	double s_bc = bc._strength;
	double c_bc = bc._confidence;

	double s_ac = s_ab * s_bc;
	double c_ac = c_ab * c_bc * s_ab;  // Simplified

	return TensorTruthValue(s_ac, std::min(c_ac, 0.99));
}

TensorTruthValue TensorTruthValue::induction(const TensorTruthValue& ab,
                                             const TensorTruthValue& cb)
{
	// PLN induction: (A->B) and (C->B) => (A->C) with low confidence
	double s_ab = ab._strength;
	double s_cb = cb._strength;

	double s_ac = s_ab * s_cb + (1 - s_ab) * (1 - s_cb);
	double c_ac = 0.1 * std::min(ab._confidence, cb._confidence);

	return TensorTruthValue(s_ac, c_ac);
}

TruthValuePtr TensorTruthValue::to_truth_value() const
{
	return SimpleTruthValue::createTV(_strength, _confidence);
}

TensorTruthValue TensorTruthValue::from_truth_value(const TruthValuePtr& tv)
{
	return TensorTruthValue(tv->get_mean(), tv->get_confidence());
}

// ============================================================
// TensorLogic Implementation
// ============================================================

TensorLogic::TensorLogic(AtomSpace* as, int64_t embedding_dim)
	: _as(as), _embedding_dim(embedding_dim)
{
	_aten = new ATenSpace(as, true);
	_attention_bank = std::make_shared<TensorAttentionBank>(as, _aten, 100, 64);
	_network_tensor = std::make_unique<NetworkAwareTensor>(as, _aten, embedding_dim, 4);

	// Create default keys
	_embedding_key = _as->add_node(PREDICATE_NODE, "*-embedding-*");
	_multiscale_key = _as->add_node(PREDICATE_NODE, "*-multiscale-*");
	_attention_key = _as->add_node(PREDICATE_NODE, "*-attention-*");
	_truth_key = _as->add_node(PREDICATE_NODE, "*-tensor-truth-*");
}

TensorLogic::~TensorLogic()
{
	delete _aten;
}

// ---- Multi-Entity Operations ----

TensorValuePtr TensorLogic::create_entity_batch(const HandleSeq& entities,
                                                const Handle& key) const
{
	return _aten->batch_get(entities, key);
}

TensorValuePtr TensorLogic::aggregate_entities(const HandleSeq& entities,
                                               const Handle& key,
                                               const std::string& aggregation) const
{
	if (entities.empty()) {
		return createTensorValue(torch::zeros({_embedding_dim}));
	}

	TensorValuePtr batch = create_entity_batch(entities, key);
	at::Tensor t = batch->tensor();  // [N, D]

	at::Tensor result;
	if (aggregation == "mean") {
		result = torch::mean(t, 0);
	} else if (aggregation == "max") {
		result = std::get<0>(torch::max(t, 0));
	} else if (aggregation == "sum") {
		result = torch::sum(t, 0);
	} else if (aggregation == "attention") {
		// Self-attention aggregation
		at::Tensor scores = torch::matmul(t, t.t()) / std::sqrt((double)_embedding_dim);
		at::Tensor weights = torch::softmax(scores.mean(0), 0);  // [N]
		result = torch::matmul(weights.unsqueeze(0), t).squeeze(0);
	} else if (aggregation == "deepsets") {
		// DeepSets: permutation-invariant aggregation
		at::Tensor sum_emb = torch::sum(t, 0);
		at::Tensor max_emb = std::get<0>(torch::max(t, 0));
		result = torch::cat({sum_emb, max_emb}, 0);
		// Project back to embedding dim
		at::Tensor proj = torch::randn({_embedding_dim, _embedding_dim * 2});
		proj = proj / std::sqrt((double)(_embedding_dim * 2));
		result = torch::matmul(proj, result);
	} else {
		result = torch::mean(t, 0);
	}

	return createTensorValue(result);
}

TensorValuePtr TensorLogic::entity_similarity_matrix(const HandleSeq& entities,
                                                     const Handle& key) const
{
	return _aten->pairwise_similarity(entities, key);
}

std::unordered_map<int, HandleSeq> TensorLogic::cluster_entities(
	const HandleSeq& entities, const Handle& key, int k) const
{
	std::unordered_map<int, HandleSeq> clusters;

	if (entities.size() <= (size_t)k) {
		for (size_t i = 0; i < entities.size(); i++) {
			clusters[i].push_back(entities[i]);
		}
		return clusters;
	}

	// Get embeddings
	TensorValuePtr batch = create_entity_batch(entities, key);
	at::Tensor embs = batch->tensor();  // [N, D]

	// Simple k-means
	int64_t N = embs.size(0);

	// Initialize centroids randomly
	std::vector<int64_t> centroid_indices;
	for (int i = 0; i < k; i++) {
		centroid_indices.push_back(i * N / k);
	}
	at::Tensor centroids = torch::index_select(embs, 0,
		torch::tensor(centroid_indices, torch::kLong));

	// Iterate
	std::vector<int> assignments(N);
	for (int iter = 0; iter < 10; iter++) {
		// Assign to nearest centroid
		at::Tensor dists = torch::cdist(embs, centroids);  // [N, k]
		at::Tensor min_dists = std::get<1>(torch::min(dists, 1));  // [N]
		auto* ptr = min_dists.data_ptr<int64_t>();
		for (int64_t i = 0; i < N; i++) {
			assignments[i] = ptr[i];
		}

		// Update centroids
		for (int c = 0; c < k; c++) {
			std::vector<int64_t> members;
			for (int64_t i = 0; i < N; i++) {
				if (assignments[i] == c) members.push_back(i);
			}
			if (!members.empty()) {
				at::Tensor member_embs = torch::index_select(embs, 0,
					torch::tensor(members, torch::kLong));
				centroids[c] = torch::mean(member_embs, 0);
			}
		}
	}

	// Build result
	for (size_t i = 0; i < entities.size(); i++) {
		clusters[assignments[i]].push_back(entities[i]);
	}

	return clusters;
}

// ---- Multi-Scale Operations ----

MultiScaleTensorPtr TensorLogic::compute_multiscale_embedding(
	const Handle& atom, const Handle& feature_key) const
{
	auto mst = createMultiScaleTensor(_embedding_dim);

	// LOCAL: Direct features
	TensorValuePtr local = _aten->get_tensor(atom, feature_key);
	if (local) {
		mst->set_scale(ScaleLevel::LOCAL, local);
	} else {
		mst->set_scale(ScaleLevel::LOCAL,
			createTensorValue(torch::randn({_embedding_dim})));
	}

	// NEIGHBOR: 1-hop aggregation
	HandleSeq neighbors;
	IncomingSet incoming = atom->getIncomingSet();
	for (const Handle& link : incoming) {
		for (const Handle& h : link->getOutgoingSet()) {
			if (h != atom) neighbors.push_back(h);
		}
	}
	if (!neighbors.empty()) {
		TensorValuePtr neigh_agg = aggregate_entities(neighbors, feature_key, "attention");
		mst->set_scale(ScaleLevel::NEIGHBOR, neigh_agg);
	}

	// COMMUNITY: 2-hop aggregation (simplified)
	HandleSeq community;
	for (const Handle& n : neighbors) {
		IncomingSet n_incoming = n->getIncomingSet();
		for (const Handle& link : n_incoming) {
			for (const Handle& h : link->getOutgoingSet()) {
				if (h != atom && h != n) {
					community.push_back(h);
				}
			}
		}
	}
	if (!community.empty()) {
		// Deduplicate
		std::unordered_set<Handle> unique_comm(community.begin(), community.end());
		HandleSeq unique_vec(unique_comm.begin(), unique_comm.end());
		TensorValuePtr comm_agg = aggregate_entities(unique_vec, feature_key, "mean");
		mst->set_scale(ScaleLevel::COMMUNITY, comm_agg);
	}

	// GLOBAL: All atoms average (expensive, use sampling in production)
	// For now, use a fixed global context
	mst->set_scale(ScaleLevel::GLOBAL,
		createTensorValue(torch::zeros({_embedding_dim})));

	return mst;
}

void TensorLogic::set_multiscale_embedding(const Handle& atom,
                                           const MultiScaleTensorPtr& mst)
{
	// Store concatenated embedding
	TensorValuePtr concat = mst->get_concatenated();
	if (concat) {
		_aten->set_tensor(atom, _multiscale_key, concat);
	}
}

MultiScaleTensorPtr TensorLogic::get_multiscale_embedding(const Handle& atom) const
{
	// Reconstruct from stored concatenated embedding
	TensorValuePtr stored = _aten->get_tensor(atom, _multiscale_key);
	if (!stored) return nullptr;

	auto mst = createMultiScaleTensor(_embedding_dim);

	// Assume 4 scales concatenated
	at::Tensor t = stored->tensor();
	int64_t total_dim = t.size(0);
	int64_t scale_dim = total_dim / 4;

	mst->set_scale(ScaleLevel::LOCAL, createTensorValue(t.slice(0, 0, scale_dim)));
	mst->set_scale(ScaleLevel::NEIGHBOR, createTensorValue(t.slice(0, scale_dim, 2*scale_dim)));
	mst->set_scale(ScaleLevel::COMMUNITY, createTensorValue(t.slice(0, 2*scale_dim, 3*scale_dim)));
	mst->set_scale(ScaleLevel::GLOBAL, createTensorValue(t.slice(0, 3*scale_dim, 4*scale_dim)));

	return mst;
}

std::vector<std::pair<Handle, double>> TensorLogic::query_at_scale(
	const TensorValuePtr& query, ScaleLevel scale, size_t k) const
{
	std::vector<std::pair<Handle, double>> results;

	// Get all atoms with multiscale embeddings
	HandleSeq all = _as->get_all_atoms();

	for (const Handle& atom : all) {
		MultiScaleTensorPtr mst = get_multiscale_embedding(atom);
		if (!mst) continue;

		TensorValuePtr scale_emb = mst->get_scale(scale);
		if (!scale_emb) continue;

		double sim = _aten->cosine_similarity(query, scale_emb);
		results.push_back({atom, sim});
	}

	std::sort(results.begin(), results.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	if (results.size() > k) {
		results.resize(k);
	}

	return results;
}

double TensorLogic::cross_scale_similarity(const Handle& a, ScaleLevel scale_a,
                                           const Handle& b, ScaleLevel scale_b) const
{
	MultiScaleTensorPtr mst_a = get_multiscale_embedding(a);
	MultiScaleTensorPtr mst_b = get_multiscale_embedding(b);

	if (!mst_a || !mst_b) return 0.0;

	TensorValuePtr emb_a = mst_a->get_scale(scale_a);
	TensorValuePtr emb_b = mst_b->get_scale(scale_b);

	if (!emb_a || !emb_b) return 0.0;

	return _aten->cosine_similarity(emb_a, emb_b);
}

// ---- Network-Aware Operations ----

TensorValuePtr TensorLogic::compute_network_embedding(const Handle& atom,
                                                      const Handle& feature_key,
                                                      int num_layers) const
{
	return _network_tensor->compute_embedding(atom, feature_key);
}

void TensorLogic::propagate_embeddings(const HandleSeq& atoms,
                                       const Handle& key,
                                       int num_iterations)
{
	auto embeddings = _network_tensor->compute_embeddings(atoms, key, num_iterations);

	for (const auto& [atom, emb] : embeddings) {
		_aten->set_tensor(atom, key, emb);
	}
}

TensorValuePtr TensorLogic::compute_relation_embedding(const Handle& link) const
{
	if (!link->is_link()) {
		throw std::runtime_error("Expected a link");
	}

	HandleSeq outgoing = link->getOutgoingSet();
	if (outgoing.size() < 2) {
		return createTensorValue(torch::zeros({_embedding_dim}));
	}

	return _network_tensor->compute_pair_embedding(
		outgoing[0], outgoing[1], _embedding_key);
}

// ---- Tensor Logic Inference ----

std::pair<Handle, TensorTruthValue> TensorLogic::tensor_deduction(
	const Handle& premise1, const Handle& premise2) const
{
	// Extract truth values
	TruthValuePtr tv1 = premise1->getTruthValue();
	TruthValuePtr tv2 = premise2->getTruthValue();

	TensorTruthValue ttv1 = TensorTruthValue::from_truth_value(tv1);
	TensorTruthValue ttv2 = TensorTruthValue::from_truth_value(tv2);

	// Compute deduction
	TensorTruthValue result_tv = TensorTruthValue::deduction(ttv1, ttv2);

	// The conclusion atom would be constructed based on the link types
	// For now, return the first premise with updated TV
	return {premise1, result_tv};
}

std::pair<Handle, TensorTruthValue> TensorLogic::tensor_induction(
	const HandleSeq& instances) const
{
	if (instances.empty()) {
		return {Handle::UNDEFINED, TensorTruthValue(0.0, 0.0)};
	}

	// Aggregate instance truth values
	double total_strength = 0.0;
	double total_confidence = 0.0;

	for (const Handle& h : instances) {
		TruthValuePtr tv = h->getTruthValue();
		total_strength += tv->get_mean() * tv->get_confidence();
		total_confidence += tv->get_confidence();
	}

	double avg_strength = total_confidence > 0 ? total_strength / total_confidence : 0.5;
	double induced_confidence = std::min(0.9, total_confidence / instances.size());

	return {instances[0], TensorTruthValue(avg_strength, induced_confidence)};
}

Handle TensorLogic::tensor_analogy(const Handle& A, const Handle& B,
                                   const Handle& C,
                                   const HandleSeq& candidates) const
{
	// Compute relationship embedding A:B
	TensorValuePtr emb_a = _aten->get_tensor(A, _embedding_key);
	TensorValuePtr emb_b = _aten->get_tensor(B, _embedding_key);
	TensorValuePtr emb_c = _aten->get_tensor(C, _embedding_key);

	if (!emb_a || !emb_b || !emb_c) {
		return Handle::UNDEFINED;
	}

	// Analogy: D = C + (B - A)
	at::Tensor target = emb_c->tensor() + emb_b->tensor() - emb_a->tensor();
	TensorValuePtr target_tv = createTensorValue(target);

	// Find closest candidate
	Handle best = Handle::UNDEFINED;
	double best_sim = -1.0;

	for (const Handle& d : candidates) {
		TensorValuePtr emb_d = _aten->get_tensor(d, _embedding_key);
		if (!emb_d) continue;

		double sim = _aten->cosine_similarity(target_tv, emb_d);
		if (sim > best_sim) {
			best_sim = sim;
			best = d;
		}
	}

	return best;
}

std::vector<std::unordered_map<Handle, Handle>> TensorLogic::tensor_unify(
	const Handle& pattern, const HandleSeq& candidates,
	double similarity_threshold) const
{
	std::vector<std::unordered_map<Handle, Handle>> results;

	// Simple unification based on embedding similarity
	TensorValuePtr pattern_emb = _aten->get_tensor(pattern, _embedding_key);
	if (!pattern_emb) {
		pattern_emb = compute_network_embedding(pattern, _embedding_key);
	}

	for (const Handle& candidate : candidates) {
		TensorValuePtr cand_emb = _aten->get_tensor(candidate, _embedding_key);
		if (!cand_emb) continue;

		double sim = _aten->cosine_similarity(pattern_emb, cand_emb);
		if (sim >= similarity_threshold) {
			std::unordered_map<Handle, Handle> binding;
			binding[pattern] = candidate;
			results.push_back(binding);
		}
	}

	return results;
}

// ---- Attention-Guided Reasoning ----

void TensorLogic::stimulate_by_query(const TensorValuePtr& query,
                                     const Handle& key,
                                     double base_stimulus)
{
	// Find similar atoms and stimulate them
	HandleSeq all = _as->get_all_atoms();

	for (const Handle& atom : all) {
		TensorValuePtr emb = _aten->get_tensor(atom, key);
		if (!emb) continue;

		double sim = _aten->cosine_similarity(query, emb);
		if (sim > 0.3) {  // Threshold
			double stimulus = base_stimulus * sim;
			_attention_bank->stimulate(atom, stimulus, query);
		}
	}
}

HandleSeq TensorLogic::attention_forward_step(const Handle& rule_base,
                                              size_t max_applications)
{
	HandleSeq results;

	// Get focus atoms
	HandleSeq focus = _attention_bank->get_attentional_focus();

	// Apply rules to focus atoms (simplified)
	for (const Handle& atom : focus) {
		if (results.size() >= max_applications) break;

		// Stimulate connected atoms
		_attention_bank->spread_attention(atom);
		results.push_back(atom);
	}

	return results;
}

bool TensorLogic::attention_backward_chain(const Handle& goal, int max_depth)
{
	// Simplified backward chaining
	_attention_bank->stimulate(goal, 2.0);

	for (int depth = 0; depth < max_depth; depth++) {
		_attention_bank->attention_cycle();

		HandleSeq focus = _attention_bank->get_attentional_focus();
		for (const Handle& h : focus) {
			if (h == goal) {
				return true;  // Found path to goal
			}
		}
	}

	return false;
}

// ---- Utility ----

void TensorLogic::initialize_embeddings(const std::vector<Type>& types,
                                        const std::string& init)
{
	for (Type t : types) {
		HandleSeq atoms;
		_as->get_handles_by_type(atoms, t);

		for (const Handle& atom : atoms) {
			_aten->set_tensor(atom, _embedding_key,
			                  {_embedding_dim}, at::kFloat, init);
		}
	}

	// Initialize type embeddings for network-aware computations
	_network_tensor->init_type_embeddings(types);
}

std::string TensorLogic::stats() const
{
	std::ostringstream ss;
	ss << "TensorLogic Statistics:\n";
	ss << "  Embedding dimension: " << _embedding_dim << "\n";
	ss << "  ATen cache size: " << _aten->cache_size() << "\n";
	ss << "  Attention focus size: " << _attention_bank->get_attentional_focus().size() << "\n";
	ss << "  CUDA available: " << (ATenSpace::cuda_available() ? "yes" : "no") << "\n";
	return ss.str();
}

#endif // HAVE_ATEN
