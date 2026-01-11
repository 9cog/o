/*
 * opencog/atoms/aten/ATenSpace.cc
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * ATenSpace bridges symbolic AI (AtomSpace) with neural tensor embeddings.
 * Based on the integration patterns from o9nn/ATen and o9nn/ATenSpace.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 */

#include <sstream>
#include <algorithm>
#include <cmath>

#include <opencog/util/exceptions.h>
#include <opencog/atoms/aten/ATenSpace.h>
#include <opencog/atomspace/AtomSpace.h>

using namespace opencog;

// ============================================================
// Constructor / Destructor

ATenSpace::ATenSpace(AtomSpace* as, int64_t embedding_dim,
                     const std::string& device)
	: _device(device)
	, _requires_grad(false)
	, _atomspace(as)
	, _embedding_dim(embedding_dim)
	, _num_scales(3)
	, _tensor_count(0)
	, _total_elements(0)
{
	if (_atomspace) {
		_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TensorKey-*");
		_embedding_key = _atomspace->add_node(PREDICATE_NODE, "*-EmbeddingKey-*");
		_attention_key = _atomspace->add_node(PREDICATE_NODE, "*-AttentionKey-*");

		_tensor_logic = std::make_shared<TensorLogic>(
			_atomspace, _embedding_dim, _num_scales, _device);
	}
}

ATenSpace::~ATenSpace()
{
	clear();
}

// ============================================================
// Configuration

void ATenSpace::set_atomspace(AtomSpace* as)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_atomspace = as;
	if (_atomspace) {
		_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TensorKey-*");
		_embedding_key = _atomspace->add_node(PREDICATE_NODE, "*-EmbeddingKey-*");
		_attention_key = _atomspace->add_node(PREDICATE_NODE, "*-AttentionKey-*");

		_tensor_logic = std::make_shared<TensorLogic>(
			_atomspace, _embedding_dim, _num_scales, _device);
	} else {
		_tensor_key = Handle::UNDEFINED;
		_embedding_key = Handle::UNDEFINED;
		_attention_key = Handle::UNDEFINED;
		_tensor_logic = nullptr;
	}
}

void ATenSpace::set_device(const std::string& device)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_device = device;
	if (_tensor_logic)
		_tensor_logic->set_device(device);
}

// ============================================================
// Named tensor management

void ATenSpace::register_tensor(const std::string& name,
                                const ATenValuePtr& tensor)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end()) {
		_total_elements -= it->second->numel();
		_tensor_count--;
	}

	_named_tensors[name] = tensor;
	_tensor_count++;
	_total_elements += tensor->numel();
}

ATenValuePtr ATenSpace::get_tensor(const std::string& name) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end())
		return it->second;
	return nullptr;
}

bool ATenSpace::has_tensor(const std::string& name) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _named_tensors.find(name) != _named_tensors.end();
}

void ATenSpace::remove_tensor(const std::string& name)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end()) {
		_total_elements -= it->second->numel();
		_tensor_count--;
		_named_tensors.erase(it);
	}
}

std::vector<std::string> ATenSpace::get_tensor_names() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<std::string> names;
	names.reserve(_named_tensors.size());
	for (const auto& pair : _named_tensors)
		names.push_back(pair.first);
	return names;
}

void ATenSpace::clear()
{
	std::lock_guard<std::mutex> lock(_mtx);

	_named_tensors.clear();
	_tensor_count = 0;
	_total_elements = 0;
}

// ============================================================
// Tensor creation utilities

ATenValuePtr ATenSpace::zeros(const std::vector<int64_t>& shape,
                              const std::string& name)
{
	ATenValuePtr tensor = createATenZeros(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::ones(const std::vector<int64_t>& shape,
                             const std::string& name)
{
	ATenValuePtr tensor = createATenOnes(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::random(const std::vector<int64_t>& shape,
                               const std::string& name)
{
	ATenValuePtr tensor = createATenRandom(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::from_vector(const std::vector<double>& data,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name)
{
	ATenValuePtr tensor = createATenFromVector(data, shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

// ============================================================
// Atom-Tensor integration (o9nn/ATenSpace style)

Handle ATenSpace::create_concept_node(const std::string& name,
                                      const ATenValuePtr& embedding)
{
	if (!_atomspace)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	Handle node = _atomspace->add_node(CONCEPT_NODE, std::string(name));

	ATenValuePtr emb = embedding;
	if (!emb) {
		// Create random embedding
		emb = createATenRandom({_embedding_dim});
	}

	attach_embedding(node, emb);
	return node;
}

void ATenSpace::attach_embedding(const Handle& atom, const ATenValuePtr& tensor)
{
	if (!_atomspace || !_embedding_key)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	_atomspace->set_value(atom, _embedding_key, tensor);

	if (_tensor_logic)
		_tensor_logic->set_embedding(atom, tensor);
}

ATenValuePtr ATenSpace::get_embedding(const Handle& atom) const
{
	if (!_embedding_key)
		return nullptr;

	ValuePtr val = atom->getValue(_embedding_key);
	return ATenValueCast(val);
}

bool ATenSpace::has_embedding(const Handle& atom) const
{
	return get_embedding(atom) != nullptr;
}

void ATenSpace::attach_tensor(const Handle& atom, const ATenValuePtr& tensor)
{
	if (!_atomspace || !_tensor_key)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	_atomspace->set_value(atom, _tensor_key, tensor);
}

ATenValuePtr ATenSpace::get_attached_tensor(const Handle& atom) const
{
	if (!_tensor_key)
		return nullptr;

	ValuePtr val = atom->getValue(_tensor_key);
	return ATenValueCast(val);
}

bool ATenSpace::has_attached_tensor(const Handle& atom) const
{
	return get_attached_tensor(atom) != nullptr;
}

// ============================================================
// Semantic similarity operations

HandleSeq ATenSpace::query_similar(const ATenValuePtr& query_tensor, size_t k)
{
	if (_tensor_logic)
		return _tensor_logic->query_similar_tensor(query_tensor, k);

	return HandleSeq();
}

HandleSeq ATenSpace::query_similar(const Handle& query_atom, size_t k)
{
	if (_tensor_logic)
		return _tensor_logic->query_similar(query_atom, k);

	return HandleSeq();
}

double ATenSpace::similarity(const Handle& a, const Handle& b)
{
	if (_tensor_logic)
		return _tensor_logic->similarity(a, b);

	// Fallback to manual computation
	ATenValuePtr emb_a = get_embedding(a);
	ATenValuePtr emb_b = get_embedding(b);

	if (!emb_a || !emb_b)
		return 0.0;

	auto vec_a = emb_a->to_vector();
	auto vec_b = emb_b->to_vector();

	if (vec_a.size() != vec_b.size())
		return 0.0;

	double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
	for (size_t i = 0; i < vec_a.size(); i++) {
		dot += vec_a[i] * vec_b[i];
		norm_a += vec_a[i] * vec_a[i];
		norm_b += vec_b[i] * vec_b[i];
	}

	if (norm_a == 0.0 || norm_b == 0.0)
		return 0.0;

	return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

double ATenSpace::distance(const Handle& a, const Handle& b)
{
	ATenValuePtr emb_a = get_embedding(a);
	ATenValuePtr emb_b = get_embedding(b);

	if (!emb_a || !emb_b)
		return std::numeric_limits<double>::max();

	auto vec_a = emb_a->to_vector();
	auto vec_b = emb_b->to_vector();

	if (vec_a.size() != vec_b.size())
		return std::numeric_limits<double>::max();

	double sum_sq = 0.0;
	for (size_t i = 0; i < vec_a.size(); i++) {
		double diff = vec_a[i] - vec_b[i];
		sum_sq += diff * diff;
	}

	return std::sqrt(sum_sq);
}

// ============================================================
// Link operations

Handle ATenSpace::create_link(Type link_type, const HandleSeq& outgoing,
                              const std::string& aggregation)
{
	if (!_atomspace)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	Handle link = _atomspace->add_link(link_type, std::move(HandleSeq(outgoing)));

	ATenValuePtr emb = compute_link_embedding(link, aggregation);
	attach_embedding(link, emb);

	return link;
}

ATenValuePtr ATenSpace::compute_link_embedding(const Handle& link,
                                                const std::string& aggregation)
{
	if (_tensor_logic)
		return _tensor_logic->compute_link_embedding(link, aggregation);

	// Fallback implementation
	if (!link->is_link())
		return get_embedding(link);

	const HandleSeq& outgoing = link->getOutgoingSet();
	if (outgoing.empty())
		return createATenZeros({_embedding_dim});

	std::vector<double> result(_embedding_dim, 0.0);

	for (const Handle& h : outgoing) {
		ATenValuePtr emb = get_embedding(h);
		if (emb) {
			auto vec = emb->to_vector();
			for (size_t i = 0; i < vec.size() && i < (size_t)_embedding_dim; i++) {
				result[i] += vec[i];
			}
		}
	}

	if (aggregation == "mean") {
		for (auto& v : result) v /= outgoing.size();
	}

	return createATenFromVector(result, {_embedding_dim});
}

// ============================================================
// Hebbian learning (ECAN-style)

Handle ATenSpace::create_hebbian_link(const Handle& source, const Handle& target,
                                      double weight)
{
	if (!_atomspace)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	Handle link = _atomspace->add_link(HEBBIAN_LINK, source, target);

	// Store weight as tensor
	ATenValuePtr weight_tensor = createATenFromVector({weight}, {1});
	attach_tensor(link, weight_tensor);

	return link;
}

void ATenSpace::hebbian_update(double learning_rate)
{
	if (_tensor_logic)
		_tensor_logic->hebbian_update(learning_rate);
}

void ATenSpace::spread_activation(const Handle& source, double amount, int hops)
{
	if (!_atomspace)
		return;

	// Get source attention
	stimulate(source, amount);

	if (hops <= 0)
		return;

	// Spread to connected atoms
	IncomingSet incoming = source->getIncomingSet();
	for (const Handle& link : incoming) {
		if (link->is_type(HEBBIAN_LINK) || link->is_type(LINK)) {
			const HandleSeq& outgoing = link->getOutgoingSet();
			for (const Handle& target : outgoing) {
				if (target != source) {
					spread_activation(target, amount * 0.5, hops - 1);
				}
			}
		}
	}
}

// ============================================================
// Attention allocation

void ATenSpace::stimulate(const Handle& atom, double amount)
{
	if (!_atomspace || !_attention_key)
		return;

	ValuePtr val = atom->getValue(_attention_key);
	double current = 0.0;

	if (val) {
		ATenValuePtr atv = ATenValueCast(val);
		if (atv) current = atv->to_vector()[0];
	}

	ATenValuePtr new_attention = createATenFromVector({current + amount}, {1});
	_atomspace->set_value(atom, _attention_key, new_attention);
}

void ATenSpace::decay_attention(double factor)
{
	if (!_atomspace)
		return;

	HandleSeq all_atoms;
	_atomspace->get_handles_by_type(all_atoms, ATOM, true);

	for (const Handle& atom : all_atoms) {
		double attention = get_attention(atom);
		if (attention > 0) {
			ATenValuePtr decayed = createATenFromVector({attention * factor}, {1});
			_atomspace->set_value(atom, _attention_key, decayed);
		}
	}
}

double ATenSpace::get_attention(const Handle& atom) const
{
	if (!_attention_key)
		return 0.0;

	ValuePtr val = atom->getValue(_attention_key);
	if (!val)
		return 0.0;

	ATenValuePtr atv = ATenValueCast(val);
	if (!atv)
		return 0.0;

	return atv->to_vector()[0];
}

// ============================================================
// Pattern matching with embeddings

HandleSeq ATenSpace::pattern_match_embedding(const Handle& pattern,
                                              double threshold,
                                              size_t max_results)
{
	HandleSeq results;

	ATenValuePtr pattern_emb = get_embedding(pattern);
	if (!pattern_emb) {
		// Try to compute from structure
		pattern_emb = compute_link_embedding(pattern);
	}

	if (!pattern_emb)
		return results;

	// Get similar atoms
	HandleSeq similar = query_similar(pattern, max_results * 2);

	for (const Handle& h : similar) {
		double sim = similarity(pattern, h);
		if (sim >= threshold) {
			results.push_back(h);
			if (results.size() >= max_results)
				break;
		}
	}

	return results;
}

// ============================================================
// PLN integration

Handle ATenSpace::tensor_deduction(const Handle& premise1, const Handle& premise2)
{
	if (!_atomspace)
		throw RuntimeException(TRACE_INFO, "ATenSpace: no AtomSpace configured");

	// Compute deduced embedding
	ATenValuePtr emb1 = get_embedding(premise1);
	ATenValuePtr emb2 = get_embedding(premise2);

	if (!emb1 || !emb2)
		return Handle::UNDEFINED;

	// Combine embeddings (element-wise product as simple deduction)
	ValuePtr combined = emb1->mul(*emb2);
	ATenValuePtr result_emb = ATenValueCast(combined);

	// Create result link
	Handle result = _atomspace->add_link(IMPLICATION_LINK, premise1, premise2);
	attach_embedding(result, result_emb);

	return result;
}

void ATenSpace::tensor_revision(const Handle& atom, double strength, double confidence)
{
	// Get current embedding
	ATenValuePtr emb = get_embedding(atom);
	if (!emb)
		return;

	// Scale embedding by confidence
	ValuePtr revised = emb->mul_scalar(confidence);
	ATenValuePtr revised_emb = ATenValueCast(revised);

	attach_embedding(atom, revised_emb);
}

// ============================================================
// Network operations

void ATenSpace::build_network()
{
	if (_tensor_logic)
		_tensor_logic->build_network_tensor();
}

void ATenSpace::message_passing(int num_iterations)
{
	if (_tensor_logic)
		_tensor_logic->message_passing(num_iterations);
}

// ============================================================
// Persistence

void ATenSpace::save_embeddings(const std::string& path)
{
	if (_tensor_logic)
		_tensor_logic->save_embeddings(path);
}

void ATenSpace::load_embeddings(const std::string& path)
{
	if (_tensor_logic)
		_tensor_logic->load_embeddings(path);
}

// ============================================================
// Statistics

std::string ATenSpace::to_string() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::ostringstream oss;
	oss << "ATenSpace {\n";
	oss << "  device: " << _device << "\n";
	oss << "  embedding_dim: " << _embedding_dim << "\n";
	oss << "  requires_grad: " << (_requires_grad ? "true" : "false") << "\n";
	oss << "  tensor_count: " << _tensor_count << "\n";
	oss << "  total_elements: " << _total_elements << "\n";

	if (_tensor_logic) {
		oss << "  tensor_logic: {\n";
		oss << "    num_embeddings: " << _tensor_logic->num_embeddings() << "\n";
		oss << "    update_count: " << _tensor_logic->update_count() << "\n";
		oss << "  }\n";
	}

	oss << "  registered_tensors: [\n";
	for (const auto& pair : _named_tensors) {
		auto shp = pair.second->shape();
		oss << "    " << pair.first << ": shape=(";
		for (size_t i = 0; i < shp.size(); i++) {
			if (i > 0) oss << ",";
			oss << shp[i];
		}
		oss << "), numel=" << pair.second->numel() << "\n";
	}
	oss << "  ]\n";
	oss << "}";

	return oss.str();
}
