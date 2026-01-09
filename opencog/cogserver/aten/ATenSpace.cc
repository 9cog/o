/*
 * ATenSpace.cc
 *
 * ATenSpace implementation
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

#ifdef HAVE_ATEN

#include <sstream>
#include <algorithm>
#include <cmath>

#include "ATenSpace.h"

using namespace opencog;

// ============================================================
// Constructor / Destructor

ATenSpace::ATenSpace(AtomSpace* as, bool use_cache)
	: _as(as), _use_cache(use_cache), _cache_max_size(10000)
{
}

ATenSpace::~ATenSpace()
{
	clear_cache();
}

// ============================================================
// Cache helpers

std::string ATenSpace::make_cache_key(const Handle& atom, const Handle& key) const
{
	return atom->to_short_string() + "::" + key->to_short_string();
}

void ATenSpace::cache_tensor(const std::string& cache_key,
                             const TensorValuePtr& tv) const
{
	if (!_use_cache) return;

	std::lock_guard<std::mutex> lock(_mtx);
	if (_cache.size() >= _cache_max_size) {
		// Simple eviction: clear half the cache
		auto it = _cache.begin();
		size_t to_remove = _cache_max_size / 2;
		while (to_remove-- > 0 && it != _cache.end()) {
			it = _cache.erase(it);
		}
	}
	_cache[cache_key] = tv;
}

// ============================================================
// Tensor storage operations

TensorValuePtr ATenSpace::set_tensor(const Handle& atom,
                                     const Handle& key,
                                     const std::vector<int64_t>& shape,
                                     at::ScalarType dtype,
                                     const std::string& init)
{
	at::Tensor t;
	auto options = torch::TensorOptions().dtype(dtype);

	if (init == "zeros") {
		t = torch::zeros(shape, options);
	} else if (init == "ones") {
		t = torch::ones(shape, options);
	} else if (init == "randn") {
		t = torch::randn(shape, options);
	} else if (init == "xavier" || init == "xavier_uniform") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) {
			torch::nn::init::xavier_uniform_(t);
		} else {
			torch::nn::init::uniform_(t, -1.0, 1.0);
		}
	} else if (init == "kaiming" || init == "he") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) {
			torch::nn::init::kaiming_uniform_(t);
		} else {
			torch::nn::init::uniform_(t, -1.0, 1.0);
		}
	} else {
		throw std::runtime_error("Unknown initialization: " + init);
	}

	auto tv = createTensorValue(t);
	atom->setValue(key, tv);

	if (_use_cache) {
		cache_tensor(make_cache_key(atom, key), tv);
	}

	return tv;
}

void ATenSpace::set_tensor(const Handle& atom,
                           const Handle& key,
                           const TensorValuePtr& tensor)
{
	atom->setValue(key, tensor);

	if (_use_cache) {
		cache_tensor(make_cache_key(atom, key), tensor);
	}
}

void ATenSpace::set_tensor(const Handle& atom,
                           const Handle& key,
                           const at::Tensor& tensor)
{
	auto tv = createTensorValue(tensor);
	set_tensor(atom, key, tv);
}

TensorValuePtr ATenSpace::get_tensor(const Handle& atom, const Handle& key) const
{
	if (_use_cache) {
		std::lock_guard<std::mutex> lock(_mtx);
		std::string cache_key = make_cache_key(atom, key);
		auto it = _cache.find(cache_key);
		if (it != _cache.end()) {
			return it->second;
		}
	}

	ValuePtr v = atom->getValue(key);
	if (nullptr == v) return nullptr;

	TensorValuePtr tv = TensorValueCast(v);
	if (tv && _use_cache) {
		cache_tensor(make_cache_key(atom, key), tv);
	}

	return tv;
}

bool ATenSpace::has_tensor(const Handle& atom, const Handle& key) const
{
	return get_tensor(atom, key) != nullptr;
}

void ATenSpace::remove_tensor(const Handle& atom, const Handle& key)
{
	atom->setValue(key, nullptr);

	if (_use_cache) {
		std::lock_guard<std::mutex> lock(_mtx);
		_cache.erase(make_cache_key(atom, key));
	}
}

HandleSeq ATenSpace::get_atoms_with_tensor(const Handle& key) const
{
	HandleSeq result;
	// This requires iterating all atoms - expensive operation
	// In practice, you'd maintain an index
	HandleSeq all = _as->get_all_atoms();
	for (const Handle& h : all) {
		if (has_tensor(h, key)) {
			result.push_back(h);
		}
	}
	return result;
}

// ============================================================
// Tensor computations

TensorValuePtr ATenSpace::matmul(const TensorValuePtr& a,
                                 const TensorValuePtr& b) const
{
	return createTensorValue(torch::matmul(a->tensor(), b->tensor()));
}

TensorValuePtr ATenSpace::elementwise(const TensorValuePtr& a,
                                      const TensorValuePtr& b,
                                      const std::string& op) const
{
	at::Tensor result;
	if (op == "add") {
		result = a->tensor() + b->tensor();
	} else if (op == "sub") {
		result = a->tensor() - b->tensor();
	} else if (op == "mul") {
		result = a->tensor() * b->tensor();
	} else if (op == "div") {
		result = a->tensor() / b->tensor();
	} else {
		throw std::runtime_error("Unknown elementwise op: " + op);
	}
	return createTensorValue(result);
}

TensorValuePtr ATenSpace::activate(const TensorValuePtr& tensor,
                                   const std::string& activation) const
{
	at::Tensor result;
	if (activation == "relu") {
		result = torch::relu(tensor->tensor());
	} else if (activation == "sigmoid") {
		result = torch::sigmoid(tensor->tensor());
	} else if (activation == "tanh") {
		result = torch::tanh(tensor->tensor());
	} else if (activation == "softmax") {
		result = torch::softmax(tensor->tensor(), -1);
	} else if (activation == "gelu") {
		result = torch::gelu(tensor->tensor());
	} else if (activation == "leaky_relu") {
		result = torch::leaky_relu(tensor->tensor());
	} else {
		throw std::runtime_error("Unknown activation: " + activation);
	}
	return createTensorValue(result);
}

double ATenSpace::cosine_similarity(const TensorValuePtr& a,
                                    const TensorValuePtr& b) const
{
	at::Tensor a_flat = a->tensor().flatten().to(at::kDouble);
	at::Tensor b_flat = b->tensor().flatten().to(at::kDouble);

	at::Tensor dot_prod = torch::dot(a_flat, b_flat);
	at::Tensor norm_a = torch::norm(a_flat);
	at::Tensor norm_b = torch::norm(b_flat);

	at::Tensor sim = dot_prod / (norm_a * norm_b + 1e-8);
	return sim.item<double>();
}

double ATenSpace::dot(const TensorValuePtr& a, const TensorValuePtr& b) const
{
	at::Tensor a_flat = a->tensor().flatten().to(at::kDouble);
	at::Tensor b_flat = b->tensor().flatten().to(at::kDouble);
	return torch::dot(a_flat, b_flat).item<double>();
}

double ATenSpace::norm(const TensorValuePtr& tensor, int p) const
{
	return torch::norm(tensor->tensor(), p).item<double>();
}

TensorValuePtr ATenSpace::normalize(const TensorValuePtr& tensor) const
{
	at::Tensor t = tensor->tensor();
	at::Tensor n = torch::norm(t);
	return createTensorValue(t / (n + 1e-8));
}

// ============================================================
// Batch operations

TensorValuePtr ATenSpace::batch_get(const HandleSeq& atoms,
                                    const Handle& key) const
{
	if (atoms.empty()) {
		throw std::runtime_error("Empty atom list for batch_get");
	}

	std::vector<at::Tensor> tensors;
	for (const Handle& h : atoms) {
		TensorValuePtr tv = get_tensor(h, key);
		if (!tv) {
			throw std::runtime_error("Atom missing tensor: " + h->to_short_string());
		}
		tensors.push_back(tv->tensor());
	}

	return createTensorValue(torch::stack(tensors, 0));
}

void ATenSpace::batch_set(const HandleSeq& atoms,
                          const Handle& key,
                          const TensorValuePtr& batch_tensor)
{
	if (atoms.size() != (size_t)batch_tensor->tensor().size(0)) {
		throw std::runtime_error("Batch dimension mismatch");
	}

	for (size_t i = 0; i < atoms.size(); i++) {
		at::Tensor slice = batch_tensor->tensor()[i];
		set_tensor(atoms[i], key, createTensorValue(slice));
	}
}

TensorValuePtr ATenSpace::pairwise_similarity(const HandleSeq& atoms,
                                              const Handle& key) const
{
	TensorValuePtr batch = batch_get(atoms, key);
	at::Tensor t = batch->tensor();  // [N, D]

	// Normalize
	at::Tensor norms = torch::norm(t, 2, 1, true);  // [N, 1]
	at::Tensor normalized = t / (norms + 1e-8);

	// Pairwise cosine similarity: A @ A^T
	at::Tensor sim = torch::matmul(normalized, normalized.transpose(0, 1));
	return createTensorValue(sim);
}

std::vector<std::pair<Handle, double>> ATenSpace::nearest_neighbors(
	const Handle& query_atom,
	const Handle& key,
	const HandleSeq& candidates,
	size_t k) const
{
	TensorValuePtr query = get_tensor(query_atom, key);
	if (!query) {
		throw std::runtime_error("Query atom has no tensor");
	}

	std::vector<std::pair<Handle, double>> scores;
	for (const Handle& h : candidates) {
		if (h == query_atom) continue;
		TensorValuePtr t = get_tensor(h, key);
		if (!t) continue;

		double sim = cosine_similarity(query, t);
		scores.push_back({h, sim});
	}

	// Sort by similarity (descending)
	std::sort(scores.begin(), scores.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	// Return top k
	if (scores.size() > k) {
		scores.resize(k);
	}
	return scores;
}

// ============================================================
// Neural network layers

TensorValuePtr ATenSpace::linear(const TensorValuePtr& input,
                                 const TensorValuePtr& weight,
                                 const TensorValuePtr& bias) const
{
	at::Tensor result = torch::matmul(input->tensor(), weight->tensor().t());
	if (bias) {
		result = result + bias->tensor();
	}
	return createTensorValue(result);
}

TensorValuePtr ATenSpace::attention(const TensorValuePtr& query,
                                    const TensorValuePtr& key,
                                    const TensorValuePtr& value,
                                    int num_heads) const
{
	// Simplified single-head attention for now
	at::Tensor q = query->tensor();
	at::Tensor k = key->tensor();
	at::Tensor v = value->tensor();

	int64_t d_k = k.size(-1);
	at::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)d_k);
	at::Tensor attn = torch::softmax(scores, -1);
	at::Tensor result = torch::matmul(attn, v);

	return createTensorValue(result);
}

TensorValuePtr ATenSpace::layer_norm(const TensorValuePtr& input,
                                     const std::vector<int64_t>& normalized_shape) const
{
	at::Tensor result = torch::layer_norm(input->tensor(), normalized_shape);
	return createTensorValue(result);
}

// ============================================================
// Utility functions

void ATenSpace::clear_cache()
{
	std::lock_guard<std::mutex> lock(_mtx);
	_cache.clear();
}

size_t ATenSpace::cache_size() const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _cache.size();
}

std::string ATenSpace::memory_stats() const
{
	std::ostringstream ss;
	ss << "ATenSpace Memory Stats:\n";
	ss << "  Cache entries: " << cache_size() << "\n";

	if (cuda_available()) {
		for (int i = 0; i < cuda_device_count(); i++) {
			ss << "  CUDA device " << i << ":\n";
			// Note: detailed memory stats would require additional torch API calls
		}
	}

	return ss.str();
}

bool ATenSpace::cuda_available()
{
	return torch::cuda::is_available();
}

int ATenSpace::cuda_device_count()
{
	return torch::cuda::device_count();
}

at::ScalarType ATenSpace::parse_dtype(const std::string& dtype_str)
{
	if (dtype_str == "float" || dtype_str == "float32") return at::kFloat;
	if (dtype_str == "double" || dtype_str == "float64") return at::kDouble;
	if (dtype_str == "half" || dtype_str == "float16") return at::kHalf;
	if (dtype_str == "int" || dtype_str == "int32") return at::kInt;
	if (dtype_str == "long" || dtype_str == "int64") return at::kLong;
	if (dtype_str == "short" || dtype_str == "int16") return at::kShort;
	if (dtype_str == "byte" || dtype_str == "uint8") return at::kByte;
	if (dtype_str == "bool") return at::kBool;
	if (dtype_str == "bfloat16") return at::kBFloat16;

	throw std::runtime_error("Unknown dtype: " + dtype_str);
}

std::string ATenSpace::dtype_to_string(at::ScalarType dtype)
{
	switch (dtype) {
		case at::kFloat: return "float32";
		case at::kDouble: return "float64";
		case at::kHalf: return "float16";
		case at::kInt: return "int32";
		case at::kLong: return "int64";
		case at::kShort: return "int16";
		case at::kByte: return "uint8";
		case at::kBool: return "bool";
		case at::kBFloat16: return "bfloat16";
		default: return "unknown";
	}
}

#endif // HAVE_ATEN
