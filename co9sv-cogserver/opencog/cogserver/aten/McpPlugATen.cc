/*
 * McpPlugATen.cc
 *
 * MCP Plugin for ATen tensor operations
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

#include <json/json.h>
#include <sstream>
#include <iomanip>

#include <opencog/persist/sexpr/Sexpr.h>
#include "McpPlugATen.h"

using namespace opencog;

// ============================================================
// Helper for tool description

static void add_tool(std::string& json, const std::string& name,
                     const std::string& description,
                     const std::string& schema,
                     bool last = false)
{
	json += "\t{\n";
	json += "\t\t\"name\": \"" + name + "\",\n";
	json += "\t\t\"description\": \"" + description + "\",\n";
	json += "\t\t\"inputSchema\": " + schema + "\n";
	json += "\t}";
	if (!last) json += ",";
	json += "\n";
}

// ============================================================
// Constructor / Destructor

McpPlugATen::McpPlugATen(ATenSpace* aten, AtomSpace* as)
	: _aten(aten), _as(as)
{
}

McpPlugATen::~McpPlugATen()
{
	delete _aten;
}

// ============================================================
// Tool descriptions

std::string McpPlugATen::get_tool_descriptions() const
{
	std::string json = "[\n";

	// tensorCreate
	add_tool(json, "tensorCreate",
		"Create a new tensor with specified shape, dtype, and initialization",
		"{\"type\": \"object\", \"properties\": {"
		"\"shape\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"description\": \"Tensor dimensions, e.g. [3, 4] for 3x4 matrix\"}, "
		"\"dtype\": {\"type\": \"string\", \"description\": \"Data type: float32 (default), float64, float16, int32, int64\", \"default\": \"float32\"}, "
		"\"init\": {\"type\": \"string\", \"description\": \"Initialization: zeros (default), ones, randn, xavier, kaiming\", \"default\": \"zeros\"}, "
		"\"atom\": {\"type\": \"string\", \"description\": \"Optional: atomese s-expression for atom to store tensor on\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Optional: atomese s-expression for key (required if atom specified)\"}}, "
		"\"required\": [\"shape\"]}");

	// tensorGet
	add_tool(json, "tensorGet",
		"Get a tensor stored on an atom at a given key",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese s-expression for the atom, e.g. (Concept \\\"embedding\\\")\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Atomese s-expression for the key, e.g. (Predicate \\\"vector\\\")\"}}, "
		"\"required\": [\"atom\", \"key\"]}");

	// tensorSet
	add_tool(json, "tensorSet",
		"Store tensor data on an atom at a given key",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese s-expression for the atom\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Atomese s-expression for the key\"}, "
		"\"data\": {\"type\": \"array\", \"items\": {\"type\": \"number\"}, \"description\": \"Flat array of tensor data\"}, "
		"\"shape\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"description\": \"Tensor dimensions\"}}, "
		"\"required\": [\"atom\", \"key\", \"data\", \"shape\"]}");

	// tensorOp
	add_tool(json, "tensorOp",
		"Perform matrix operation on tensors (matmul, transpose, reshape)",
		"{\"type\": \"object\", \"properties\": {"
		"\"op\": {\"type\": \"string\", \"description\": \"Operation: matmul, transpose, reshape, sum, mean\"}, "
		"\"a\": {\"type\": \"object\", \"description\": \"First tensor reference: {atom, key} or {data, shape}\"}, "
		"\"b\": {\"type\": \"object\", \"description\": \"Second tensor reference (for matmul)\"}, "
		"\"dim0\": {\"type\": \"integer\", \"description\": \"First dimension for transpose\"}, "
		"\"dim1\": {\"type\": \"integer\", \"description\": \"Second dimension for transpose\"}, "
		"\"shape\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"description\": \"New shape for reshape\"}, "
		"\"result_atom\": {\"type\": \"string\", \"description\": \"Optional: store result on this atom\"}, "
		"\"result_key\": {\"type\": \"string\", \"description\": \"Optional: key for storing result\"}}, "
		"\"required\": [\"op\", \"a\"]}");

	// tensorMath
	add_tool(json, "tensorMath",
		"Element-wise math operations on tensors",
		"{\"type\": \"object\", \"properties\": {"
		"\"op\": {\"type\": \"string\", \"description\": \"Operation: add, sub, mul, div\"}, "
		"\"a\": {\"type\": \"object\", \"description\": \"First tensor reference\"}, "
		"\"b\": {\"type\": \"object\", \"description\": \"Second tensor reference\"}, "
		"\"result_atom\": {\"type\": \"string\", \"description\": \"Optional: store result on this atom\"}, "
		"\"result_key\": {\"type\": \"string\", \"description\": \"Optional: key for storing result\"}}, "
		"\"required\": [\"op\", \"a\", \"b\"]}");

	// tensorActivation
	add_tool(json, "tensorActivation",
		"Apply activation function to tensor",
		"{\"type\": \"object\", \"properties\": {"
		"\"activation\": {\"type\": \"string\", \"description\": \"Activation: relu, sigmoid, tanh, softmax, gelu, leaky_relu\"}, "
		"\"tensor\": {\"type\": \"object\", \"description\": \"Tensor reference\"}, "
		"\"result_atom\": {\"type\": \"string\", \"description\": \"Optional: store result on this atom\"}, "
		"\"result_key\": {\"type\": \"string\", \"description\": \"Optional: key for storing result\"}}, "
		"\"required\": [\"activation\", \"tensor\"]}");

	// tensorSimilarity
	add_tool(json, "tensorSimilarity",
		"Compute similarity between two tensors",
		"{\"type\": \"object\", \"properties\": {"
		"\"metric\": {\"type\": \"string\", \"description\": \"Similarity metric: cosine (default), dot, euclidean\", \"default\": \"cosine\"}, "
		"\"a\": {\"type\": \"object\", \"description\": \"First tensor reference\"}, "
		"\"b\": {\"type\": \"object\", \"description\": \"Second tensor reference\"}}, "
		"\"required\": [\"a\", \"b\"]}");

	// tensorNearest
	add_tool(json, "tensorNearest",
		"Find k nearest neighbors by tensor similarity",
		"{\"type\": \"object\", \"properties\": {"
		"\"query\": {\"type\": \"string\", \"description\": \"Atomese for query atom\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Atomese for tensor key\"}, "
		"\"candidates\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of candidate atom s-expressions\"}, "
		"\"k\": {\"type\": \"integer\", \"description\": \"Number of neighbors to return\", \"default\": 10}}, "
		"\"required\": [\"query\", \"key\", \"candidates\"]}");

	// tensorToFloatValue
	add_tool(json, "tensorToFloatValue",
		"Convert TensorValue to FloatValue (for AtomSpace compatibility)",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese for atom with TensorValue\"}, "
		"\"tensor_key\": {\"type\": \"string\", \"description\": \"Key of TensorValue\"}, "
		"\"float_key\": {\"type\": \"string\", \"description\": \"Key to store FloatValue\"}}, "
		"\"required\": [\"atom\", \"tensor_key\", \"float_key\"]}");

	// tensorFromFloatValue
	add_tool(json, "tensorFromFloatValue",
		"Convert FloatValue to TensorValue with specified shape",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese for atom with FloatValue\"}, "
		"\"float_key\": {\"type\": \"string\", \"description\": \"Key of FloatValue\"}, "
		"\"tensor_key\": {\"type\": \"string\", \"description\": \"Key to store TensorValue\"}, "
		"\"shape\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"description\": \"Tensor shape (product must equal FloatValue length)\"}}, "
		"\"required\": [\"atom\", \"float_key\", \"tensor_key\", \"shape\"]}");

	// tensorInfo
	add_tool(json, "tensorInfo",
		"Get ATen/CUDA status and memory information",
		"{\"type\": \"object\", \"properties\": {}, \"required\": []}", true);

	json += "]";
	return json;
}

// ============================================================
// Tool invocation

std::string McpPlugATen::invoke_tool(const std::string& tool_name,
                                     const std::string& arguments) const
{
	try {
		if (tool_name == "tensorCreate") return do_create(arguments);
		if (tool_name == "tensorGet") return do_get(arguments);
		if (tool_name == "tensorSet") return do_set(arguments);
		if (tool_name == "tensorOp") return do_op(arguments);
		if (tool_name == "tensorMath") return do_math(arguments);
		if (tool_name == "tensorActivation") return do_activation(arguments);
		if (tool_name == "tensorSimilarity") return do_similarity(arguments);
		if (tool_name == "tensorNearest") return do_nearest(arguments);
		if (tool_name == "tensorToFloatValue") return do_to_float_value(arguments);
		if (tool_name == "tensorFromFloatValue") return do_from_float_value(arguments);
		if (tool_name == "tensorInfo") return do_info(arguments);

		return format_error("Unknown tool: " + tool_name);
	}
	catch (const std::exception& e) {
		return format_error(e.what());
	}
}

// ============================================================
// Helpers

Handle McpPlugATen::parse_atom(const std::string& atomese) const
{
	return Sexpr::decode_atom(atomese, *_as);
}

TensorValuePtr McpPlugATen::parse_tensor_ref(const std::string& json) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(json);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		throw std::runtime_error("Invalid JSON: " + errors);
	}

	// If has atom/key, fetch from AtomSpace
	if (root.isMember("atom") && root.isMember("key")) {
		Handle atom = parse_atom(root["atom"].asString());
		Handle key = parse_atom(root["key"].asString());
		TensorValuePtr tv = _aten->get_tensor(atom, key);
		if (!tv) throw std::runtime_error("No tensor found on atom");
		return tv;
	}

	// If has data/shape, create tensor
	if (root.isMember("data") && root.isMember("shape")) {
		std::vector<double> data;
		for (const auto& v : root["data"]) {
			data.push_back(v.asDouble());
		}
		std::vector<int64_t> shape;
		for (const auto& v : root["shape"]) {
			shape.push_back(v.asInt64());
		}
		auto fv = createFloatValue(data);
		return createTensorValue(TensorValue::from_float_value(fv, shape).tensor());
	}

	throw std::runtime_error("Tensor ref must have {atom,key} or {data,shape}");
}

std::string McpPlugATen::format_tensor(const TensorValuePtr& tv) const
{
	Json::Value result;
	result["type"] = "TensorValue";

	Json::Value shape(Json::arrayValue);
	for (auto d : tv->shape()) {
		shape.append((Json::Int64)d);
	}
	result["shape"] = shape;

	result["dtype"] = ATenSpace::dtype_to_string(tv->dtype());
	result["device"] = tv->device_string();
	result["numel"] = (Json::UInt64)tv->numel();

	// Include data preview
	std::vector<double> data = tv->to_vector();
	Json::Value data_preview(Json::arrayValue);
	size_t n = std::min(data.size(), (size_t)16);
	for (size_t i = 0; i < n; i++) {
		data_preview.append(data[i]);
	}
	result["data_preview"] = data_preview;
	if (data.size() > 16) {
		result["truncated"] = true;
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugATen::format_error(const std::string& msg) const
{
	Json::Value result;
	result["error"] = true;
	result["message"] = msg;
	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugATen::format_success(const std::string& msg) const
{
	Json::Value result;
	result["success"] = true;
	result["result"] = msg;
	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

// ============================================================
// Tool implementations

std::string McpPlugATen::do_create(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::vector<int64_t> shape;
	for (const auto& v : root["shape"]) {
		shape.push_back(v.asInt64());
	}

	std::string dtype_str = root.get("dtype", "float32").asString();
	std::string init = root.get("init", "zeros").asString();
	at::ScalarType dtype = ATenSpace::parse_dtype(dtype_str);

	auto options = torch::TensorOptions().dtype(dtype);
	at::Tensor t;

	if (init == "zeros") {
		t = torch::zeros(shape, options);
	} else if (init == "ones") {
		t = torch::ones(shape, options);
	} else if (init == "randn") {
		t = torch::randn(shape, options);
	} else if (init == "xavier") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) {
			torch::nn::init::xavier_uniform_(t);
		}
	} else if (init == "kaiming") {
		t = torch::empty(shape, options);
		if (shape.size() >= 2) {
			torch::nn::init::kaiming_uniform_(t);
		}
	} else {
		return format_error("Unknown init: " + init);
	}

	TensorValuePtr tv = createTensorValue(t);

	// Optionally store on atom
	if (root.isMember("atom") && root.isMember("key")) {
		Handle atom = parse_atom(root["atom"].asString());
		Handle key = parse_atom(root["key"].asString());
		_aten->set_tensor(atom, key, tv);
	}

	return format_tensor(tv);
}

std::string McpPlugATen::do_get(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle atom = parse_atom(root["atom"].asString());
	Handle key = parse_atom(root["key"].asString());

	TensorValuePtr tv = _aten->get_tensor(atom, key);
	if (!tv) {
		return format_error("No tensor found at specified key");
	}

	return format_tensor(tv);
}

std::string McpPlugATen::do_set(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle atom = parse_atom(root["atom"].asString());
	Handle key = parse_atom(root["key"].asString());

	std::vector<double> data;
	for (const auto& v : root["data"]) {
		data.push_back(v.asDouble());
	}

	std::vector<int64_t> shape;
	for (const auto& v : root["shape"]) {
		shape.push_back(v.asInt64());
	}

	auto fv = createFloatValue(data);
	TensorValue tv = TensorValue::from_float_value(fv, shape);
	_aten->set_tensor(atom, key, createTensorValue(tv.tensor()));

	return format_success("Tensor stored successfully");
}

std::string McpPlugATen::do_op(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::string op = root["op"].asString();

	Json::StreamWriterBuilder writer;
	TensorValuePtr a = parse_tensor_ref(Json::writeString(writer, root["a"]));
	TensorValuePtr result;

	if (op == "matmul") {
		TensorValuePtr b = parse_tensor_ref(Json::writeString(writer, root["b"]));
		result = _aten->matmul(a, b);
	} else if (op == "transpose") {
		int64_t dim0 = root.get("dim0", 0).asInt64();
		int64_t dim1 = root.get("dim1", 1).asInt64();
		result = createTensorValue(a->transpose(dim0, dim1).tensor());
	} else if (op == "reshape") {
		std::vector<int64_t> new_shape;
		for (const auto& v : root["shape"]) {
			new_shape.push_back(v.asInt64());
		}
		result = createTensorValue(a->reshape(new_shape).tensor());
	} else if (op == "sum") {
		int64_t dim = root.get("dim", -1).asInt64();
		bool keepdim = root.get("keepdim", false).asBool();
		result = createTensorValue(a->sum(dim, keepdim).tensor());
	} else if (op == "mean") {
		int64_t dim = root.get("dim", -1).asInt64();
		bool keepdim = root.get("keepdim", false).asBool();
		result = createTensorValue(a->mean(dim, keepdim).tensor());
	} else {
		return format_error("Unknown op: " + op);
	}

	// Optionally store result
	if (root.isMember("result_atom") && root.isMember("result_key")) {
		Handle r_atom = parse_atom(root["result_atom"].asString());
		Handle r_key = parse_atom(root["result_key"].asString());
		_aten->set_tensor(r_atom, r_key, result);
	}

	return format_tensor(result);
}

std::string McpPlugATen::do_math(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::string op = root["op"].asString();
	Json::StreamWriterBuilder writer;
	TensorValuePtr a = parse_tensor_ref(Json::writeString(writer, root["a"]));
	TensorValuePtr b = parse_tensor_ref(Json::writeString(writer, root["b"]));

	TensorValuePtr result = _aten->elementwise(a, b, op);

	// Optionally store result
	if (root.isMember("result_atom") && root.isMember("result_key")) {
		Handle r_atom = parse_atom(root["result_atom"].asString());
		Handle r_key = parse_atom(root["result_key"].asString());
		_aten->set_tensor(r_atom, r_key, result);
	}

	return format_tensor(result);
}

std::string McpPlugATen::do_activation(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::string activation = root["activation"].asString();
	Json::StreamWriterBuilder writer;
	TensorValuePtr tensor = parse_tensor_ref(Json::writeString(writer, root["tensor"]));

	TensorValuePtr result = _aten->activate(tensor, activation);

	// Optionally store result
	if (root.isMember("result_atom") && root.isMember("result_key")) {
		Handle r_atom = parse_atom(root["result_atom"].asString());
		Handle r_key = parse_atom(root["result_key"].asString());
		_aten->set_tensor(r_atom, r_key, result);
	}

	return format_tensor(result);
}

std::string McpPlugATen::do_similarity(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::string metric = root.get("metric", "cosine").asString();
	Json::StreamWriterBuilder writer;
	TensorValuePtr a = parse_tensor_ref(Json::writeString(writer, root["a"]));
	TensorValuePtr b = parse_tensor_ref(Json::writeString(writer, root["b"]));

	double sim;
	if (metric == "cosine") {
		sim = _aten->cosine_similarity(a, b);
	} else if (metric == "dot") {
		sim = _aten->dot(a, b);
	} else if (metric == "euclidean") {
		// Euclidean distance (not similarity)
		auto diff = a->sub(*b);
		sim = -_aten->norm(createTensorValue(diff.tensor()));
	} else {
		return format_error("Unknown metric: " + metric);
	}

	Json::Value result;
	result["metric"] = metric;
	result["similarity"] = sim;
	return Json::writeString(writer, result);
}

std::string McpPlugATen::do_nearest(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle query = parse_atom(root["query"].asString());
	Handle key = parse_atom(root["key"].asString());
	size_t k = root.get("k", 10).asUInt();

	HandleSeq candidates;
	for (const auto& c : root["candidates"]) {
		candidates.push_back(parse_atom(c.asString()));
	}

	auto neighbors = _aten->nearest_neighbors(query, key, candidates, k);

	Json::Value result(Json::arrayValue);
	for (const auto& [h, sim] : neighbors) {
		Json::Value entry;
		entry["atom"] = h->to_short_string();
		entry["similarity"] = sim;
		result.append(entry);
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugATen::do_to_float_value(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle atom = parse_atom(root["atom"].asString());
	Handle tensor_key = parse_atom(root["tensor_key"].asString());
	Handle float_key = parse_atom(root["float_key"].asString());

	TensorValuePtr tv = _aten->get_tensor(atom, tensor_key);
	if (!tv) {
		return format_error("No tensor found at specified key");
	}

	FloatValuePtr fv = tv->to_float_value();
	atom->setValue(float_key, fv);

	return format_success("Converted to FloatValue with " +
	                      std::to_string(fv->value().size()) + " elements");
}

std::string McpPlugATen::do_from_float_value(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle atom = parse_atom(root["atom"].asString());
	Handle float_key = parse_atom(root["float_key"].asString());
	Handle tensor_key = parse_atom(root["tensor_key"].asString());

	std::vector<int64_t> shape;
	for (const auto& v : root["shape"]) {
		shape.push_back(v.asInt64());
	}

	ValuePtr v = atom->getValue(float_key);
	FloatValuePtr fv = FloatValueCast(v);
	if (!fv) {
		return format_error("No FloatValue found at specified key");
	}

	TensorValue tv = TensorValue::from_float_value(fv, shape);
	_aten->set_tensor(atom, tensor_key, createTensorValue(tv.tensor()));

	return format_success("Converted to TensorValue");
}

std::string McpPlugATen::do_info(const std::string& args) const
{
	Json::Value result;
	result["aten_available"] = true;
	result["cuda_available"] = ATenSpace::cuda_available();
	result["cuda_device_count"] = ATenSpace::cuda_device_count();
	result["cache_size"] = (Json::UInt64)_aten->cache_size();

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

#endif // HAVE_ATEN
