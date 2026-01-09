/*
 * McpPlugTensorLogic.cc
 *
 * MCP Plugin for Tensor Logic operations
 *
 * Copyright (c) 2025 OpenCog Foundation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation.
 */

#ifdef HAVE_ATEN

#include <json/json.h>
#include <sstream>

#include <opencog/persist/sexpr/Sexpr.h>
#include "McpPlugTensorLogic.h"

using namespace opencog;

// Helper for tool descriptions
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

McpPlugTensorLogic::McpPlugTensorLogic(std::shared_ptr<TensorLogic> logic,
                                       AtomSpace* as)
	: _logic(logic), _as(as)
{
}

std::string McpPlugTensorLogic::get_tool_descriptions() const
{
	std::string json = "[\n";

	// Multi-Entity Operations
	add_tool(json, "tensorEntityBatch",
		"Create batch tensor embedding for multiple entities",
		"{\"type\": \"object\", \"properties\": {"
		"\"entities\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of atomese s-expressions for entities\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Atomese for embedding key\"}}, "
		"\"required\": [\"entities\", \"key\"]}");

	add_tool(json, "tensorAggregateEntities",
		"Aggregate entity embeddings into single representation",
		"{\"type\": \"object\", \"properties\": {"
		"\"entities\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of entity atomese\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Embedding key\"}, "
		"\"aggregation\": {\"type\": \"string\", \"description\": \"Method: mean, max, sum, attention, deepsets\", \"default\": \"attention\"}}, "
		"\"required\": [\"entities\", \"key\"]}");

	add_tool(json, "tensorSimilarityMatrix",
		"Compute pairwise similarity matrix for entities",
		"{\"type\": \"object\", \"properties\": {"
		"\"entities\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of entity atomese\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Embedding key\"}}, "
		"\"required\": [\"entities\", \"key\"]}");

	add_tool(json, "tensorClusterEntities",
		"Cluster entities by embedding similarity",
		"{\"type\": \"object\", \"properties\": {"
		"\"entities\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of entity atomese\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Embedding key\"}, "
		"\"k\": {\"type\": \"integer\", \"description\": \"Number of clusters\", \"default\": 5}}, "
		"\"required\": [\"entities\", \"key\"]}");

	// Multi-Scale Operations
	add_tool(json, "tensorMultiscaleCompute",
		"Compute multi-scale embeddings (LOCAL, NEIGHBOR, COMMUNITY, GLOBAL) for an atom",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese for the atom\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Feature embedding key\"}, "
		"\"store\": {\"type\": \"boolean\", \"description\": \"Store result on atom\", \"default\": true}}, "
		"\"required\": [\"atom\", \"key\"]}");

	add_tool(json, "tensorQueryAtScale",
		"Query atoms similar to embedding at specific scale level",
		"{\"type\": \"object\", \"properties\": {"
		"\"query\": {\"type\": \"object\", \"description\": \"Query tensor: {atom, key} or {data, shape}\"}, "
		"\"scale\": {\"type\": \"string\", \"description\": \"Scale: LOCAL, NEIGHBOR, COMMUNITY, GLOBAL\"}, "
		"\"k\": {\"type\": \"integer\", \"description\": \"Number of results\", \"default\": 10}}, "
		"\"required\": [\"query\", \"scale\"]}");

	add_tool(json, "tensorCrossScaleSimilarity",
		"Compute similarity between atoms at different scales",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom_a\": {\"type\": \"string\", \"description\": \"First atom atomese\"}, "
		"\"scale_a\": {\"type\": \"string\", \"description\": \"Scale for atom_a\"}, "
		"\"atom_b\": {\"type\": \"string\", \"description\": \"Second atom atomese\"}, "
		"\"scale_b\": {\"type\": \"string\", \"description\": \"Scale for atom_b\"}}, "
		"\"required\": [\"atom_a\", \"scale_a\", \"atom_b\", \"scale_b\"]}");

	// Network-Aware Operations
	add_tool(json, "tensorNetworkEmbed",
		"Compute network-aware embedding incorporating graph structure",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese for atom\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Feature key\"}, "
		"\"layers\": {\"type\": \"integer\", \"description\": \"GNN layers\", \"default\": 2}}, "
		"\"required\": [\"atom\", \"key\"]}");

	add_tool(json, "tensorPropagate",
		"Propagate embeddings through graph via message passing",
		"{\"type\": \"object\", \"properties\": {"
		"\"atoms\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Atoms to update\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Embedding key\"}, "
		"\"iterations\": {\"type\": \"integer\", \"description\": \"Number of iterations\", \"default\": 3}}, "
		"\"required\": [\"atoms\", \"key\"]}");

	add_tool(json, "tensorRelationEmbed",
		"Compute relationship embedding for a link",
		"{\"type\": \"object\", \"properties\": {"
		"\"link\": {\"type\": \"string\", \"description\": \"Atomese for the link\"}}, "
		"\"required\": [\"link\"]}");

	// Attention Operations
	add_tool(json, "tensorStimulate",
		"Stimulate attention on an atom",
		"{\"type\": \"object\", \"properties\": {"
		"\"atom\": {\"type\": \"string\", \"description\": \"Atomese for atom to stimulate\"}, "
		"\"stimulus\": {\"type\": \"number\", \"description\": \"Stimulus amount\", \"default\": 1.0}}, "
		"\"required\": [\"atom\"]}");

	add_tool(json, "tensorGetFocus",
		"Get current attentional focus (high-attention atoms)",
		"{\"type\": \"object\", \"properties\": {"
		"\"limit\": {\"type\": \"integer\", \"description\": \"Max atoms to return\", \"default\": 20}}, "
		"\"required\": []}");

	add_tool(json, "tensorSpreadAttention",
		"Spread attention from source atom to connected atoms",
		"{\"type\": \"object\", \"properties\": {"
		"\"source\": {\"type\": \"string\", \"description\": \"Source atom atomese\"}}, "
		"\"required\": [\"source\"]}");

	add_tool(json, "tensorAttentionQuery",
		"Attention-weighted similarity query",
		"{\"type\": \"object\", \"properties\": {"
		"\"query\": {\"type\": \"object\", \"description\": \"Query tensor reference\"}, "
		"\"key\": {\"type\": \"string\", \"description\": \"Embedding key\"}, "
		"\"k\": {\"type\": \"integer\", \"description\": \"Number of results\", \"default\": 10}}, "
		"\"required\": [\"query\", \"key\"]}");

	add_tool(json, "tensorAttentionCycle",
		"Run one attention allocation cycle (decay, spread, update)",
		"{\"type\": \"object\", \"properties\": {}, \"required\": []}");

	// Inference Operations
	add_tool(json, "tensorDeduction",
		"Tensor-enhanced deduction: (A->B) + (B->C) => (A->C)",
		"{\"type\": \"object\", \"properties\": {"
		"\"premise1\": {\"type\": \"string\", \"description\": \"First premise atomese\"}, "
		"\"premise2\": {\"type\": \"string\", \"description\": \"Second premise atomese\"}}, "
		"\"required\": [\"premise1\", \"premise2\"]}");

	add_tool(json, "tensorInduction",
		"Tensor-enhanced induction from instances",
		"{\"type\": \"object\", \"properties\": {"
		"\"instances\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Instance atomese list\"}}, "
		"\"required\": [\"instances\"]}");

	add_tool(json, "tensorAnalogy",
		"Analogy by embedding: A:B :: C:? finds D",
		"{\"type\": \"object\", \"properties\": {"
		"\"A\": {\"type\": \"string\", \"description\": \"First term atomese\"}, "
		"\"B\": {\"type\": \"string\", \"description\": \"Second term atomese\"}, "
		"\"C\": {\"type\": \"string\", \"description\": \"Third term atomese\"}, "
		"\"candidates\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Candidate D atoms\"}}, "
		"\"required\": [\"A\", \"B\", \"C\", \"candidates\"]}");

	add_tool(json, "tensorUnify",
		"Tensor-guided pattern unification",
		"{\"type\": \"object\", \"properties\": {"
		"\"pattern\": {\"type\": \"string\", \"description\": \"Pattern atomese\"}, "
		"\"candidates\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Candidate atoms\"}, "
		"\"threshold\": {\"type\": \"number\", \"description\": \"Similarity threshold\", \"default\": 0.5}}, "
		"\"required\": [\"pattern\", \"candidates\"]}");

	// Utility
	add_tool(json, "tensorLogicInit",
		"Initialize embeddings for atoms of given types",
		"{\"type\": \"object\", \"properties\": {"
		"\"types\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Atom type names\"}, "
		"\"init\": {\"type\": \"string\", \"description\": \"Initialization: zeros, randn, xavier\", \"default\": \"randn\"}}, "
		"\"required\": [\"types\"]}");

	add_tool(json, "tensorLogicStats",
		"Get TensorLogic system statistics",
		"{\"type\": \"object\", \"properties\": {}, \"required\": []}", true);

	json += "]";
	return json;
}

std::string McpPlugTensorLogic::invoke_tool(const std::string& tool_name,
                                            const std::string& arguments) const
{
	try {
		// Multi-Entity
		if (tool_name == "tensorEntityBatch") return do_entity_batch(arguments);
		if (tool_name == "tensorAggregateEntities") return do_aggregate_entities(arguments);
		if (tool_name == "tensorSimilarityMatrix") return do_similarity_matrix(arguments);
		if (tool_name == "tensorClusterEntities") return do_cluster_entities(arguments);

		// Multi-Scale
		if (tool_name == "tensorMultiscaleCompute") return do_multiscale_compute(arguments);
		if (tool_name == "tensorQueryAtScale") return do_query_at_scale(arguments);
		if (tool_name == "tensorCrossScaleSimilarity") return do_cross_scale_similarity(arguments);

		// Network-Aware
		if (tool_name == "tensorNetworkEmbed") return do_network_embed(arguments);
		if (tool_name == "tensorPropagate") return do_propagate(arguments);
		if (tool_name == "tensorRelationEmbed") return do_relation_embed(arguments);

		// Attention
		if (tool_name == "tensorStimulate") return do_stimulate(arguments);
		if (tool_name == "tensorGetFocus") return do_get_focus(arguments);
		if (tool_name == "tensorSpreadAttention") return do_spread_attention(arguments);
		if (tool_name == "tensorAttentionQuery") return do_attention_query(arguments);
		if (tool_name == "tensorAttentionCycle") return do_attention_cycle(arguments);

		// Inference
		if (tool_name == "tensorDeduction") return do_deduction(arguments);
		if (tool_name == "tensorInduction") return do_induction(arguments);
		if (tool_name == "tensorAnalogy") return do_analogy(arguments);
		if (tool_name == "tensorUnify") return do_unify(arguments);

		// Utility
		if (tool_name == "tensorLogicInit") return do_initialize(arguments);
		if (tool_name == "tensorLogicStats") return do_stats(arguments);

		return format_error("Unknown tool: " + tool_name);
	}
	catch (const std::exception& e) {
		return format_error(e.what());
	}
}

// ============================================================
// Helpers
// ============================================================

Handle McpPlugTensorLogic::parse_atom(const std::string& atomese) const
{
	return Sexpr::decode_atom(atomese, *_as);
}

HandleSeq McpPlugTensorLogic::parse_atom_list(const std::string& json_array) const
{
	HandleSeq result;
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(json_array);

	if (Json::parseFromStream(builder, stream, &root, &errors)) {
		for (const auto& item : root) {
			result.push_back(parse_atom(item.asString()));
		}
	}
	return result;
}

ScaleLevel McpPlugTensorLogic::parse_scale(const std::string& scale_str) const
{
	if (scale_str == "LOCAL") return ScaleLevel::LOCAL;
	if (scale_str == "NEIGHBOR") return ScaleLevel::NEIGHBOR;
	if (scale_str == "COMMUNITY") return ScaleLevel::COMMUNITY;
	if (scale_str == "GLOBAL") return ScaleLevel::GLOBAL;
	throw std::runtime_error("Unknown scale: " + scale_str);
}

std::string McpPlugTensorLogic::format_error(const std::string& msg) const
{
	Json::Value result;
	result["error"] = true;
	result["message"] = msg;
	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::format_success(const std::string& msg) const
{
	Json::Value result;
	result["success"] = true;
	result["result"] = msg;
	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::format_tensor(const TensorValuePtr& tv) const
{
	if (!tv) return "null";

	Json::Value result;
	result["type"] = "TensorValue";

	Json::Value shape(Json::arrayValue);
	for (auto d : tv->shape()) {
		shape.append((Json::Int64)d);
	}
	result["shape"] = shape;
	result["numel"] = (Json::UInt64)tv->numel();

	std::vector<double> data = tv->to_vector();
	Json::Value preview(Json::arrayValue);
	size_t n = std::min(data.size(), (size_t)8);
	for (size_t i = 0; i < n; i++) {
		preview.append(data[i]);
	}
	result["data_preview"] = preview;

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

// ============================================================
// Multi-Entity Operations
// ============================================================

std::string McpPlugTensorLogic::do_entity_batch(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq entities;
	for (const auto& e : root["entities"]) {
		entities.push_back(parse_atom(e.asString()));
	}
	Handle key = parse_atom(root["key"].asString());

	TensorValuePtr batch = _logic->create_entity_batch(entities, key);
	return format_tensor(batch);
}

std::string McpPlugTensorLogic::do_aggregate_entities(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq entities;
	for (const auto& e : root["entities"]) {
		entities.push_back(parse_atom(e.asString()));
	}
	Handle key = parse_atom(root["key"].asString());
	std::string agg = root.get("aggregation", "attention").asString();

	TensorValuePtr result = _logic->aggregate_entities(entities, key, agg);
	return format_tensor(result);
}

std::string McpPlugTensorLogic::do_similarity_matrix(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq entities;
	for (const auto& e : root["entities"]) {
		entities.push_back(parse_atom(e.asString()));
	}
	Handle key = parse_atom(root["key"].asString());

	TensorValuePtr matrix = _logic->entity_similarity_matrix(entities, key);
	return format_tensor(matrix);
}

std::string McpPlugTensorLogic::do_cluster_entities(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq entities;
	for (const auto& e : root["entities"]) {
		entities.push_back(parse_atom(e.asString()));
	}
	Handle key = parse_atom(root["key"].asString());
	int k = root.get("k", 5).asInt();

	auto clusters = _logic->cluster_entities(entities, key, k);

	Json::Value result;
	for (const auto& [cluster_id, members] : clusters) {
		Json::Value cluster_members(Json::arrayValue);
		for (const Handle& h : members) {
			cluster_members.append(h->to_short_string());
		}
		result[std::to_string(cluster_id)] = cluster_members;
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

// ============================================================
// Multi-Scale Operations
// ============================================================

std::string McpPlugTensorLogic::do_multiscale_compute(const std::string& args) const
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
	bool store = root.get("store", true).asBool();

	MultiScaleTensorPtr mst = _logic->compute_multiscale_embedding(atom, key);

	if (store) {
		_logic->set_multiscale_embedding(atom, mst);
	}

	Json::Value result;
	result["atom"] = atom->to_short_string();
	result["scales"] = mst->to_string();

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::do_query_at_scale(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	// Parse query tensor
	Json::Value query_ref = root["query"];
	TensorValuePtr query;
	if (query_ref.isMember("atom") && query_ref.isMember("key")) {
		Handle atom = parse_atom(query_ref["atom"].asString());
		Handle key = parse_atom(query_ref["key"].asString());
		query = _logic->get_aten_space()->get_tensor(atom, key);
	}

	if (!query) {
		return format_error("Could not resolve query tensor");
	}

	ScaleLevel scale = parse_scale(root["scale"].asString());
	size_t k = root.get("k", 10).asUInt();

	auto results = _logic->query_at_scale(query, scale, k);

	Json::Value result_array(Json::arrayValue);
	for (const auto& [h, sim] : results) {
		Json::Value item;
		item["atom"] = h->to_short_string();
		item["similarity"] = sim;
		result_array.append(item);
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result_array);
}

std::string McpPlugTensorLogic::do_cross_scale_similarity(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle a = parse_atom(root["atom_a"].asString());
	ScaleLevel scale_a = parse_scale(root["scale_a"].asString());
	Handle b = parse_atom(root["atom_b"].asString());
	ScaleLevel scale_b = parse_scale(root["scale_b"].asString());

	double sim = _logic->cross_scale_similarity(a, scale_a, b, scale_b);

	Json::Value result;
	result["similarity"] = sim;
	result["atom_a"] = a->to_short_string();
	result["scale_a"] = root["scale_a"].asString();
	result["atom_b"] = b->to_short_string();
	result["scale_b"] = root["scale_b"].asString();

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

// ============================================================
// Network-Aware Operations
// ============================================================

std::string McpPlugTensorLogic::do_network_embed(const std::string& args) const
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
	int layers = root.get("layers", 2).asInt();

	TensorValuePtr emb = _logic->compute_network_embedding(atom, key, layers);
	return format_tensor(emb);
}

std::string McpPlugTensorLogic::do_propagate(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq atoms;
	for (const auto& a : root["atoms"]) {
		atoms.push_back(parse_atom(a.asString()));
	}
	Handle key = parse_atom(root["key"].asString());
	int iters = root.get("iterations", 3).asInt();

	_logic->propagate_embeddings(atoms, key, iters);

	return format_success("Propagated embeddings for " +
	                      std::to_string(atoms.size()) + " atoms");
}

std::string McpPlugTensorLogic::do_relation_embed(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle link = parse_atom(root["link"].asString());
	TensorValuePtr emb = _logic->compute_relation_embedding(link);
	return format_tensor(emb);
}

// ============================================================
// Attention Operations
// ============================================================

std::string McpPlugTensorLogic::do_stimulate(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle atom = parse_atom(root["atom"].asString());
	double stimulus = root.get("stimulus", 1.0).asDouble();

	_logic->get_attention_bank()->stimulate(atom, stimulus);

	return format_success("Stimulated " + atom->to_short_string() +
	                      " with " + std::to_string(stimulus));
}

std::string McpPlugTensorLogic::do_get_focus(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	Json::parseFromStream(builder, stream, &root, &errors);

	size_t limit = root.get("limit", 20).asUInt();

	HandleSeq focus = _logic->get_attention_bank()->get_attentional_focus();

	Json::Value result(Json::arrayValue);
	for (size_t i = 0; i < std::min(focus.size(), limit); i++) {
		auto av = _logic->get_attention_bank()->get_attention(focus[i]);
		Json::Value item;
		item["atom"] = focus[i]->to_short_string();
		item["sti"] = av.sti;
		item["lti"] = av.lti;
		result.append(item);
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::do_spread_attention(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle source = parse_atom(root["source"].asString());
	_logic->get_attention_bank()->spread_attention(source);

	return format_success("Spread attention from " + source->to_short_string());
}

std::string McpPlugTensorLogic::do_attention_query(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Json::Value query_ref = root["query"];
	TensorValuePtr query;
	if (query_ref.isMember("atom") && query_ref.isMember("key")) {
		Handle atom = parse_atom(query_ref["atom"].asString());
		Handle qkey = parse_atom(query_ref["key"].asString());
		query = _logic->get_aten_space()->get_tensor(atom, qkey);
	}

	Handle key = parse_atom(root["key"].asString());
	size_t k = root.get("k", 10).asUInt();

	auto results = _logic->get_attention_bank()->attention_weighted_query(query, key, k);

	Json::Value result_array(Json::arrayValue);
	for (const auto& [h, score] : results) {
		Json::Value item;
		item["atom"] = h->to_short_string();
		item["score"] = score;
		result_array.append(item);
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result_array);
}

std::string McpPlugTensorLogic::do_attention_cycle(const std::string& args) const
{
	_logic->attention_cycle();
	return format_success("Attention cycle completed");
}

// ============================================================
// Inference Operations
// ============================================================

std::string McpPlugTensorLogic::do_deduction(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle p1 = parse_atom(root["premise1"].asString());
	Handle p2 = parse_atom(root["premise2"].asString());

	auto [conclusion, tv] = _logic->tensor_deduction(p1, p2);

	Json::Value result;
	result["conclusion"] = conclusion->to_short_string();
	result["strength"] = tv.strength();
	result["confidence"] = tv.confidence();

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::do_induction(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	HandleSeq instances;
	for (const auto& i : root["instances"]) {
		instances.push_back(parse_atom(i.asString()));
	}

	auto [generalization, tv] = _logic->tensor_induction(instances);

	Json::Value result;
	if (generalization)
		result["generalization"] = generalization->to_short_string();
	result["strength"] = tv.strength();
	result["confidence"] = tv.confidence();

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::do_analogy(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle A = parse_atom(root["A"].asString());
	Handle B = parse_atom(root["B"].asString());
	Handle C = parse_atom(root["C"].asString());

	HandleSeq candidates;
	for (const auto& c : root["candidates"]) {
		candidates.push_back(parse_atom(c.asString()));
	}

	Handle D = _logic->tensor_analogy(A, B, C, candidates);

	Json::Value result;
	result["A"] = A->to_short_string();
	result["B"] = B->to_short_string();
	result["C"] = C->to_short_string();
	if (D) {
		result["D"] = D->to_short_string();
	} else {
		result["D"] = Json::nullValue;
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

std::string McpPlugTensorLogic::do_unify(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	Handle pattern = parse_atom(root["pattern"].asString());

	HandleSeq candidates;
	for (const auto& c : root["candidates"]) {
		candidates.push_back(parse_atom(c.asString()));
	}

	double threshold = root.get("threshold", 0.5).asDouble();

	auto bindings = _logic->tensor_unify(pattern, candidates, threshold);

	Json::Value result(Json::arrayValue);
	for (const auto& binding : bindings) {
		Json::Value b;
		for (const auto& [var, val] : binding) {
			b[var->to_short_string()] = val->to_short_string();
		}
		result.append(b);
	}

	Json::StreamWriterBuilder writer;
	return Json::writeString(writer, result);
}

// ============================================================
// Utility Operations
// ============================================================

std::string McpPlugTensorLogic::do_initialize(const std::string& args) const
{
	Json::Value root;
	Json::CharReaderBuilder builder;
	std::string errors;
	std::istringstream stream(args);
	if (!Json::parseFromStream(builder, stream, &root, &errors)) {
		return format_error("Invalid JSON: " + errors);
	}

	std::vector<Type> types;
	for (const auto& t : root["types"]) {
		Type type = nameserver().getType(t.asString());
		if (type != NOTYPE) {
			types.push_back(type);
		}
	}

	std::string init = root.get("init", "randn").asString();

	_logic->initialize_embeddings(types, init);

	return format_success("Initialized embeddings for " +
	                      std::to_string(types.size()) + " types");
}

std::string McpPlugTensorLogic::do_stats(const std::string& args) const
{
	return _logic->stats();
}

#endif // HAVE_ATEN
