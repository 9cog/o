/*
 * McpPlugTensorLogic.h
 *
 * MCP Plugin for Multi-Entity & Multi-Scale Network-Aware Tensor Logic
 *
 * Copyright (c) 2025 OpenCog Foundation
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation.
 */

#ifndef _OPENCOG_MCP_PLUG_TENSOR_LOGIC_H
#define _OPENCOG_MCP_PLUG_TENSOR_LOGIC_H

#ifdef HAVE_ATEN

#include <opencog/cogserver/mcp-tools/McpPlugin.h>
#include "TensorLogic.h"

namespace opencog {

/**
 * MCP Plugin providing Tensor Logic operations as MCP tools.
 *
 * This plugin exposes the Multi-Entity & Multi-Scale Network-Aware
 * Tensor Logic functionality through MCP, enabling LLMs to:
 *
 * Multi-Entity Operations:
 * - tensorEntityBatch: Create batch embeddings for entity sets
 * - tensorAggregateEntities: Aggregate entity embeddings (attention, deepsets)
 * - tensorClusterEntities: Cluster entities by embedding similarity
 *
 * Multi-Scale Operations:
 * - tensorMultiscaleCompute: Compute multi-scale embeddings
 * - tensorQueryAtScale: Query at specific scale level
 * - tensorCrossScaleSimilarity: Compare atoms at different scales
 *
 * Network-Aware Operations:
 * - tensorNetworkEmbed: Compute network-aware embeddings
 * - tensorPropagate: Message-passing embedding propagation
 * - tensorRelationEmbed: Compute relationship embeddings
 *
 * Attention Operations:
 * - tensorStimulate: Stimulate atom attention
 * - tensorGetFocus: Get attentional focus
 * - tensorSpreadAttention: Spread attention through graph
 * - tensorAttentionQuery: Attention-weighted similarity query
 *
 * Inference Operations:
 * - tensorDeduction: Tensor-enhanced deduction
 * - tensorInduction: Tensor-enhanced induction
 * - tensorAnalogy: Analogy by embedding similarity
 * - tensorUnify: Tensor-guided pattern matching
 */
class McpPlugTensorLogic : public McpPlugin
{
private:
	std::shared_ptr<TensorLogic> _logic;
	AtomSpace* _as;

	// Tool implementations - Multi-Entity
	std::string do_entity_batch(const std::string& args) const;
	std::string do_aggregate_entities(const std::string& args) const;
	std::string do_similarity_matrix(const std::string& args) const;
	std::string do_cluster_entities(const std::string& args) const;

	// Tool implementations - Multi-Scale
	std::string do_multiscale_compute(const std::string& args) const;
	std::string do_query_at_scale(const std::string& args) const;
	std::string do_cross_scale_similarity(const std::string& args) const;

	// Tool implementations - Network-Aware
	std::string do_network_embed(const std::string& args) const;
	std::string do_propagate(const std::string& args) const;
	std::string do_relation_embed(const std::string& args) const;

	// Tool implementations - Attention
	std::string do_stimulate(const std::string& args) const;
	std::string do_get_focus(const std::string& args) const;
	std::string do_spread_attention(const std::string& args) const;
	std::string do_attention_query(const std::string& args) const;
	std::string do_attention_cycle(const std::string& args) const;

	// Tool implementations - Inference
	std::string do_deduction(const std::string& args) const;
	std::string do_induction(const std::string& args) const;
	std::string do_analogy(const std::string& args) const;
	std::string do_unify(const std::string& args) const;

	// Tool implementations - Utility
	std::string do_initialize(const std::string& args) const;
	std::string do_stats(const std::string& args) const;

	// Helpers
	Handle parse_atom(const std::string& atomese) const;
	HandleSeq parse_atom_list(const std::string& json_array) const;
	ScaleLevel parse_scale(const std::string& scale_str) const;
	std::string format_error(const std::string& msg) const;
	std::string format_success(const std::string& result) const;
	std::string format_tensor(const TensorValuePtr& tv) const;

public:
	McpPlugTensorLogic(std::shared_ptr<TensorLogic> logic, AtomSpace* as);
	virtual ~McpPlugTensorLogic() = default;

	virtual std::string get_tool_descriptions() const override;
	virtual std::string invoke_tool(const std::string& tool_name,
	                                const std::string& arguments) const override;
};

} // namespace opencog

#endif // HAVE_ATEN
#endif // _OPENCOG_MCP_PLUG_TENSOR_LOGIC_H
