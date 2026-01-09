/*
 * McpPlugATen.h
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

#ifndef _OPENCOG_MCP_PLUG_ATEN_H
#define _OPENCOG_MCP_PLUG_ATEN_H

#ifdef HAVE_ATEN

#include <opencog/cogserver/mcp-tools/McpPlugin.h>
#include "ATenSpace.h"

namespace opencog {

/**
 * MCP Plugin providing ATen tensor operations as MCP tools.
 *
 * This plugin exposes ATen/PyTorch tensor functionality through the MCP
 * protocol, enabling LLMs and other MCP clients to:
 *
 * - Create and manipulate tensors
 * - Store tensors as Values on AtomSpace atoms
 * - Perform tensor computations (matmul, activations, etc.)
 * - Compute similarities and embeddings
 * - Execute neural network layer operations
 *
 * Tools provided:
 * - tensorCreate: Create new tensors
 * - tensorGet: Retrieve tensor from atom
 * - tensorSet: Store tensor on atom
 * - tensorOp: Matrix operations (matmul, transpose)
 * - tensorMath: Element-wise math operations
 * - tensorActivation: Apply activation functions
 * - tensorSimilarity: Compute similarity between tensors
 * - tensorNearest: Find nearest neighbors by embedding
 * - tensorInfo: Get information about ATen/CUDA status
 */
class McpPlugATen : public McpPlugin
{
private:
	ATenSpace* _aten;
	AtomSpace* _as;

	// Tool implementations
	std::string do_create(const std::string& args) const;
	std::string do_get(const std::string& args) const;
	std::string do_set(const std::string& args) const;
	std::string do_op(const std::string& args) const;
	std::string do_math(const std::string& args) const;
	std::string do_activation(const std::string& args) const;
	std::string do_similarity(const std::string& args) const;
	std::string do_nearest(const std::string& args) const;
	std::string do_info(const std::string& args) const;
	std::string do_to_float_value(const std::string& args) const;
	std::string do_from_float_value(const std::string& args) const;

	// Helpers
	Handle parse_atom(const std::string& atomese) const;
	TensorValuePtr parse_tensor_ref(const std::string& json) const;
	std::string format_tensor(const TensorValuePtr& tv) const;
	std::string format_error(const std::string& msg) const;
	std::string format_success(const std::string& result) const;

public:
	/**
	 * Construct plugin with ATenSpace instance.
	 * @param aten ATenSpace instance (takes ownership)
	 * @param as AtomSpace for atom parsing
	 */
	McpPlugATen(ATenSpace* aten, AtomSpace* as);
	virtual ~McpPlugATen();

	/**
	 * Get the list of tensor tools provided by this plugin.
	 */
	virtual std::string get_tool_descriptions() const override;

	/**
	 * Invoke a tensor tool.
	 * @param tool_name The name of the tool to invoke
	 * @param arguments JSON string containing the arguments
	 * @return JSON string containing the response
	 */
	virtual std::string invoke_tool(const std::string& tool_name,
	                                const std::string& arguments) const override;
};

} // namespace opencog

#endif // HAVE_ATEN
#endif // _OPENCOG_MCP_PLUG_ATEN_H
