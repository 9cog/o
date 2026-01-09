# ATen/PyTorch Integration for CogServer

This module provides integration between OpenCog's AtomSpace and PyTorch's
ATen tensor library, enabling neuro-symbolic AI by bridging symbolic
knowledge graphs with neural network computations.

Implements **Multi-Entity & Multi-Scale Network-Aware Tensor Logic**
following the [ATen](https://github.com/o9nn/ATen) and
[ATenSpace](https://github.com/o9nn/ATenSpace) architectural patterns.

## Overview

The ATenSpace extension provides:

- **TensorValue**: AtomSpace Value type wrapping ATen tensors
- **ATenSpace**: High-level API for tensor operations on knowledge graphs
- **TensorLogic**: Multi-entity, multi-scale, network-aware reasoning
- **TensorAttentionBank**: ECAN-style attention allocation with tensors
- **MCP Plugins**: JSON-based tools for LLM interaction
- **Scheme Bindings**: Guile/Scheme API for all operations

## Key Features

### Multi-Entity Operations
- Create batch embeddings for entity collections
- Aggregate entities using attention, DeepSets, max/mean pooling
- Cluster entities by embedding similarity
- Compute entity similarity matrices

### Multi-Scale Embeddings
Hierarchical representations at different granularities:
- **LOCAL**: Direct node features
- **NEIGHBOR**: 1-hop neighborhood aggregation
- **COMMUNITY**: Cluster-level context (2-hop)
- **GLOBAL**: Whole-graph summary

### Network-Aware Tensors
Graph-structure-aware embeddings that encode:
- Degree centrality
- Type information
- Positional encodings
- Message-passing aggregation (GNN-style)

### ECAN Attention Allocation
Economic Attention Networks with tensors:
- Short-Term Importance (STI) / Long-Term Importance (LTI)
- Attention spreading via Hebbian links
- Attention-weighted similarity queries
- Cognitive-inspired focus management

### Tensor Logic Inference
- **Tensor Deduction**: (A→B) + (B→C) ⇒ (A→C) with tensor truth values
- **Tensor Induction**: Generalize from instances using embeddings
- **Tensor Analogy**: A:B :: C:? by embedding arithmetic
- **Tensor Unification**: Pattern matching guided by similarity

## Building with ATen Support

### Prerequisites

Install PyTorch (CPU or CUDA version):

```bash
# CPU only
pip install torch

# With CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Or download LibTorch from https://pytorch.org/get-started/locally/

### CMake Configuration

```bash
mkdir build && cd build
cmake -DWITH_ATEN=ON ..
make -j
sudo make install
```

## Usage

### Scheme API

```scheme
(use-modules (opencog aten))

;; Basic tensor operations
(define emb (cog-new-tensor '(128) #:init 'randn))
(cog-set-tensor! (Concept "cat") (Predicate "embedding") emb)

;; Multi-scale embeddings
(cog-multiscale-embed (Concept "cat") (Predicate "embedding"))

;; Network-aware embeddings (GNN-style)
(cog-network-embed (Concept "cat") (Predicate "embedding") 2)

;; Attention-guided reasoning
(cog-stimulate (Concept "cat") 2.0)
(cog-spread-attention (Concept "cat"))
(cog-get-focus)  ; Get high-attention atoms

;; Tensor analogy: king - man + woman = ?
(cog-tensor-analogy
  (Concept "man") (Concept "king")
  (Concept "woman")
  (list (Concept "queen") (Concept "princess")))

;; Cluster entities by embedding
(cog-cluster-entities
  (list (Concept "cat") (Concept "dog") (Concept "fish"))
  (Predicate "embedding")
  2)  ; 2 clusters
```

### MCP Tools

**Basic Tensor Tools:**
- `tensorCreate`, `tensorGet`, `tensorSet`
- `tensorOp` (matmul, transpose, reshape)
- `tensorMath` (add, sub, mul, div)
- `tensorActivation` (relu, sigmoid, softmax)

**TensorLogic Tools:**
- `tensorEntityBatch`: Batch embeddings for entities
- `tensorAggregateEntities`: Aggregate with attention/deepsets
- `tensorClusterEntities`: K-means clustering
- `tensorMultiscaleCompute`: Multi-scale embeddings
- `tensorQueryAtScale`: Query at specific scale
- `tensorNetworkEmbed`: GNN-style embedding
- `tensorPropagate`: Message passing
- `tensorStimulate`: ECAN attention
- `tensorGetFocus`: Attentional focus
- `tensorDeduction`: Tensor-enhanced inference
- `tensorAnalogy`: A:B :: C:D analogy

### C++ API

```cpp
#include <opencog/cogserver/aten/TensorLogic.h>

using namespace opencog;

// Create TensorLogic instance
auto logic = createTensorLogic(&atomspace, 128);

// Multi-scale embeddings
auto mst = logic->compute_multiscale_embedding(atom, feature_key);
TensorValuePtr local = mst->get_scale(ScaleLevel::LOCAL);
TensorValuePtr neighbor = mst->get_scale(ScaleLevel::NEIGHBOR);

// Network-aware embedding
TensorValuePtr net_emb = logic->compute_network_embedding(atom, key, 2);

// Attention operations
logic->get_attention_bank()->stimulate(atom, 1.0);
logic->get_attention_bank()->spread_attention(atom);
HandleSeq focus = logic->get_attention_bank()->get_attentional_focus();

// Tensor analogy
Handle D = logic->tensor_analogy(A, B, C, candidates);

// Entity clustering
auto clusters = logic->cluster_entities(entities, key, 5);
```

## Architecture

```
                        ┌─────────────────────┐
                        │     AtomSpace       │
                        │  (Knowledge Graph)  │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  TensorValue    │    │   TensorLogic       │    │ AttentionBank   │
│  (Value wrap)   │    │ (Reasoning Engine)  │    │ (ECAN-style)    │
└────────┬────────┘    └──────────┬──────────┘    └────────┬────────┘
         │                        │                        │
         │              ┌─────────┴─────────┐              │
         │              │                   │              │
         ▼              ▼                   ▼              │
┌─────────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  ATenSpace      │  │ MultiScale  │  │ Network     │     │
│  (Tensor Ops)   │  │ Tensor      │  │ AwareTensor │     │
└────────┬────────┘  └─────────────┘  └─────────────┘     │
         │                                                 │
         └─────────────────────┬───────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    ATen/PyTorch     │
                    │  (Tensor Library)   │
                    └─────────────────────┘
```

## Key Concepts

### Multi-Entity Reasoning
Process collections of entities simultaneously:
- Batch embeddings for efficiency
- Set-level representations (DeepSets, attention pooling)
- Pairwise similarity matrices
- Entity clustering for structure discovery

### Multi-Scale Representations
Different levels of context:
- **Local**: What is this atom itself?
- **Neighbor**: What is directly connected?
- **Community**: What cluster does it belong to?
- **Global**: How does it fit the whole graph?

### Network-Aware Embeddings
Encode graph structure into tensors:
- Type embeddings for different atom types
- Positional encodings from graph structure
- Graph attention for neighbor aggregation
- Message-passing for iterative refinement

### ECAN Attention
Cognitive-inspired attention allocation:
- STI (Short-Term Importance): Current focus
- LTI (Long-Term Importance): Persistent relevance
- Hebbian links: Strengthen co-activated atoms
- Attention spreading: Propagate through graph

## Files

| File | Description |
|------|-------------|
| `TensorValue.h/cc` | AtomSpace Value wrapping at::Tensor |
| `ATenSpace.h/cc` | Core tensor operations API |
| `TensorLogic.h/cc` | Multi-entity/scale/network reasoning |
| `McpPlugATen.h/cc` | Basic tensor MCP tools |
| `McpPlugTensorLogic.h/cc` | TensorLogic MCP tools |
| `ATenSCM.cc` | Guile/Scheme bindings |
| `aten.scm` | Scheme module definition |

## Dependencies

- **PyTorch/LibTorch**: Core tensor library
- **jsoncpp**: JSON parsing for MCP
- **Guile 3.0+**: Scheme bindings (optional)

## References

- [ATen](https://github.com/o9nn/ATen) - C++11 tensor library
- [ATenSpace](https://github.com/o9nn/ATenSpace) - Symbolic-neural bridge
- [AtomSpace](https://wiki.opencog.org/w/AtomSpace) - Knowledge graph
- [ECAN](https://wiki.opencog.org/w/ECAN) - Economic Attention Networks
- [PyTorch](https://pytorch.org/docs/) - Deep learning framework
