# OpenCog AtomSpace - Claude Code Guide

This document provides an overview of the OpenCog AtomSpace codebase for AI-assisted development.

## Project Overview

OpenCog AtomSpace is an in-RAM knowledge representation (KR) database with an associated query engine and graph-rewriting system. It is a generalized hypergraph (metagraph) database designed for AI/AGI applications.

**Version**: 5.2.0
**License**: AGPL-3.0-or-later

## Directory Structure

```
/home/user/co9as-atomspace/
├── opencog/                  # Main source code
│   ├── atoms/               # Atom type implementations
│   │   ├── atom_types/      # Type system definitions (atom_types.script)
│   │   ├── aten/            # ATen tensor integration (NEW)
│   │   ├── base/            # Base classes: Atom, Node, Link, Handle
│   │   ├── value/           # Value classes: FloatValue, StringValue, etc.
│   │   ├── core/            # Core atoms: NumberNode, FunctionLink
│   │   ├── scope/           # Lambda, variable scoping
│   │   ├── pattern/         # Pattern matching
│   │   ├── flow/            # Data flow: ValueOfLink, SetValueLink
│   │   ├── columnvec/       # Column operations for GPU
│   │   ├── execution/       # Execution: ExecutionOutputLink
│   │   ├── reduct/          # Arithmetic: PlusLink, TimesLink
│   │   └── ...              # Other atom modules
│   ├── atomspace/           # AtomSpace database implementation
│   ├── query/               # Query engine
│   ├── eval/                # Evaluation system
│   ├── guile/               # Scheme/Guile bindings
│   ├── cython/              # Python/Cython bindings
│   └── persist/             # Persistence layer
├── tests/                   # Test suite (CxxTest)
├── examples/                # Usage examples
├── cmake/                   # CMake build configuration
└── Design-Notes-*.md        # Design documentation
```

## Core Architecture

### Class Hierarchy

```
Value (base class)
├── Atom
│   ├── Node (has name)
│   └── Link (has outgoing set)
└── FloatValue, StringValue, LinkValue, etc.
```

### Key Classes

- **Value** (`opencog/atoms/value/Value.h`): Base class for all values
- **Atom** (`opencog/atoms/base/Atom.h`): Base class for all atoms
- **Node** (`opencog/atoms/base/Node.h`): Atom with a name
- **Link** (`opencog/atoms/base/Link.h`): Atom with an outgoing set
- **Handle** (`opencog/atoms/base/Handle.h`): Smart pointer wrapper
- **AtomSpace** (`opencog/atomspace/AtomSpace.h`): Database container

## Naming Conventions

### Classes & Types
- **Classes**: CamelCase (e.g., `FloatValue`, `NumberNode`, `PlusLink`)
- **Type enums**: UPPER_SNAKE_CASE (e.g., `FLOAT_VALUE`, `NUMBER_NODE`)
- **Pointer types**: `XxxPtr` (e.g., `FloatValuePtr`, `AtomSpacePtr`)

### Macros

```cpp
// For Value subclasses
VALUE_PTR_DECL(ClassName)      // Creates ClassNamePtr typedef and ClassNameCast
CREATE_VALUE_DECL(ClassName)   // Creates createClassName factory

// For Atom subclasses
ATOM_PTR_DECL(ClassName)       // Creates ClassNamePtr typedef and ClassNameCast
NODE_PTR_DECL(ClassName)       // Same as ATOM_PTR_DECL
LINK_PTR_DECL(ClassName)       // Same as ATOM_PTR_DECL
```

### Namespace
- All code is in the `opencog` namespace

## Adding New Types

### 1. Define Type in atom_types.script

```
// In opencog/atoms/atom_types/atom_types.script
MY_NEW_VALUE <- VALUE,FLOAT_VEC_ARG
MY_NEW_LINK <- FUNCTION_LINK,NUMERIC_OUTPUT_SIG
```

### 2. Create Header File

```cpp
// MyNewValue.h
#ifndef _OPENCOG_MY_NEW_VALUE_H
#define _OPENCOG_MY_NEW_VALUE_H

#include <opencog/atoms/value/Value.h>

namespace opencog {

class MyNewValue : public Value
{
protected:
    // Data members
public:
    MyNewValue(/* args */);
    virtual ~MyNewValue() {}

    virtual std::string to_string(const std::string& indent) const;
    virtual bool operator==(const Value&) const;
};

VALUE_PTR_DECL(MyNewValue)
CREATE_VALUE_DECL(MyNewValue)

} // namespace opencog
#endif
```

### 3. Create Source File

```cpp
// MyNewValue.cc
#include <opencog/atoms/value/MyNewValue.h>
#include <opencog/atoms/value/ValueFactory.h>

using namespace opencog;

// Implementation...

// Register factory
DEFINE_VALUE_FACTORY(MY_NEW_VALUE, createMyNewValue, /* args */)
```

### 4. Update CMakeLists.txt

Add the new source file to the library and header to install list.

## Build System

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make test
sudo make install
```

### Dependencies
- CogUtil 2.2.1+
- CMake 3.12+
- Boost
- Guile 3.0+ (optional)
- Python 3.8+ (optional)
- CxxTest (for tests)

## Testing

Tests are in the `tests/` directory using CxxTest framework.

```bash
cd build
make test                    # Run all tests
ctest -R MyTest             # Run specific test
```

## Key Patterns

### Creating Atoms

```cpp
// Create nodes
Handle h = as->add_node(CONCEPT_NODE, "foo");

// Create links
Handle l = as->add_link(LIST_LINK, h1, h2);

// Create values
FloatValuePtr fv = createFloatValue({1.0, 2.0, 3.0});
```

### Attaching Values to Atoms

```cpp
Handle key = as->add_node(PREDICATE_NODE, "my-key");
atom->setValue(key, createFloatValue({1.0, 2.0}));
ValuePtr v = atom->getValue(key);
```

### Executing Links

```cpp
ValuePtr result = link->execute(atomspace);
```

## ATen Integration & Tensor Logic

The `opencog/atoms/aten/` directory provides comprehensive tensor integration based on
[o9nn/ATen](https://github.com/o9nn/ATen) and [o9nn/ATenSpace](https://github.com/o9nn/ATenSpace).

This creates a **hybrid symbolic-neural AI architecture** that bridges:
- Symbolic knowledge representation (AtomSpace hypergraph)
- Neural tensor embeddings (ATen tensor operations)

### Core Components

#### ATenValue (`ATenValue.h/.cc`)
Value type wrapping ATen tensors with fallback implementation:
```cpp
// Create tensors
ATenValuePtr tensor = createATenFromVector({1.0, 2.0, 3.0}, {3});
ATenValuePtr zeros = createATenZeros({128});
ATenValuePtr random = createATenRandom({64, 64});

// Tensor operations
ValuePtr result = a->add(*b);
ValuePtr product = a->matmul(*b);
ValuePtr activated = tensor->relu();
```

#### ATenSpace (`ATenSpace.h/.cc`)
Bridge between symbolic AI and neural embeddings:
```cpp
ATenSpace space(atomspace, 128); // 128-dim embeddings

// Create concept with embedding
Handle cat = space.create_concept_node("cat", random_embedding);

// Query similar atoms
HandleSeq similar = space.query_similar(cat, 10);

// Compute similarity
double sim = space.similarity(cat, dog);
```

#### TensorLogic (`TensorLogic.h/.cc`)
Multi-entity & multi-scale network-aware operations:

- **EntityEmbedding**: Maps atoms to dense vectors
- **MultiScaleTensor**: Hierarchical tensor representations
- **NetworkAwareTensor**: Graph-aware message passing
- **TruthValueTensor**: PLN integration with tensors

```cpp
TensorLogic logic(atomspace, 128, 3); // 128-dim, 3 scales

// Entity embeddings
logic.set_embedding(atom, tensor);
auto emb = logic.get_embedding(atom);

// Similarity search
HandleSeq similar = logic.query_similar(query, 10);

// Network operations
logic.build_network_tensor();
logic.message_passing(2); // 2 hops
logic.hebbian_update(0.01);
```

#### TensorLink (`TensorLink.h/.cc`)
Executable links for tensor operations:
- `TensorAddLink`, `TensorSubLink`, `TensorMulLink`, `TensorDivLink`
- `TensorMatmulLink`, `TensorTransposeLink`, `TensorReshapeLink`
- `TensorReluLink`, `TensorSigmoidLink`, `TensorTanhLink`, `TensorSoftmaxLink`
- `TensorSumLink`, `TensorMeanLink`
- `TensorOfLink`, `SetTensorLink`

#### HebbianLink
ECAN-style Hebbian learning connections:
```cpp
Handle hebb = space.create_hebbian_link(source, target, 1.0);
space.hebbian_update(0.01); // Learning rate
space.spread_activation(source, 10.0, 2); // Spread activation 2 hops
```

### Building with ATen

The module auto-detects PyTorch or o9nn/ATen:
```bash
# With PyTorch
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Or falls back to CPU-only implementation
cmake ..
```

### Key Patterns

#### Creating Tensor-Enhanced Atoms
```cpp
ATenSpace space(atomspace, 128);

// Create concept with embedding
Handle concept = space.create_concept_node("entity", embedding);

// Create link with aggregated embedding
Handle link = space.create_link(INHERITANCE_LINK, {child, parent}, "mean");
```

#### Semantic Similarity Search
```cpp
// Find similar atoms
HandleSeq similar = space.query_similar(query_atom, k);

// Pattern match using embeddings
HandleSeq matches = space.pattern_match_embedding(pattern, 0.8);
```

#### Neural-Symbolic Inference
```cpp
TensorLogic logic(atomspace);

// Combine embeddings with PLN
Handle conclusion = space.tensor_deduction(premise1, premise2);

// Attention-guided pattern matching
HandleSeq results = logic.attention_query(pattern, 100);
```

## Common Pitfalls

1. **Always use Handle/ValuePtr** - Never store raw pointers to Atoms
2. **Atoms are immutable** - Create new atoms instead of modifying
3. **Type checking** - Use `is_type()` or `isa()` for type checks
4. **Thread safety** - AtomSpace operations are thread-safe

## Resources

- Wiki: http://wiki.opencog.org/
- GitHub: https://github.com/opencog/atomspace
- Design Notes: See `Design-Notes-*.md` files in repo root
