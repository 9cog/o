# CLAUDE.MD - OpenCog Utilities (cogutil)

## Project Overview

This is the **OpenCog Utilities** library (`cogutil`) - a foundational C++ utility library providing essential building blocks for OpenCog projects. Version 2.2.1.

## Build Commands

```bash
# Standard build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
make check

# Install (requires sudo)
sudo make install

# Build documentation
make doxygen
```

## Project Structure

```
cogutil/
├── opencog/util/     # Main source code (headers and implementations)
├── tests/util/       # CXXTest unit tests
├── cmake/            # CMake helper modules
├── doc/              # Doxygen documentation
├── scripts/          # Build and utility scripts
└── etc/              # Configuration files
```

## Key Components

### Concurrent Data Structures
- `concurrent_queue.h` - Thread-safe queue with blocking/non-blocking operations
- `concurrent_set.h` - Thread-safe set with atomic operations
- `concurrent_stack.h` - Thread-safe stack implementation
- `pool.h` - Resource pool for reusable objects

### Asynchronous Operations
- `async_buffer.h` - De-duplicating async write buffer
- `async_method_caller.h` - Asynchronous method invocation

### Signal-Slot System
- `sigslot.h` - Lightweight signal-slot mechanism (replaces Boost.Signals2)

### Mathematical Utilities
- `numeric.h` - Entropy, distance metrics, floating-point comparisons
- `random.h` - Random number generation utilities
- `zipf.h` - Zipfian distribution implementation

### ATen-Style Tensor System
- `tensor.h` - Core tensor operations (ATen-inspired)
  - Multi-dimensional arrays with shared storage
  - Factory methods: zeros, ones, rand, randn, eye, arange, linspace
  - Element-wise operations and mathematical functions
  - Reductions: sum, mean, var, min, max, norm
  - Linear algebra: matmul, dot, outer, transpose

### Tensor-Enhanced AtomSpace (ATenSpace Integration)
- `tensor_atom.h` - Symbolic atoms with tensor embeddings
  - TensorAtom: Base class with UUID, TruthValue, AttentionValue, embedding
  - TensorNode: Named entities/concepts with embeddings
  - TensorLink: Hypergraph edges connecting atoms
  - PLN truth values and ECAN attention values
  - Atom types: ConceptNode, PredicateNode, InheritanceLink, etc.

- `tensor_space.h` - Hypergraph container with tensor operations
  - Thread-safe atom storage with indices (by type, name, UUID)
  - Tensor similarity search (k-nearest neighbors)
  - Attention bank for ECAN-style focus management
  - Neighborhood embedding computation

- `tensor_logic.h` - Multi-entity, multi-scale tensor logic
  - Fuzzy logic: fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_implication
  - T-norms: Gödel, product, Lukasiewicz
  - Multi-entity operations: interaction tensors, attention, message passing
  - Multi-scale: aggregation, coarse-to-fine propagation
  - Network-aware: centrality, PageRank, GNN-style updates
  - MultiEntityNetwork: Complete system for multi-entity tensor reasoning

### Core Utilities
- `Logger.h` - Logging framework (DEBUG, INFO, WARN, ERROR, NONE levels)
- `exceptions.h` - StandardException hierarchy
- `oc_assert.h` - Assertion framework with stack traces
- `algorithm.h` - Set operations, for_each, accumulate extensions
- `Counter.h` - Python-style Counter container
- `misc.h` - Tokenization, bitcount utilities

## Coding Style

- Modern C++ (C++11/14/17/20 features)
- Namespace: `opencog`
- Header guards: `#ifndef _OPENCOG_<FILENAME>_H`
- License header: AGPL v3 (see LICENSE file)
- Template-heavy for generic programming
- Prefer header-only implementations where practical

## Dependencies

**Required:**
- CMake 3.12+
- C++11-compatible compiler (GCC recommended)
- CXXTest (for testing)

**Optional:**
- Binutils-dev + libiberty-dev (for pretty stack traces)
- Doxygen (for documentation)
- OpenMP (for parallelism)

**Note:** All Boost dependencies have been removed as of v2.2.1.

## Testing

Tests use the CXXTest framework. Test files are in `tests/util/` with `.cxxtest` extension.

```bash
# Run all tests
make check

# Run specific test
./tests/util/numericUTest
```

## Adding New Components

1. Create header in `opencog/util/`
2. Add implementation (.cc) if needed
3. Update `opencog/util/CMakeLists.txt`:
   - Add .cc files to `ADD_LIBRARY(cogutil ...)`
   - Add .h files to `INSTALL(FILES ...)`
4. Create corresponding test in `tests/util/`
