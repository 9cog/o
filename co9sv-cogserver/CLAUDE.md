# CLAUDE.md - OpenCog CogServer Project Guide

## Project Overview

The **OpenCog CogServer** (v3.4.0) is a multi-protocol network server for the [AtomSpace](https://github.com/opencog/atomspace) hypergraph database. It provides real-time, multi-user access to AtomSpace via telnet, WebSocket, HTTP, and MCP (Model Context Protocol) interfaces.

## Build Commands

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
make check              # Run tests
```

### Build Options
- `CMAKE_BUILD_TYPE`: Release (default), Debug, Coverage, Profile, RelWithDebInfo
- Requires: CogUtil >= 2.2.1, AtomSpace >= 5.2.0, AtomSpaceStorage >= 4.3.0, ASIO
- Optional: OpenSSL (WebSockets), jsoncpp (MCP), Python3 (Python shell), Guile 3.0+ (Scheme shell)

## Repository Structure

```
cogserver/
├── opencog/
│   ├── network/              # Network layer (TCP, WebSocket, HTTP)
│   │   ├── NetworkServer.h/cc    # Main server thread management
│   │   ├── ServerSocket.h/cc     # Per-client connection handler
│   │   ├── WebSocket.h/cc        # WebSocket protocol
│   │   └── GenericShell.h/cc     # Base shell class
│   ├── cogserver/
│   │   ├── server/           # Core server
│   │   │   ├── CogServer.h/cc    # Main server class
│   │   │   ├── Module.h          # Plugin interface
│   │   │   └── ModuleManager.h/cc # Dynamic module loading
│   │   ├── shell/            # Shell implementations
│   │   │   ├── SchemeShell.*     # Guile REPL
│   │   │   ├── PythonShell.*     # Python REPL
│   │   │   ├── JsonShell.*       # JSON interface
│   │   │   ├── SexprShell.*      # S-expression bulk transfer
│   │   │   └── McpShell.*        # Model Context Protocol
│   │   ├── atoms/            # Atom types (CogServerNode)
│   │   ├── types/            # Type system definitions
│   │   ├── mcp-eval/         # MCP protocol evaluator
│   │   ├── mcp-tools/        # MCP plugin system
│   │   ├── mcp-docs/         # MCP documentation resources
│   │   ├── aten/             # ATen tensor integration (NEW)
│   │   ├── scm/              # Scheme bindings
│   │   └── python/           # Python module
├── examples/                 # Usage examples
├── tests/                    # Unit tests
└── cmake/                    # CMake configuration
```

## Key Architectural Patterns

### Multi-User Architecture
- All clients share the same AtomSpace (thread-safe)
- Each client gets its own ServerSocket + thread
- Changes are immediately visible to all users

### Plugin System
1. **MCP Plugins** (recommended): JSON-based tool interface
   - Derive from `McpPlugin` class
   - Implement `get_tool_descriptions()` and `invoke_tool()`

2. **Legacy Modules** (deprecated): Use Scheme/Python instead
   - Use `DECLARE_MODULE` macro

### Shell Architecture
- `GenericShell` → spawns eval thread → `GenericEval` subclass
- Input via `concurrent_queue`, poll results with `poll_result()`

## ATenSpace Integration

### Overview
The ATenSpace extension bridges PyTorch's ATen tensor library with OpenCog's AtomSpace, enabling:
- Tensor storage and retrieval as AtomSpace Values
- Neural network integration with symbolic reasoning
- GPU-accelerated tensor operations within the knowledge graph

### TensorValue
A new Value type extending AtomSpace's Value system:
```cpp
// Store tensor data on an atom
TensorValue tv({3, 4}, torch::kFloat32);  // 3x4 float tensor
atom->setValue(key, createTensorValue(tv));

// Retrieve tensor
TensorValuePtr tvp = TensorValueCast(atom->getValue(key));
at::Tensor tensor = tvp->tensor();
```

### MCP Tools for Tensors
- `createTensor`: Create tensors of various shapes/types
- `tensorOp`: Matrix operations (matmul, transpose, etc.)
- `tensorMath`: Element-wise math (add, mul, etc.)
- `getTensor`: Retrieve tensor data
- `tensorToValue`: Convert between TensorValue and FloatValue

### Scheme API
```scheme
(use-modules (opencog aten))

; Create a tensor
(define t (cog-new-tensor '(2 3) 'float))

; Matrix multiply
(cog-tensor-matmul t1 t2)

; Store on atom
(cog-set-value! atom key (TensorValue t))
```

## Default Ports
- **17001**: Console (telnet shell)
- **18080**: Web server (HTTP + WebSocket)
- **18888**: MCP server

## Common Development Tasks

### Adding a New MCP Tool
1. Create plugin in `mcp-tools/` deriving from `McpPlugin`
2. Implement `get_tool_descriptions()` returning JSON schema
3. Implement `invoke_tool()` for command handling
4. Register in `McpEval.cc`
5. Add to `mcp-tools/CMakeLists.txt`

### Adding New Value Types
1. Define class inheriting from `Value` in AtomSpace
2. Implement serialization/deserialization
3. Add type to `types/cogserver_types.script` if needed
4. Create MCP tools for manipulation
5. Add Scheme/Python bindings

### Testing
```bash
cd build
make check              # All tests
ctest -R <pattern>      # Specific tests
```

## Dependencies for ATen Support
```bash
# PyTorch/LibTorch
pip install torch  # or download libtorch from pytorch.org

# CMake detection
cmake -DWITH_ATEN=ON ..
```

## Code Style
- C++17 standard
- OpenCog coding conventions (see cogutil)
- Use `opencog::` namespace
- Prefer smart pointers (`std::shared_ptr`)
- Thread-safe by design

## Important Notes
- No authentication by default - wrap with auth server if needed
- S-expression format used for all Atomese I/O
- Values are attached to Atoms but NOT stored in AtomSpace graph
- Use MCP or Scheme/Python for new functionality (avoid legacy Request system)
