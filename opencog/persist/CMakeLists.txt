# Unified Persist CMakeLists.txt
# Merged from atomspace-storage and atomspace-rocks

# ============================================================
# AtomSpace Storage (base persistence layer)
# ============================================================
ADD_SUBDIRECTORY (storage)
ADD_SUBDIRECTORY (api)
ADD_SUBDIRECTORY (csv)
ADD_SUBDIRECTORY (file)
ADD_SUBDIRECTORY (flow)
ADD_SUBDIRECTORY (json)
ADD_SUBDIRECTORY (prolog)
ADD_SUBDIRECTORY (proxy)
ADD_SUBDIRECTORY (sexcom)
ADD_SUBDIRECTORY (sexpr)
ADD_SUBDIRECTORY (tlb)

# ============================================================
# AtomSpace Rocks (RocksDB backend)
# ============================================================
# The atom_types.h file is written to the build directory
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

IF (HAVE_ROCKSDB)
	ADD_SUBDIRECTORY(rocks-types)
	ADD_SUBDIRECTORY(monospace)
	ADD_SUBDIRECTORY(rocks)
ENDIF (HAVE_ROCKSDB)
