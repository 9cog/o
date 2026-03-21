#
# Build the RocksDB driver for the AtomSpace
#

ADD_LIBRARY (persist-rocks SHARED
	RocksDAG.cc
	RocksFrame.cc
	RocksIO.cc
	RocksStorage.cc
	RocksPersistSCM.cc
)

# The atom_types.h file is written to the build directory
TARGET_INCLUDE_DIRECTORIES(persist-rocks PRIVATE ${CMAKE_BINARY_DIR})

TARGET_LINK_LIBRARIES(persist-rocks
	persist-rocks-types
	${ATOMSPACE_STORAGE_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
	rocksdb
)

# In unified builds, ATOMSPACE_LIBRARIES / ATOMSPACE_STORAGE_LIBRARIES may
# be unset; add the cmake targets directly so the required symbols are
# available without polluting the install-time export interface.
TARGET_LINK_LIBRARIES(persist-rocks PRIVATE
	persist
	smob
	sexpr
)

ADD_DEPENDENCIES(persist-rocks persist_rocks_atom_types)

ADD_GUILE_EXTENSION(SCM_CONFIG persist-rocks
	"opencog-ext-path-persist-rocks")

# The EXPORT is need to autogenerate CMake boilerplate in the lib
# directory that lets other packages FIND_PACKAGE(AtomSpaceRocks)
INSTALL (TARGETS persist-rocks
	EXPORT AtomSpaceRocksTargets
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES
	RocksStorage.h
	DESTINATION "include/opencog/persist/rocks"
)
