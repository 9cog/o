
# Unified OpenCog CMakeLists.txt
# Combines cogutil, atomspace, and cogserver components

# The atom_types.h file is written to the build directory
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

# ============================================================
# cogutil components
# ============================================================
ADD_SUBDIRECTORY(util)

# ============================================================
# atomspace components
# ============================================================
ADD_SUBDIRECTORY(atoms)
ADD_SUBDIRECTORY(atomspace)
ADD_SUBDIRECTORY(persist)
ADD_SUBDIRECTORY(query)
ADD_SUBDIRECTORY(eval)
ADD_SUBDIRECTORY(guile)
ADD_SUBDIRECTORY(scm)
ADD_SUBDIRECTORY(sheaf)
ADD_SUBDIRECTORY(cython)

# ============================================================
# cogserver components
# ============================================================
# Enable testing without installing first.
DECLARE_GUILE_CONFIG_TARGET(SCM_CONFIG "opencog cs-config" "OPENCOG_TEST")

ADD_SUBDIRECTORY(network)
ADD_SUBDIRECTORY(cogserver)

WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/cs-config.scm SCM_CONFIG TRUE)
WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/cs-config-installable.scm SCM_CONFIG FALSE)
INSTALL(FILES ${GUILE_BIN_DIR}/opencog/cs-config-installable.scm DESTINATION ${GUILE_SITE_DIR}/opencog RENAME cs-config.scm)
