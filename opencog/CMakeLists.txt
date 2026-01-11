
# Unified OpenCog CMakeLists.txt
# Combines cogutil, atomspace, and cogserver components

# Include directories for the unified build
# CMAKE_SOURCE_DIR is needed for includes like <opencog/util/platform.h>
# CMAKE_BINARY_DIR is needed for generated atom_types.h
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

# Enable testing without installing first.
# Must be declared early as subdirectories use it.
IF (HAVE_GUILE)
    DECLARE_GUILE_CONFIG_TARGET(SCM_CONFIG "opencog cs-config" "OPENCOG_TEST")
ENDIF (HAVE_GUILE)

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
IF (HAVE_GUILE)
    ADD_SUBDIRECTORY(guile)
    ADD_SUBDIRECTORY(scm)
ENDIF (HAVE_GUILE)
# ADD_SUBDIRECTORY(sheaf) -- no CMakeLists.txt, documentation only
ADD_SUBDIRECTORY(cython)

# ============================================================
# cogserver components
# ============================================================
ADD_SUBDIRECTORY(network)
ADD_SUBDIRECTORY(cogserver)

IF (HAVE_GUILE)
    WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/cs-config.scm SCM_CONFIG TRUE)
    WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/cs-config-installable.scm SCM_CONFIG FALSE)
    INSTALL(FILES ${GUILE_BIN_DIR}/opencog/cs-config-installable.scm DESTINATION ${GUILE_SITE_DIR}/opencog RENAME cs-config.scm)
ENDIF (HAVE_GUILE)
