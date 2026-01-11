#
# Master OpenCog Unified CMake file.
#
# This is a unified build integrating:
#   - cogutil: Low-level C++ programming utilities
#   - atomspace: Hypergraph database and query engine
#   - cogserver: Distributed AtomSpace network server
#
# General organization:
# -- check for different compilers, OS'es
# -- search for various required & optional libraries/tools
# -- decide what to build based on above results.
# -- configure various config files.
# -- print pretty summary
#
include(cmake/Summary.cmake)

CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(opencog-unified VERSION 1.0.0)

# Python venv seems to need this.
IF(CMAKE_VERSION VERSION_GREATER 3.31)
   CMAKE_POLICY(SET CMP0177 NEW)
ENDIF(CMAKE_VERSION VERSION_GREATER 3.31)

# ----------------------------------------------------------
# User-modifiable options. Feel free to change these!
#
# Uncomment to be in Release mode [default].
# SET(CMAKE_BUILD_TYPE Release)

# Uncomment to build in debug mode.
# SET(CMAKE_BUILD_TYPE Debug)

# Uncomment to be in coverage testing mode.
# SET(CMAKE_BUILD_TYPE Coverage)

# Uncomment to build in profile mode.
# SET(CMAKE_BUILD_TYPE Profile)

# Uncomment to build in release mode with debug information.
# SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# Default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Building unified OpenCog stack (cogutil + atomspace + cogserver)...")

ADD_DEFINITIONS(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                -DPROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}"
                -DPROJECT_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

# Add the 'cmake' dir to cmake's module search path.
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# Include OpenCog helper macros and options
INCLUDE(OpenCogFunctions)
INCLUDE(OpenCogGccOptions)
INCLUDE(OpenCogInstallOptions)
INCLUDE(OpenCogLibOptions)
INCLUDE(OpenCogTestOptions)
INCLUDE(OpenCogAtomTypes)
INCLUDE(AddCxxtest)

# Find required packages
FIND_PACKAGE(Cxxtest)
INCLUDE(OpenCogFindGuile)
INCLUDE(OpenCogFindPython)

# Include guile macros if available
IF (HAVE_GUILE)
    INCLUDE(OpenCogGuile)
ENDIF (HAVE_GUILE)

# ============================================================
# Check for optional packages
# ============================================================

# RocksDB (optional - for persistent storage backend)
FIND_PACKAGE(RocksDB)
IF (ROCKSDB_FOUND)
	MESSAGE(STATUS "RocksDB found - enabling RocksDB storage backend.")
	SET(HAVE_ROCKSDB 1)
ELSE (ROCKSDB_FOUND)
	MESSAGE(STATUS "RocksDB not found - RocksDB storage backend disabled.")
ENDIF (ROCKSDB_FOUND)

# ============================================================
# Build subdirectories
# ============================================================

# Documentation
ADD_SUBDIRECTORY(doc)

# Configuration files
ADD_SUBDIRECTORY(etc)

# Main OpenCog source (util, atoms, atomspace, cogserver, network)
# Must be before lib, as lib exports targets defined here.
ADD_SUBDIRECTORY(opencog)

# Library configuration (exports targets defined in opencog)
ADD_SUBDIRECTORY(lib)

# Examples
ADD_SUBDIRECTORY(examples)

# Tests
IF (CXXTEST_FOUND)
    ADD_CUSTOM_TARGET(tests)
    ADD_SUBDIRECTORY(tests)
ENDIF (CXXTEST_FOUND)

MESSAGE(STATUS "Unified OpenCog build configured successfully!")
