#
# Master OpenCog Unified CMake file.
#
# This file integrates cogutil, atomspace, and cogserver for native build.
# Build order follows the dependency hierarchy:
#   cogutil -> atomspace -> cogserver
#
CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(opencog-unified VERSION 1.0.0)

# Default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Building unified OpenCog stack...")

# Add subdirectories in dependency order
# 1. cogutil - base utilities (no dependencies)
ADD_SUBDIRECTORY(co9as-cogutil)

# 2. atomspace - hypergraph database (depends on cogutil)
ADD_SUBDIRECTORY(co9as-atomspace)

# 3. cogserver - network server (depends on atomspace)
ADD_SUBDIRECTORY(co9sv-cogserver)

MESSAGE(STATUS "Unified OpenCog build configured successfully!")
