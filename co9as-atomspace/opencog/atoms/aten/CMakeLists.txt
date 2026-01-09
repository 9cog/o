# ATen tensor integration for OpenCog AtomSpace
#
# This module provides PyTorch ATen tensor support for AtomSpace.
# Based on the integration patterns from o9nn/ATen and o9nn/ATenSpace.
#
# Key features:
# - ATenValue: Tensor wrapper as OpenCog Value
# - ATenSpace: Bridge between symbolic AI and neural embeddings
# - TensorLogic: Multi-entity & multi-scale network-aware operations
# - HebbianLink: ECAN-style attention and learning
#
# When ATen is available, it uses native PyTorch tensors.
# When ATen is not available, it falls back to a basic implementation.

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

# Try to find PyTorch/ATen (o9nn/ATen or standard PyTorch)
FIND_PACKAGE(Torch QUIET)

IF(Torch_FOUND)
    MESSAGE(STATUS "PyTorch/ATen found: ${TORCH_LIBRARIES}")
    ADD_DEFINITIONS(-DHAVE_ATEN)
    SET(ATEN_LIBRARIES ${TORCH_LIBRARIES})
    INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ELSE()
    # Try to find o9nn/ATen as external library
    FIND_PATH(O9NN_ATEN_INCLUDE_DIR
        NAMES ATen/ATen.h
        PATHS /usr/local/include /usr/include
        PATH_SUFFIXES aten
    )
    FIND_LIBRARY(O9NN_ATEN_LIBRARY
        NAMES aten
        PATHS /usr/local/lib /usr/lib
    )

    IF(O9NN_ATEN_INCLUDE_DIR AND O9NN_ATEN_LIBRARY)
        MESSAGE(STATUS "o9nn/ATen found: ${O9NN_ATEN_LIBRARY}")
        ADD_DEFINITIONS(-DHAVE_ATEN)
        SET(ATEN_LIBRARIES ${O9NN_ATEN_LIBRARY})
        INCLUDE_DIRECTORIES(${O9NN_ATEN_INCLUDE_DIR})
    ELSE()
        MESSAGE(STATUS "ATen not found - using fallback tensor implementation")
        MESSAGE(STATUS "  For GPU acceleration, install PyTorch or o9nn/ATen")
        SET(ATEN_LIBRARIES "")
    ENDIF()
ENDIF()

# Try to find o9nn/ATenSpace
FIND_PATH(O9NN_ATENSPACE_INCLUDE_DIR
    NAMES ATenSpace/AtomSpace.h
    PATHS /usr/local/include /usr/include
    PATH_SUFFIXES atenspace
)
FIND_LIBRARY(O9NN_ATENSPACE_LIBRARY
    NAMES atenspace
    PATHS /usr/local/lib /usr/lib
)

IF(O9NN_ATENSPACE_INCLUDE_DIR AND O9NN_ATENSPACE_LIBRARY)
    MESSAGE(STATUS "o9nn/ATenSpace found: ${O9NN_ATENSPACE_LIBRARY}")
    ADD_DEFINITIONS(-DHAVE_ATENSPACE)
    LIST(APPEND ATEN_LIBRARIES ${O9NN_ATENSPACE_LIBRARY})
    INCLUDE_DIRECTORIES(${O9NN_ATENSPACE_INCLUDE_DIR})
ELSE()
    MESSAGE(STATUS "o9nn/ATenSpace not found - using built-in implementation")
ENDIF()

ADD_LIBRARY(aten
    ATenValue.cc
    ATenSpace.cc
    TensorLink.cc
    TensorLogic.cc
)

# Without this, parallel make will race and crap up the generated files.
ADD_DEPENDENCIES(aten opencog_atom_types)

TARGET_LINK_LIBRARIES(aten
    atom_types
    value
    base
    ${ATEN_LIBRARIES}
    ${COGUTIL_LIBRARY}
)

# Install the library
INSTALL(TARGETS aten EXPORT AtomSpaceTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

# Install headers
INSTALL(FILES
    ATenValue.h
    ATenSpace.h
    TensorLink.h
    TensorLogic.h
    DESTINATION "include/opencog/atoms/aten"
)
