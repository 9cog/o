#
# ATen/PyTorch integration for CogServer
#
# Provides Multi-Entity & Multi-Scale Network-Aware Tensor Logic
# for the OpenCog AtomSpace, integrating with ATen and ATenSpace patterns.
#

# Source files
SET(ATEN_SOURCES
	TensorValue.cc
	ATenSpace.cc
	TensorLogic.cc
	McpPlugATen.cc
	McpPlugTensorLogic.cc
)

# Add Scheme bindings if Guile is available
IF (HAVE_GUILE)
	LIST(APPEND ATEN_SOURCES ATenSCM.cc)
ENDIF (HAVE_GUILE)

# Build ATen integration library
ADD_LIBRARY (aten SHARED ${ATEN_SOURCES})

TARGET_LINK_LIBRARIES(aten
	PUBLIC
		atomspace
		atombase
		${COGUTIL_LIBRARY}
		${TORCH_LIBRARIES}
	PRIVATE
		${JSONCPP_LIBRARIES}
)

# Add Guile libraries if available
IF (HAVE_GUILE)
	TARGET_LINK_LIBRARIES(aten
		PUBLIC smob
		PRIVATE ${GUILE_LIBRARIES}
	)
	TARGET_INCLUDE_DIRECTORIES(aten PRIVATE ${GUILE_INCLUDE_DIRS})
ENDIF (HAVE_GUILE)

# Include directories for Torch
TARGET_INCLUDE_DIRECTORIES(aten
	PUBLIC ${TORCH_INCLUDE_DIRS}
)

# Torch requires specific compile features
TARGET_COMPILE_FEATURES(aten PUBLIC cxx_std_17)

# Disable torch-specific warnings that clutter output
IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	TARGET_COMPILE_OPTIONS(aten PRIVATE -Wno-deprecated-declarations)
ENDIF()

INSTALL (TARGETS aten EXPORT CogServerTargets
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES
	TensorValue.h
	ATenSpace.h
	TensorLogic.h
	McpPlugATen.h
	McpPlugTensorLogic.h
	DESTINATION "include/opencog/cogserver/aten"
)

# Install Scheme bindings
IF (HAVE_GUILE)
	INSTALL (FILES
		aten.scm
		DESTINATION "${GUILE_SITE_DIR}/opencog"
	)
ENDIF (HAVE_GUILE)
