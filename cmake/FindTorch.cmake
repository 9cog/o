#
# FindTorch.cmake
#
# Find PyTorch/LibTorch and ATen libraries
#
# This module defines:
#   TORCH_FOUND        - True if Torch was found
#   TORCH_INCLUDE_DIRS - Include directories for Torch
#   TORCH_LIBRARIES    - Libraries to link against
#   TORCH_CXX_FLAGS    - C++ flags for Torch
#
# Usage:
#   find_package(Torch)
#   if(TORCH_FOUND)
#       target_link_libraries(mylib ${TORCH_LIBRARIES})
#   endif()
#
# You can set TORCH_ROOT or CMAKE_PREFIX_PATH to help find Torch.
#

# Try to find Torch using CMake config (preferred method)
find_package(Torch QUIET CONFIG)

if(Torch_FOUND OR TORCH_FOUND)
	set(TORCH_FOUND TRUE)
	message(STATUS "Found Torch via CMake config")
	message(STATUS "  TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
	message(STATUS "  TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
	return()
endif()

# Manual search if CMake config not found
message(STATUS "Torch CMake config not found, searching manually...")

# Common locations
set(_TORCH_SEARCH_PATHS
	${TORCH_ROOT}
	$ENV{TORCH_ROOT}
	${CMAKE_PREFIX_PATH}
	/usr/local/lib/python3/dist-packages/torch
	/usr/local/lib/python3.10/dist-packages/torch
	/usr/local/lib/python3.11/dist-packages/torch
	/usr/local/lib/python3.12/dist-packages/torch
	~/.local/lib/python3/dist-packages/torch
	~/.local/lib/python3.10/dist-packages/torch
	~/.local/lib/python3.11/dist-packages/torch
	~/.local/lib/python3.12/dist-packages/torch
	/opt/pytorch
	/opt/libtorch
)

# Try to find using Python
if(NOT TORCH_INCLUDE_DIRS)
	execute_process(
		COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
		OUTPUT_VARIABLE _TORCH_CMAKE_PATH
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET
	)
	if(_TORCH_CMAKE_PATH)
		list(APPEND _TORCH_SEARCH_PATHS "${_TORCH_CMAKE_PATH}")
	endif()
endif()

# Find include directory
find_path(TORCH_INCLUDE_DIRS
	NAMES torch/torch.h ATen/ATen.h
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES include
)

# Find library
find_library(TORCH_LIBRARY
	NAMES torch torch_cpu
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

find_library(C10_LIBRARY
	NAMES c10
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

find_library(TORCH_CPU_LIBRARY
	NAMES torch_cpu
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

# Optional CUDA support
find_library(TORCH_CUDA_LIBRARY
	NAMES torch_cuda
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

find_library(C10_CUDA_LIBRARY
	NAMES c10_cuda
	PATHS ${_TORCH_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

# Compose library list
set(TORCH_LIBRARIES "")
if(TORCH_LIBRARY)
	list(APPEND TORCH_LIBRARIES ${TORCH_LIBRARY})
endif()
if(TORCH_CPU_LIBRARY)
	list(APPEND TORCH_LIBRARIES ${TORCH_CPU_LIBRARY})
endif()
if(C10_LIBRARY)
	list(APPEND TORCH_LIBRARIES ${C10_LIBRARY})
endif()

# Add CUDA libraries if found
if(TORCH_CUDA_LIBRARY AND C10_CUDA_LIBRARY)
	list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARY} ${C10_CUDA_LIBRARY})
	set(TORCH_CUDA_FOUND TRUE)
endif()

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Torch
	REQUIRED_VARS TORCH_INCLUDE_DIRS TORCH_LIBRARIES
)

if(TORCH_FOUND)
	message(STATUS "Found Torch:")
	message(STATUS "  TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
	message(STATUS "  TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
	if(TORCH_CUDA_FOUND)
		message(STATUS "  CUDA support: enabled")
	endif()
endif()

mark_as_advanced(
	TORCH_INCLUDE_DIRS
	TORCH_LIBRARIES
	TORCH_LIBRARY
	TORCH_CPU_LIBRARY
	C10_LIBRARY
	TORCH_CUDA_LIBRARY
	C10_CUDA_LIBRARY
)
