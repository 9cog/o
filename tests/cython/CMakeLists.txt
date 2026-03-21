# Unified Cython Tests CMakeLists.txt
# Merged from atomspace and cogserver

# ============================================================
# AtomSpace Cython Tests
# ============================================================
INCLUDE_DIRECTORIES(
	${Python3_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/opencog/atomspace
	${PROJECT_SOURCE_DIR}/opencog/guile
	${PROJECT_SOURCE_DIR}/opencog/util
)

LINK_DIRECTORIES(
	${PROJECT_BINARY_DIR}/opencog/atomspace
	${PROJECT_BINARY_DIR}/opencog/guile
	${PROJECT_BINARY_DIR}/opencog/util
)

set(TEST_ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
	"PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
	"PYTHONPATH=${PROJECT_BINARY_DIR}/opencog/cython:${PROJECT_SOURCE_DIR}/tests/cython/atomspace:${PROJECT_SOURCE_DIR}/tests/cython"
	"GUILE_LOAD_PATH=${GUILE_LOAD_PATH}"
	"LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/opencog/cython:$ENV{LD_LIBRARY_PATH}")

# Test the evaluator wrapper for Python.
ADD_CXXTEST(PythonEvalUTest)

TARGET_LINK_LIBRARIES(PythonEvalUTest
	atomspace
	PythonEval
	atomspace_cython
	cogutil
)

SET_PROPERTY(TEST PythonEvalUTest
	APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

ADD_CXXTEST(PythonInitFiniUTest)

TARGET_LINK_LIBRARIES(PythonInitFiniUTest
	PythonEval
	atomspace_cython
	atomspace
	cogutil
)

SET_PROPERTY(TEST PythonInitFiniUTest
	APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

IF (HAVE_PYTEST)

	# Basic test, just does the cython wrapper for opencog/util
	ADD_TEST(CythonLogger ${PYTEST_EXECUTABLE} -v
		${CMAKE_SOURCE_DIR}/tests/cython/logger)
	SET_PROPERTY(TEST CythonLogger
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

	ADD_TEST(CythonAtomSpace ${PYTEST_EXECUTABLE} -v
		${CMAKE_SOURCE_DIR}/tests/cython/atomspace/)
	SET_PROPERTY(TEST CythonAtomSpace
		PROPERTY DEPENDS atomspace_cython)
	SET_PROPERTY(TEST CythonAtomSpace
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

	ADD_TEST(CythonValues ${PYTEST_EXECUTABLE} -v
		${CMAKE_SOURCE_DIR}/tests/cython/values/)
	SET_PROPERTY(TEST CythonValues
		PROPERTY DEPENDS atomspace_cython)
	SET_PROPERTY(TEST CythonValues
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

	# All of the subsequent cython unit tests depend on having
	# guile working. So test guile first, and then the others.
	ADD_TEST(CythonGuile ${PYTEST_EXECUTABLE} -v
		${CMAKE_SOURCE_DIR}/tests/cython/guile/)
	SET_PROPERTY(TEST CythonGuile
		PROPERTY DEPENDS atomspace_cython scheme)
	SET_PROPERTY(TEST CythonGuile
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})

	ADD_TEST(CythonBindlink ${PYTEST_EXECUTABLE} -v
			${CMAKE_SOURCE_DIR}/tests/cython/querylink/)
	SET_PROPERTY(TEST CythonBindlink
		PROPERTY DEPENDS atomspace_cython querylink_cython)
	SET_PROPERTY(TEST CythonBindlink
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})
	SET_PROPERTY(TEST CythonBindlink
		APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/opencog/cython:${PROJECT_SOURCE_DIR}/opencog/python/:${PROJECT_SOURCE_DIR}/tests/cython/querylink")

	# Multi-threading tests for Python bindings
	ADD_TEST(CythonMultiThread ${PYTEST_EXECUTABLE} -v
			${CMAKE_SOURCE_DIR}/tests/cython/multithread/)
	SET_PROPERTY(TEST CythonMultiThread
		PROPERTY DEPENDS atomspace_cython PythonEval)
	SET_PROPERTY(TEST CythonMultiThread
		APPEND PROPERTY ENVIRONMENT ${TEST_ENVIRONMENT})
	SET_PROPERTY(TEST CythonMultiThread
		APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/opencog/cython:${PROJECT_SOURCE_DIR}/tests/cython/multithread")

ENDIF (HAVE_PYTEST)

# ============================================================
# CogServer Cython Tests
# ============================================================

IF (HAVE_COGSERVER)

ADD_CXXTEST(PyEvalUTest)

TARGET_LINK_LIBRARIES(PyEvalUTest
	py-shell
	servernode
	cogserver-types
	${COGSERVER_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
	cogutil
)

# Set COGSERVER_MODULE_PATH so modules can be found in the build directory
SET(COGSERVER_TEST_MODULE_PATH
	"COGSERVER_MODULE_PATH=${PROJECT_BINARY_DIR}/opencog/cogserver/atoms:${PROJECT_BINARY_DIR}/opencog/cogserver/modules:${PROJECT_BINARY_DIR}/opencog/cogserver/shell")
SET_PROPERTY(TEST PyEvalUTest APPEND PROPERTY ENVIRONMENT ${COGSERVER_TEST_MODULE_PATH})

# --------------------------

ADD_TEST(NAME CogServerPythonTest
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cogserver.py
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

SET_PROPERTY(TEST CogServerPythonTest
	PROPERTY ENVIRONMENT
	"PYTHONPATH=${PROJECT_SOURCE_DIR}/opencog/cogserver/python:${PYTHON_ROOT}:$ENV{PYTHONPATH}"
	"LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/opencog/cogserver/server:$ENV{LD_LIBRARY_PATH}"
	${COGSERVER_TEST_MODULE_PATH}
)

ENDIF (HAVE_COGSERVER)

# --------------------------
