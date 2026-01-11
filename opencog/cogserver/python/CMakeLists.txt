#
# Install pure Python cogserver module
#
# Only install if Cython is available (which sets PYTHON_DEST)
IF (CYTHON_FOUND)
    INSTALL(FILES
        opencog/__init__.py
        opencog/cogserver.py
        DESTINATION "${PYTHON_DEST}"
    )
ENDIF (CYTHON_FOUND)
