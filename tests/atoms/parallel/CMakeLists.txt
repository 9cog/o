
IF(HAVE_GUILE)
	LINK_LIBRARIES(smob atomspace cogutil)
	ADD_CXXTEST(ParallelUTest)
	ADD_CXXTEST(ThreadedUTest)
ENDIF(HAVE_GUILE)
