
LINK_LIBRARIES(atomspace cogutil)

ADD_CXXTEST(CheckersUTest)
ADD_CXXTEST(FindUtilsUTest)
ADD_CXXTEST(NumberNodeUTest)

ADD_CXXTEST(DeleteLinkUTest)

IF(HAVE_GUILE)
	ADD_CXXTEST(FreeLinkUTest)
	TARGET_LINK_LIBRARIES(FreeLinkUTest smob atomspace cogutil)
	ADD_CXXTEST(CondLinkUTest)
	TARGET_LINK_LIBRARIES(CondLinkUTest smob atomspace cogutil)

	ADD_GUILE_TEST(PresentAbsentTest present-absent-test.scm)
ENDIF(HAVE_GUILE)
