
LINK_LIBRARIES(atomspace cogutil)

ADD_CXXTEST(AlphaConvertUTest)
ADD_CXXTEST(BetaReduceUTest)
ADD_CXXTEST(VariablesUTest)

IF(HAVE_GUILE)
	ADD_CXXTEST(ScopeLinkUTest)
	TARGET_LINK_LIBRARIES(ScopeLinkUTest smob atomspace cogutil)
	ADD_CXXTEST(RewriteLinkUTest)
	TARGET_LINK_LIBRARIES(RewriteLinkUTest smob atomspace cogutil)
	ADD_CXXTEST(PutLinkUTest)
	TARGET_LINK_LIBRARIES(PutLinkUTest smob atomspace cogutil)
	ADD_CXXTEST(DontExecUTest)
	TARGET_LINK_LIBRARIES(DontExecUTest smob atomspace cogutil)
	ADD_CXXTEST(QuotationUTest)
	TARGET_LINK_LIBRARIES(QuotationUTest smob atomspace cogutil)

	ADD_GUILE_TEST(PutRecursiveTest put-recursive-test.scm)
ENDIF(HAVE_GUILE)
