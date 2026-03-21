
LINK_LIBRARIES(atomspace cogutil smob)

ADD_CXXTEST(DefinedSchemaUTest)

ADD_GUILE_TEST(ExecEvalTest exec-eval-test.scm)
