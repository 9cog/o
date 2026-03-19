
ADD_CXXTEST(NameServerUTest)
TARGET_LINK_LIBRARIES(NameServerUTest atomspace cogutil)

ADD_CXXTEST(AtomNamesUTest)
TARGET_LINK_LIBRARIES(AtomNamesUTest smob atomspace cogutil scope)
