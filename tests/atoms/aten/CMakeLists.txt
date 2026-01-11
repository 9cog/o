# ATen Tensor tests

ADD_CXXTEST(ATenUTest)
TARGET_LINK_LIBRARIES(ATenUTest
    aten
    atomspace
    cogutil
)

# Tensor Logic tests (Pedro Domingos' Tensor Logic)
ADD_CXXTEST(TensorLogicUTest)
TARGET_LINK_LIBRARIES(TensorLogicUTest
    aten
    atomspace
    cogutil
)
