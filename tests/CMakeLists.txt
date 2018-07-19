file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp)

file (GLOB __unitest_sources
  unitests/elk_gemm.cpp
  unitests/elk_trans_output.cpp
  unitests/elt_unitests.cpp
  unitests/elk_trans_input.cpp
  unitests/elk_trans_weights.cpp)

add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 boost_program_options)
add_test(NAME elt_conv
    COMMAND elt_conv)

add_executable(elt_unitests ${__unitest_sources})
add_definitions("-Dlest_FEATURE_AUTO_REGISTER=1")
target_link_libraries(elt_unitests ${lib_name} iomp5 boost_program_options)
add_test(NAME elt_unitests
    COMMAND elt_unitests)
