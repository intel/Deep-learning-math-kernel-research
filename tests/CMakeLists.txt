file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp)

file (GLOB __unitest_sources
  main.cpp
  elt_conv_gtest.cpp
  elt_conv_utils.cpp
)

add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 boost_program_options)
add_test(NAME elt_conv
    COMMAND elt_conv)


include_directories(BEFORE . unittests unittests/gtest)


add_executable(elt_conv_gtest ${__unitest_sources})
target_link_libraries(elt_conv_gtest ${lib_name} euler_gtest iomp5 boost_program_options)
add_test(NAME elt_conv_gtest
    COMMAND elt_conv_gtest)


add_subdirectory (unittests)
