file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp
  elt_gflag.cpp)

add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 boost_program_options gflags)
add_test(NAME elt_conv COMMAND elt_conv)

if (WITH_GTEST)
  file (GLOB __unitest_sources
    main.cpp
    elt_conv_gtest.cpp
    elt_conv_utils.cpp)

  include_directories(BEFORE
    .
    unittests
    unittests/googletest/googletest/gtest
    unittests/googletest/googletest/include
    unittests/googletest/googlemock/include)

  add_executable(elt_conv_gtest ${__unitest_sources})
  target_link_libraries(elt_conv_gtest
    ${lib_name} gtest gtest_main gmock iomp5 boost_program_options)
  add_test(NAME elt_conv_gtest COMMAND elt_conv_gtest)

  if (WITH_GK)
    add_subdirectory (unittests)
  else()
    add_subdirectory (unittests/googletest)
  endif()
endif()
