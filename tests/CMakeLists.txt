file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp
  elt_gflag.cpp)

include (Findgflags.cmake)
if (GFLAGS_FOUND)
  message(STATUS "Found gflags include dirs: ${GFLAGS_INCLUDE_DIRS}")
  message(STATUS "Found gflags libraries: ${GFLAGS_LIBRARIES}")
  message(STATUS "Found gflags namespace: ${GFLAGS_NAMESPACE}")
else (GFLAGS_FOUND)
  message(FATAL_ERROR "Gflags not found")
endif (GFLAGS_FOUND)

include_directories("${GFLAGS_INCLUDE_DIRS}")
add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 ${GFLAGS_LIBRARIES})
add_test(NAME elt_conv COMMAND elt_conv)
