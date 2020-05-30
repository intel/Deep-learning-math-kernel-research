# supress gflags warning #3280
remove_definitions(-Werror)
remove_definitions(-Wshadow)

file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp
  elt_gflag.cpp)

set(gflags_BUILD_STATIC_LIBS ON)
set(GFLAGS_SRC_DIR gflags CACHE STRING "gflags source from submodules")
add_subdirectory("${GFLAGS_SRC_DIR}")
include_directories(gflags)

add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 gflags)
add_test(NAME elt_conv COMMAND elt_conv)
