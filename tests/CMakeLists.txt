file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp
  elt_gflag.cpp)

# workaround the find_package issue on Ubuntu 16
#find_package(gflags REQUIRED)
#include_directories("${gflags_INCLUDE_DIR}")

add_executable(elt_conv ${__test_sources})
target_link_libraries(elt_conv ${lib_name} iomp5 gflags)
add_test(NAME elt_conv COMMAND elt_conv)
