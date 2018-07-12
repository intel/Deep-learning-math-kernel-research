file (GLOB __test_sources
  elt_conv.cpp
  elt_conv_utils.cpp
  unitests/elk_gemm.cpp
  unitests/elk_trans_output.cpp
  unitests/elt_unitests.cpp
  unitests/elk_trans_input.cpp
  unitests/elk_trans_weights.cpp)

foreach(__test_file ${__test_sources})
  get_filename_component(__exe_file_name ${__test_file} NAME_WE)
  add_executable(${__exe_file_name} ${__test_file})
  target_link_libraries(${__exe_file_name} ${lib_name} iomp5)
  add_test(${__exe_file_name} ${__exe_file_name})
endforeach()
