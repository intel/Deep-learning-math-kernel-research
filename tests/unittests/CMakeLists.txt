#===============================================================================
# Copyright 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
add_subdirectory (googletest)

set(APP_NAME "gtest")
set(MAIN_SRC_GTEST main.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include

                    )
# TODO: enable me!
#file(GLOB API_TEST_CASES_SRC api_tests/*.cpp)
#file(GLOB PRIM_TEST_CASES_SRC test_*.cpp)
file(GLOB PRIM_TEST_CASES_SRC
                              elk_gemm.cpp
                              elk_trans_input.cpp
                              elk_trans_output.cpp
                              elk_trans_weights.cpp
                              parameter_tuner.cpp
                              ) #temporary
foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    message(STATUS ${TEST_FILE} )
    get_filename_component (TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(CURR_APP_NAME ${TEST_FILE_WE})
    add_executable(${CURR_APP_NAME} ${MAIN_SRC_GTEST} ${TEST_FILE})
    target_link_libraries(${CURR_APP_NAME}
       el ${LIB_NAME} gtest gtest_main gmock ${EXTRA_LIBS} iomp5 boost_program_options)
    add_test(${CURR_APP_NAME} ${CURR_APP_NAME})
endforeach()
# we need to have either this or the add_test above; disabling for now...
# add_custom_command(TARGET ${APP_NAME}
#                   POST_BUILD
#                   WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#                   COMMAND ${APP_NAME} )

# vim: et ts=4 sw=4

#file(GLOB __native_test_src
#                          sample1.cpp
#                          )

#foreach(__test_file ${__native_test_src})
  #get_filename_component(__exe_file_name ${__test_file} NAME_WE)
#  add_executable(sample1_unittest ${__native_test_src})
#  target_link_libraries(sample1_unittest ${CURR_APP_NAME}
#       el ${LIB_NAME} euler_gtest ${EXTRA_LIBS} iomp5 boost_program_options)
#  add_test(NAME sample1_unittest
#    COMMAND sample1_unittest)
#endforeach()
