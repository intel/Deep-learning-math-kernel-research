cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
include(CMakeDependentOption)

# euler version info
project(Euler CXX C)

option(WITH_TEST "Build with test cases" ON)
option(WITH_VNNI "With CLX VNNI support" OFF)
option(ENABLE_USER_FP16 "User FP16 type support" OFF)

if (NOT DEFINED MT_RUNTIME)
  set(MT_RUNTIME "omp")
endif()
STRING(TOLOWER "${MT_RUNTIME}" MT_RUNTIME)

# Build Euler version string
find_package (Git)
if (${GIT_FOUND})
  # Get current branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE retval
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (${retval})
    set(GIT_BRANCH "not-a-git-repo-branch")
  endif ()

  # Get current commit hash
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE retval
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (${retval})
    set(GIT_COMMIT_HASH "not-a-git-repo-hash")
  endif ()
else()
  set(GIT_BRANCH "no-git-found-branch")
  set(GIT_COMMIT_HASH "no-git-found-hash")
endif()

set(EULER_VERSION_MAJOR 0)
set(EULER_VERSION_MINOR 0)
set(EULER_VERSION_PATCH 1)
set(EULER_VERSION
  "v${EULER_VERSION_MAJOR}.${EULER_VERSION_MINOR}.${EULER_VERSION_PATCH}+${GIT_BRANCH}.${GIT_COMMIT_HASH}")
message("-- Euler version: " ${EULER_VERSION})

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0042 NEW)

# expose compiler info
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# general var
set(lib_name "el")

# set build type
if(CMAKE_BUILD_TYPE)
  string(TOUPPER ${CMAKE_BUILD_TYPE} __build_type)
  set (CMAKE_CXX_FLAGS_${__build_type} "")
endif()

# set build flags
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message(STATUS "Build type is Debug")
  set(__debug ON)
  set(WITH_TEST ON)
elseif(${CMAKE_BUILD_TYPE} MATCHES "Profiling")
  message(STATUS "Build type is Profiling")
  set(__profiling ON)
endif()

include("cmake/flags.cmake")
include("cmake/required.cmake")


# find src files
file (GLOB __euler_source
  src/eld_conv.cpp
  src/elx_conv.cpp
  src/elx_conv_wino_trans_input.cpp
  src/elx_conv_wino_trans_weights.cpp
  src/elx_conv_wino_gemm.cpp
  src/elx_conv_wino_u8s8_gemm.cpp
  src/elx_conv_wino_trans_output.cpp
  src/elx_conv_wino.cpp
  src/elx_conv_wino_xopt.cpp
  src/elx_conv_wino_bind.cpp
  src/elx_conv_wino_lp.cpp
  src/elx_conv_wino_lp_xopt.cpp
  src/elx_conv_wino_lp_bind.cpp
  src/elx_conv_direct_1x1_bind.cpp
  src/elx_conv_direct_1x1.cpp
  src/elx_conv_direct_1x1_xopt.cpp
  src/elx_conv_direct_1x1_lp_bind.cpp
  src/elx_conv_direct_1x1_lp.cpp
  src/elx_conv_direct_1x1_lp_xopt.cpp
  src/elx_conv_direct_bind.cpp
  src/elx_conv_direct.cpp
  src/elx_conv_direct_xopt.cpp
  src/elx_conv_direct_vmg_bind.cpp
  src/elx_conv_direct_vmg.cpp
  src/elx_conv_direct_vmg_xopt.cpp
  src/elx_conv_direct_lp_bind.cpp
  src/elx_conv_direct_lp.cpp
  src/elx_conv_direct_lp_xopt.cpp
  src/elx_deconv_direct_bind.cpp
  src/elx_deconv_direct.cpp
  src/elx_deconv_direct_xopt.cpp
  src/elx_conv_direct_depthwise_lp.cpp
  src/elx_conv_direct_depthwise_lp_xopt.cpp
  src/elx_conv_direct_depthwise_lp_bind.cpp
  src/elx_stream.cpp
  src/elx_reorder.cpp
  src/el_init.cpp)

set(KGEMM_GEN_DIR ${CMAKE_BINARY_DIR}/kgen)
set(BUILD_KGEMM_GEN_CMD ${KGEMM_GEN_DIR}/kgemm_gen.sh)
set(BUILD_U8S8_KGEMM_GEN_CMD ${KGEMM_GEN_DIR}/u8s8_kgemm_gen.sh)
set(BUILD_KCONV_GEN_CMD ${KGEMM_GEN_DIR}/kconv_gen.sh)
set(BUILD_U8S8_KCONV_GEN_CMD ${KGEMM_GEN_DIR}/u8s8_kconv_gen.sh)
set(BUILD_U8S8_DEPTHWISE_KCONV_GEN_CMD ${KGEMM_GEN_DIR}/u8s8_depthwise_kconv_gen.sh)
set(BUILD_VMG_KCONV_GEN_CMD ${KGEMM_GEN_DIR}/vmg_kconv_gen.sh)
set(BUILD_KGEMM_GEN_SRC1 ${KGEMM_GEN_DIR}/elk_gemm_otj_binder.hxx)
set(BUILD_KGEMM_GEN_SRC2 ${KGEMM_GEN_DIR}/elk_u8s8_gemm_otj_binder.hxx)
set(BUILD_KCONV_GEN_SRC1 ${KGEMM_GEN_DIR}/elk_conv_otj_binder.hxx)
set(BUILD_KCONV_GEN_SRC2 ${KGEMM_GEN_DIR}/elk_u8s8_conv_otj_binder.hxx)
set(BUILD_KCONV_GEN_SRC3 ${KGEMM_GEN_DIR}/elk_vmg_conv_otj_binder.hxx)
set(BUILD_KCONV_GEN_SRC4 ${KGEMM_GEN_DIR}/elk_u8s8_depthwise_conv_otj_binder.hxx)

set(KGEMM_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/kgemm_gen.sh)
set(U8S8_KGEMM_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/u8s8_kgemm_gen.sh)
set(KCONV_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/kconv_gen.sh)
set(VMG_KCONV_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/vmg_kconv_gen.sh)
set(U8S8_KCONV_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/u8s8_kconv_gen.sh)
set(U8S8_DEPTHWISE_KCONV_GEN_CMD ${CMAKE_HOME_DIRECTORY}/src/kernel/u8s8_depthwise_kconv_gen.sh)
set(KGEMM_GEN_SRC1 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_gemm_otj_binder.hxx)
set(KGEMM_GEN_SRC2 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_u8s8_gemm_otj_binder.hxx)
set(KCONV_GEN_SRC1 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_conv_otj_binder.hxx)
set(KCONV_GEN_SRC2 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_u8s8_conv_otj_binder.hxx)
set(KCONV_GEN_SRC3 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_vmg_conv_otj_binder.hxx)
set(KCONV_GEN_SRC4 ${CMAKE_HOME_DIRECTORY}/src/kernel/elk_u8s8_depthwise_conv_otj_binder.hxx)

execute_process(COMMAND mkdir -p ${KGEMM_GEN_DIR})
configure_file(${KGEMM_GEN_CMD} ${BUILD_KGEMM_GEN_CMD} COPYONLY)
configure_file(${U8S8_KGEMM_GEN_CMD} ${BUILD_U8S8_KGEMM_GEN_CMD} COPYONLY)
configure_file(${KCONV_GEN_CMD} ${BUILD_KCONV_GEN_CMD} COPYONLY)
configure_file(${VMG_KCONV_GEN_CMD} ${BUILD_VMG_KCONV_GEN_CMD} COPYONLY)
configure_file(${U8S8_KCONV_GEN_CMD} ${BUILD_U8S8_KCONV_GEN_CMD} COPYONLY)
configure_file(${U8S8_DEPTHWISE_KCONV_GEN_CMD} ${BUILD_U8S8_DEPTHWISE_KCONV_GEN_CMD} COPYONLY)
configure_file(${KGEMM_GEN_SRC1} ${BUILD_KGEMM_GEN_SRC1} COPYONLY)
configure_file(${KGEMM_GEN_SRC2} ${BUILD_KGEMM_GEN_SRC2} COPYONLY)
configure_file(${KCONV_GEN_SRC1} ${BUILD_KCONV_GEN_SRC1} COPYONLY)
configure_file(${KCONV_GEN_SRC2} ${BUILD_KCONV_GEN_SRC2} COPYONLY)
configure_file(${KCONV_GEN_SRC3} ${BUILD_KCONV_GEN_SRC3} COPYONLY)
configure_file(${KCONV_GEN_SRC4} ${BUILD_KCONV_GEN_SRC4} COPYONLY)

execute_process(COMMAND ${BUILD_KGEMM_GEN_CMD} ${BUILD_KGEMM_GEN_SRC1}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})
execute_process(COMMAND ${BUILD_U8S8_KGEMM_GEN_CMD} ${BUILD_KGEMM_GEN_SRC2}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})
execute_process(COMMAND ${BUILD_KCONV_GEN_CMD} ${BUILD_KCONV_GEN_SRC1}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})
execute_process(COMMAND ${BUILD_U8S8_KCONV_GEN_CMD} ${BUILD_KCONV_GEN_SRC2}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})
execute_process(COMMAND ${BUILD_VMG_KCONV_GEN_CMD} ${BUILD_KCONV_GEN_SRC3}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})
execute_process(COMMAND ${BUILD_U8S8_DEPTHWISE_KCONV_GEN_CMD} ${BUILD_KCONV_GEN_SRC4}
  ${KGEMM_GEN_DIR} ${CMAKE_CXX_COMPILER} ${ENABLE_USER_FP16})

file (GLOB __euler_gemm_kernel_source ${KGEMM_GEN_DIR}/*.cpp)

include_directories(AFTER . include src tests src/common)
link_directories(lib)

# build lib
add_library(${lib_name} SHARED ${__euler_source} ${__euler_gemm_kernel_source})
set_target_properties(${lib_name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
if (MT_RUNTIME STREQUAL "tbb")
  target_link_libraries(${lib_name} PUBLIC tbb)
elseif (MT_RUNTIME STREQUAL "omp")
  target_link_libraries(${lib_name} PUBLIC iomp5)
endif()
target_link_libraries(${lib_name} PUBLIC rt)

if (WITH_TEST)
  add_subdirectory(tests)
endif()
