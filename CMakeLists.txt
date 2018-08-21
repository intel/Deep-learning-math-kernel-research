cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
include(CMakeDependentOption)

# euler version info
project(Euler CXX C)

option(WITH_TEST "Build with test cases" ON)
cmake_dependent_option(WITH_GTEST
  "Build with gtest/unit tests if WITH_TEST is ON" ON "WITH_TEST" OFF)
option(WITH_GK "Build with generic kernel and unit tests" ON)

set(EULER_VERSION_MAJOR 0)
set(EULER_VERSION_MINOR 0)
set(EULER_VERSION_PATCH 1)
set(EULER_VERSION
  "${EULER_VERSION_MAJOR}.${EULER_VERSION_MINOR}.${EULER_VERSION_PATCH}")

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
  set(WITH_GTEST ON)
  set(WITH_GK ON)
endif()

include("cmake/flags.cmake")
include("cmake/required.cmake")

# find src files
file (GLOB __euler_source
  src/eld_conv.cpp
  src/elk_conv_direct_1x1.cpp
  src/elx_conv.cpp
  src/elx_conv_wino.cpp
  src/elx_conv_direct_1x1.cpp)

include_directories(AFTER . include src tests xsimd/include)
link_directories(lib)

# build lib
add_library(${lib_name} SHARED ${__euler_source})
target_link_libraries(${lib_name} PUBLIC iomp5)

if (WITH_TEST)
  add_subdirectory(tests)
endif()
