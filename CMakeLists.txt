cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
include(CheckTypeSize)

# euler version info
project(Euler CXX C)

set(EULER_VERSION_MAJOR 0)
set(EULER_VERSION_MINOR 0)
set(EULER_VERSION_PATCH 1)
set(EULER_VERSION "${EULER_VERSION_MAJOR}.${EULER_VERSION_MINOR}.${EULER_VERSION_PATCH}")

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
endif()

include("cmake/flags.cmake")
include("cmake/required.cmake")

# find src files
file (GLOB __euler_source
  src/eld_conv.cpp
  src/elk_conv_wino.cpp
  src/elx_conv.cpp
  src/elx_conv_wino.cpp)

include_directories(AFTER . include src)
link_directories(lib)

# build lib
add_library(${lib_name} SHARED ${__euler_source})
target_link_libraries(${lib_name} PUBLIC iomp5)

add_subdirectory(tests)
