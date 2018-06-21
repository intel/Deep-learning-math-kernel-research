CMAKE_MINIMUM_REQUIRED(VERSION 3.2 FATAL_ERROR)
INCLUDE(CheckTypeSize)

# PLEASE set the path to icc libraries before make it
SET(icc_lib_path "/opt/intel/lib")

# choose compiler
SET(CMAKE_CXX_COMPILER icc)
SET(CMAKE_C_COMPILER icc)

# euler version info
PROJECT(Euler CXX C)

SET(EULER_VERSION_MAJOR 0)
SET(EULER_VERSION_MINOR 0)
SET(EULER_VERSION_PATCH 1)
SET(EULER_VERSION "${EULER_VERSION_MAJOR}.${EULER_VERSION_MINOR}.${EULER_VERSION_PATCH}")

# expose compiler info
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# general var
SET(lib_name "el")

CHECK_TYPE_SIZE("void*" size_of_voidptr)
IF ("${size_of_voidptr}" EQUAL 8)
  SET(icc_lib_path "${icc_lib_path}/intel64")
ELSE()
  SET(icc_lib_path "${icc_lib_path}/ia32")
ENDIF()

# set build type
IF(NOT CMAKE_BUILD_TYPE)
  MESSAGE(STATUS "Build type is Release by default")
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Set the build type from: Debug Release." FORCE)
ENDIF()

# set build flags
IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Build type is Relase")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Build type is Debug")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
ELSE()
  MESSAGE(FATAL_ERROR "Build type is invalid")
ENDIF()

IF(CMAKE_CXX_COMPILER MATCHES "icc")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-zmm-usage=high")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCore-AVX512")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=${icc_lib_path}")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# find dependence libs
SET(__euler_libs iomp5)
SET(euler_dependency_libs "")

IF(CMAKE_CXX_COMPILER MATCHES "icc")
  LIST(APPEND __euler_libs imf svml irng intlc)
  SET(lib_link_path ${icc_lib_path})
ELSE()
  SET(lib_link_path ${CMAKE_SOURCE_DIR}/lib)
ENDIF()

FOREACH(__euler_lib ${__euler_libs})
  STRING(TOUPPER ${__euler_lib} __euler_lib_upper)
  FIND_LIBRARY(${__euler_lib_upper}_LIBRARY
    NAME ${__euler_lib}
    PATHS ${lib_link_path}
    DOC "The path to library ${__euler_lib}")
  LIST(APPEND euler_dependency_libs ${__euler_lib})
ENDFOREACH()

# find src files
SET(src_dir "${CMAKE_SOURCE_DIR}/src")
FILE(GLOB_RECURSE euler_src_files ${src_dir}/*.cpp)

# find include dirs
SET(euler_include_dirs "${CMAKE_SOURCE_DIR}/include")
LIST(APPEND euler_include_dirs ${src_dir})

# build lib
ADD_LIBRARY(${lib_name} SHARED ${euler_src_files})
TARGET_INCLUDE_DIRECTORIES(${lib_name} PUBLIC ${euler_include_dirs})
TARGET_LINK_LIBRARIES(${lib_name} PUBLIC ${euler_dependency_libs})

ADD_SUBDIRECTORY(test)
