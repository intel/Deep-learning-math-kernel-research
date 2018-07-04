# find boost package
FIND_PACKAGE(Boost COMPONENTS system filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

# set build flags
SET(CMAKE_CXX_FLAGS "-g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pie")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_program_options")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -DDEBUG")
ELSE()
  MESSAGE(FATAL_ERROR "Build type is invalid")
ENDIF()

IF(CMAKE_CXX_COMPILER MATCHES "icc")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=${icc_lib_path}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dlest_FEATURE_AUTO_REGISTER=1")
ENDIF()

# find dependence libs
SET(__test_libs iomp5 ${lib_name})
FOREACH(__test_lib ${__test_libs})
  STRING(TOUPPER ${__test_lib} __test_lib_upper)
  FIND_LIBRARY(${__test_lib_upper}_LIBRARY
    NAME ${__test_lib}
    PATHS ${lib_link_path} ${CMAKE_BINARY_DIR}
    DOC "The path to library ${__test_lib}")
  LIST(APPEND test_dependency_libs ${__test_lib})
ENDFOREACH()
LIST(APPEND test_dependency_libs ${Boost_LIBRARIES})

# find test/utest src files
SET(test_src_dir "${CMAKE_SOURCE_DIR}/test")
SET(utest_src_dir "${test_src_dir}/unitests")
FILE(GLOB_RECURSE test_src_files ${test_src_dir}/*.cpp)
FILE(GLOB_RECURSE utest_src_files ${utest_src_dir}/*.cpp)
FOREACH(__utest ${utest_src_files})
  LIST(REMOVE_ITEM test_src_files "${__utest}")
ENDFOREACH()

# find test includes
SET(test_include_dirs "${CMAKE_SOURCE_DIR}/include")
LIST(APPEND test_include_dirs ${Boost_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

# build test
SET(test_name elt_conv)
ADD_EXECUTABLE(${test_name} ${test_src_files})
TARGET_INCLUDE_DIRECTORIES(${test_name} PUBLIC ${test_include_dirs})
TARGET_LINK_LIBRARIES(${test_name} PUBLIC ${test_dependency_libs})

# find utest includes
SET(utest_include_dirs "${CMAKE_SOURCE_DIR}/include")
LIST(APPEND utest_include_dirs ${Boost_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

# build utest
SET(utest_name elt_unitests)
ADD_EXECUTABLE(${utest_name} ${utest_src_files})
TARGET_INCLUDE_DIRECTORIES(${utest_name} PUBLIC ${utest_include_dirs})
TARGET_LINK_LIBRARIES(${utest_name} PUBLIC ${test_dependency_libs})

INSTALL(TARGETS ${test_name} ${utest_name} RUNTIME DESTINATION bin)
