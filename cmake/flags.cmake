if (__flags_included)
  return ()
endif()

set(__basic_flags "-fopenmp -Wall -Werror -Wextra -Wno-unused-parameter")
set(__cxx_flags "-std=c++11")

if (__debug)
  set(__opt_flags "-O0 -g -DDEBUG")
else ()
  set(__opt_flags "-O3 -DNDEBUG")
endif ()

if (CMAKE_CXX_COMPILER MATCHES "icpc")
  set(__opt_flags "${__opt_flags} -qopt-report=5 -xHost")
else ()
  set(__opt_flags "${__opt_flags} -mavx512f")
endif()

add_definitions(${__basic_flags} ${__cxx_flags} ${__opt_flags})
