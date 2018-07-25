if (__flags_included)
  return ()
endif()

set(__basic_flags " \
-fopenmp -Wall -Werror -Wextra -Wno-unused-parameter -Wno-unused-variable \
-Wno-sign-compare -Wno-uninitialized \
")

set(__cxx_flags "-std=c++11")

if (__debug)
  set(__opt_flags "-O0 -g -DDEBUG")
endif ()

if (CMAKE_CXX_COMPILER MATCHES "icpc")
  if (NOT __debug)
    set(__opt_flags "-O2 -DNDEBUG")
  endif ()
  set(__opt_flags "${__opt_flags} -qopt-report=5 -xHost -qopt-zmm-usage=high")

  # FIXME: Workaround ICC 18.0.2 inline bug. Remove this if ICC bug fixed.
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 18.0.1)
    set_source_files_properties(src/elx_conv_wino.cpp PROPERTIES COMPILE_FLAGS -finline-limit=80)
  endif ()
else ()
  set(__basic_flags "${__basic_flags} -Wno-unused-result")
  if (NOT __debug)
    set(__opt_flags "-O2 -DNDEBUG")
  endif ()
  set(__opt_flags "${__opt_flags} -mavx512f")
endif()

add_definitions(${__basic_flags} ${__cxx_flags} ${__opt_flags})
