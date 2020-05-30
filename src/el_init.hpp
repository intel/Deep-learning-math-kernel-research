#pragma once

#include "el_utils.hpp"

#if __ICC_COMPILER
#include "xmmintrin.h"
#include "pmmintrin.h"
#endif

namespace euler {

struct el_global_option {
  int log_level = __INFO;
  bool verbose = false; // for EULER_VERBOSE
  bool initialized = false;
};

void el_init();

extern el_global_option ego;

}; // namespace euler
