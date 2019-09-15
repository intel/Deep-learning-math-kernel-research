#include "el_utils.hpp"
#include "el_init.hpp"

namespace euler {

el_global_option ego;

__attribute__((constructor)) void el_init() {

  if (ego.initialized)
    return;

#if __ICC_COMPILER
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

  auto env_verbose = getenv("EULER_VERBOSE");
  if (env_verbose != nullptr && env_verbose[0] == '1') {
    ego.verbose = 1;
  }

  if (ego.verbose > 0)
    printf("\nEuler version: %s, MT_RUNTIME: %s\n",
           XSTRINGIFY(EULER_VERSION), mt_runtime_to_string(MT_RUNTIME));

  ego.initialized = true;
}

} // namespace euler
