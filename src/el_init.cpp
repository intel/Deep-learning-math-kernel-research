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

  // PERF_TRACE
  auto env_verbose = ::getenv("EULER_VERBOSE");
  if (env_verbose != nullptr && env_verbose[0] == '1') {
    ego.verbose = 1;
  }

  auto env_log_level = ::getenv("EULER_LOG_LEVEL");
  if (env_log_level != nullptr) {
    ego.log_level = atoi(env_log_level);
  }

  if (ego.verbose > 0)
    el_log(__INFO, "Version: %s, MT_RUNTIME: %s",
           XSTRINGIFY(EULER_VERSION), mt_runtime_to_string(MT_RUNTIME));

  ego.initialized = true;
}

} // namespace euler
