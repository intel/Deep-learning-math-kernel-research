#pragma once

#include "el_def.hpp"
#include "el_utils.hpp"

namespace euler {

int el_log(int severity, const char *fmt, ...);

static inline void el_error(const char *msg) {
  el_log(ERROR, "%s", msg);
}

static inline void el_warn(const char *msg) {
  el_log(WARN, "%s", msg);
}

static inline void el_info(const char *msg) {
  el_log(ERROR, "%s", msg);
}

} // euler
