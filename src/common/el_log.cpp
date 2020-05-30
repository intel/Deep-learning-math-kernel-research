#include <stdio.h>
#include "el_utils.hpp"
#include "el_init.hpp"

namespace euler {

int el_log(int severity, const char *fmt, ...) {
  int ret = 0;
  if (severity >= ego.log_level) {
    va_list ap;

    fprintf(stdout, "euler-%s, ", log_severity_to_string(severity));
    va_start(ap, fmt);
    ret = vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputs("\n", stdout);
  }

  if (severity >= __ERROR && severity != __PERF_TRACE) {
    fputs("Euler abort due to error\n", stdout);
    abort();
  }

  return ret;
}

}; // euler
