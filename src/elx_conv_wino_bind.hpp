#pragma once

#include "elx_conv_wino.hpp"

namespace euler {

Template_elx_conv_wino_t void
Instance_elx_conv_wino_t::bind_execute_functions() {
#define EXECUTE_CASE(n)                                                      \
  case 0x##n:                                                                \
    execute_opt_ = &Instance_elx_conv_wino_t::__execute_##n;                 \
    break

  switch (xopt_) {
  EXECUTE_CASE(a000);
  EXECUTE_CASE(a033);
  EXECUTE_CASE(a061);
  EXECUTE_CASE(a071);
  EXECUTE_CASE(a073);
  default:
    el_error("Unimplemented");
    break;
  }
}

} // namespace euler
