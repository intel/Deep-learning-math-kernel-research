#pragma once

namespace euler {

struct el_global_option {
  int verbose = 0; // TODO: log level
  bool initialized = false;
};

void el_init();

extern el_global_option ego;

}; // namespace euler
