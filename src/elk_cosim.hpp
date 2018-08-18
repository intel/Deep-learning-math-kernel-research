#pragma once
#include <cmath>
#include <mutex>
#include <iostream>
#include <cstring>

namespace euler {

template <typename Type>
class cosim_base {
  constexpr static int max_errors = 20;
public:
  static bool near_eq(const Type l, const Type r, double acc = 1e-5) {
    if (l == r)
      return true;

    auto l_d = static_cast<double>(l);
    auto r_d = static_cast<double>(r);
    auto diff = std::fabs(l_d - r_d);

    auto min = static_cast<double>(std::numeric_limits<Type>::min());
    auto max = static_cast<double>(std::numeric_limits<Type>::max());

    if (l_d == 0 || r_d == 0 || diff < min) {
      return diff < acc * min;
    } else {
      return (diff /
          std::min(std::fabs(l_d + r_d), max)) < acc;
    }
  }

  static void compare_small(const Type *__restrict l,
      const Type * __restrict r, int elem_count, double acc = 3e-3) {
    int errors = 0;
    static std::mutex mu;
    for (int i = 0; i < elem_count; ++ i) {
      if (!near_eq(l[i], r[i])) {
        mu.lock();
        std::cerr<<"Exceeded "<<acc<<" at index:"<<i<<"! "
          <<r[i]<<" expected than "<<l[i]<<"."<<std::endl;
        mu.unlock();
        if (errors ++ > max_errors)
          break;
      }
    }
  }
};

};
