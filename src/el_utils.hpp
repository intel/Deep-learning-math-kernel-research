#include <stdlib.h>
#include <assert.h>
#include <cxxabi.h>
#include <chrono>

#ifndef __EL_UTILS_HPP__
#define __EL_UTILS_HPP__

#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                   \
  _T(Time::time_point __e##n = Time::now());                        \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, omp_get_thread_num(), \
            Duration(__e##n - __s##n).count()));

#define for_each(i, lim) for (int(i) = 0; (i) < (lim); ++(i))

namespace euler {

template <typename T, typename P>
inline bool one_of(T val, P item) {
  return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) {
  return one_of(nullptr, ptrs...);
}

template <typename T>
inline T accumulate(T tn) {
  return tn;
}
template <typename T, typename... Args>
inline T accumulate(T a, Args... args) {
  return a * accumulate(args...);
}

#define MD(type, array, dims, ptr) \
  type(&array) dims = *reinterpret_cast<type(*) dims>(ptr)

template <typename T, int N>
class mdarray {
 public:
  template <typename... Args>
  mdarray(T *p, Args... dims) : _p(p), _dims{dims...} {}
  template <typename... Args>
  inline T &operator()(Args... dims) {
    return *(_p + offset(1, dims...));
  }

 private:
  template <typename... Args>
  inline size_t offset(size_t index, size_t off, size_t dim, Args... dims) {
    off = _dims[index] * off + dim;
    return offset(index + 1, off, dims...);
  }
  inline size_t offset(size_t index, size_t off, size_t dim) {
    return _dims[index] * off + dim;
  }

  T *_p;
  const int _dims[N];
};

}  // namespace euler

#endif  // __EL_UTILS_HPP__
