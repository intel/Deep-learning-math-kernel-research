#include <stdlib.h>
#include <assert.h>
#include <cxxabi.h>
#include <chrono>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#ifndef __EL_UTILS_HPP__
#define __EL_UTILS_HPP__

#define _T(x) x
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define __tstart(n) _T(Time::time_point __s##n = Time::now());
#define __tend(n)                                                              \
  _T(Time::time_point __e##n = Time::now());                                   \
  _T(printf("time: %s, th=%d, %.2f ms\n", #n, omp_get_thread_num(),            \
      Duration(__e##n - __s##n).count()));

#define for_each(i, lim) for (int(i) = 0; (i) < (lim); ++(i))

namespace euler {

template <typename T> inline T accumulate(T tn) { return tn; }
template <typename T, typename... Args> inline T accumulate(T a, Args... args)
{
  return a * accumulate(args...);
}

#define MD(type, array, dims, ptr)                                             \
  auto &array = *reinterpret_cast<float (*) dims>(ptr)

#define MEMALIGN64(ptr, size) posix_memalign((void **)ptr, 64, size);

} // namespace euler

#endif // __EL_UTILS_HPP__
