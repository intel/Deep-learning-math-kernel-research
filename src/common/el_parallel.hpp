#pragma once

#include <type_traits>
#include <cstdlib>
#include <cassert>
#include "el_def.hpp"
#if MT_RUNTIME == MT_RUNTIME_OMP
# include <omp.h>
#elif MT_RUNTIME == MT_RUNTIME_TBB
# include "tbb/parallel_for.h"
# include "tbb/task_arena.h"
#else
# error Invalid MT_RUNTIME
#endif

#define _CAT(a, b) __CAT(a, b)
#define __CAT(a, b) a##_##b

#define _LOOP_ORDER_1(a1) _CAT(__loop_order, a1)
#define _LOOP_ORDER_2(a1, a2) _CAT(_LOOP_ORDER_1(a1), a2)
#define _LOOP_ORDER_3(a1, a2, a3) _CAT(_LOOP_ORDER_2(a1, a2), a3)
#define _LOOP_ORDER_4(a1, a2, a3, a4) _CAT(_LOOP_ORDER_3(a1, a2, a3), a4)
#define _LOOP_ORDER_5(a1, a2, a3, a4, a5) _CAT(_LOOP_ORDER_4(a1, a2, a3, a4), a5)
#define _LOOP_ORDER_6(a1, a2, a3, a4, a5, a6) _CAT(_LOOP_ORDER_5(a1, a2, a3, a4, a5), a6)
#define _LOOP_ORDER_7(a1, a2, a3, a4, a5, a6, a7) _CAT(_LOOP_ORDER_6(a1, a2, a3, a4, a5, a6), a7)
#define _LOOP_ORDER_8(a1, a2, a3, a4, a5, a6, a7, a8) _CAT(_LOOP_ORDER_7(a1, a2, a3, a4, a5, a6, a7), a8)

#define INIT_LOOP_ORDER(n)  int __loop_order_##n = 0;
#define SET_LOOP_ORDER(n, ...) __loop_order_##n = _LOOP_ORDER_##n(__VA_ARGS__)
#define CHECK_LOOP_ORDER(n, ...)  (__loop_order_##n == _LOOP_ORDER_##n(__VA_ARGS__))
#define CREATE_LOOP_ORDER(n, ...) const int _LOOP_ORDER_##n(__VA_ARGS__) = ++__loop_order_##n;

namespace euler {
namespace estl {

#if MT_RUNTIME == MT_RUNTIME_OMP
static inline int current_thread_index() {
  return omp_get_thread_num();
}
static inline int max_concurrency() {
  return omp_get_max_threads();
}
#define THREAD_PARALLEL() _Pragma("omp parallel")
#define THREAD_BARRIER() _Pragma("omp barrier")
#define THREAD_FOR(N, mthr, ithr, ...) estl::thread_parallel_for<N>(mthr, ithr, __VA_ARGS__)
#define THREAD_FOR2(N, M, mthr, ithr, ...) estl::thread_parallel_for<N, M>(mthr, ithr, __VA_ARGS__)

#elif MT_RUNTIME == MT_RUNTIME_TBB
static inline int current_thread_index() {
  return tbb::this_task_arena::current_thread_index();
}
static inline int max_concurrency() {
  return tbb::this_task_arena::max_concurrency();
}
#define THREAD_PARALLEL()
#define THREAD_BARRIER()
#define THREAD_FOR(N, mthr, ithr, ...) estl::parallel_for<N>(mthr, __VA_ARGS__)
#define THREAD_FOR2(N, M, mthr, ithr, ...) estl::parallel_for<N, M>(mthr, __VA_ARGS__)
#endif // MT_RUNTIME

// Loops over N loops in current thread.
// The M-th loop will not be used for task allocation.
template <int N, int M = -1> struct thread_parallel_for {
  static_assert(N > 0, "N > 0 required");
  static_assert(M == -1 || (M >= 0 && M < N), "M should be -1 or in [0, N)");

  template <typename F, typename... Args>
  thread_parallel_for(int mthr, int ithr, F func, Args... Xs)
      : loops_{ Xs... }
  {
    static_assert(N == sizeof...(Xs), "N != sizeof(Xs...)");

    nb_tasks_ = 1;
    for (int i = 0; i < N; ++i) {
      if (i != M) {
        nb_tasks_ *= loops_[i];
      }
    }
    alloc_thread_task(nb_tasks_, mthr, ithr, task_start_, task_end_);

    if (task_start_ <= task_end_) {
      build_loop_index(task_start_, iterators_lim_from_);
      build_loop_index(task_end_, iterators_lim_to_);

      int next_L = M == 0 ? 1 : 0;
      loop_for<F, 0>(func, iterators_lim_from_[next_L],
                     iterators_lim_to_[next_L], true, true);
    }
  }

  private:
  template <typename F, int L, typename... Args>
  inline typename std::enable_if<L == N - 1>::type loop_for(
      F func, int from, int to, bool, bool, Args... iterators)
  {
    if (L == M) {
      for (int i = 0; i < loops_[L]; ++i) {
        func(iterators..., i);
      }
    } else {
      for (int i = from; i <= to; ++i) {
        func(iterators..., i);
      }
    }
  }

  template <typename F, int L, typename... Args>
  inline typename std::enable_if<L == M && L != N - 1>::type loop_for(
      F func, int from, int to, bool is_first, bool is_last, Args... iterators)
  {
    for (int i = 0; i < loops_[M]; ++i) {
      loop_for<F, L + 1>(func, from, to, is_first, is_last, iterators..., i);
    }
  }

  template <typename F, int L, typename... Args>
  inline typename std::enable_if<(L < (N - 1)) && (L != M)>::type loop_for(
      F func, int from, int to, bool is_first, bool is_last, Args... iterators)
  {
    for (int i = from; i <= to; ++i) {
      bool is_first_L = is_first && i == iterators_lim_from_[L];
      bool is_last_L = is_last && i == iterators_lim_to_[L];
      int next_L = (L + 1) == M ? L + 2 : L + 1;
      int next_from = is_first_L ? iterators_lim_from_[next_L] : 0;
      int next_to = is_last_L ? iterators_lim_to_[next_L] : loops_[next_L] - 1;
      loop_for<F, L + 1>(
          func, next_from, next_to, is_first_L, is_last_L, iterators..., i);
    }
  }

  inline void alloc_thread_task(int n, int mthr, int ithr, int &start, int &end)
  {
    assert(n > 0);

    auto res = std::div(n, mthr);
    auto base = res.quot;
    auto more = res.rem;

    auto select = static_cast<int>((ithr - 1) < more);
    start = (base + select) * ithr + more * (1 - select);

    // right close set??? How about open to save a op
    end = start + base + static_cast<int>(ithr < more) -1;
  }

  inline void build_loop_index(int taskid, int indices[N])
  {
    int n = taskid;
    for (auto i = N - 1; i >= 0; --i) {
      if (i != M) {
        auto res = std::div(n, loops_[i]);
        indices[i] = res.rem;
        n = res.quot;
      } else {
        indices[i] = 0;
      }
    }
  }

  int iterators_lim_from_[N], iterators_lim_to_[N], loops_[N];
  int nb_tasks_, task_start_, task_end_;
};

template <int N, int M, typename F, typename... Args>
static inline void parallel_for(int mthr, F func, Args... args)
{
#if MT_RUNTIME == MT_RUNTIME_OMP
#pragma omp parallel num_threads(mthr) proc_bind(close)
  {
    int ithr = omp_get_thread_num();
    thread_parallel_for<N, M>(mthr, ithr, func, args...);
  }
#elif MT_RUNTIME == MT_RUNTIME_TBB
  tbb::parallel_for(0, mthr, [&](int ithr) {
    thread_parallel_for<N, M>(mthr, ithr, func, args...);
  }, tbb::static_partitioner());
#endif
}

template <int N, int M, typename F, typename... Args>
static inline void parallel_for(F func, Args... args)
{
  int mthr = estl::max_concurrency();
#if MT_RUNTIME == MT_RUNTIME_OMP
#pragma omp parallel num_threads(mthr) proc_bind(close)
  {
    int ithr = omp_get_thread_num();
    thread_parallel_for<N, M>(mthr, ithr, func, args...);
  }
#elif MT_RUNTIME == MT_RUNTIME_TBB
  tbb::parallel_for(0, mthr, [&](int ithr) {
    thread_parallel_for<N, M>(mthr, ithr, func, args...);
  }, tbb::static_partitioner());
#endif
}

template <int N, typename F, typename... Args>
static inline void parallel_for(int mthr, F func, Args... args)
{
#if MT_RUNTIME == MT_RUNTIME_OMP
#pragma omp parallel num_threads(mthr) proc_bind(close)
  {
    int ithr = omp_get_thread_num();
    thread_parallel_for<N, -1>(mthr, ithr, func, args...);
  }
#elif MT_RUNTIME == MT_RUNTIME_TBB
  tbb::parallel_for(0, mthr, [&](int ithr) {
    thread_parallel_for<N, -1>(mthr, ithr, func, args...);
  }, tbb::static_partitioner());
#endif
}

template <int N, typename F, typename... Args>
static inline void parallel_for(F func, Args... args)
{
#if MT_RUNTIME == MT_RUNTIME_OMP
#pragma omp parallel proc_bind(close)
  {
    int mthr = omp_get_max_threads();
    int ithr = omp_get_thread_num();
    thread_parallel_for<N, -1>(mthr, ithr, func, args...);
  }
#elif MT_RUNTIME == MT_RUNTIME_TBB
  int mthr = estl::max_concurrency();
  tbb::parallel_for(0, mthr, [&](int ithr) {
    thread_parallel_for<N, -1>(mthr, ithr, func, args...);
  }, tbb::static_partitioner());
#endif
}

} // namespace estl
} // namespace euler
