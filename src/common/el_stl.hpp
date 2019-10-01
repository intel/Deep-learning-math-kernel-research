#pragma once

#include <type_traits>
#include "el_def.hpp"

// Euler non-standard STL utilities
namespace euler {
namespace estl {

// Integer container
template <int N>
struct integer {
  // For generic Lambda
  constexpr static inline int get() { return N; }
  constexpr static int value = N;
};

// Base sequence class
struct sequence {
  template <int I> constexpr static int get() { return 0; }
};

template <int... Is> struct integer_sequence_indexer;

// Integer sequence container
template <int... Ns> struct integer_sequence : sequence {
  template <int I> constexpr static int get()
  {
    return integer_sequence_indexer<Ns...>::template get<I>();
  }
};

// Make a sequence: entry
template <bool enabled, int from, int... Ns> struct make_integer_sequence {};

// Make a sequence: recursion
template <int from, int to, int... Ns>
struct make_integer_sequence<true, from, to, Ns...> {
  using type =
      typename make_integer_sequence<true, from, to - 1, to - 1, Ns...>::type;
};

// Make a sequence: end condition
template <int from, int... Ns>
struct make_integer_sequence<true, from, from, Ns...> {
  using type = integer_sequence<Ns...>;
};

// Assert invalid sequence
template <int from, int to, int... Ns>
struct make_integer_sequence<false, from, to, Ns...> {
  static_assert(from < to, "sequence (from, to) is invalid when from >= to!");
};

template <int from, int to>
using valid_range = typename make_integer_sequence <from < to, from, to>::type;

// Helper interfaces for make_interger_sequence
template <int from, int to> using range = valid_range<from, to>;
template <int to> using arange = valid_range<0, to>;

// Iterate the sequence
template <typename F, int... Ns>
inline void for_each(integer_sequence<Ns...>, F func)
{
#if __cpp_generic_lambdas && !__GCC_COMPILER
  (void)(int[]){ (func(integer<Ns>{}), 0)... };
#else
  // Workaround the lambda issue. XXX. It will depends on
  // compiler to optimize the runtime parameter passing.
  (void)(int[]){ (func(Ns), 0)... };
#endif
}

//
// Looping unrolling
//
// /* C++14 */
// estl::for_each (estl::range<4, 13>{}, [&](const auto n) {
//   constexpr N = n.get();
//   print<N>();
// });
//
// /* C++11 */
// estl::for_each (estl::range<4, 13>{}, [&](const int n) {
//   print(n);
// });
//

struct sequence_indexer {};

// Get sequece element by index
template <int...Vs>
struct integer_sequence_indexer : sequence_indexer {
  // Index, value pair
  template <int N, int V> struct pair {
    using type = decltype(integer<V>{});
  };

  // Base
  template <typename T, int ...Ms>
  struct zip;

  // Map index with value
  template <int ...Is>
  struct zip<integer_sequence<Is...>, Vs...> : pair<Is, Vs>... {};

  // Return value by index
  template <int N, int V> constexpr static int index(pair<N, V>) { return V; };

  template <int I> constexpr static int get()
  {
    static_assert(I < sizeof...(Vs), "Index overflow!");
    return index<I>(zip<arange<sizeof...(Vs)>, Vs...>{});
  }
};

// Helper interface to integer_sequence_indexer
// I: index, R: return type, T: sequence
template <int I, typename R, typename S>
static constexpr
typename std::enable_if<std::is_base_of<sequence, S>{}, R>::type get()
{
  return S::template get<I>();
}

// Indexing integer (bool, short, int, ...) sequence
//
// template <int ...Ns> struct {
//   using seq = integer_sequence<Ns...>;
//   constexpr auto a = estl::get<0, int, seq>();
// };
//

template <typename T, int n>
struct mm_reg {
  T& get() { return t_; }
  using type = T;
  T t_;
};

template<typename... Types>
struct mm_regs;

template <typename R, typename... Rs> struct mm_regs<R, Rs...> {
  template <typename _R>
  inline typename std::enable_if<std::is_same<R, _R>::value, _R &>::type
  get_element()
  {
    return r_;
  }

  template <typename _R>
  inline typename std::enable_if<!std::is_same<R, _R>::value, _R &>::type
  get_element()
  {
    return rs_.template get_element<_R>();
  }

  template <typename T, int n> inline T &get()
  {
    return get_element<mm_reg<T, n>>().get();
  }

  R r_;
  mm_regs<Rs...> rs_;
};

template <typename R> struct mm_regs<R> {
  template <typename _R> inline R& get_element() { return r_; }

  template <typename T, int n> inline T& get()
  {
    return get_element<mm_reg<T, n>>().get();
  }

  R r_;
};

template <typename T, typename N>
struct make_mm_regs {};

template <typename T, int ...Ns>
struct make_mm_regs<T, integer_sequence<Ns...>> {
  using type = mm_regs<mm_reg<T, Ns>...>;
};

template <typename T, int N>
using make_mm_arange = typename make_mm_regs<T, arange<N>>::type;

template <typename T, int M, int N>
struct mm_regs_matrix {
  using regs_matrix = typename make_mm_regs<T, arange<M * N>>::type;

  template <int m, int n>
  T& get() {
    return matrix_.template get<T, m * N + n>();
  }
  regs_matrix matrix_;
};

template <typename T, int N>
struct mm_regs_arange {
  using regs_arange = typename make_mm_regs<T, arange<N>>::type;

  template <int n>
  T& get() {
    return arange_.template get<T, n>();
  }
  regs_arange arange_;
};

// Regs array/matrix
//
// /* Allocate a 4*8 ZMM matrix */
// mm_regs_matrix<__m<V>, 4, 8> mm;
//
// /* Indexing */
// mm.get<3, 4>() = _mm<V>::set1_ps(0.0f);
//

template <typename T, typename U> inline bool any_of(T val, U last)
{ return val == last; }
template <typename T, typename U, typename... Args> inline bool any_of(
    T val, U first, Args... rest) {
  return (val == first) || any_of(val, rest...);
}

template <typename T, typename... Args> inline bool none_of(
    T val, Args... args) {
  return !any_of(val, args...);
}

template <typename T, typename U> inline bool all_of(T val, U last) {
  return val == last;
}

template <typename T, typename U, typename... Args> inline bool all_of(
    T val, U first, Args... rest) {
  return (val == first) && all_of(val, rest...);
}

template<typename T>
inline const T& max(const T& m, const T& n) {
    return m > n ? m : n;
}

template<typename T>
inline const T& min(const T& m, const T& n) {
    return m < n ? m : n;
}

template <typename T, std::size_t N>
std::size_t size(T (&)[N])
{
    return N;
}

} // namespace estl
} // namepsace euler
