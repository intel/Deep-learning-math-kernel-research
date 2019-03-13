#pragma once
#include <algorithm>

namespace euler {

#define IF
#define THEN ?
#define ELSE :

// (weights) pipeline length
//
// Wtype = fp32 || fp16:
//   O == 1: T + P <= 32
//   O > 1: O (T + P) + 1 <= 32
template <int O, int T, typename Wtype>
struct P_traits {
  static constexpr int P = IF (O == 1) THEN (
    IF (T <= 28) THEN (4) ELSE (
      IF (T == 29 || T == 30) THEN (2) ELSE (1)
    )
  ) ELSE (
    IF (O > 1 && (31 / O - T) >= 4) THEN (4) ELSE (
      IF (O > 1 && (31 / O - T == 2 || 31 / O - T == 3)) THEN (2) ELSE (1)
    )
  );
};

// Wtype = int8_t:
//   O == 1: T + P + 1(one) + 1(t0) <= 32
//   O > 1: O (T + P) + 1(bcast) + 1(one) + 1(t0) <= 32
template <int O, int T>
struct P_traits<O, T, int8_t> {
  static constexpr int P = IF (O == 1) THEN (
    IF (T <= 26) THEN (4) ELSE (
      IF (T == 27 || T == 28) THEN (2) ELSE (1)
    )
  ) ELSE (
    IF (O > 1 && (29 / O - T) >= 4) THEN (4) ELSE (
      IF (O > 1 && (29 / O - T == 2 || 29 / O - T == 3)) THEN (2) ELSE (1)
    )
  );
};

// Jamming
template <int O, int T, typename Wtype = float, typename C = void>
struct J_traits {};

template <int T, typename Wtype>
struct J_traits<8, T, Wtype, typename std::enable_if<T == 6>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, typename Wtype>
struct J_traits<8, T, Wtype,
    typename std::enable_if<T == 7 || T == 8, void>::type> {
  static constexpr int J = 3;
  static constexpr int O0 = 3;
  static constexpr int O1 = 3;
  static constexpr int O2 = 2;
  static constexpr int P0 = P_traits<O0, T, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Wtype>::P;
  static constexpr int P2 = P_traits<O2, T, Wtype>::P;
};

template <int T, typename Wtype>
struct J_traits<8, T, Wtype,
    typename std::enable_if<(T >= 3 && T < 6), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, typename Wtype>
struct J_traits<4, T, Wtype,
    typename std::enable_if<(T >= 7 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 2;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, typename Wtype>
struct J_traits<3, T, Wtype,
    typename std::enable_if<(T >= 10 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 1;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int O, int T>
struct J_traits<O, T, float,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, float>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T>
struct J_traits<O, T, short,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, short>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T>
struct J_traits<O, T, int8_t,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, int8_t>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int F>
struct F_traits {
  static constexpr bool is_compact_input = (F & 0xF00) == 0xC00;
  static constexpr bool is_blocked_input = (F & 0xF00) == 0xD00;
  static constexpr bool is_nchw_input = (F & 0xF00) == 0xE00;
  static constexpr bool is_nhwc_input = (F & 0xF00) == 0xF00;
  static constexpr bool is_compact_weights = (F & 0xF0) == 0xC0;
  static constexpr bool is_compact_ir_weights = (F & 0xF0) == 0xB0;
  static constexpr bool is_compact_output = (F & 0xF) == 0xC;
  static constexpr bool is_blocked_output = (F & 0xF) == 0xD;
  static constexpr bool is_nhwc_output = (F & 0xF) == 0xF;
};

} // namespace euler
