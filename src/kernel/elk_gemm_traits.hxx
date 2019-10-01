#pragma once
#include <algorithm>

namespace euler {

enum {
  K_GEMM,
  K_CONV
};

#define IF
#define THEN ?
#define ELSE :

// (weights) pipeline length
//
// K_GEMM, Wtype = fp32 || fp16:
//   O == 1: T + P <= 32
//   O > 1: O (T + P) + 1 <= 32
template <int O, int T, int Ktype, typename Wtype>
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

#if defined(WITH_VNNI)
// Wtype = int8_t:
//   O == 1: T + P <= 32
//   O > 1: O (T + P) + 1(bcast) <= 32
template <int O, int T>
struct P_traits<O, T, K_GEMM, int8_t> {
  static constexpr int P = IF (O == 1) THEN (
    IF (T <= 28) THEN (4) ELSE (
      IF (T == 29 || T == 30) THEN (2) ELSE (1)
    )
  ) ELSE (
    IF (O > 1 && (32 / O - T) >= 4) THEN (4) ELSE (
      IF (O > 1 && (32 / O - T == 2 || 32 / O - T == 3)) THEN (2) ELSE (1)
    )
  );
};

template <int O, int T>
struct P_traits<O, T, K_CONV, int8_t> {
  static constexpr int P = IF (O == 1) THEN (
    IF (T <= 27) THEN (4) ELSE (
      IF (T == 28 || T == 29) THEN (2) ELSE (1)
    )
  ) ELSE (
    IF (O > 1 && (31 / O - T) >= 4) THEN (4) ELSE (
      IF (O > 1 && (31 / O - T == 2 || 31 / O - T == 3)) THEN (2) ELSE (1)
    )
  );
};

#else

// Wtype = int8_t:
//   O == 1: T + P + 1(one) + 1(t0) <= 32
//   O > 1: O (T + P) + 1(bcast) + 1(one) + 1(t0) <= 32
template <int O, int T, int Ktype>
struct P_traits<O, T, Ktype, int8_t> {
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
#endif

// Loop splitting
template <int O, int T, int Ktype, typename Wtype = float, typename C = void>
struct J_traits {};

template <int T, int Ktype, typename Wtype>
struct J_traits<8, T, Ktype, Wtype, typename std::enable_if<T == 6>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Ktype, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, int Ktype, typename Wtype>
struct J_traits<8, T, Ktype, Wtype,
    typename std::enable_if<T == 7 || T == 8, void>::type> {
  static constexpr int J = 3;
  static constexpr int O0 = 3;
  static constexpr int O1 = 3;
  static constexpr int O2 = 2;
  static constexpr int P0 = P_traits<O0, T, Ktype, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Ktype, Wtype>::P;
  static constexpr int P2 = P_traits<O2, T, Ktype, Wtype>::P;
};

template <int T, int Ktype, typename Wtype>
struct J_traits<8, T, Ktype, Wtype,
    typename std::enable_if<(T >= 3 && T < 6), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Ktype, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, int Ktype, typename Wtype>
struct J_traits<4, T, Ktype, Wtype,
    typename std::enable_if<(T >= 7 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 2;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Ktype, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, int Ktype, typename Wtype>
struct J_traits<3, T, Ktype, Wtype,
    typename std::enable_if<(T >= 10 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 1;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, Ktype, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int O, int T, int Ktype>
struct J_traits<O, T, Ktype, float,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, float>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T, int Ktype>
struct J_traits<O, T, Ktype, short,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, short>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T, int Ktype>
struct J_traits<O, T, Ktype, int8_t,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, Ktype, int8_t>::P;
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
