#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename InputType, int format, bool is_border,
    int V>
struct elk_conv_wino_trans_input<float, InputType, format, is_border,
    ISA_SKX_AVX512, 5, V> {
  constexpr static int A = 5;

  static void execute(elx_conv_params_t &xc, float atinput[A][A][V],
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end)
  {
    auto f_cb = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(InputType, ainput, input, A, A, V);
        if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else if (format == TKF_BLOCKED) {
        MD3(InputType, ainput, input, xc.ih, xc.iw, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return _mm<V>::setzero_ps();
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else {
        MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.ic);
        // TODO: overflow on last V
        MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0),
            xc.ic4 * xc.ic3 * xc.I2, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return _mm<V>::setzero_ps();
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md2(ainput1, 0, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md2(ainput1, 0, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      }
    };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m, n) f(m, n) = F(m, n)

    __m<V> M[5][5];

    auto z0 = _mm<V>::set1_ps(0.5f);
    auto z1 = _mm<V>::set1_ps(1.5f);

#pragma unroll
    for (int i = 0; i < 5; i++) {
      auto f0 = F(0, i);
      auto f1 = F(1, i);
      auto f2 = F(2, i);
      auto f3 = F(3, i);
      auto f4 = F(4, i);

      auto t0 = f1 * z0;
      auto t1 = f2 * z0;
      auto t2 = f3 - f1;

      M[0][i] = f0 * z0 - t1 - t2;
      M[1][i] = f3 - t0 - t1;
      M[2][i] = f2 * z1 + f3 + t0;
      M[3][i] = t2;
      M[4][i] = f3 * z0 + f4 - f2 - t0;
    }

#pragma unroll
    for (int i = 0; i < 5; i++) {
      auto f0 = M[i][0];
      auto f1 = M[i][1];
      auto f2 = M[i][2];
      auto f3 = M[i][3];
      auto f4 = M[i][4];

      auto t0 = f1 * z0;
      auto t1 = f2 * z0;
      auto t2 = f3 - f1;

      *(__m<V> *)T(i, 0) = f0 * z0 - t1 - t2;
      *(__m<V> *)T(i, 1) = f3 - t0 - t1;
      *(__m<V> *)T(i, 2) = f2 * z1 + f3 + t0;
      *(__m<V> *)T(i, 3) = t2;
      *(__m<V> *)T(i, 4) = f3 * z0 + f4 - f2 - t0;
    }
  }
}; // elk_conv_wino_trans_input

} // namespace euler
