#include <cmath>
#include <string.h>
#include <float.h>
#include "el_intrin.hpp"
#include "el_parallel.hpp"
#include "elx_conv_wino_trans_input.hpp"

namespace euler {

#define t2spati(                                                             \
    __t2, __T, __n, __ih, __iw, __hA_start, __hA_end, __wA_start, __wA_end)  \
  do {                                                                       \
    int _t = __t2 * ep->T + __T;                                           \
    int _nt = _t % ep->nt;                                                 \
    int _ht = _nt / ep->wt;                                                \
    int _wt = _nt % ep->wt;                                                \
    __n = _t / ep->nt;                                                     \
    __ih = _ht * (A - K + 1) - ep->tp;                                     \
    __iw = _wt * (A - K + 1) - ep->lp;                                     \
    __hA_start = (_ht > 0) ? 0 : ep->tp;                                   \
    __wA_start = (_wt > 0) ? 0 : ep->lp;                                   \
    __hA_end = (_ht < ep->ht - 1) ? A - 1 : hA_end_;                       \
    __wA_end = (_wt < ep->wt - 1) ? A - 1 : wA_end_;                       \
  } while (0)

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_blocked(TinputType *__restrict tinput,
    InputType *__restrict input, int Tz, int _t2, int _I4) {
  // n, ic2, ih, iw, V => t2 | hA, wA, I3, I2, T, V
  MD7(InputType, ainput, input,
      ep->n, ep->I4, ep->I3, ep->I2, ep->ih, ep->iw, V);
  alignas(64) op_type aout[A][A][V];

  auto res = std::div(_t2 * ep->T, ep->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  iter_each (_I3, ep->I3) {
  iter_each (_I2, ep->I2) {
  input_tile_iter<A, K> t2spati_o(_n, _t_off, ep->ht, ep->wt,
      ep->ih, ep->iw, ep->tp, ep->lp);
  iter_each (_T, Tz) {
    auto _ih = t2spati_o.anchor_t_;
    auto _iw = t2spati_o.anchor_l_;

    InputType *in = &md7(ainput, t2spati_o.n_, _I4, _I3, _I2, _ih, _iw, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*ep, (float *)&aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);
    __execute_post(tinput, (op_type *)aout, Tz, _I3, _I2, _T);

    ++ t2spati_o;
  }}}
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_blocked(TinputType *__restrict tinput,
    InputType *__restrict input, int _I4) {
  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr, [&](int _t2, int _I3, int _I2) {
    // n, ic2, ih, iw, V => t2, hA, wA, I3, I2, T, V
    MD2(TinputType, atinput, tinput, ep->t2,
        A * A * ep->T * ep->I3 * ep->I2 * V);
    MD7(InputType, ainput, input, ep->n, ep->I4, ep->I3,
        ep->I2, ep->ih, ep->iw, V);
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    alignas(64) op_type aout[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      auto *in = &md7(ainput, _n, _I4, _I3, _I2, _ih, _iw, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *ep, (float *)&aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
      __execute_post(&md2(atinput, _t2, 0), (op_type *)&aout, Tz, _I3, _I2, _T);
    }
  }, ep->t2, ep->I3, ep->I2);
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_post(TinputType *__restrict tinput,
    op_type *tbuf, int Tz, int _I3, int _I2, int _T) {
  MD6(TinputType, atinput6, tinput, A, A, ep->I3, ep->I2, Tz, V);

  MD3(op_type, at, tbuf, A, A, V);
  if (I == ISA_AVX512 && std::is_same<op_type, float>::value
      && std::is_same<TinputType, float>::value) {
    if (stream_in_) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        _mm<V>::stream_ps(&md6(atinput6, _hA, _wA, _I3, _I2, _T, 0),
                       *((__m<V> *)&md3(at, _hA, _wA, 0)));
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        _mm<V>::store_ps(&md6(atinput6, _hA, _wA, _I3, _I2, _T, 0),
                      *((__m<V> *)&md3(at, _hA, _wA, 0)));
      }}
    }
  } else if (I == ISA_AVX512 && std::is_same<op_type, float>::value
     && std::is_same<TinputType, float16>::value) {
    if (stream_in_) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        auto fp16v = _mm<V>::cvt_f32_b16(*(__i<V> *)&md3(at, _hA, _wA, 0));
        _mm<V/2>::stream_si256(
            (__m256i *)&md6(atinput6, _hA, _wA, _I3, _I2, _T, 0), fp16v);
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        auto fp16v = _mm<V>::cvt_f32_b16(*(__i<V> *)&md3(at, _hA, _wA, 0));
        _mm<V/2>::store_si256(
            (__m256i *)&md6(atinput6, _hA, _wA, _I3, _I2, _T, 0), fp16v);
      }}
    }
  } else {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
#pragma omp simd
    iter_each (_V, V) {
      md6(atinput6, _hA, _wA, _I3, _I2, _T, _V) = md3(at, _hA, _wA, _V);
    }}}
  }
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_nhwc(TinputType *__restrict tinput,
    InputType *__restrict input, int _I4) {

  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr, [&](int _t2, int _I3, int _I2) {
    // n, ih, iw, ic => t2, hA, wA, I3, I2, T, V
    MD2(TinputType, atinput, tinput, ep->t2,
        A * A * ep->T * ep->I3 * ep->I2 * V);
    MD4(InputType, ainput0, input, ep->n, ep->ih, ep->iw, ep->ic);
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    alignas(64) op_type aout[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
      MD4(InputType, ainput1, &md4(ainput0, _n, _ih, _iw, 0), ep->I4,
          ep->I3, ep->I2, V);
      InputType *in = &md4(ainput1, _I4, _I3, _I2, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *ep, (float *)&aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
      __execute_post(&md2(atinput, _t2, 0), (op_type *)&aout, Tz, _I3, _I2, _T);
    }
  }, ep->t2, ep->I3, ep->I2);
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_nhwc(TinputType *__restrict tinput,
    InputType *__restrict input, int Tz, int _t2, int _I4) {
  // n, ih, iw, ic2, V => t2 | hA, wA, I3, I2, T, V
  MD4(InputType, ainput0, input, ep->n, ep->ih, ep->iw, ep->ic);
  alignas(64) op_type aout[A][A][V];

  auto res = std::div(_t2 * ep->T, ep->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  iter_each (_I3, ep->I3) {
  iter_each (_I2, ep->I2) {
  input_tile_iter<A, K> t2spati_o(_n, _t_off, ep->ht, ep->wt,
      ep->ih, ep->iw, ep->tp, ep->lp);
  iter_each (_T, Tz) {
    auto _ih = t2spati_o.anchor_t_;
    auto _iw = t2spati_o.anchor_l_;

    MD4(InputType, ainput1, &md4(ainput0, t2spati_o.n_, _ih, _iw, 0),
        ep->I4, ep->I3, ep->I2, V);
    InputType *in = &md4(ainput1, _I4, _I3, _I2, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*ep, (float *)&aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);
    __execute_post(tinput, (op_type *)&aout, Tz, _I3, _I2, _T);

    ++ t2spati_o;
  }}}
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_nchw(TinputType *__restrict tinput,
    InputType *__restrict input, int _I4) {
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ih * ep->iw));

  auto readin = [&](InputType ain[A][A][V], int _t2, int _I3, int _I2, int _T,
                    bool is_Ir) {
    MD2(InputType, ainput0, input, ep->n, ep->ic * ep->ih * ep->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD6(InputType, ainput1, &md2(ainput0, _n, 0), ep->I4, ep->I3,
        ep->I2, V, ep->ih, ep->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            iter_each(_V, ep->Ir) ain[_hA][_wA][_V] =
              md6(ainput1, _I4, _I3, _I2, _V, _ih + _hA, _iw + _wA);
          }
        }
      }
    } else {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
            if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
              constexpr int scale = sizeof(InputType);
              __m<V> t = _mm<V>::i32gather_ps(vindex,
                  &md6(ainput1, _I4, _I3, _I2, 0, _ih + _hA, _iw + _wA), scale);
              _mm<V>::store_ps(ain[_hA][_wA], t);
            } else {
#pragma omp simd
              iter_each(_V, V) ain[_hA][_wA][_V] =
                  md6(ainput1, _I4, _I3, _I2, _V, _ih + _hA, _iw + _wA);
            }
          }
        }
      }
    }
  };

  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr, [&](int _t2, int _I3, int _I2) {
    // n, ic2, ih, iw, V => t2, hA, wA, I3, I2, T, V
    MD2(TinputType, atinput, tinput, ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1 &&
         _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    alignas(64) op_type aout[A][A][V];
    alignas(64) InputType ain[A][A][V];
    iter_each(_T, Tz) {
      readin(ain, _t2, _I3, _I2, _T, is_Ir);
      ker_trans_input_(*ep, (float *)&aout, (InputType *)ain, 0, 0, 0, -1);
      __execute_post(&md2(atinput, _t2, 0), (op_type *)&aout, Tz, _I3, _I2, _T);
    }
  }, ep->t2, ep->I3, ep->I2);
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::__execute_nchw(TinputType *__restrict tinput,
    InputType *__restrict input, int Tz, int _t2, int _I4) {
  // n, IC, ih, iw => t2 | hA, wA, I3, I2, T, V
  alignas(64) op_type aout[A][A][V];
  alignas(64) InputType ain[A][A][V];
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ih * ep->iw));

  auto readin = [&](InputType ain[A][A][V], int _I3, int _I2, int _T, bool is_Ir) {
    MD2(InputType, ainput0, input, ep->n, ep->ic * ep->ih * ep->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD6(InputType, ainput1, &md2(ainput0, _n, 0), ep->I4, ep->I3,
        ep->I2, V, ep->ih, ep->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0.0f;
        } else {
          iter_each(_V, ep->Ir) {
            ain[_hA][_wA][_V] =
                md6(ainput1, _I4, _I3, _I2, _V, _ih + _hA, _iw + _wA);
          }
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0.0f;
        } else {
          if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md6(ainput1, _I4, _I3, _I2, 0, _ih + _hA, _iw + _wA),
                scale);
            _mm<V>::store_ps(ain[_hA][_wA], t);
          } else {
#pragma omp simd
            iter_each (_V, V)
              ain[_hA][_wA][_V] = md6(ainput1, _I4, _I3, _I2, _V,
                                      _ih + _hA, _iw + _wA);
          }
        }
      }}
    }
  };

  iter_each (_I3, ep->I3) {
  iter_each (_I2, ep->I2) {
    bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1 &&
                 _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    iter_each (_T, Tz) {
      readin(ain, _I3, _I2, _T, is_Ir);
      ker_trans_input_(*ep, (float *)&aout, (InputType *)ain, 0, 0, 0, -1);
      __execute_post(tinput, (op_type *)&aout, Tz, _I3, _I2, _T);
    }
  }}
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::execute(TinputType *tinput, InputType *input, int _I4) {
  if (input_is_bfmt_ || input_as_bfmt_)
    __execute_blocked(tinput, input, _I4);
  else if (ep->input_fmt == nhwc)
    __execute_nhwc(tinput, input, _I4);
  else
    __execute_nchw(tinput, input, _I4);
}

template <typename TinputType, typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
::execute(TinputType *tinput, InputType *input, int Tz, int _t2, int _I4) {
  if (input_is_bfmt_ || input_as_bfmt_)
    __execute_blocked(tinput, input, Tz, _t2, _I4);
  else if (ep->input_fmt == nhwc)
    __execute_nhwc(tinput, input, Tz, _t2, _I4);
  else
    __execute_nchw(tinput, input, Tz, _t2, _I4);
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::__execute_blocked_nhwc(float *tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _I4)
{
  __m<V> mrepS = _mm<V>::set1_ps(ep->input_quant_S * ep->tinput_quant_repS);
  __m<V> mz = _mm<V>::set1_ps(ep->tinput_quant_z);

  int ithr = estl::current_thread_index();
  THREAD_FOR(4, mthr_, ithr, [&](int _t2, int _I3, int _I2, int _T) {
    MD2(uint8_t, atinput2_u8, tinput_u8,
        ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    MD2(TinputType, atinput2, tinput,
        ep->t2, A * A * ep->I3 * ep->I2 * ep->T * V);
    MD7(InputType, ainput_blocked, input,
        ep->n, ep->I4, ep->I3, ep->I2, ep->ih, ep->iw, V);
    MD7(InputType, ainput_nhwc, input,
        ep->n, ep->ih, ep->iw, ep->I4, ep->I3, ep->I2, V); // TODO: Ir

    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;

    if (_T < Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      InputType *in = ep->input_fmt == nhwc
                    ? &md7(ainput_nhwc, _n, _ih, _iw, _I4, _I3, _I2, 0)
                    : &md7(ainput_blocked, _n, _I4, _I3, _I2, _ih, _iw, 0);

      if (ep->sampling_kind == CALIBRATED) {
        MD6(uint8_t, atinput_u8, &md2(atinput2_u8, _t2, 0),
            A, A, ep->I3, ep->I2, Tz, V);
        alignas(64) op_type aout[A][A][V];

        if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
            && _wA_end == A - 1)
          ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(*ep, (float *)&aout, in, _hA_start, _hA_end, _wA_start, _wA_end);

        iter_each (_hA, A) {
        iter_each (_wA, A) {
          // Min-Max quantization
          __m<V> a = *(__m<V> *)&aout[_hA][_wA][0];
          __m<V> mresf32 = a * mrepS + mz;
          // convert to uint8
          __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(
              mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
          // store
          _mm_store_si128((__m128i *)&md6(
              atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
        }}
      } else {
        MD4(TinputType, atinput4, &md2(atinput2, _t2, 0),
            ep->I3, ep->I2, Tz, A * A * V);
        auto aout = &md4(atinput4, _I3, _I2, _T, 0);

        if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
            && _wA_end == A - 1)
          ker_trans_input_(*ep, aout, in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(
              *ep, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
      }
    }
  }, ep->t2, ep->I3, ep->I2, ep->T);

  if (ep->sampling_kind == CALIBRATED)
    return;

  THREAD_BARRIER()

  THREAD_FOR(4, mthr_, ithr, [&](int _t2, int _hA, int _wA, int _I3) {
    MD2(uint8_t, atinput2_u8, tinput_u8,
        ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    MD6(float, atinput_scale, tinput_scale,
        ep->t2, A, A, ep->I3, 2, ep->T);
    MD2(TinputType, atinput2, tinput,
        ep->t2, A * A * ep->I3 * ep->I2 * ep->T * V);
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0),
        ep->I3, ep->I2, Tz, A, A, V);
    MD6(uint8_t, atinput_u8, &md2(atinput2_u8, _t2, 0),
        A, A, ep->I3, ep->I2, Tz, V);
    iter_each (_T, Tz) {
      __m<V> mmax, mmin;
      bool flush = true;
      iter_each (_I2, ep->I2) {
        __m<V> mcur = *(__m<V> *)&md6(atinput6, _I3, _I2, _T, _hA, _wA, 0);
        if (flush) {
          mmax = mcur;
          mmin = mcur;
          flush = false;
        } else {
          mmax = _mm<V>::max_ps(mcur, mmax);
          mmin = _mm<V>::min_ps(mcur, mmin);
        }
      }

      TinputType max, min;
      if (I == ISA_AVX512 && std::is_same<TinputType, float>::value) {
        max = _mm<V>::reduce_max_ps(mmax);
        min = _mm<V>::reduce_min_ps(mmin);
      } else {
        TinputType *_mmax = (TinputType *)&mmax;
        TinputType *_mmin = (TinputType *)&mmin;
        max = _mmax[0];
        min = _mmin[0];
        for (int _V = 1; _V < V; _V++) {
          max = _mmax[_V] > max ? _mmax[_V] : max;
          min = _mmin[_V] < min ? _mmin[_V] : min;
        }
      }

      auto delta = max - min + 0.000001;
      float S = delta / EL_UINT8_MAX;
      float repS = EL_UINT8_MAX / delta;
      float z = std::ceil(- min * repS);
      md6(atinput_scale, _t2, _hA, _wA, _I3, 0, _T) = S;
      md6(atinput_scale, _t2, _hA, _wA, _I3, 1, _T) = z;

      __m<V> mrepS = _mm<V>::set1_ps(repS);
      __m<V> mz = _mm<V>::set1_ps(z);
      iter_each (_I2, ep->I2) {
        __m<V> f = *(__m<V> *)&md6(atinput6, _I3, _I2, _T, _hA, _wA, 0);
        __m<V> mresf32 = f * mrepS + mz;
        __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(
            mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
        _mm_store_si128((__m128i *)&md6(
            atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
      }
    }
  }, ep->t2, A, A, ep->I3);
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::__execute_nchw(float *tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _I4) {
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ih * ep->iw));

  auto readin = [&](InputType ain[A][A][V], int _t2, int _I3, int _I2, int _T,
                    bool is_Ir) {
    MD2(InputType, ainput0, input, ep->n, ep->ic * ep->ih * ep->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD6(InputType, ainput1, &md2(ainput0, _n, 0), ep->I4, ep->I3,
        ep->I2, V, ep->ih, ep->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            iter_each(_V, ep->Ir) ain[_hA][_wA][_V] =
              md6(ainput1, 0, _I3, _I2, _V, _ih + _hA, _iw + _wA);
          }
        }
      }
    } else {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
            if (I == ISA_AVX512 && std::is_same<InputType, float>::value) {
              constexpr int scale = sizeof(InputType);
              __m<V> t = _mm<V>::i32gather_ps(vindex,
                  &md6(ainput1, 0, _I3, _I2, 0, _ih + _hA, _iw + _wA), scale);
              _mm<V>::store_ps(ain[_hA][_wA], t);
            } else {
#pragma omp simd
              iter_each(_V, V) ain[_hA][_wA][_V] =
                  md6(ainput1, 0, _I3, _I2, _V, _ih + _hA, _iw + _wA);
            }
          }
        }
      }
    }
  };

  __m<V> mrepS = _mm<V>::set1_ps(ep->input_quant_S * ep->tinput_quant_repS);
  __m<V> mz = _mm<V>::set1_ps(ep->tinput_quant_z);
  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr, [&](int _t2, int _I3, int _I2) {
    // n, ic2, ih, iw, V => t2, hA, wA, I3, I2, T, V
    MD2(uint8_t, atinput2_u8, tinput_u8,
        ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1 &&
         _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    alignas(64) op_type aout[A][A][V];
    alignas(64) InputType ain[A][A][V];
    iter_each(_T, Tz) {
      readin(ain, _t2, _I3, _I2, _T, is_Ir);
      ker_trans_input_(*ep, (float *)&aout, (InputType *)ain, 0, 0, 0, -1);

      MD6(uint8_t, atinput_u8, &md2(atinput2_u8, _t2, 0),
          A, A, ep->I3, ep->I2, Tz, V);
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        // Min-Max quantization
        __m<V> a = *(__m<V> *)&aout[_hA][_wA][0];
        __m<V> mresf32 = a * mrepS + mz;
        // convert to uint8
        __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(
            mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
        // store
        _mm_store_si128((__m128i *)&md6(
            atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
      }}
    }
  }, ep->t2, ep->I3, ep->I2);
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::execute(float *__restrict tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _I4) {
  if (input_is_bfmt_ || input_as_bfmt_ || ep->input_fmt == nhwc)
    __execute_blocked_nhwc(tinput_scale, tinput_u8, tinput, input, _I4);
  else
    __execute_nchw(tinput_scale, tinput_u8, tinput, input, _I4);
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::__execute_blocked_nhwc(float *tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _t2, int Tz)
{
  MD7(InputType, ainput_blocked, input,
      ep->n, ep->I4, ep->I3, ep->I2, ep->ih, ep->iw, V);
  MD7(InputType, ainput_nhwc, input,
      ep->n, ep->ih, ep->iw, ep->I4, ep->I3, ep->I2, V);

  MD6(uint8_t, atinput_u8, tinput_u8, A, A, ep->I3, ep->I2, Tz, V);
  MD5(float, atinput_scale, tinput_scale, ep->I3, A, A, 2, ep->T);

  auto res = std::div(_t2 * ep->T, ep->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  if (ep->sampling_kind == CALIBRATED) {
    __m<V> mrepS = _mm<V>::set1_ps(ep->input_quant_S * ep->tinput_quant_repS);
    __m<V> mz = _mm<V>::set1_ps(ep->tinput_quant_z);
    alignas(64) op_type aout[A][A][V];

    iter_each(_I3, ep->I3) {
    iter_each (_I2, ep->I2) {
      input_tile_iter<A, K> t2spati_o(_n, _t_off, ep->ht, ep->wt, ep->ih,
                                      ep->iw, ep->tp, ep->lp);
      iter_each (_T, Tz) {
        auto _ih = t2spati_o.anchor_t_;
        auto _iw = t2spati_o.anchor_l_;

        InputType *in = ep->input_fmt == nhwc
                ? &md7(ainput_nhwc, t2spati_o.n_, _ih, _iw, 0, _I3, _I2, 0)
                : &md7(ainput_blocked, t2spati_o.n_, 0, _I3, _I2, _ih, _iw, 0);
        if (!t2spati_o.is_border())
          ker_trans_input_(*ep, (float *)&aout, in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(*ep, (float *)&aout, in,
              t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

        ++ t2spati_o;

        iter_each (_hA, A) {
        iter_each (_wA, A) {
          // Min-Max quantization
          __m<V> a = *(__m<V> *)&aout[_hA][_wA][0];
          __m<V> mresf32 = a * mrepS + mz;
          // convert to uint8
          __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(
              mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
          // store
          _mm_store_si128((__m128i *)&md6(atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
        }}
      }
    }}
    return;
  } else if (ep->sampling_kind == COARSE) {
    MD6(TinputType, atinput, tinput, ep->I3, ep->I2, Tz, A, A, V);
    auto mmin = _mm<V>::set1_ps(FLT_MAX);
    auto mmax = _mm<V>::set1_ps(-FLT_MAX);

    iter_each(_I3, ep->I3) {
    iter_each (_I2, ep->I2) {
      input_tile_iter<A, K> t2spati_o(_n, _t_off, ep->ht, ep->wt, ep->ih,
                                      ep->iw, ep->tp, ep->lp);
      iter_each (_T, Tz) {
        auto _ih = t2spati_o.anchor_t_;
        auto _iw = t2spati_o.anchor_l_;

        MD3(TinputType, aout, &md6(atinput, _I3, _I2, _T, 0, 0, 0), A, A, V);
        InputType *in = ep->input_fmt == nhwc
                ? &md7(ainput_nhwc, t2spati_o.n_, _ih, _iw, 0, _I3, _I2, 0)
                : &md7(ainput_blocked, t2spati_o.n_, 0, _I3, _I2, _ih, _iw, 0);
        if (!t2spati_o.is_border())
          ker_trans_input_(*ep, &md3(aout, 0, 0, 0), in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(*ep, &md3(aout, 0, 0, 0), in,
              t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

        ++ t2spati_o;

        iter_each (_hA, A) {
        iter_each (_wA, A) {
          mmin = _mm<V>::min_ps(mmin, *(__m<V> *)&md3(aout, _hA, _wA, 0));
          mmax = _mm<V>::max_ps(mmax, *(__m<V> *)&md3(aout, _hA, _wA, 0));
        }}
      }
    }}

    TinputType min = _mm<V>::reduce_min_ps(mmin);
    TinputType max = _mm<V>::reduce_max_ps(mmax);

    TinputType delta = max - min + 0.000001;
    TinputType S = delta / EL_UINT8_MAX;
    TinputType repS = EL_UINT8_MAX / delta;
    TinputType z = std::ceil(-min * repS);

    iter_each(_T, Tz) {
      md5(atinput_scale, 0, 0, 0, 0, _T) = S;
      md5(atinput_scale, 0, 0, 0, 1, _T) = z;
    }

    iter_each (_I3, ep->I3) {
    iter_each (_I2, ep->I2) {
    iter_each (_T, Tz) {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
      // Min-Max quantization
      __m<V> mrepS = _mm<V>::set1_ps(repS);
      __m<V> mz = _mm<V>::set1_ps(z);
      __m<V> a = *(__m<V> *)&md6(atinput, _I3, _I2, _T, _hA, _wA, 0);
      __m<V> mresf32 = a * mrepS + mz;
      // convert to uint8
      __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
      __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
      // store
      _mm_store_si128((__m128i *)&md6(atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
    }}}}}
    return;
  } else if (ep->sampling_kind == FINE) {
    MD4(TinputType, atinput, tinput, ep->I2, A, A, V);
    iter_each(_I3, ep->I3) {
      input_tile_iter<A, K> t2spati_o(_n, _t_off, ep->ht, ep->wt, ep->ih,
                                      ep->iw, ep->tp, ep->lp);
      iter_each (_T, Tz) {
        alignas(64) TinputType mmax[A][A][V];
        alignas(64) TinputType mmin[A][A][V];
        bool flush = true;
        iter_each (_I2, ep->I2) {
          auto _ih = t2spati_o.anchor_t_;
          auto _iw = t2spati_o.anchor_l_;

          MD3(TinputType, aout, &md4(atinput, _I2, 0, 0, 0), A, A, V);
          InputType *in = ep->input_fmt == nhwc
                ? &md7(ainput_nhwc, t2spati_o.n_, _ih, _iw, 0, _I3, _I2, 0)
                : &md7(ainput_blocked, t2spati_o.n_, 0, _I3, _I2, _ih, _iw, 0);
          if (!t2spati_o.is_border())
            ker_trans_input_(*ep, &md3(aout, 0, 0, 0), in, 0, A - 1, 0, A - 1);
          else
            ker_trans_input0_(*ep, &md3(aout, 0, 0, 0), in,
                t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

          if (flush) {
            iter_each (_hA, A) {
            iter_each (_wA, A) {
              __m<V> &_mmax = *(__m<V> *)&mmax[_hA][_wA][0];
              _mmax = *(__m<V> *)&md3(aout, _hA, _wA, 0);
              __m<V> &_mmin = *(__m<V> *)&mmin[_hA][_wA][0];
              _mmin = *(__m<V> *)&md3(aout, _hA, _wA, 0);
            }}
            flush = false;
          } else {
            iter_each (_hA, A) {
            iter_each (_wA, A) {
              __m<V> &_mmax = *(__m<V> *)&mmax[_hA][_wA][0];
              _mmax = _mm<V>::max_ps(_mmax, *(__m<V> *)&md3(aout, _hA, _wA, 0));
              __m<V> &_mmin = *(__m<V> *)&mmin[_hA][_wA][0];
              _mmin = _mm<V>::min_ps(_mmin, *(__m<V> *)&md3(aout, _hA, _wA, 0));
            }}
          }
        }

        iter_each (_hA, A) {
        iter_each (_wA, A) {
          if (I == ISA_AVX512 && std::is_same<TinputType, float>::value) {
            mmax[_hA][_wA][0] = _mm<V>::reduce_max_ps(*(__m<V> *)&mmax[_hA][_wA][0]);
            mmin[_hA][_wA][0] = _mm<V>::reduce_min_ps(*(__m<V> *)&mmin[_hA][_wA][0]);
          } else {
            for (int _V = 1; _V < V; _V++) {
              mmax[_hA][_wA][0] =
                  mmax[_hA][_wA][_V] > mmax[_hA][_wA][0] ?
                  mmax[_hA][_wA][_V] : mmax[_hA][_wA][0];
              mmin[_hA][_wA][0] =
                  mmin[_hA][_wA][_V] < mmin[_hA][_wA][0] ?
                  mmin[_hA][_wA][_V] : mmin[_hA][_wA][0];
            }
          }
          float delta = mmax[_hA][_wA][0] - mmin[_hA][_wA][0] + 0.000001;
          float S = delta / EL_UINT8_MAX;
          float repS = EL_UINT8_MAX / delta;
          float z = std::ceil(- mmin[_hA][_wA][0] * repS);
          mmax[_hA][_wA][0] = repS;
          mmin[_hA][_wA][0] = z;

          md5(atinput_scale, _I3, _hA, _wA, 0, _T) = S;
          md5(atinput_scale, _I3, _hA, _wA, 1, _T) = z;
        }}

        // quantization
        iter_each (_I2, ep->I2) {
        iter_each (_hA, A) {
        iter_each (_wA, A) {
          // Min-Max quantization
          __m<V> mrepS = _mm<V>::set1_ps(mmax[_hA][_wA][0]);
          __m<V> mz = _mm<V>::set1_ps(mmin[_hA][_wA][0]);
          __m<V> f = *(__m<V> *)&md4(atinput, _I2, _hA, _wA, 0);
          __m<V> mresf32 = f * mrepS + mz;
          // convert to uint8
          __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
          // store
          _mm_store_si128((__m128i *)&md6(atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
        }}}
        ++ t2spati_o;
      }
    }
    return;
  }
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::__execute_nchw(float *__restrict tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _t2, int Tz) {
  // n, IC, ih, iw => t2 | hA, wA, I3, I2, T, V
  alignas(64) op_type aout[A][A][V];
  alignas(64) InputType ain[A][A][V];
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ih * ep->iw));

  auto readin = [&](InputType ain[A][A][V], int _I3, int _I2, int _T, bool is_Ir) {
    MD2(InputType, ainput0, input, ep->n, ep->ic * ep->ih * ep->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD5(InputType, ainput5, &md2(ainput0, _n, 0), ep->I3,
        ep->I2, V, ep->ih, ep->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0;
        } else {
          iter_each(_V, ep->Ir) {
            ain[_hA][_wA][_V] =
                md5(ainput5, _I3, _I2, _V, _ih + _hA, _iw + _wA);
          }
          iter_each(_V, V - ep->Ir) {
            ain[_hA][_wA][_V + ep->Ir] = 0;
          }
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0;
        } else {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] =
                md5(ainput5, _I3, _I2, _V, _ih + _hA, _iw + _wA);
        }
      }}
    }
  };

  __m<V> mrepS = _mm<V>::set1_ps(ep->input_quant_S * ep->tinput_quant_repS);
  __m<V> mz = _mm<V>::set1_ps(ep->tinput_quant_z);
  MD6(uint8_t, atinput_u8, tinput_u8, A, A, ep->I3, ep->I2, Tz, V);

  iter_each (_I3, ep->I3) {
  iter_each (_I2, ep->I2) {
    bool is_Ir = ep->Ir != V && _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    iter_each (_T, Tz) {
      readin(ain, _I3, _I2, _T, is_Ir);
      ker_trans_input_(*ep, (float *)&aout, (InputType *)ain, 0, 0, 0, -1);

      iter_each (_hA, A) {
      iter_each (_wA, A) {
        // Min-Max quantization
        __m<V> a = *(__m<V> *)&aout[_hA][_wA][0];
        __m<V> mresf32 = a * mrepS + mz;
        // convert to uint8
        __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(
            mresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mresu32);
        // store
        _mm_store_si128((__m128i *)&md6(
            atinput_u8, _hA, _wA, _I3, _I2, _T, 0), mmresu8);
      }}
    }
  }}
}

template <typename InputType, int I, int A, int K, int V>
void elx_conv_wino_trans_input_t<uint8_t, InputType, I, A, K, V>
::execute(float *__restrict tinput_scale,
    uint8_t *__restrict tinput_u8, TinputType *__restrict tinput,
    InputType *__restrict input, int _t2, int Tz) {
  if (input_is_bfmt_ || input_as_bfmt_ || ep->input_fmt == nhwc)
    __execute_blocked_nhwc(tinput_scale, tinput_u8, tinput, input, _t2, Tz);
  else
    __execute_nchw(tinput_scale, tinput_u8, tinput, input, _t2, Tz);
}

template class elx_conv_wino_trans_input_t<float, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_input_t<float16, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<float16, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<float16, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_input_t<float16, float, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_input_t<uint8_t, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_input_t<uint8_t, uint8_t, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, uint8_t, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, uint8_t, ISA_AVX512, 6, 3, 16>;

#ifdef ENABLE_USER_FP16
template class elx_conv_wino_trans_input_t<float, float16, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float16, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float16, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_input_t<float, float16, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_input_t<uint8_t, float16, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float16, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float16, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_input_t<uint8_t, float16, ISA_AVX512, 7, 3, 16>;
#endif

} // namespace euler
