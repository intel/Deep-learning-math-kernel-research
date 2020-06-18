#include <string.h>
#include <float.h>
#include "el_intrin.hpp"
#include "el_parallel.hpp"
#include "elx_conv_wino_trans_output.hpp"

namespace euler {

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::setup(elx_param_t *conv_ep)
{
  ep = conv_ep;
  mthr_ = ep->nthreads;
  stream_out_ = ep->streaming_output
      ? (ep->streaming_output == STORE_STREAMING)
      : false;
  output_is_bfmt_ = ep->output_fmt == nChw16c;
  output_as_bfmt_ = ep->output_fmt == nchw && ep->output_as_blocked;

  if (ep->Or != V && ep->output_fmt == nhwc) {
    el_error("Unimplemented: nhwc output with Or");
  }

  hOA_end_ = ep->oh % (A - K + 1) - 1;
  if (hOA_end_ == -1)
    hOA_end_ = A - K;
  wOA_end_ = ep->ow % (A - K + 1) - 1;
  if (wOA_end_ == -1)
    wOA_end_ = A - K;
  hA_end_ = (ep->ih + ep->tp) - (ep->ht - 1) * (A - K + 1) - 1;
  wA_end_ = (ep->iw + ep->lp) - (ep->wt - 1) * (A - K + 1) - 1;

  bind_kernel_functions();
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::bind_kernel_functions()
{
#undef E
#define E(format, border, bias, relu, sum)                                     \
  elk_conv_wino_trans_output<TrOpType, OutputType, BiasType, format,           \
      border, bias, relu, sum, I, A, K, V>::execute
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } C_ktable[2][2][2] = {
      {{{E(TKF_COMPACT, 0, 0, 0, 0), E(TKF_COMPACT, 1, 0, 0, 0)},
        {E(TKF_COMPACT, 0, 0, 0, 1), E(TKF_COMPACT, 1, 0, 0, 1)}},
       {{E(TKF_COMPACT, 0, 0, 1, 0), E(TKF_COMPACT, 1, 0, 1, 0)},
        {E(TKF_COMPACT, 0, 0, 1, 1), E(TKF_COMPACT, 1, 0, 1, 1)}}},
      {{{E(TKF_COMPACT, 0, 1, 0, 0), E(TKF_COMPACT, 1, 1, 0, 0)},
        {E(TKF_COMPACT, 0, 1, 0, 1), E(TKF_COMPACT, 1, 1, 0, 1)}},
       {{E(TKF_COMPACT, 0, 1, 1, 0), E(TKF_COMPACT, 1, 1, 1, 0)},
        {E(TKF_COMPACT, 0, 1, 1, 1), E(TKF_COMPACT, 1, 1, 1, 1)}}}};
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } D_ktable[2][2][2] = {
      {{{E(TKF_BLOCKED, 0, 0, 0, 0), E(TKF_BLOCKED, 1, 0, 0, 0)},
        {E(TKF_BLOCKED, 0, 0, 0, 1), E(TKF_BLOCKED, 1, 0, 0, 1)}},
       {{E(TKF_BLOCKED, 0, 0, 1, 0), E(TKF_BLOCKED, 1, 0, 1, 0)},
        {E(TKF_BLOCKED, 0, 0, 1, 1), E(TKF_BLOCKED, 1, 0, 1, 1)}}},
      {{{E(TKF_BLOCKED, 0, 1, 0, 0), E(TKF_BLOCKED, 1, 1, 0, 0)},
        {E(TKF_BLOCKED, 0, 1, 0, 1), E(TKF_BLOCKED, 1, 1, 0, 1)}},
       {{E(TKF_BLOCKED, 0, 1, 1, 0), E(TKF_BLOCKED, 1, 1, 1, 0)},
        {E(TKF_BLOCKED, 0, 1, 1, 1), E(TKF_BLOCKED, 1, 1, 1, 1)}}}};
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } F_ktable[2][2][2] = {
      {{{E(TKF_NHWC, 0, 0, 0, 0), E(TKF_NHWC, 1, 0, 0, 0)},
        {E(TKF_NHWC, 0, 0, 0, 1), E(TKF_NHWC, 1, 0, 0, 1)}},
       {{E(TKF_NHWC, 0, 0, 1, 0), E(TKF_NHWC, 1, 0, 1, 0)},
        {E(TKF_NHWC, 0, 0, 1, 1), E(TKF_NHWC, 1, 0, 1, 1)}}},
      {{{E(TKF_NHWC, 0, 1, 0, 0), E(TKF_NHWC, 1, 1, 0, 0)},
        {E(TKF_NHWC, 0, 1, 0, 1), E(TKF_NHWC, 1, 1, 0, 1)}},
       {{E(TKF_NHWC, 0, 1, 1, 0), E(TKF_NHWC, 1, 1, 1, 0)},
        {E(TKF_NHWC, 0, 1, 1, 1), E(TKF_NHWC, 1, 1, 1, 1)}}}};

  if (output_is_bfmt_ || output_as_bfmt_) {
    ker_trans_output_ =
      D_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f1_;
    ker_trans_output0_ =
      D_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f2_;
    ker_trans_output_acc_ = D_ktable[ep->with_bias][ep->with_relu][1].f1_;
    ker_trans_output0_acc_ = D_ktable[ep->with_bias][ep->with_relu][1].f2_;
  } else if (ep->output_fmt == nhwc) {
    ker_trans_output_ =
      F_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f1_;
    ker_trans_output0_ =
      F_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f2_;
    ker_trans_output_acc_ = F_ktable[ep->with_bias][ep->with_relu][1].f1_;
    ker_trans_output0_acc_ = F_ktable[ep->with_bias][ep->with_relu][1].f2_;
  } else {  // nchw
    ker_trans_output_ =
      C_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f1_;
    ker_trans_output0_ =
      C_ktable[ep->with_bias][ep->with_relu][ep->with_ip_sum].f2_;
    ker_trans_output_acc_ = C_ktable[ep->with_bias][ep->with_relu][1].f1_;
    ker_trans_output0_acc_ = C_ktable[ep->with_bias][ep->with_relu][1].f2_;
  }
}

#define t2spato(__t2, __T, __n, __oh, __ow, __hOA_end, __wOA_end)              \
  do {                                                                         \
    int _t = __t2 * ep->T + __T;                                             \
    int _nt = _t % ep->nt;                                                   \
    int _ht = _nt / ep->wt;                                                  \
    int _wt = _nt % ep->wt;                                                  \
    __n = _t / ep->nt;                                                       \
    __oh = _ht * (A - K + 1);                                                  \
    __ow = _wt * (A - K + 1);                                                  \
    __hOA_end = (_ht < ep->ht - 1) ? A - K : hOA_end_;                       \
    __wOA_end = (_wt < ep->wt - 1) ? A - K : wOA_end_;                       \
  } while (0)

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_nchw(OutputType *__restrict output,
    ToutputType *__restrict toutput, BiasType *__restrict bias, int Tz, int _t2,
    int _O4, int _I4)
{
  // A, A, O3, O2, T, V -> n, OC, oh, ow
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD3(BiasType, abias, bias, ep->O3, ep->O2, V);

  alignas(64) OutputType aout[A - K + 1][A - K + 1][V];
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->oh * ep->ow));

  auto writeout = [&](OutputType aout[A - K + 1][A - K + 1][V], int _O3,
                      int _O2, int _T, bool is_Or) {
    MD2(OutputType, aoutput0, output, ep->n, ep->oc * ep->oh * ep->ow);
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
    MD6(OutputType, aoutput1, &md2(aoutput0, _n, 0), ep->O4, ep->O3,
        ep->O2, V, ep->oh, ep->ow);

    for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
      for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
        if (is_Or) {
          if ((ep->with_ip_sum && !output_as_bfmt_) || _I4 > 0) {
            iter_each (_V, ep->Or)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
            iter_each (_V, ep->Or)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((ep->with_ip_sum && !output_as_bfmt_) || _I4 > 0) {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else if (I == ISA_AVX512
              && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr int scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md6(aoutput1, _O4, _O3, _O2, 0, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  union {
    __m<V> vin;
    TrOpType ain[V];
  } In[A][A];

  iter_each (_O3, ep->O3) {
    iter_each (_O2, ep->O2) {
      bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
          && _O3 == ep->O3 - 1 && _O2 == ep->O2 - 1;
      iter_each (_T, Tz) {
        iter_each (_hA, A) {
          iter_each (_wA, A) {
            if (std::is_same<ToutputType, float>::value) {
              In[_hA][_wA].vin
                  = _mm<V>::load_ps(&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
            } else {
              auto fp16v = _mm<V / 2>::load_si256(
                  (__m256i *)&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
              In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
            }
          }
        }
        ker_trans_output_(*ep, (OutputType *)aout, (float *)&In,
            (_I4 == ep->I4 - 1 || _I4 == -1) ? &md3(abias, _O3, _O2, 0)
                                                  : nullptr,
            0, -1);

        writeout(aout, _O3, _O2, _T, is_Or);
      }
    }
  }
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_blocked(OutputType *output, ToutputType *toutput,
    BiasType *bias, int Tz, int _t2, int _O4, int _I4)
{
  auto ker_trans_output = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output_acc_
      : ker_trans_output_;
  auto ker_trans_output_tail = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output0_acc_
      : ker_trans_output0_;

  // A, A, O3, O2, T, V -> n, oc2, oh, ow, V
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD7(OutputType, aoutput, output, ep->n, ep->O4, ep->O3, ep->O2,
      ep->oh, ep->ow, V);
  MD3(BiasType, abias, bias, ep->O3, ep->O2, V);

  auto res = std::div(_t2 * ep->T, ep->nt);
  auto _n_off = res.quot;
  auto _t_off = res.rem;
  alignas(64) union {
    __m<V> vin;
    TrOpType ain[V];
  } In[A][A];

  iter_each (_O3, ep->O3) {
    iter_each (_O2, ep->O2) {
      output_tile_iter<A, K> t2spato_o(
          _n_off, _t_off, ep->ht, ep->wt, ep->oh, ep->ow);
      iter_each (_T, Tz) {
        iter_each (_hA, A) {
          iter_each (_wA, A) {
            if (std::is_same<ToutputType, float>::value) {
              In[_hA][_wA].vin
                  = _mm<V>::load_ps(&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
            } else {
              auto fp16v = _mm<V / 2>::load_si256(
                  (__m256i *)&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
              In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
            }
          }
        }

        auto _n = t2spato_o.n_;
        auto _oh = t2spato_o.t_;
        auto _ow = t2spato_o.l_;
        OutputType *out = &md7(aoutput, _n, _O4, _O3, _O2, _oh, _ow, 0);

        if (t2spato_o.is_border())
          ker_trans_output_tail(*ep, out, (float *)&In,
              (_I4 == -1 || _I4 == ep->I4 - 1) ? &md3(abias, _O3, _O2, 0)
                                                    : nullptr,
              t2spato_o.d_, t2spato_o.r_);
        else
          ker_trans_output(*ep, out, (float *)&In,
              (_I4 == -1 || _I4 == ep->I4 - 1) ? &md3(abias, _O3, _O2, 0)
                                                    : nullptr,
              A - K, A - K);

        ++t2spato_o;
      }
    }
  }
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_nhwc(OutputType *output, ToutputType *toutput, BiasType *bias,
    int Tz, int _t2, int _O4, int _I4)
{
  auto ker_trans_output = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output_acc_
      : ker_trans_output_;
  auto ker_trans_output_tail = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output0_acc_
      : ker_trans_output0_;

  // A, A, O3, O2, T, V -> n, oc2, oh, ow, V
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD3(BiasType, abias, bias, ep->O3, ep->O2, V);
  MD4(OutputType, aoutput0, output, ep->n, ep->oh, ep->ow, ep->oc);

  auto res = std::div(_t2 * ep->T, ep->nt);
  auto _n_off = res.quot;
  auto _t_off = res.rem;
  alignas(64) union {
    __m<V> vin;
    TrOpType ain[V];
  } In[A][A];

  iter_each (_O3, ep->O3) {
    iter_each (_O2, ep->O2) {
      output_tile_iter<A, K> t2spato_o(
          _n_off, _t_off, ep->ht, ep->wt, ep->oh, ep->ow);
      iter_each (_T, Tz) {
        iter_each (_hA, A) {
          iter_each (_wA, A) {
            if (std::is_same<ToutputType, float>::value) {
              In[_hA][_wA].vin
                  = _mm<V>::load_ps(&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
            } else {
              auto fp16v = _mm<V / 2>::load_si256(
                  (__m256i *)&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
              In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
            }
          }
        }
        auto _n = t2spato_o.n_;
        auto _oh = t2spato_o.t_;
        auto _ow = t2spato_o.l_;
        MD4(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), ep->O4,
            ep->O3, ep->O2, V);
        OutputType *out = &md4(aoutput1, _O4, _O3, _O2, 0);
        if (t2spato_o.is_border())
          ker_trans_output_tail(*ep, out, (float *)&In,
              (_I4 == -1 || _I4 == ep->I4 - 1) ? &md3(abias, _O3, _O2, 0)
                                                    : nullptr,
              t2spato_o.d_, t2spato_o.r_);
        else
          ker_trans_output(*ep, out, (float *)&In,
              (_I4 == -1 || _I4 == ep->I4 - 1) ? &md3(abias, _O3, _O2, 0)
                                                    : nullptr,
              A - K, A - K);

        ++t2spato_o;
      }
    }
  }
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::execute(OutputType *output, ToutputType *toutput, BiasType *bias,
    int Tz, int _t2, int _O4, int _I4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __execute_blocked(output, toutput, bias, Tz, _t2, _O4, _I4);
  else if (ep->output_fmt == nhwc)
    __execute_nhwc(output, toutput, bias, Tz, _t2, _O4, _I4);
  else
    __execute_nchw(output, toutput, bias, Tz, _t2, _O4, _I4);
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_blocked(OutputType *output, ToutputType *toutput,
    BiasType *bias, int _O4, int _I4)
{
  auto ker_trans_output = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output_acc_
      : ker_trans_output_;
  auto ker_trans_output_tail = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output0_acc_
      : ker_trans_output0_;

  int ithr = estl::current_thread_index();
  THREAD_FOR(4, mthr_, ithr,
      [&](int _t2, int _O3, int _O2, int _T) {
        // A, A, O3, O2, T, V -> n, oc2, oh, ow, V
        MD7(OutputType, aoutput, output, ep->n, ep->O4, ep->O3, ep->O2,
            ep->oh, ep->ow, V);
        MD2(ToutputType, atoutput2, toutput, ep->t2,
            A * A * ep->T * ep->O3 * ep->O2 * V);
        MD3(BiasType, abias, bias, ep->O3, ep->O2, V);

        int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
        MD6(ToutputType, atoutput, &md2(atoutput2, _t2, 0), A, A, ep->O3,
            ep->O2, Tz, V);
        alignas(64) union {
          __m<V> vin;
          TrOpType ain[V];
        } In[A][A];

        //iter_each (_T, Tz) {
        if (_T < Tz) {
          iter_each (_hA, A) {
            iter_each (_wA, A) {
              if (std::is_same<ToutputType, float>::value) {
                In[_hA][_wA].vin = _mm<V>::load_ps(
                    &md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
              } else {
                auto fp16v = _mm<V / 2>::load_si256(
                    (__m256i *)&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
                In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
              }
            }
          }

          int _n, _oh, _ow, _hOA_end, _wOA_end;
          t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
          OutputType *out = &md7(aoutput, _n, _O4, _O3, _O2, _oh, _ow, 0);

          if (_hOA_end < A - K || _wOA_end < A - K)
            ker_trans_output_tail(*ep, out, (float *)&In,
                (_I4 == -1 || _I4 == ep->I4 - 1)
                    ? &md3(abias, _O3, _O2, 0)
                    : nullptr,
                _hOA_end, _wOA_end);
          else
            ker_trans_output(*ep, out, (float *)&In,
                (_I4 == -1 || _I4 == ep->I4 - 1)
                    ? &md3(abias, _O3, _O2, 0)
                    : nullptr,
                A - K, A - K);
        }
      }, ep->t2, ep->O3, ep->O2, ep->T);
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_nhwc(OutputType *output, ToutputType *toutput, BiasType *bias,
    int _O4, int _I4)
{
  auto ker_trans_output = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output_acc_
      : ker_trans_output_;
  auto ker_trans_output_tail = (ep->with_ip_sum || _I4 > 0)
      ? ker_trans_output0_acc_
      : ker_trans_output0_;

  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr,
      [&](int _t2, int _O3, int _O2) {
        // A, A, O3, O2, T, V -> n, oh, ow, oc
        MD4(OutputType, aoutput0, output, ep->n, ep->oh, ep->ow, ep->oc);
        MD2(ToutputType, atoutput2, toutput, ep->t2,
            A * A * ep->T * ep->O3 * ep->O2 * V);
        MD3(BiasType, abias, bias, ep->O3, ep->O2, V);
        int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
        MD6(ToutputType, atoutput, &md2(atoutput2, _t2, 0), A, A, ep->O3,
            ep->O2, Tz, V);
        alignas(64) union {
          __m<V> vin;
          TrOpType ain[V];
        } In[A][A];

        iter_each (_T, Tz) {
          iter_each (_hA, A) {
            iter_each (_wA, A) {
              if (std::is_same<ToutputType, float>::value) {
                In[_hA][_wA].vin = _mm<V>::load_ps(
                    &md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
              } else {
                auto fp16v = _mm<V / 2>::load_si256(
                    (__m256i *)&md6(atoutput, _hA, _wA, _O3, _O2, _T, 0));
                In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
              }
            }
          }
          int _n, _oh, _ow, _hOA_end, _wOA_end;
          t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
          MD4(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), ep->O4,
              ep->O3, ep->O2, V);
          OutputType *out = &md4(aoutput1, _O4, _O3, _O2, 0);

          if (_hOA_end < A - K || _wOA_end < A - K)
            ker_trans_output_tail(*ep, out, (float *)&In,
                (_I4 == -1 || _I4 == ep->I4 - 1)
                    ? &md3(abias, _O3, _O2, 0)
                    : nullptr,
                _hOA_end, _wOA_end);
          else
            ker_trans_output(*ep, out, (float *)&In,
                (_I4 == -1 || _I4 == ep->I4 - 1)
                    ? &md3(abias, _O3, _O2, 0)
                    : nullptr,
                A - K, A - K);
        }
      }, ep->t2, ep->O3, ep->O2);
}
template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::__execute_nchw(OutputType *__restrict output,
    ToutputType *__restrict toutput, BiasType *bias, int _O4, int _I4)
{
  // A, A, O3, O2, T, V -> n, OC, oh, ow
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->oh * ep->ow));

  auto writeout = [&](OutputType aout[A - K + 1][A - K + 1][V], int _t2,
                      int _O3, int _O2, int _T, bool is_Or) {
    MD2(OutputType, aoutput0, output, ep->n, ep->oc * ep->oh * ep->ow);
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
    MD6(OutputType, aoutput1, &md2(aoutput0, _n, 0), ep->O4, ep->O3,
        ep->O2, V, ep->oh, ep->ow);

    for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
      for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
        if (is_Or) {
          if ((ep->with_ip_sum && !output_as_bfmt_) || (_I4 > 0)) {
#pragma omp simd
            iter_each (_V, ep->Or)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, ep->Or)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((ep->with_ip_sum && !output_as_bfmt_) || (_I4 > 0)) {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else if (I == ISA_AVX512
              && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md6(aoutput1, _O4, _O3, _O2, 0, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _O4, _O3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  int ithr = estl::current_thread_index();
  THREAD_FOR(3, mthr_, ithr,
      [&](int _t2, int _O3, int _O2) {
        MD3(BiasType, abias, bias, ep->O3, ep->O2, V);
        MD2(ToutputType, atoutput2, toutput, ep->t2,
            A * A * ep->T * ep->O3 * ep->O2 * V);
        int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
        MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, ep->O3,
            ep->O2, Tz, V);
        alignas(64) OutputType aout[A - K + 1][A - K + 1][V];
        alignas(64) union {
          __m<V> vin;
          TrOpType ain[V];
        } In[A][A];
        bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
            && _O3 == ep->O3 - 1 && _O2 == ep->O2 - 1;
        iter_each (_T, Tz) {
          iter_each (_hA, A) {
            iter_each (_wA, A) {
              if (std::is_same<ToutputType, float>::value) {
                In[_hA][_wA].vin = _mm<V>::load_ps(
                    &md6(atoutput6, _hA, _wA, _O3, _O2, _T, 0));
              } else {
                auto fp16v = _mm<V / 2>::load_si256(
                    (__m256i *)&md6(atoutput6, _hA, _wA, _O3, _O2, _T, 0));
                In[_hA][_wA].vin = _mm<V>::cvtph_ps(fp16v);
              }
            }
          }

          ker_trans_output_(*ep, (OutputType *)aout, (float *)&In,
              (_I4 == -1 || _I4 == ep->I4 - 1) ? &md3(abias, _O3, _O2, 0)
                                                  : nullptr,
              0, -1);
          writeout(aout, _t2, _O3, _O2, _T, is_Or);
        }
      }, ep->t2, ep->O3, ep->O2);
}

template <typename OutputType, typename BiasType, typename ToutputType, int I,
    int A, int K, int V>
void elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K,
    V>::execute(OutputType *output, ToutputType *toutput, BiasType *bias,
    int _O4, int _I4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __execute_blocked(output, toutput, bias, _O4, _I4);
  else if (ep->output_fmt == nhwc)
    __execute_nhwc(output, toutput, bias, _O4, _I4);
  else
    __execute_nchw(output, toutput, bias, _O4, _I4);
}


// user: float, tarray: float
template class elx_conv_wino_trans_output_t<float, float, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float, ISA_AVX512, 7, 3, 16>;

// user: float, tarray: fp16
template class elx_conv_wino_trans_output_t<float, float, float16, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float16, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float16, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_output_t<float, float, float16, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_output_t<uint8_t, float, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<uint8_t, float, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<uint8_t, float, float, ISA_AVX512, 6, 3, 16>;

template class elx_conv_wino_trans_output_t<int8_t, float, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<int8_t, float, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<int8_t, float, float, ISA_AVX512, 6, 3, 16>;

#ifdef ENABLE_USER_FP16
// user: fp16, tarray: float
template class elx_conv_wino_trans_output_t<float16, float16, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float, ISA_AVX512, 7, 3, 16>;

// user: fp16, tarray: fp16
template class elx_conv_wino_trans_output_t<float16, float16, float16, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float16, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float16, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_output_t<float16, float16, float16, ISA_AVX512, 7, 3, 16>;
#endif

} // namespace euler
