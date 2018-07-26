#include <string.h>
#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_wino.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {
#define t2spato(__t2, __T, __n, __oh, __ow, __hOA_end, __wOA_end)            \
  do {                                                                       \
    int _t = __t2 * this->T + __T;                                           \
    int _nt = _t % this->nt;                                                 \
    int _ht = _nt / this->wt;                                                \
    int _wt = _nt % this->wt;                                                \
    __n = _t / this->nt;                                                     \
    __oh = _ht * (A - K + 1);                                                \
    __ow = _wt * (A - K + 1);                                                \
    __hOA_end = (_ht < this->ht - 1) ? A - K : hOA_end_;                     \
    __wOA_end = (_wt < this->wt - 1) ? A - K : wOA_end_;                     \
  } while (0)

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_plain(
    Type * __restrict output, Type * __restrict toutput, Type * __restrict bias
    , int _t2, int Tz)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

  alignas(64) Type ain[A][A][V];
  alignas(64) Type aout[A - K + 1][A - K + 1][V];

  int s = this->oh * this->ow;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto writeout_v = [&](int _oc3, int _O2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m512 t = _mm512_load_ps(aout[_hA][_wA]);
          constexpr int scale = sizeof(Type);
          _mm512_i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          for_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _oc3, int _O2, int _T,
                        Type aout[A - K + 1][A - K + 1][V]) {
    MD4(Type, aoutput, output, this->n, this->oc, this->oh, this->ow);

    assert(this->oc4 == 1);
    int is_Or = _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
#pragma omp simd
        for_each (_V, this->Or)
          md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
              = aout[_hA][_wA][_V];
        } else {
#pragma omp simd
        for_each (_V, V)
          md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
              _ow + _wA)
              = aout[_hA][_wA][_V];

        }
      }
    }
  };

  for_each (_oc3, this->oc3) {
  for_each (_O2, this->O2) {
    for_each (_T, Tz) {
      for_each (_wA, A) {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      ker_trans_output_(
          *this, (Type *)aout, ain, &md3(abias, _oc3, _O2, 0), 0, -1);

      if (this->Or != V)
        writeout_r(_oc3, _O2, _T, aout);
      else
        writeout_v(_oc3, _O2, _T, aout);
    }
  }}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, this->oh, this->ow, V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

  alignas(64) Type ain[A][A][V];

  for_each (_oc3, this->oc3) {
  for_each (_O2, this->O2) {
  for_each (_T, Tz) {
    for_each (_wA, A) {
    for_each (_hA, A) {
#pragma omp simd
    for_each (_V, V) {
      ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
    }}}

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
    Type *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

    if (_hOA_end < A - K || _wOA_end < A - K)
      ker_trans_output0_(*this, out, ain, &md3(abias, _oc3, _O2, 0), _hOA_end, _wOA_end);
    else
      ker_trans_output_(*this, out, ain, &md3(abias, _oc3, _O2, 0), A - K, A - K);
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _t2, Tz);
  else
    __trans_output_plain(output, toutput, bias, _t2, Tz);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_blocked(Type *output,
    Type *output_tmp, Type *toutput, Type *bias, int _t2, int Tz, int ic4,
    int oc4, bool inline_reduce)
{
  MD6(Type, atoutput, toutput, A, A, this->ic4, this->oc3 * this->O2 / this->ic4, Tz, V);
  MD7(Type, aoutput, output, this->n, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, this->oh, this->ow, V);
  MD7(Type, aoutput_tmp, output_tmp, this->n, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, this->oh, this->ow, V);
  MD8(Type, aroutput, routput_, this->ic4, this->n, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, this->oh, this->ow, V);
  MD4(Type, abias, bias, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, V);

  // TODO: pause
  auto sync_on = [this](int _t2, int oc4) {
    MD2(unsigned char, cntr, routput_cntr_, this->t2, this->oc4);
#pragma omp atomic
    md2(cntr, _t2, oc4)++;
    unsigned char c = 0;
#pragma omp atomic read
    c = md2(cntr, _t2, oc4);
    while (c != this->ic4) {
      _mm_pause();
#pragma omp atomic read
      c = md2(cntr, _t2, oc4);
    }
  };

  alignas(64) Type ain[A][A][V];

  for_each (_ic4, this->ic4) {
    for_each (_oc, this->oc3 * this->O2/this->ic4) {
      for_each (_T, Tz) {
        for_each (_wA, A) {
        for_each (_hA, A) {
#pragma omp simd
        for_each (_V, V) {
          ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _ic4, _oc, _T, _V);
        }}}
        int _n, _oh, _ow, _hOA_end, _wOA_end;
        t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
        Type *out = &md7(aoutput_tmp, _n, oc4, _ic4, _oc, _oh, _ow, 0);

        if (bias == nullptr) {
          if (_hOA_end < A - K || _wOA_end < A - K)
            ker_trans_output0_nobias_(*this, out, ain, nullptr, _hOA_end, _wOA_end);
          else
            ker_trans_output_nobias_(*this, out, ain, nullptr, A - K, A - K);
        } else {
          if (_hOA_end < A - K || _wOA_end < A - K)
            ker_trans_output0_(*this, out, ain, &md4(abias, oc4, _ic4, _oc, 0),
                _hOA_end, _wOA_end);
          else
            ker_trans_output_(
                *this, out, ain, &md4(abias, oc4, _ic4, _oc, 0), A - K, A - K);
        }
      }
    }
  }

  if (inline_reduce) {
    sync_on(_t2, oc4);

    for_each (_oc, this->oc3 * this->O2 / this->ic4) {
    for_each (_T, Tz) {
      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      for_each (_hA, _hOA_end + 1) {
      for_each (_wA, _wOA_end + 1) {
      for (int __ic4 = 1; __ic4 < this->ic4; __ic4++) {
#pragma omp simd
      for_each (_V, V) {
        md7(aoutput, _n, oc4, ic4, _oc, _oh + _hA, _ow + _wA, _V) +=
          md8(aroutput, __ic4 - 1, _n, oc4, ic4, _oc, _oh + _hA, _ow + _wA, _V);
      }}}}
    }}
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_plain(Type *output,
    Type * __restrict output_tmp, Type * __restrict toutput, Type *bias, int _t2, int Tz, int ic4,
    int oc4, bool inline_reduce)
{
  MD6(Type, atoutput, toutput, A, A, this->ic4, this->oc3 * this->O2 / this->ic4, Tz, V);
  MD8(Type, aroutput, routput_, this->ic4, this->n, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, this->oh, this->ow, V);
  MD4(Type, abias, bias, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, V);

  alignas(64) Type ain[A][A][V];
  alignas(64) Type aout[A - K + 1][A - K + 1][V];

  int s = this->oh * this->ow;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto writeout_v = [&](int _ic4, int _oc, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD7(Type, aoutput, output_tmp, this->n, this->oc4, this->ic4, this->oc3 * this->O2 / this->ic4, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m512 t = _mm512_load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(Type);
          _mm512_i32scatter_ps(
              &md7(aoutput, _n, oc4, _ic4, _oc, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          for_each (_V, V)
            md7(aoutput, _n, oc4, _ic4, _oc, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _oc, int _T, Type aout[A - K + 1][A - K + 1][V]) {
    MD4(Type, aoutput, output_tmp, this->n, this->oc, this->oh, this->ow);

    assert(this->ic4 == 1 && this->oc4 == 1);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    bool is_Or =  _oc == this->oc2  - 1;

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
#pragma omp simd
          for_each (_V, this->Or)
            md4(aoutput, _n, _oc * V + _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        } else {
#pragma omp simd
          for_each (_V, V)
            md4(aoutput, _n, _oc * V + _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  for_each (_ic4, this->ic4) {
  for_each (_oc, this->oc3 * this->O2/this->ic4) {
    for_each (_T, Tz) {
      for_each (_wA, A) {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _ic4, _oc, _T, _V);
      }}}

      if (bias == nullptr)
        ker_trans_output_nobias_(*this, (Type *)aout, ain, nullptr, 0, -1);
      else
        ker_trans_output_(
            *this, (Type *)aout, ain, &md4(abias, oc4, _ic4, _oc, 0), 0, -1);

      if (this->Or != V)
        writeout_r(_oc, _T, aout);
      else
        writeout_v(_ic4, _oc, _T, aout);
    }
  }}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_output(Type *output,
    Type *output_tmp, Type *toutput, Type *bias, int _t2, int Tz, int ic4,
    int oc4, bool inline_reduce)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(
        output, output_tmp, toutput, bias, _t2, Tz, ic4, oc4, inline_reduce);
  else
    __trans_output_plain(
        output, output_tmp, toutput, bias, _t2, Tz, ic4, oc4, false);
}

// toutput:  mthr | hA/A, oc3, O2, T, V
// toutputa: t2, oc4 | oc3, O2, T, wA/A | hA/A-K+1, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_outputa_th(
    Type *toutputa, Type *toutput, int Tz)
{
  MD4(Type, atoutput, toutput, A, this->oc3 * this->O2, Tz, V);
  MD4(Type, atoutputa, toutputa, this->oc3 * this->O2, Tz, A, (A - K + 1) * V);

  for_each (_oc, this->oc3 * this->O2) {
    for_each (_T, Tz) {
      ker_trans_outputa_th_(*this, &md4(atoutputa, _oc, _T, 0, 0),
        &md4(atoutput, 0, _oc, _T, 0), Tz, stream_out_);
    }
  }
}

// output: n, oc2, h, w, V
// toutputa: t2, oc2, T, wA/A | hA/A-K+1, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_outputa_bh_blocked(
    Type *output, Type *toutputa, Type *bias)
{
  MD5(Type, aoutput, output, this->n, this->oc2, this->oh, this->ow, V);
  MD2(Type, abias, bias, this->oc2, V);
  MD2(Type, atoutputa2, toutputa, this->t2, A * (A - K + 1) * this->T * this->OC);

#pragma omp for nowait collapse(2)
  for_each (_t2, this->t2) {
  for_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(Type, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz, A * (A - K + 1) * V);

    for_each (_T, Tz) {
      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      Type *out = &md5(aoutput, _n, _oc2, _oh, _ow, 0);
      using Array1 = Type[A][A - K + 1][V];
      Array1 *in = (Array1 *)&md3(atoutputa3, _oc2, _T, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_outputa0_bh_(
            *this, out, *in, &md2(abias, _oc2, 0), _hOA_end, _wOA_end);
      else
        ker_trans_outputa_bh_(
            *this, out, *in, &md2(abias, _oc2, 0), A - K, A - K);
    }
  }}
}

// output: n, OC, h, w
// toutputa: t2, oc2, T, wA/A | hA/A-K+1, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_outputa_bh_plain(
    Type * __restrict output, Type * __restrict toutputa, Type *bias)
{
  MD2(Type, abias, bias, this->oc2, V);
  MD2(Type, atoutputa2, toutputa, this->t2, A * (A - K + 1) * this->T * this->OC);

  int s = this->oh * this->ow;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto writeout_v = [&](int _t2, int _oc2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD5(Type, aoutput, output, this->n, this->oc2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m512 t = _mm512_load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(Type);
          _mm512_i32scatter_ps(
              &md5(aoutput, _n, _oc2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          for_each (_V, V)
            md5(aoutput, _n, _oc2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _t2, int _oc2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD4(Type, aoutput, output, this->n, this->oc, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    assert(this->oc4 == 1);
    bool is_Or = _oc2 == this->oc2 - 1;
    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
#pragma omp simd
          for_each (_V, this->Or)
            md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        } else {
#pragma omp simd
          for_each (_V, V)
            md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

#pragma omp for nowait collapse(2)
  for_each (_t2, this->t2) {
  for_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(Type, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz, A * (A - K + 1) * V);
    alignas(64) Type aout[A - K + 1][A - K + 1][V];

    for_each (_T, Tz) {
      using Array1 = Type[A][A - K + 1][V];
      Array1 *in = (Array1 *)&md3(atoutputa3, _oc2, _T, 0);

      ker_trans_outputa_bh_(
          *this, (Type *)aout, *in, &md2(abias, _oc2, 0), 0, -1);

      if (this->Or != V)
        writeout_r(_t2, _oc2, _T, aout);
      else
        writeout_v(_t2, _oc2, _T, aout);
    }
  }}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_outputa_bh(
    Type *output, Type *toutputa, Type *bias)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_outputa_bh_blocked(output, toutputa, bias);
  else
    __trans_outputa_bh_plain(output, toutputa, bias);

}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, Type *bias)
{
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, this->oh, this->ow, V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
  for_each (_oc3, this->oc3) {
  for_each (_O2, this->O2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atoutput, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    alignas(64) Type ain[A][A][V];

    for_each (_T, Tz) {
      for_each (_wA, A) {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      Type *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_output0_(
            *this, out, ain, &md3(abias, _oc3, _O2, 0), _hOA_end, _wOA_end);
      else
        ker_trans_output_(
            *this, out, ain, &md3(abias, _oc3, _O2, 0), A - K, A - K);
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_plain(
    Type * __restrict output, Type * __restrict toutput, Type *bias)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD3(Type, abias, bias, this->oc3, this->O2, V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);

  int s = this->oh * this->ow;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto writeout_v = [&](int _t2, int _oc3, int _O2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m512 t = _mm512_load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(Type);
          _mm512_i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          for_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _t2, int _oc3, int _O2, int _T,
                        Type aout[A - K + 1][A - K + 1][V]) {
    MD4(Type, aoutput, output, this->n, this->oc, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    assert(this->oc4 == 1);
    bool is_Or = _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
#pragma omp simd
          for_each (_V, this->Or)
            md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        } else {
#pragma omp simd
          for_each (_V, V)
            md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
  for_each (_oc3, this->oc3) {
  for_each (_O2, this->O2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    alignas(64) Type ain[A][A][V];
    alignas(64) Type aout[A - K + 1][A - K + 1][V];

    for_each (_T, Tz) {
      for_each (_wA, A) {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput6, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      ker_trans_output_(
          *this, (Type *)aout, ain, &md3(abias, _oc3, _O2, 0), 0, -1);

      if (this->Or != V)
        writeout_r(_t2, _oc3, _O2, _T, aout);
      else
        writeout_v(_t2, _oc3, _O2, _T, aout);
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias);
  else
    __trans_output_plain(output, toutput, bias);
}

} // namespace euler
