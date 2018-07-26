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

const unsigned XOPT_MSK = 0xA000;

const unsigned TTM_MSK = 0xF00;
const unsigned TTM_I   = 0x100;
const unsigned TTM_O   = 0x200;
const unsigned TTM_T   = 0x400;

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;
const unsigned FUS_T   = 0x40;
const unsigned FUS_A   = 0x80;

const unsigned DUP_MSK = 0xF;
const unsigned DUP_I   = 0x1;
const unsigned DUP_O   = 0x2;
const unsigned DUP_W   = 0x8;

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

#define t2spati(                                                             \
    __t2, __T, __n, __ih, __iw, __hA_start, __hA_end, __wA_start, __wA_end)  \
  do {                                                                       \
    int _t = __t2 * this->T + __T;                                           \
    int _nt = _t % this->nt;                                                 \
    int _ht = _nt / this->wt;                                                \
    int _wt = _nt % this->wt;                                                \
    __n = _t / this->nt;                                                     \
    __ih = _ht * (A - K + 1) - this->tp;                                     \
    __iw = _wt * (A - K + 1) - this->lp;                                     \
    __hA_start = (_ht > 0) ? 0 : this->tp;                                   \
    __wA_start = (_wt > 0) ? 0 : this->lp;                                   \
    __hA_end = (_ht < this->ht - 1) ? A - 1 : hA_end_;                       \
    __wA_end = (_wt < this->wt - 1) ? A - 1 : wA_end_;                       \
  } while (0)

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weights_plain(
    Type * __restrict tweights, Type * __restrict weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD10(Type, aweights_v, weights, oc4, this->oc3, this->O2, V, this->ic4, this->ic3, this->I2, V, K, K);
  MD10(Type, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);

  int s = this->IC * this->kh * this->kw;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto readin_v = [&](Type ain[K][K][V][V], Type *wei) {
    MD5(Type, awei, wei, V, this->ic2, V, K, K);

    for_each (_hK, K) {
    for_each (_wK, K) {
    for_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm512_i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm512_store_ps(ain[_hK][_wK][_iV], t);
      } else {
        for_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](Type ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(Type, awei, weights, this->oc, this->ic, K, K);

    assert(this->ic4 == 1 && this->oc4 == 1);
    int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
    int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;
    int iV = is_Ir ? this->Ir : V;

    if (is_Or) {
      for_each (_hK, K) {
      for_each (_wK, K) {
      for_each (_iV, iV) {
#pragma omp simd
      for_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      for_each (_hK, K) {
      for_each (_wK, K) {
      for_each (_iV, iV) {
#pragma omp simd
      for_each (_oV, V) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    }
  };

#pragma omp for nowait collapse(6) schedule(static)
  for_each (_oc4, oc4) {
  for_each (_ic4, this->ic4) {
  for_each (_oc3, this->oc3) {
  for_each (_ic3, this->ic3) {
  for_each (_O2, this->O2) {
  for_each (_I2, this->I2) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;

    alignas(64) Type ain[K][K][V][V];
    alignas(64) Type aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O2, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md10(aweights_v, _oc4, _oc3, _O2, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_stream_ps(&md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                               _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_store_ps(&md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                              _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      for_each (_wA, A) {
      for_each (_hA, A) {
      for_each (_iV, V) {
#pragma omp simd
      for_each (_oV, V) {
        md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O2, _I2, _iV, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD10(Type, aweights, weights, oc4, this->oc3, this->O2, this->ic4, this->ic3, this->I2, K, K, V, V);
  MD10(Type, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);

#pragma omp for nowait collapse(6) schedule(static)
  for_each (_oc4, oc4) {
  for_each (_ic4, this->ic4) {
  for_each (_oc3, this->oc3) {
  for_each (_ic3, this->ic3) {
  for_each (_O2, this->O2) {
  for_each (_I2, this->I2) {
    alignas(64) Type aout[A][A][V][V];
    Type *in = &md10(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = Type[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_stream_ps(&md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                               _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_store_ps(&md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                              _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      for_each (_wA, A) {
      for_each (_hA, A) {
      for_each (_iV, V) {
#pragma omp simd
        for_each (_oV, V)
          md10(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O2, _I2, _iV, _oV)
              = aout[_wA][_hA][_iV][_oV];
      }}}
    }
  }}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_weights(
    Type *tweights, Type *weights, int oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights, oc4);
  else
    __trans_weights_plain(tweights, weights, oc4);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsa_blocked(
    Type *tweights, Type *weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD10(Type, aweights, weights, this->oc4, this->oc3, this->O2, this->ic4, this->ic3, this->I2, K, K, V, V);
  MD10(Type, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3, this->ic3, this->O2, this->I2, V, V);

#pragma omp for nowait collapse(6) schedule(static)
  for_each (_oc4, this->oc4) {
  for_each (_ic4, this->ic4) {
  for_each (_oc3, this->oc3) {
  for_each (_ic3, this->ic3) {
  for_each (_O2, this->O2) {
  for_each (_I2, this->I2) {
    alignas(64) Type aout[A][A][V][V];
    Type *in = &md10(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = Type[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_stream_ps(&md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                               _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_store_ps(&md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                              _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      for_each (_wA, A) {
      for_each (_hA, A) {
      for_each (_iV, V) {
#pragma omp simd
      for_each (_oV, V) {
        md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O2, _I2, _iV, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsa_plain(
    Type * __restrict tweights, Type * __restrict weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD10(Type, aweights, weights, this->oc4, this->oc3, this->O2, V, this->ic4, this->ic3, this->I2, V, K, K);
  MD10(Type, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3, this->ic3, this->O2, this->I2, V, V);

  int s = this->IC * this->kh * this->kw;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto readin_v = [&](Type ain[K][K][V][V], Type *wei) {
    MD5(Type, awei, wei, V, this->ic2, V, K, K);

    for_each (_hK, K) {
    for_each (_wK, K) {
    for_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm512_i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm512_store_ps(ain[_hK][_wK][_iV], t);
      } else {
        for_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](Type ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(Type, awei, weights, this->oc, this->ic, K, K);

    int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
    int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;
    int iV = is_Ir ? this->Ir : V;
    int oV = is_Or ? this->Or : V;

    for_each (_hK, K) {
    for_each (_wK, K) {
    for_each (_iV, iV) {
#pragma omp simd
    for_each (_oV, oV) {
      ain[_hK][_wK][_iV][_oV]
              = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
    }}}}
  };


#pragma omp for nowait collapse(6) schedule(static)
  for_each (_oc4, this->oc4) {
  for_each (_ic4, this->ic4) {
  for_each (_oc3, this->oc3) {
  for_each (_ic3, this->ic3) {
  for_each (_O2, this->O2) {
  for_each (_I2, this->I2) {

    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;

    alignas(64) Type ain[K][K][V][V];
    alignas(64) Type aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O2, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md10(aweights, _oc4, _oc3, _O2, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_stream_ps(&md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                               _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
        for_each (_iV, V) {
          _mm512_store_ps(&md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                              _O2, _I2, _iV, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      for_each (_wA, A) {
      for_each (_hA, A) {
      for_each (_iV, V) {
#pragma omp simd
      for_each (_oV, V) {
        md10(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O2, _I2, _iV, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_weightsa(
    Type *tweights, Type *weights)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weightsa_blocked(tweights, weights);
  else
    __trans_weightsa_plain(tweights, weights);
}

} // namespace euler
