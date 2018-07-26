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
void elx_conv_wino_t<Type, A, K, V, I>::__trans_input_plain(
    Type * __restrict tinput, Type * __restrict input, int _t2, int Tz)
{
  // n, IC, ih, iw => t2 | wA, hA, ic3, I2, T, V
  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];
  alignas(64) Type ain[A][A][V];
  int s = this->ih * this->iw;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto readin_v = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    for_each (_hA, A) {
    for_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        for_each (_V, V)
          ain[_hA][_wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m512 t = _mm512_i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm512_store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          for_each (_V, V)
            ain[_hA][_wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD4(Type, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    assert(this->ic4 == 1);
    if (is_Ir) {
      for_each (_hA, A) {
      for_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          for_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
#pragma omp simd
          for_each (_v, this->Ir)
            ain[_hA][_wA][_v] = md4(ainput, _n,
                (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
        }
      }}
    } else {
      for_each (_hA, A) {
      for_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          for_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
#pragma omp simd
          for_each (_v, V)
            ain[_hA][_wA][_v] = md4(ainput, _n,
                (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
        }
      }}
    }
  };

  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
    for_each (_T, Tz) {
      if (this->Ir != V) {
        readin_r(_ic3, _I2, _T, ain);
      } else
        readin_v(_ic3, _I2, _T, ain);

      ker_trans_input_(*this, aout, (Type *)ain, 0, 0, 0, -1);

      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_) {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m512 *)&aout[_wA][_hA][0]));
          }}
        } else {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m512 *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
#pragma omp simd
        for_each (_V, V) {
          md6(atinput, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
        }}}
      }
    }
  }}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_input_blocked(
    Type * __restrict tinput, Type * __restrict input, int _t2, int Tz)
{
  // n, ic2, ih, iw, V => t2 | wA, hA, ic3, I2, T, V
  MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, this->ih, this->iw, V);
  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];

  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
  for_each (_T, Tz) {
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    Type *in = &md7(ainput, _n, 0, _ic3, _I2, _ih, _iw, 0);
    if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
        && _wA_end == A - 1)
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(
          *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_) {
        for_each (_wA, A) {
        for_each (_hA, A) {
          _mm512_stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_wA][_hA][0]));
        }}
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
          _mm512_store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_wA][_hA][0]));
        }}
      }
    } else {
      for_each (_wA, A) {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        md6(atinput, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
      }}}
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_input(
    Type * __restrict tinput, Type * __restrict input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input, _t2, Tz);
  else
    __trans_input_plain(tinput, input, _t2, Tz);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_input_blocked(
    Type * __restrict tinput, Type * __restrict input)
{
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, this->ih, this->iw, V);
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->IC);

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    alignas(64) Type aout[A][A][V];

    for_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      Type *in = &md7(ainput, _n, 0, _ic3, _I2, _ih, _iw, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);

      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_) {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                           *((__m512 *)&aout[_wA][_hA][0]));
          }}
        } else {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                          *((__m512 *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
#pragma omp simd
        for_each (_V, V) {
          md6(atinput6, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
        }}}
      }
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_input_plain(
    Type * __restrict tinput, Type * __restrict input)
{
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->IC);

  int s = this->ih * this->iw;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto readin_v = [&](int _t2, int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    for_each (_hA, A) {
    for_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        for_each (_V, V)
          ain[_hA][_wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m512 t = _mm512_i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm512_store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          for_each (_V, V)
            ain[_hA][_wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _t2, int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD4(Type, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    assert(this->ic4 == 1);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    if (is_Ir) {
      for_each (_hA, A) {
        for_each (_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
              || _wA > _wA_end) {
#pragma omp simd
            for_each (_V, V)
              ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            for_each (_v, this->Ir)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }
    } else {
      for_each (_hA, A) {
        for_each (_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
              || _wA > _wA_end) {
#pragma omp simd
            for_each (_V, V)
              ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            for_each (_v, V)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }

    }
  };

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    alignas(64) Type aout[A][A][V];
    alignas(64) Type ain[A][A][V];

    for_each (_T, Tz) {
      if (this->Ir != V)
        readin_r(_t2, _ic3, _I2, _T, ain);
      else
        readin_v(_t2, _ic3, _I2, _T, ain);
      ker_trans_input_(*this, aout, (Type *)ain, 0, 0, 0, -1);

      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_) {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m512 *)&aout[_wA][_hA][0]));
          }}
        } else {
          for_each (_wA, A) {
          for_each (_hA, A) {
            _mm512_store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m512 *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        for_each (_wA, A) {
        for_each (_hA, A) {
#pragma omp simd
        for_each (_V, V) {
          md6(atinput6, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
        }}}
      }
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_input(
    Type *tinput, Type *input)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input);
  else
    __trans_input_plain(tinput, input);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_inputa_blocked(
    Type *tinput, Type *input, int _t2, int _wA, int Tz)
{
  // n, ic2, ih, iw, V => t2, wA | hA, ic3, I2, T, V
  MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, this->ih, this->iw, V);
  MD5(Type, atinput, tinput, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];

  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
  for_each (_T, Tz) {
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    Type *in = &md7(ainput, _n, 0, _ic3, _I2, _ih, _iw, 0);
    if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
        && _wA_end == A - 1) {
      ker_trans_inputa_(*this, aout, in, _wA, 0, A - 1, 0, A - 1);
    } else {
      ker_trans_inputa0_(
          *this, aout, in, _wA, _hA_start, _hA_end, _wA_start, _wA_end);
    }

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_) {
        for_each (_hA, A) {
          _mm512_stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_hA][_wA][0]));
        }
      } else {
        for_each (_hA, A) {
          _mm512_store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_hA][_wA][0]));
        }
      }
    } else {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        md5(atinput, _hA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
      }}
    }
  }}}
}
 
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_inputa_plain(
    Type * __restrict tinput, Type * __restrict input, int _t2, int _wA, int Tz)
{
  // n, ic2, ih, iw, V => t2, wA | hA, ic3, I2, T, V
  MD5(Type, atinput, tinput, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];
  alignas(64) Type ain[A][A][V];
  int s = this->ih * this->iw;
  const __m512i vindex
      = _mm512_set_epi32(15 * s, 14 * s, 13 * s, 12 * s, 11 * s, 10 * s,
          9 * s, 8 * s, 7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);

  auto readin_v = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    for_each (__wA, A) {
    for_each (__hA, A) {
      if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
          || __wA > _wA_end) {
#pragma omp simd
        for_each (_V, V)
          ain[__hA][__wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m512 t = _mm512_i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + __hA, _iw + __wA),
              scale);
          _mm512_store_ps(ain[__hA][__wA], t);
        } else {
#pragma omp simd
          for_each (_V, V)
            ain[__hA][__wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + __hA, _iw + __wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD4(Type, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    assert(this->ic4 == 1);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    if (is_Ir) {
      for_each (__wA, A) {
        for_each (__hA, A) {
          if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
              || __wA > _wA_end) {
#pragma omp simd
            for_each (_V, V)
              ain[__hA][__wA][_V] = 0.0f;
          } else {
#pragma omp simd
            for_each (_V, this->Ir)
              ain[__hA][__wA][_V] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
          }
        }
      }
    } else {
      for_each (__wA, A) {
        for_each (__hA, A) {
          if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
              || __wA > _wA_end) {
#pragma omp simd
            for_each (_V, V)
              ain[__hA][__wA][_V] = 0.0f;
          } else {
#pragma omp simd
            for_each (_V, V)
              ain[__hA][__wA][_V] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
          }
        }
      }
    }
  };

  for_each (_ic3, this->ic3) {
  for_each (_I2, this->I2) {
  for_each (_T, Tz) {
    if (this->Ir != V)
      readin_r(_ic3, _I2, _T, ain);
    else
      readin_v(_ic3, _I2, _T, ain);
    ker_trans_inputa_(*this, aout, (Type *)ain, _wA, 0, A - 1, 0, -1);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_) {
        for_each (_hA, A)
          _mm512_stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_hA][_wA][0]));
      } else {
        for_each (_hA, A)
          _mm512_store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m512 *)&aout[_hA][_wA][0]));
      }
    } else {
      for_each (_hA, A) {
#pragma omp simd
      for_each (_V, V) {
        md5(atinput, _hA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
      }}
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_inputa(
    Type *tinput, Type *input, int _t2, int _wA, int Tz)
{
  if(input_is_bfmt_ || input_as_bfmt_)
    __trans_inputa_blocked(tinput, input, _t2, _wA, Tz);
  else
    __trans_inputa_plain(tinput, input, _t2, _wA, Tz);
}
} // namespace euler
