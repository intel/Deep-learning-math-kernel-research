#include <cmath>
#include <string.h>
#include <float.h>
#include "el_intrin.hpp"
#include "el_parallel.hpp"
#include "elx_conv_wino_trans_weights.hpp"

namespace euler {

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_post(TweightsType *__restrict tweights, op_type at[A][A][V][V],
    int _O4, int _I4, int _O3, int _I3, int _O1, int _I2, int _O) {
  MD9(TweightsType, atweights, tweights, ep->O3, ep->I3, A, A,
      ep->O1, ep->I2, V, ep->O, V);
  if (I == ISA_AVX512 && std::is_same<op_type, float>::value
      && std::is_same<TweightsType, float>::value) {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
    iter_each (_iV, V) {
      _mm512_store_ps(&md9(atweights, _O3, _I3, _hA, _wA,
          _O1, _I2, _iV, _O, 0), *((__m512 *)&at[_hA][_wA][_iV][0]));
    }}}
  } else if (I == ISA_AVX512 && std::is_same<op_type, float>::value
     && std::is_same<TweightsType, float16>::value) {
    if (ep->O == 2) { // fp32 -> bf16  combine two _O
      auto mask = _mm<V>::set1_epi32(0xFFFF0000);
      if (_O ==  0) {
        iter_each (_hA, A) {
        iter_each (_wA, A) {
        iter_each (_iV, V) {
          auto si512 = _mm<V>::load_si512(&at[_hA][_wA][_iV][0]);
          auto w0 = _mm<V>::and_epi32(si512, mask);
          _mm<V>::store_si512((__i<V> *)&md9(atweights, _O3, _I3, _hA, _wA,
                                             _O1, _I2, _iV, 0, 0), w0);
        }}}
      } else {
        iter_each (_hA, A) {
        iter_each (_wA, A) {
        iter_each (_iV, V) {
          auto si512 = _mm<V>::load_si512(&at[_hA][_wA][_iV][0]);
          auto w1 = _mm<V>::and_epi32(si512, mask);
          auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

          auto w0 = _mm<V>::load_si512(
              &md9(atweights, _O3, _I3, _hA, _wA, _O1, _I2, _iV, 0, 0));

          auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
          _mm<V>::store_si512((__i<V> *)&md9(atweights, _O3, _I3, _hA, _wA,
                                             _O1, _I2, _iV, 0, 0), w0w1);
        }}}
      }
    } else {       // fp32 -> fp16
      iter_each (_hA, A) {
      iter_each (_wA, A) {
      iter_each (_iV, V) {
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&at[_hA][_wA][_iV][0],
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256((__m256i *)&md9(atweights, _O3,
                             _I3, _hA, _wA, _O1, _I2, _iV, _O, 0), fp16v);
      }}}
    }
  } else {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
    iter_each (_iV, V) {
#pragma omp simd
    iter_each (_oV, V) {
      md9(atweights, _O3, _I3, _hA, _wA, _O1, _I2, _iV, _O, _oV)
          = at[_hA][_wA][_iV][_oV];
    }}}}
  }
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_oihw(
    TweightsType *__restrict tweights, WeightsType *__restrict weights, int O4) {
  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ic * ep->kh * ep->kw));

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, ep->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
        constexpr auto scale = sizeof(WeightsType);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _O4, int _O3, int _O2,
                      int _I4, int _I3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, ep->oc, ep->ic, K, K);

    assert(ep->I4 == 1 && ep->O4 == 1);
    int _oc2 = _O4 * ep->O3 * ep->O2 + _O3 * ep->O2 + _O2;
    int _ic2 = _I4 * ep->I3 * ep->I2 + _I3 * ep->I2 + _I2;
    int iV = is_Ir ? ep->Ir : V;

    __m<V> z = _mm<V>::set1_ps(0.0);
    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      _mm<V>::store_ps(ain[_hK][_wK][_iV], z);
    }}}

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, ep->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
          constexpr auto scale = sizeof(WeightsType);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md4(awei, _oc2 * V, _ic2 * V + _iV, _hK, _wK), scale);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            ain[_hK][_wK][_iV][_oV]
                = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
          }
        }
      }}}
    }
  };

  estl::parallel_for<6>([&](int _O4, int _I4, int _O3,
                            int _I3, int _O1, int _I2) {
    // oc2, ic2, hK, wK, V, V => O4, I4, O3, I3, wA, hA, O2, I2, V, V
    MD11(WeightsType, aweights_v, weights, O4, ep->O3, ep->O1, ep->O, V,
        ep->I4, ep->I3, ep->I2, V, K, K);
    MD3(TweightsType, atweights, tweights, ep->O4, ep->I4,
        ep->O3 * ep->I3 * A * A * ep->O2 * ep->I2 * V * V);

    iter_each (_O, ep->O) {
      bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1
          && _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
      bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
          && _O3 == ep->O3 - 1 && _O1 == ep->O1 - 1
          && _O == ep->O - 1;

      alignas(64) WeightsType ain[K][K][V][V];
      alignas(64) op_type aout[A][A][V][V];

      if (ep->Ir != V || is_Ir || is_Or)
        readin_r(ain, _O4, _O3, _O1 * ep->O + _O, _I4, _I3, _I2, is_Ir, is_Or);
      else
        readin_v(
            ain, &md11(aweights_v, _O4, _O3, _O1, _O, 0, _I4, _I3, _I2, 0, 0, 0));

      ker_trans_weights_(aout, ain);
      __execute_post(&md3(atweights, _O4, _I4, 0), aout, _O4, _I4, _O3,
                           _I3, _O1, _I2, _O);
    }
  }, O4, ep->I4, ep->O3, ep->I3, ep->O1, ep->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_blocked(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int O4) {
  estl::parallel_for<6>([&](int _O4, int _I4, int _O3,
                            int _I3, int _O1, int _I2) {
    // oc2, ic2, hK, wK, V, V => O4, I4, O3, I3, wA, hA, O2, I2, V, V
    MD8(WeightsType, aweights, weights, O4, ep->O3, ep->O1, ep->O,
        ep->I4, ep->I3, ep->I2, K * K * V * V);
    MD3(TweightsType, atweights, tweights, ep->O4, ep->I4,
        ep->O3 * ep->I3 * A * A * ep->O2 * ep->I2 * V * V);
    iter_each (_O, ep->O) {
      alignas(64) op_type aout[A][A][V][V];
      WeightsType *in = &md8(aweights, _O4, _O3, _O1, _O, _I4, _I3, _I2, 0);
      using Array = WeightsType[K][K][V][V];
      ker_trans_weights_(aout, *(Array *)in);
      __execute_post(&md3(atweights, _O4, _I4, 0), aout, _O4, _I4, _O3,
                           _I3, _O1, _I2, _O);
    }
  }, O4, ep->I4, ep->O3, ep->I3, ep->O1, ep->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_hwio(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int O4) {
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _O4, int _O3, int _O1, int _O,
                    int _I4, int _I3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, ep->ic, ep->oc);
    int iV = is_Ir ? ep->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), ep->I4,
          ep->I3, ep->I2, V, ep->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _I4, _I3, _I2, _iV, 0),
          ep->O4, ep->O3, ep->O1, ep->O, V);
      if (is_Or) {
        iter_each (_oV, ep->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _O4, _O3, _O1, _O, _oV);
      } else {
        if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _O4, _O3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _O4, _O3, _O1, _O, _oV);
        }
      }
    }}}
  };
  estl::parallel_for<6>([&](int _O4, int _I4, int _O3,
                            int _I3, int _O1, int _I2) {
    MD3(TweightsType, atweights, tweights, ep->O4, ep->I4,
        ep->O3 * ep->I3 * A * A * ep->O2 * ep->I2 * V * V);
    iter_each (_O, ep->O) {
      bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1
          && _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
      bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
          && _O3 == ep->O3 - 1 && _O1 == ep->O1 - 1
          && _O == ep->O - 1;
      alignas(64) WeightsType ain[K][K][V][V];
      alignas(64) op_type aout[A][A][V][V];

      readin(ain, weights, _O4, _O3, _O1, _O, _I4, _I3, _I2, is_Ir, is_Or);
      ker_trans_weights_(aout, ain);
      __execute_post(&md3(atweights, _O4, _I4, 0), aout, _O4, _I4, _O3,
                           _I3, _O1, _I2, _O);
    }
  }, O4, ep->I4, ep->O3, ep->I3, ep->O1, ep->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_oihw(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _I4, int _O4) {
  MD11(WeightsType, aweights_v, weights, ep->O4, ep->O3, ep->O1, ep->O, V,
      ep->I4, ep->I3, ep->I2, V, K, K);

  const __i<V> vindex = _mm<V>::set_epi32(SET_VINDEX_16(ep->ic * ep->kh * ep->kw));

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, ep->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
        constexpr auto scale = sizeof(WeightsType);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _O4, int _O3, int _O2,
                      int _I4, int _I3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, ep->oc, ep->ic, K, K);

    assert(ep->I4 == 1 && ep->O4 == 1);
    int _oc2 = _O4 * ep->O3 * ep->O2 + _O3 * ep->O2 + _O2;
    int _ic2 = _I4 * ep->I3 * ep->I2 + _I3 * ep->I2 + _I2;
    int iV = is_Ir ? ep->Ir : V;

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, ep->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
          constexpr auto scale = sizeof(WeightsType);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md4(awei, _oc2 * V, _ic2 * V + _iV, _hK, _wK), scale);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            ain[_hK][_wK][_iV][_oV]
                = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
          }
        }
      }}}
    }
  };

  iter_each (_O3, ep->O3) {
  iter_each (_I3, ep->I3) {
  iter_each (_O1, ep->O1) {
  iter_each (_I2, ep->I2) {
  iter_each (_O, ep->O) {
    bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1
        && _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
        && _O3 == ep->O3 - 1 && _O1 == ep->O1 - 1
        && _O == ep->O - 1;

    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) op_type aout[A][A][V][V];

    if (ep->Ir != V || is_Ir || is_Or)
      readin_r(ain, _O4, _O3, _O1 * ep->O + _O, _I4, _I3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _O4, _O3, _O1, _O, 0, _I4, _I3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);
    __execute_post(tweights, aout, _O4, _I4, _O3, _I3, _O1, _I2, _O);
  }}}}}
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_hwio(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _I4, int _O4) {
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _O4, int _O3, int _O1, int _O,
                    int _I4, int _I3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, ep->ic, ep->oc);
    int iV = is_Ir ? ep->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), ep->I4,
          ep->I3, ep->I2, V, ep->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _I4, _I3, _I2, _iV, 0),
          ep->O4, ep->O3, ep->O1, ep->O, V);
      if (is_Or) {
        iter_each (_oV, ep->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _O4, _O3, _O1, _O, _oV);
      } else {
        if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _O4, _O3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _O4, _O3, _O1, _O, _oV);
        }
      }
    }}}
  };

  iter_each (_O3, ep->O3) {
  iter_each (_I3, ep->I3) {
  iter_each (_O1, ep->O1) {
  iter_each (_I2, ep->I2) {
  iter_each (_O, ep->O) {
    bool is_Ir = ep->Ir != V && _I4 == ep->I4 - 1
        && _I3 == ep->I3 - 1 && _I2 == ep->I2 - 1;
    bool is_Or = ep->Or != V && _O4 == ep->O4 - 1
        && _O3 == ep->O3 - 1 && _O1 == ep->O1 - 1
        && _O == ep->O - 1;
    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) op_type aout[A][A][V][V];

    readin(ain, weights, _O4, _O3, _O1, _O, _I4, _I3, _I2, is_Ir, is_Or);
    ker_trans_weights_(aout, ain);
    __execute_post(tweights, aout, _O4, _I4, _O3, _I3, _O1, _I2, _O);
  }}}}}
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_blocked(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _I4, int _O4) {
  MD11(WeightsType, aweights, weights, ep->O4, ep->O3, ep->O1, ep->O,
      ep->I4, ep->I3, ep->I2, K, K, V, V);

  iter_each (_O3, ep->O3) {
  iter_each (_I3, ep->I3) {
  iter_each (_O1, ep->O1) {
  iter_each (_I2, ep->I2) {
  iter_each (_O, ep->O) {
    alignas(64) op_type aout[A][A][V][V];
    WeightsType *in = &md11(
        aweights, _O4, _O3, _O1, _O, _I4, _I3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);
    __execute_post(tweights, aout, _O4, _I4, _O3, _I3, _O1, _I2, _O);
  }}}}};
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<TweightsType, WeightsType, I, A, K, V>
::execute(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int O4) {
  if (weights_is_bfmt_ || weights_as_bfmt_)
    this->__execute_blocked(tweights, weights, O4);
  else if (ep->weights_fmt == hwio)
    this->__execute_hwio(tweights, weights, O4);
  else
    this->__execute_oihw(tweights, weights, O4);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<TweightsType, WeightsType, I, A, K, V>
::execute(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _I4, int _O4) {
  if (weights_is_bfmt_ || weights_as_bfmt_)
    this->__execute_blocked(tweights, weights, _I4, _O4);
  else if (ep->weights_fmt == hwio)
    this->__execute_hwio(tweights, weights, _I4, _O4);
  else
    this->__execute_oihw(tweights, weights, _I4, _O4);
}

template <typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
::quantization(float *__restrict tweights_scale,
    float *__restrict tweights_shift,
    int8_t *__restrict tweights_s8,
    TweightsType *__restrict tweights, int O4) {
  __m<V> zero = _mm<V>::set1_ps(0.0);
  __m<V> mmscale = _mm<V>::set1_ps(EL_INT8_MAX);

  // abs-max
  estl::parallel_for<7>([&](int _O4, int _I4, int _O3, int _hA, int _wA,
                            int _O1, int _O) {
    MD11(TweightsType, atweights, tweights,
        O4, ep->I4, ep->O3, ep->I3, A, A, ep->O1, ep->I2, V, ep->O, V);
    MD8(float, atweights_scale, tweights_scale,
        O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);
    __m<V> mmabs_max = _mm<V>::set1_ps(0.0);

    iter_each (_I3, ep->I3) {
    iter_each (_I2, ep->I2) {
    iter_each (_iV, V) {
      mmabs_max =
          _mm<V>::max_ps(mmabs_max, _mm512_abs_ps(*(__m<V> *)&md11(atweights,
          _O4, _I4, _O3, _I3, _hA, _wA, _O1, _I2, _iV, _O, 0)));
    }}}
    _mm512_store_ps(&md8(atweights_scale,
        _O4, _I4, _O3, _hA, _wA, _O1, _O, 0), mmabs_max);
  }, O4, ep->I4, ep->O3, A, A, ep->O1, ep->O);

  // quantization
  estl::parallel_for<11>([&](int _O4, int _I4, int _O3,
      int _I3, int _hA, int _wA, int _O1, int _I2, int _V1, int _O, int _iVx) {
    MD12(int8_t, atweights_s8, tweights_s8, O4, ep->I4, ep->O3, ep->I3,
        A, A, ep->O1, ep->I2, ep->V1, ep->O, V, ep->Vx);
    MD8(float, atweights_scale, tweights_scale,
        O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);

    // I2 V => I2 V1 Vx
    MD12(TweightsType, _atweights, tweights, O4, ep->I4, ep->O3, ep->I3,
         A, A, ep->O1, ep->I2, ep->V1, ep->Vx, ep->O, V);
    __m<V> t0;
    // multi scal
    t0 = _mm<V>::mul_ps(*(__m<V> *)&md12(_atweights,
        _O4, _I4, _O3, _I3, _hA, _wA, _O1, _I2, _V1, _iVx, _O, 0), mmscale);
    t0 = _mm<V>::div_ps(t0, *(__m<V> *)&md8(atweights_scale,
        _O4, _I4, _O3, _hA, _wA, _O1, _O, 0));
    // rounding
    t0 = _mm<V>::roundscale_ps(
        t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // int8_t
    TweightsType *rounded = (TweightsType *)&t0;
#pragma omp simd
    iter_each (_oV, V) {
      md12(atweights_s8,
          _O4, _I4, _O3, _I3, _hA, _wA, _O1, _I2, _V1, _O, _oV, _iVx) =
          (int8_t)rounded[_oV];
    }
  }, O4, ep->I4, ep->O3, ep->I3, A, A, ep->O1, ep->I2, ep->V1, ep->O, ep->Vx);

  // weights-acc
  estl::parallel_for<8>(mthr_,
      [&](int _O4, int _I4, int _O3, int _hA, int _wA, int _O1, int _O, int _oV) {
    MD12(int8_t, atweights_s8, tweights_s8, O4, ep->I4, ep->O3, ep->I3,
        A, A, ep->O1, ep->I2, ep->V1, ep->O, V, ep->Vx);
    MD8(float, atweights_shift, tweights_shift,
        O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);

    float acc = 0;
    iter_each (_I3, ep->I3) {
    iter_each (_I2, ep->I2) {
    iter_each (_V1, ep->V1) {
    iter_each (_iVx, ep->Vx) {
      acc += (float)md12(atweights_s8,
          _O4, _I4, _O3, _I3, _hA, _wA, _O1, _I2, _V1, _O, _oV, _iVx);
    }}}}
    md8(atweights_shift, _O4, _I4, _O3, _hA, _wA, _O1, _O, _oV) = acc;
  }, O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);

  // weights-scale, combine with restore
  estl::parallel_for<8>([&](int _O4, int _I4, int _O3,
                            int _hA, int _wA, int _O1, int _O, int _oV) {
    MD8(float, atweights_scale, tweights_scale,
        O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);
    MD8(float, atweights_shift, tweights_shift,
        O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);

    float Sw =
        md8(atweights_scale, _O4, _I4, _O3, _hA, _wA, _O1, _O, _oV);
    Sw /= EL_INT8_MAX;
    float Zw =
        md8(atweights_shift, _O4, _I4, _O3, _hA, _wA, _O1, _O, _oV);
    if (ep->sampling_kind == CALIBRATED) {
      Sw = Sw * ep->tinput_quant_S;
      Zw = -Zw * Sw * ep->tinput_quant_z;
      md8(atweights_shift, _O4, _I4, _O3, _hA, _wA, _O1, _O, _oV) = Zw;
    }
    md8(atweights_scale, _O4, _I4, _O3, _hA, _wA, _O1, _O, _oV) = Sw;
  }, O4, ep->I4, ep->O3, A, A, ep->O1, ep->O, V);
}

template <typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
::execute(float *__restrict tweights_scale,
    float *__restrict tweights_shift,
    int8_t *__restrict tweights_s8,
    TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int O4) {
  {
    if (weights_is_bfmt_ || weights_as_bfmt_)
      this->__execute_blocked(tweights, weights, O4);
    else
      this->__execute_oihw(tweights, weights, O4);

    quantization(tweights_scale, tweights_shift,
                 tweights_s8, tweights, O4);
  }
}

template class elx_conv_wino_trans_weights_t<float, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, float, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<short, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, float, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, float, ISA_AVX512, 7, 3, 16>;

#ifdef ENABLE_USER_FP16
template class elx_conv_wino_trans_weights_t<float, short, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<float, short, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<short, short, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<short, short, ISA_AVX512, 7, 3, 16>;

template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_AVX512, 4, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_AVX512, 5, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_AVX512, 6, 3, 16>;
template class elx_conv_wino_trans_weights_t<int8_t, short, ISA_AVX512, 7, 3, 16>;
#endif

} // namespace euler
