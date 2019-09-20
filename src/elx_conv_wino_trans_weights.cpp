#include <cmath>
#include <string.h>
#include <float.h>
#include "el_intrin.hpp"
#include "el_parallel.hpp"
#include "elx_conv_wino_trans_weights.hpp"

namespace euler {

static constexpr float INT8GEMM_TWT_QTSCALE = 127.0;

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_post(TweightsType *__restrict tweights, op_type at[A][A][V][V],
    int _oc4, int _ic4, int _oc3, int _ic3, int _O1, int _I2, int _O) {
  MD9(TweightsType, atweights, tweights, xc->oc3, xc->ic3, A, A,
      xc->O1, xc->I2, V, xc->O, V);
  if (I == ISA_SKX_AVX512 && std::is_same<op_type, float>::value
      && std::is_same<TweightsType, float>::value) {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
    iter_each (_iV, V) {
      _mm512_store_ps(&md9(atweights, _oc3, _ic3, _hA, _wA,
          _O1, _I2, _iV, _O, 0), *((__m512 *)&at[_hA][_wA][_iV][0]));
    }}}
  } else if (I == ISA_SKX_AVX512 && std::is_same<op_type, float>::value
     && std::is_same<TweightsType, float16>::value) {
    if (xc->O == 2) { // fp32 -> bf16  combine two _O
      auto mask = _mm<V>::set1_epi32(0xFFFF0000);
      if (_O ==  0) {
        iter_each (_hA, A) {
        iter_each (_wA, A) {
        iter_each (_iV, V) {
          auto si512 = _mm<V>::load_si512(&at[_hA][_wA][_iV][0]);
          auto w0 = _mm<V>::and_epi32(si512, mask);
          _mm<V>::store_si512((__i<V> *)&md9(atweights, _oc3, _ic3, _hA, _wA,
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
              &md9(atweights, _oc3, _ic3, _hA, _wA, _O1, _I2, _iV, 0, 0));

          auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
          _mm<V>::store_si512((__i<V> *)&md9(atweights, _oc3, _ic3, _hA, _wA,
                                             _O1, _I2, _iV, 0, 0), w0w1);
        }}}
      }
    } else {       // fp32 -> fp16
      iter_each (_hA, A) {
      iter_each (_wA, A) {
      iter_each (_iV, V) {
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&at[_hA][_wA][_iV][0],
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256((__m256i *)&md9(atweights, _oc3,
                             _ic3, _hA, _wA, _O1, _I2, _iV, _O, 0), fp16v);
      }}}
    }
  } else {
    iter_each (_hA, A) {
    iter_each (_wA, A) {
    iter_each (_iV, V) {
#pragma omp simd
    iter_each (_oV, V) {
      md9(atweights, _oc3, _ic3, _hA, _wA, _O1, _I2, _iV, _O, _oV)
          = at[_hA][_wA][_iV][_oV];
    }}}}
  }
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_oihw(
    TweightsType *__restrict tweights, WeightsType *__restrict weights, int oc4) {
  SET_EPI32(xc->ic * xc->kh * xc->kw)

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, xc->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
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

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, xc->oc, xc->ic, K, K);

    assert(xc->ic4 == 1 && xc->oc4 == 1);
    int _oc2 = _oc4 * xc->oc3 * xc->O2 + _oc3 * xc->O2 + _O2;
    int _ic2 = _ic4 * xc->ic3 * xc->I2 + _ic3 * xc->I2 + _I2;
    int iV = is_Ir ? xc->Ir : V;

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
      iter_each (_oV, xc->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
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

  parallel_for<6>(mthr_, [&](int _oc4, int _ic4, int _oc3,
                             int _ic3, int _O1, int _I2) {
    // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
    MD11(WeightsType, aweights_v, weights, oc4, xc->oc3, xc->O1, xc->O, V,
        xc->ic4, xc->ic3, xc->I2, V, K, K);
    MD3(TweightsType, atweights, tweights, xc->oc4, xc->ic4,
        xc->oc3 * xc->ic3 * A * A * xc->O2 * xc->I2 * V * V);

    iter_each (_O, xc->O) {
      bool is_Ir = xc->Ir != V && _ic4 == xc->ic4 - 1
          && _ic3 == xc->ic3 - 1 && _I2 == xc->I2 - 1;
      bool is_Or = xc->Or != V && _oc4 == xc->oc4 - 1
          && _oc3 == xc->oc3 - 1 && _O1 == xc->O1 - 1
          && _O == xc->O - 1;

      alignas(64) WeightsType ain[K][K][V][V];
      alignas(64) op_type aout[A][A][V][V];

      if (xc->Ir != V || is_Ir || is_Or)
        readin_r(ain, _oc4, _oc3, _O1 * xc->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
      else
        readin_v(
            ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

      ker_trans_weights_(aout, ain);
      __execute_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                           _ic3, _O1, _I2, _O);
    }
  }, oc4, xc->ic4, xc->oc3, xc->ic3, xc->O1, xc->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_blocked(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int oc4) {
  parallel_for<6>(mthr_, [&](int _oc4, int _ic4, int _oc3,
                             int _ic3, int _O1, int _I2) {
    // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
    MD8(WeightsType, aweights, weights, oc4, xc->oc3, xc->O1, xc->O,
        xc->ic4, xc->ic3, xc->I2, K * K * V * V);
    MD3(TweightsType, atweights, tweights, xc->oc4, xc->ic4,
        xc->oc3 * xc->ic3 * A * A * xc->O2 * xc->I2 * V * V);
    iter_each (_O, xc->O) {
      alignas(64) op_type aout[A][A][V][V];
      WeightsType *in = &md8(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0);
      using Array = WeightsType[K][K][V][V];
      ker_trans_weights_(aout, *(Array *)in);
      __execute_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                           _ic3, _O1, _I2, _O);
    }
  }, oc4, xc->ic4, xc->oc3, xc->ic3, xc->O1, xc->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_hwio(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int oc4) {
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _oc4, int _oc3, int _O1, int _O,
                    int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, xc->ic, xc->oc);
    int iV = is_Ir ? xc->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), xc->ic4,
          xc->ic3, xc->I2, V, xc->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _ic4, _ic3, _I2, _iV, 0),
          xc->oc4, xc->oc3, xc->O1, xc->O, V);
      if (is_Or) {
        iter_each (_oV, xc->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _oc4, _oc3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
        }
      }
    }}}
  };
  parallel_for<6>(mthr_, [&](int _oc4, int _ic4, int _oc3,
                            int _ic3, int _O1, int _I2) {
    MD3(TweightsType, atweights, tweights, xc->oc4, xc->ic4,
        xc->oc3 * xc->ic3 * A * A * xc->O2 * xc->I2 * V * V);
    iter_each (_O, xc->O) {
      bool is_Ir = xc->Ir != V && _ic4 == xc->ic4 - 1
          && _ic3 == xc->ic3 - 1 && _I2 == xc->I2 - 1;
      bool is_Or = xc->Or != V && _oc4 == xc->oc4 - 1
          && _oc3 == xc->oc3 - 1 && _O1 == xc->O1 - 1
          && _O == xc->O - 1;
      alignas(64) WeightsType ain[K][K][V][V];
      alignas(64) op_type aout[A][A][V][V];

      readin(ain, weights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, is_Ir, is_Or);
      ker_trans_weights_(aout, ain);
      __execute_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                           _ic3, _O1, _I2, _O);
    }
  }, oc4, xc->ic4, xc->oc3, xc->ic3, xc->O1, xc->I2);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_oihw(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _ic4, int _oc4) {
  MD11(WeightsType, aweights_v, weights, xc->oc4, xc->oc3, xc->O1, xc->O, V,
      xc->ic4, xc->ic3, xc->I2, V, K, K);

  SET_EPI32(xc->ic * xc->kh * xc->kw)

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, xc->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
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

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, xc->oc, xc->ic, K, K);

    assert(xc->ic4 == 1 && xc->oc4 == 1);
    int _oc2 = _oc4 * xc->oc3 * xc->O2 + _oc3 * xc->O2 + _O2;
    int _ic2 = _ic4 * xc->ic3 * xc->I2 + _ic3 * xc->I2 + _I2;
    int iV = is_Ir ? xc->Ir : V;

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, xc->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
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

  iter_each (_oc3, xc->oc3) {
  iter_each (_ic3, xc->ic3) {
  iter_each (_O1, xc->O1) {
  iter_each (_I2, xc->I2) {
  iter_each (_O, xc->O) {
    bool is_Ir = xc->Ir != V && _ic4 == xc->ic4 - 1
        && _ic3 == xc->ic3 - 1 && _I2 == xc->I2 - 1;
    bool is_Or = xc->Or != V && _oc4 == xc->oc4 - 1
        && _oc3 == xc->oc3 - 1 && _O1 == xc->O1 - 1
        && _O == xc->O - 1;

    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) op_type aout[A][A][V][V];

    if (xc->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * xc->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);
    __execute_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}}
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_hwio(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _ic4, int _oc4) {
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _oc4, int _oc3, int _O1, int _O,
                    int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, xc->ic, xc->oc);
    int iV = is_Ir ? xc->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), xc->ic4,
          xc->ic3, xc->I2, V, xc->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _ic4, _ic3, _I2, _iV, 0),
          xc->oc4, xc->oc3, xc->O1, xc->O, V);
      if (is_Or) {
        iter_each (_oV, xc->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _oc4, _oc3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
        }
      }
    }}}
  };

  iter_each (_oc3, xc->oc3) {
  iter_each (_ic3, xc->ic3) {
  iter_each (_O1, xc->O1) {
  iter_each (_I2, xc->I2) {
  iter_each (_O, xc->O) {
    bool is_Ir = xc->Ir != V && _ic4 == xc->ic4 - 1
        && _ic3 == xc->ic3 - 1 && _I2 == xc->I2 - 1;
    bool is_Or = xc->Or != V && _oc4 == xc->oc4 - 1
        && _oc3 == xc->oc3 - 1 && _O1 == xc->O1 - 1
        && _O == xc->O - 1;
    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) op_type aout[A][A][V][V];

    readin(ain, weights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    ker_trans_weights_(aout, ain);
    __execute_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}}
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_base<TweightsType, WeightsType, I, A, K, V>
::__execute_blocked(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _ic4, int _oc4) {
  MD11(WeightsType, aweights, weights, xc->oc4, xc->oc3, xc->O1, xc->O,
      xc->ic4, xc->ic3, xc->I2, K, K, V, V);

  iter_each (_oc3, xc->oc3) {
  iter_each (_ic3, xc->ic3) {
  iter_each (_O1, xc->O1) {
  iter_each (_I2, xc->I2) {
  iter_each (_O, xc->O) {
    alignas(64) op_type aout[A][A][V][V];
    WeightsType *in = &md11(
        aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);
    __execute_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}};
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<TweightsType, WeightsType, I, A, K, V>
::execute(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int oc4) {
  if (weights_is_bfmt_ || weights_as_bfmt_)
    this->__execute_blocked(tweights, weights, oc4);
  else if (xc->weights_fmt == hwio)
    this->__execute_hwio(tweights, weights, oc4);
  else
    this->__execute_oihw(tweights, weights, oc4);
}

template <typename TweightsType, typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<TweightsType, WeightsType, I, A, K, V>
::execute(TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int _ic4, int _oc4) {
  if (weights_is_bfmt_ || weights_as_bfmt_)
    this->__execute_blocked(tweights, weights, _ic4, _oc4);
  else if (xc->weights_fmt == hwio)
    this->__execute_hwio(tweights, weights, _ic4, _oc4);
  else
    this->__execute_oihw(tweights, weights, _ic4, _oc4);
}

template <typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
::quantization(TscaleType *__restrict tweights_quant_scale,
    TscaleType *__restrict tweights_quant_factor,
    int8_t *__restrict tweights_s8,
    TweightsType *__restrict tweights, int oc4) {
  __m<V> zero = _mm<V>::set1_ps(0.0);
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);

  // abs-max
  parallel_for<7>(mthr_,
      [&](int _oc4, int _ic4, int _oc3, int _hA, int _wA, int _O1, int _O) {
    MD11(TweightsType, atweights, tweights,
        oc4, xc->ic4, xc->oc3, xc->ic3, A, A, xc->O1, xc->I2, V, xc->O, V);
    MD8(TscaleType, atweights_quant_scale, tweights_quant_scale,
        oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);
    __m<V> mmabs_max = _mm<V>::set1_ps(0.0);

    iter_each (_ic3, xc->ic3) {
    iter_each (_I2, xc->I2) {
    iter_each (_iV, V) {
      mmabs_max =
          _mm<V>::max_ps(mmabs_max, _mm512_abs_ps(*(__m<V> *)&md11(atweights,
          _oc4, _ic4, _oc3, _ic3, _hA, _wA, _O1, _I2, _iV, _O, 0)));
    }}}
    _mm512_store_ps(&md8(atweights_quant_scale,
        _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, 0), mmabs_max);
  }, oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O);

  // quantization
  parallel_for<11>(mthr_, [&](int _oc4, int _ic4, int _oc3,
      int _ic3, int _hA, int _wA, int _O1, int _I2, int _V1, int _O, int _iVx) {
    MD12(int8_t, atweights_s8, tweights_s8, oc4, xc->ic4, xc->oc3, xc->ic3,
        A, A, xc->O1, xc->I2, xc->V1, xc->O, V, xc->Vx);
    MD8(TscaleType, atweights_quant_scale, tweights_quant_scale,
        oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);

    // I2 V => I2 V1 Vx
    MD12(TweightsType, _atweights, tweights, oc4, xc->ic4, xc->oc3, xc->ic3,
         A, A, xc->O1, xc->I2, xc->V1, xc->Vx, xc->O, V);
    __m<V> t0;
    // multi scal
    t0 = _mm<V>::mul_ps(*(__m<V> *)&md12(_atweights,
        _oc4, _ic4, _oc3, _ic3, _hA, _wA, _O1, _I2, _V1, _iVx, _O, 0), mmscale);
    t0 = _mm<V>::div_ps(t0, *(__m<V> *)&md8(atweights_quant_scale,
        _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, 0));
    // rounding
    t0 = _mm<V>::roundscale_ps(
        t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // int8_t
    TweightsType *rounded = (TweightsType *)&t0;
#pragma omp simd
    iter_each (_oV, V) {
      md12(atweights_s8,
          _oc4, _ic4, _oc3, _ic3, _hA, _wA, _O1, _I2, _V1, _O, _oV, _iVx) =
          (int8_t)rounded[_oV];
    }
  }, oc4, xc->ic4, xc->oc3, xc->ic3, A, A, xc->O1, xc->I2, xc->V1, xc->O, xc->Vx);

  // weights-acc
  parallel_for<8>(mthr_,
      [&](int _oc4, int _ic4, int _oc3, int _hA, int _wA, int _O1, int _O, int _oV) {
    MD12(int8_t, atweights_s8, tweights_s8, oc4, xc->ic4, xc->oc3, xc->ic3,
        A, A, xc->O1, xc->I2, xc->V1, xc->O, V, xc->Vx);
    MD8(TscaleType, atweights_quant_factor, tweights_quant_factor,
        oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);

    float acc = 0;
    iter_each (_ic3, xc->ic3) {
    iter_each (_I2, xc->I2) {
    iter_each (_V1, xc->V1) {
    iter_each (_iVx, xc->Vx) {
      acc += (float)md12(atweights_s8,
          _oc4, _ic4, _oc3, _ic3, _hA, _wA, _O1, _I2, _V1, _O, _oV, _iVx);
    }}}}
    md8(atweights_quant_factor, _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, _oV) = acc;
  }, oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);

  // weights-scale, combine with restore
  parallel_for<8>(mthr_, [&](int _oc4, int _ic4, int _oc3,
                             int _hA, int _wA, int _O1, int _O, int _oV) {
    MD8(TscaleType, atweights_quant_scale, tweights_quant_scale,
        oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);
    MD8(TscaleType, atweights_quant_factor, tweights_quant_factor,
        oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);

    float Sw =
        md8(atweights_quant_scale, _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, _oV);
    Sw /= INT8GEMM_TWT_QTSCALE;
    float Zw =
        md8(atweights_quant_factor, _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, _oV);
    if (xc->sampling_kind == CALIBRATED) {
      Sw = Sw * xc->tinput_quant_S;
      Zw = -Zw * Sw * xc->tinput_quant_z;
      md8(atweights_quant_factor, _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, _oV) =
          Zw;
    }
    md8(atweights_quant_scale, _oc4, _ic4, _oc3, _hA, _wA, _O1, _O, _oV) = Sw;
  }, oc4, xc->ic4, xc->oc3, A, A, xc->O1, xc->O, V);
}

template <typename WeightsType, int I, int A, int K, int V>
void elx_conv_wino_trans_weights_t<int8_t, WeightsType, I, A, K, V>
::execute(TscaleType *__restrict tweights_quant_scale,
    TscaleType *__restrict tweights_quant_factor,
    int8_t *__restrict tweights_s8,
    TweightsType *__restrict tweights,
    WeightsType *__restrict weights, int oc4) {
  {
    if (weights_is_bfmt_ || weights_as_bfmt_)
      this->__execute_blocked(tweights, weights, oc4);
    else
      this->__execute_oihw(tweights, weights, oc4);

    quantization(tweights_quant_scale, tweights_quant_factor,
                 tweights_s8, tweights, oc4);
  }
}
} // namespace euler
