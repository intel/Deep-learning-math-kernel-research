#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "euler.hpp"

namespace euler {

const unsigned XOPT_MSK = 0xA000;

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;
const unsigned FUS_T   = 0x40;
const unsigned FUS_A   = 0x80;

const unsigned DUP_MSK = 0xF;
const unsigned DUP_I   = 0x1;
const unsigned DUP_O   = 0x2;
const unsigned DUP_W   = 0x8;

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_t<Type, A, K, V, I>::elx_conv_wino_t(
    eld_conv_t<Type>& dc)
    : elx_conv_t<Type>(dc)
{
  // TODO: error when V!=16 && fmt=OIhw16i16o

  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  this->A = A;
  this->ht = (this->oh + A - K) / (A - K + 1);
  this->wt = (this->ow + A - K) / (A - K + 1);
  this->nt = this->ht * this->wt;
  this->t = this->nt * this->n;

  hOA_end_ = this->oh % (A - K + 1) - 1;
  if (hOA_end_ == -1) hOA_end_ = A - K;
  wOA_end_ = this->ow % (A - K + 1) - 1;
  if (wOA_end_ == -1) wOA_end_ = A - K;
  hA_end_ = (this->ih + this->tp) - (this->ht - 1) * (A - K + 1) - 1;
  wA_end_ = (this->iw + this->lp) - (this->wt - 1) * (A - K + 1) - 1;

  // TODO: santize user settings
  if (this->O == 0) this->O = 1; // TODO: O selection
  if (this->O1 == 0) this->O1 = 1; // TODO: O1 selection
  if (this->I2 == 0) this->I2 = 1; // TODO: I2 selection
  if (this->T == 0)  this->T = 1; // TODO: T selection
  this->O2 = this->O * this->O1;

  // Tailing
  this->Tr = this->t % this->T ? this->t % this->T : this->T;
  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  if (this->nthreads == 0 || this->nthreads > mthr_) {
    this->nthreads = mthr_;
  } else {
    mthr_ = this->nthreads;
  }
  inference_acc_ = this->prop_kind == forward_inference;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  // further divide packed oc/ic
  this->oc3 = this->oc2 / this->O2;
  this->ic3 = this->ic2 / this->I2;

  this->t2 = (this->t + this->T - 1) / this->T;

  // In case of Ir != V && blocked-format, assume bias also
  // padded to Vx.

  xopt_ = this->execution_mode;
  if (!(xopt_ & XOPT_MSK)) {
    // TODO: deduce xopt
    xopt_ = FUS_T | DUP_I;
  }

  prepare_execute_opt();

  bind_execute_functions();

  // dbg
  printf("############################################################\n");
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, OC=%d\n", this->V, this->Or, this->O2, this->O, this->O1, this->oc3, this->oc4, this->OC);

#ifdef DEBUG
  if (this->V * this->I2 * this->ic3 * this->ic4 != this->IC) {
      el_warn("V * I2 * ic3 * ic4 != this->IC\n Force ic4 = IC / (V * I2 * ic3)");
      this->ic4 = this->IC / (this->V * this->I2 * this->ic3);
  }

  if (this->V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_warn("V * O2 * oc3 * oc4 != this->OC\n Force oc4 = OC / (V * O2 * oc3)");
      this->oc4 = this->OC / (this->V * this->O2 * this->oc3);
  }
#else
  if ((xopt_ == 0xa073 || xopt_ == 0xa07b || this->with_ip_sum)
      && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (this->V * this->I2 * this->ic3 * this->ic4 != this->IC) {
      el_error("V * I2 * ic3 * ic4 != this->IC\n)");
  }

  if (this->V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_error("V * O2 * oc3 * oc4 != this->OC\n)");
  }
#endif
}
template <typename Type, const int A, const int K, const int V, const int I>
int  elx_conv_wino_t<Type, A, K, V, I>::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t toutputa_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING)
      : false;

  if (xopt_ & FUS_O) {
    this->oc3 /= this->oc4;
    if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_error("Config error!");
      return -1;
    }
  }
  if (xopt_ & FUS_I) {
    this->ic3 /= this->ic4;
    if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
      el_error("Config error!");
      return -1;
    }
  }

  input_is_bfmt_ = this->input_fmt == nchw ? false : true;
  weights_is_bfmt_ = this->weights_fmt == oihw ? false : true;
  output_is_bfmt_ = this->output_fmt == nchw ? false : true;
  input_as_bfmt_ = !input_is_bfmt_ && this->input_as_blocked;
  weights_as_bfmt_ = !weights_is_bfmt_ && this->weights_as_blocked;
  output_as_bfmt_ = !output_is_bfmt_ && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->ic4 > 1 && this->Ir != V) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V
      && (!output_as_bfmt_ || !weights_as_bfmt_)) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw;
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC * this->kh * this->kw;
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow;

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  toutputa_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case 0xa000:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->t;
    toutput_size = A * A * this->OC * this->t;
    break;
  case 0xa033:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->ic3 * this->I2 * V * this->t;
    toutput_size = A * A * (this->OC / this->oc4) * this->t;
    break;
  case 0xa061:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->T * mthr_;
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_;
    break;
  case 0xa071:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_;
    toutput_size = A * A * this->OC * this->t;
    break;
  case 0xa073:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_;
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_;
    break;
  case 0xa079:
    tweights_size = A * A * (this->IC / this->ic4) * (this->OC / this->oc4) * mthr_;
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_;
    toutput_size = A * A * this->OC * this->t;
    break;
  case 0xa07b:
    tweights_size = A * A * (this->IC / this->ic4) * (this->OC / this->oc4) * mthr_;
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_;
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_;
    break;
  case 0xa0e0:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->t;
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_;
    toutputa_size = A * (A - K + 1) * this->OC * this->t;
    break;
  case 0xa0e1:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * this->IC * this->T * mthr_;
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_;
    toutputa_size = A * (A - K + 1) * this->OC * this->t;
    break;
  default:
      el_error("Config error!");
      return -1;
    break;
  }

#define WEIGHTS_MAX_PRELOAD 4
  const size_t align = PAGE_SIZE / sizeof(Type);
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  toutputa_size_ = toutputa_size > 0 ? alignup(toutputa_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;

  workspace_ = nullptr, scratch_ = nullptr;
  size_t workspace_size = tweights_size_;
  size_t scratch_size = tinput_size_ + toutput_size_ + toutputa_size_
      + binput_size_ + bweights_size_ + boutput_size_;
  if (xopt_ == 0xa079 || xopt_ == 0xa07b) {
    scratch_size += tweights_size_;
    workspace_size = 0;
  }
  // TODO: user provided buffer
  if (scratch_size != 0)
    scratch_ = (Type *)galloc::acquire(scratch_size * sizeof(Type));
  if (workspace_size != 0)
    MEMALIGN64(&workspace_, workspace_size * sizeof(Type));

  set_trans_buffers();

  // dbg
  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  printf("gemmker_input_footprint = %ld\n", gemmker_input_footprint());
  printf("gemmker_weights_footprint = %ld\n", gemmker_weights_footprint());
  printf("gemmker_output_footprint = %ld\n", gemmker_output_footprint());
  printf("gemm_input_reuse_set = %ld\n", gemm_input_reuse_set());
  printf("gemm_output_reuse_set = %ld\n", gemm_output_reuse_set());

  auto plan = execute_plan(this->nthreads, 1, 1024 * 1024, 32 * 1024);
  plan.dump();

  return 0;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::set_trans_buffers()
{
  if (workspace_ != nullptr) {
    tweights_ = workspace_;
    tinput_ = (Type *)galloc::get();
  } else {
    tweights_ = (Type *)galloc::get();
    tinput_ = tweights_ + tweights_size_;
  }
  toutput_ = tinput_ + tinput_size_;
  toutputa_ = toutput_ + toutput_size_;
  binput_ = toutputa_ + toutputa_size_;
  bweights_ = binput_ + binput_size_;
  boutput_ = bweights_ + bweights_size_;
}

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_t<Type, A, K, V, I>::~elx_conv_wino_t()
{
  if (workspace_ != nullptr)
    ::free(workspace_);

  galloc::release();
}

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
  MD11(Type, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V, this->ic4, this->ic3, this->I2, V, K, K);
  MD11(Type, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3, A, A, this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](Type ain[K][K][V][V], Type *wei) {
    MD5(Type, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
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
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr auto scale = sizeof(Type);
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

#pragma omp for nowait collapse(6) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1 && _O == this->O - 1;

    alignas(64) Type ain[K][K][V][V];
    alignas(64) Type aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                               _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                              _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weights_blocked(
    Type *tweights, Type *weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD11(Type, aweights, weights, oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2, K, K, V, V);
  MD11(Type, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3, A, A, this->O1, this->I2, V, this->O, V);

#pragma omp for nowait collapse(6) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) Type aout[A][A][V][V];
    Type *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = Type[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                               _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                              _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V)
          md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
              = aout[_wA][_hA][_iV][_oV];
      }}}
    }
  }}}}}}}
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
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsf_plain(
    Type * __restrict tweights, Type * __restrict weights, int _ic4, int _oc4)
{
  MD11(Type, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V, this->ic4, this->ic3, this->I2, V, K, K);
  MD9(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](Type ain[K][K][V][V], Type *wei) {
    MD5(Type, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
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
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr auto scale = sizeof(Type);
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

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1 && _O == this->O - 1;

    alignas(64) Type ain[K][K][V][V];
    alignas(64) Type aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md9(atweights, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsf_blocked(
    Type *tweights, Type *weights, int _ic4, int _oc4)
{
  MD11(Type, aweights, weights, oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2, K, K, V, V);
  MD9(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O1, this->I2, V, this->O, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) Type aout[A][A][V][V];
    Type *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = Type[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V)
          md9(atweights, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
              = aout[_wA][_hA][_iV][_oV];
      }}}
    }
  }}}}};
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_weightsf(
    Type *tweights, Type *weights, int _ic4, int _oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weightsf_blocked(tweights, weights, _ic4, _oc4);
  else
    __trans_weightsf_plain(tweights, weights, _ic4, _oc4);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsa_blocked(
    Type *tweights, Type *weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD11(Type, aweights, weights, this->oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2, K, K, V, V);
  MD11(Type, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3, this->ic3, this->O1, this->I2, V, this->O, V);

#pragma omp for nowait collapse(6) schedule(static)
  iter_each (_oc4, this->oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) Type aout[A][A][V][V];
    Type *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = Type[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                               _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                              _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_weightsa_plain(
    Type * __restrict tweights, Type * __restrict weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD11(Type, aweights, weights, this->oc4, this->oc3, this->O1, this->O, V, this->ic4, this->ic3, this->I2, V, K, K);
  MD11(Type, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3, this->ic3, this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](Type ain[K][K][V][V], Type *wei) {
    MD5(Type, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        constexpr auto scale = sizeof(Type);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
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

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr auto scale = sizeof(Type);
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

#pragma omp for nowait collapse(6) schedule(static)
  iter_each (_oc4, this->oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {

    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1 && _O == this->O - 1;

    alignas(64) Type ain[K][K][V][V];
    alignas(64) Type aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_stream_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                               _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
                              _O1, _I2, _iV, _O, 0),
              *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
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


template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_input_plain(
    Type * __restrict tinput, Type * __restrict input, int _t2, int Tz)
{
  // n, IC, ih, iw => t2 | wA, hA, ic3, I2, T, V
  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];
  alignas(64) Type ain[A][A][V];
  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (_hA, A) {
    iter_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V)
          ain[_hA][_wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm<V>::store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
#pragma omp simd
          iter_each (_v, this->Ir)
            ain[_hA][_wA][_v] = md4(ainput, _n,
                (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            constexpr int scale = sizeof(Type);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + _hA, _iw + _wA),
                scale);
            _mm<V>::store_ps(ain[_hA][_wA], t);
          } else {
#pragma omp simd
            iter_each (_v, V)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }}
    }
  };

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
    iter_each (_T, Tz) {
      if (this->Ir != V) {
        readin_r(_ic3, _I2, _T, ain);
      } else
        readin_v(_ic3, _I2, _T, ain);

      ker_trans_input_(*this, aout, (Type *)ain, 0, 0, 0, -1);

      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
#pragma omp simd
        iter_each (_V, V) {
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

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt, this->ih, this->iw,
      this->tp, this->lp);
  iter_each (_T, Tz) {
    auto _ih = t2spati_o.anchor_t_;
    auto _iw = t2spati_o.anchor_l_;

    Type *in = &md7(ainput, t2spati_o.n_, 0, _ic3, _I2, _ih, _iw, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*this, aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

    ++ t2spati_o;

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
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
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    alignas(64) Type aout[A][A][V];

    iter_each (_T, Tz) {
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
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                           *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                          *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
#pragma omp simd
        iter_each (_V, V) {
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
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);

  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _t2, int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (_hA, A) {
    iter_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V)
          ain[_hA][_wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm<V>::store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
      iter_each (_hA, A) {
        iter_each (_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
              || _wA > _wA_end) {
#pragma omp simd
            iter_each (_V, V)
              ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            iter_each (_v, this->Ir)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }
    } else {
      iter_each (_hA, A) {
        iter_each (_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
              || _wA > _wA_end) {
#pragma omp simd
            iter_each (_V, V)
              ain[_hA][_wA][_V] = 0.0f;
          } else {
            if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
              constexpr int scale = sizeof(Type);
              __m<V> t = _mm<V>::i32gather_ps(vindex,
                  &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + _hA, _iw + _wA),
                  scale);
              _mm<V>::store_ps(ain[_hA][_wA], t);
            } else {
#pragma omp simd
              iter_each (_v, V)
                ain[_hA][_wA][_v] = md4(ainput, _n,
                    (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
            }
          }
        }
      }

    }
  };

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    alignas(64) Type aout[A][A][V];
    alignas(64) Type ain[A][A][V];

    iter_each (_T, Tz) {
      if (this->Ir != V)
        readin_r(_t2, _ic3, _I2, _T, ain);
      else
        readin_v(_t2, _ic3, _I2, _T, ain);
      ker_trans_input_(*this, aout, (Type *)ain, 0, 0, 0, -1);

      if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
        if (stream_in_) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
#pragma omp simd
        iter_each (_V, V) {
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

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
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
        iter_each (_hA, A) {
          _mm<V>::stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
        }
      } else {
        iter_each (_hA, A) {
          _mm<V>::store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
        }
      }
    } else {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
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
  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _ic3, int _I2, int _T, Type ain[A][A][V]) {
    MD7(Type, ainput, input, this->n, this->ic4, this->ic3, this->I2, V, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (__wA, A) {
    iter_each (__hA, A) {
      if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
          || __wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V)
          ain[__hA][__wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          constexpr int scale = sizeof(Type);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + __hA, _iw + __wA),
              scale);
          _mm<V>::store_ps(ain[__hA][__wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
      iter_each (__wA, A) {
        iter_each (__hA, A) {
          if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
              || __wA > _wA_end) {
#pragma omp simd
            iter_each (_V, V)
              ain[__hA][__wA][_V] = 0.0f;
          } else {
#pragma omp simd
            iter_each (_V, this->Ir)
              ain[__hA][__wA][_V] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
          }
        }
      }
    } else {
      iter_each (__wA, A) {
        iter_each (__hA, A) {
          if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
              || __wA > _wA_end) {
#pragma omp simd
            iter_each (_V, V)
              ain[__hA][__wA][_V] = 0.0f;
          } else {
            if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
              constexpr int scale = sizeof(Type);
              __m<V> t = _mm<V>::i32gather_ps(vindex,
                  &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + __hA, _iw + __wA),
                  scale);
              _mm<V>::store_ps(ain[__hA][__wA], t);
            } else {
#pragma omp simd
              iter_each (_V, V)
                ain[__hA][__wA][_V] = md4(ainput, _n,
                    (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
            }
          }
        }
      }
    }
  };

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    if (this->Ir != V)
      readin_r(_ic3, _I2, _T, ain);
    else
      readin_v(_ic3, _I2, _T, ain);
    ker_trans_inputa_(*this, aout, (Type *)ain, _wA, 0, A - 1, 0, -1);

    if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
      if (stream_in_) {
        iter_each (_hA, A)
          _mm<V>::stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
      } else {
        iter_each (_hA, A)
          _mm<V>::store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
      }
    } else {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
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

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

  bool scramble = (this->T == this->Tr) || (this->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _hA = n % A;
      int _wA = n / A;
      iter_each(_oc3, this->oc3) {
        bool last_ic4 = _ic4 == this->ic4 - 1;
        int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
        iter_each(_ic3, ic3) {
          int attr =
              _ic3 == 0 && _ic4 == 0 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                   &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                   &md5(atweights, _oc3, _ic3, _wA, _hA, 0), nullptr, attr);
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 && this->ic4 == 1
                          ? set_attr(attr_, r_output_idx)
                          : attr_;
          ker_gemm_tail(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                        &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                        &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                        nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_wA, A) {
      iter_each(_hA, A) {
        iter_each(_oc3, this->oc3) {
          bool last_ic4 = _ic4 == this->ic4 - 1;
          int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
          iter_each(_ic3, ic3) {
            int attr =
                _ic3 == 0 && _ic4 == 0 ? set_attr(attr_, r_output_idx) : attr_;
            ker_gemm(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                     &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                     &md5(atweights, _oc3, _ic3, _wA, _hA, 0), nullptr, attr);
          }
          if (last_ic4) {
            auto attr = this->ic3 == 1 && this->ic4 == 1
                            ? set_attr(attr_, r_output_idx)
                            : attr_;
            ker_gemm_tail(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                          &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                          nullptr, attr);
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm_non_acc(
    Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

  bool scramble = (this->T == this->Tr) || (this->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _hA = n % A;
      int _wA = n / A;
      iter_each(_oc3, this->oc3) {
        bool last_ic4 = _ic4 == this->ic4 - 1;
        int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
        iter_each(_ic3, ic3) {
          int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                   &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                   &md5(atweights, _oc3, _ic3, _wA, _hA, 0), nullptr, attr);
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm_tail(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                        &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                        &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                        nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_wA, A) {
      iter_each(_hA, A) {
        iter_each(_oc3, this->oc3) {
          bool last_ic4 = _ic4 == this->ic4 - 1;
          int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
          iter_each(_ic3, ic3) {
            int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
            ker_gemm(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                     &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                     &md5(atweights, _oc3, _ic3, _wA, _hA, 0), nullptr, attr);
          }
          if (last_ic4) {
            auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
            ker_gemm_tail(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                          &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                          nullptr, attr);
          }
        }
      }
    }
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm(
    Type * __restrict toutput, Type * __restrict tinput, Type * __restrict tweights, int _ic4)
{
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD5(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

#pragma omp for nowait collapse(4)
  iter_each (_wA, A) {
    iter_each (_hA, A) {
      iter_each (_oc3, this->oc3) {
        iter_each (_t2, this->t2) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
          auto ker_gemm_tail
              = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;
          MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
          MD6(Type, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
          bool last_ic4 = _ic4 == this->ic4 - 1;
          int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;

          iter_each (_ic3, ic3) {
            int attr = _ic3 == 0 && _ic4 == 0 ?
                set_attr(attr_, r_output_idx) : attr_;
            ker_gemm(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
                &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
                &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
		        nullptr, attr);
          }
          if (last_ic4) {
            int attr = this->ic3 == 1 && this->ic4 == 1 ?
                set_attr(attr_, r_output_idx) : attr_;
            ker_gemm_tail(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
                &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
		        nullptr, attr);
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm_non_acc(
    Type * __restrict toutput, Type * __restrict tinput, Type * __restrict tweights, int _ic4)
{
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD5(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

#pragma omp for nowait collapse(4)
  iter_each (_wA, A) {
    iter_each (_hA, A) {
      iter_each (_oc3, this->oc3) {
        iter_each (_t2, this->t2) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
          auto ker_gemm_tail
              = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;
          MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
          MD6(Type, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
          bool last_ic4 = _ic4 == this->ic4 - 1;
          int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;

          iter_each (_ic3, ic3) {
            int attr = _ic3 == 0 ?
                set_attr(attr_, r_output_idx) : attr_;
            ker_gemm(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
                &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
                &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
		        nullptr, attr);
          }
          if (last_ic4) {
            int attr = this->ic3 == 1 ?
                set_attr(attr_, r_output_idx) : attr_;
            ker_gemm_tail(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
                &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
		        nullptr, attr);
          }
        }
      }
    }
  }
}


// tweights:    oc4, A | A, oc3, ic3, O2, I2, V, V
// tinputs:      t2, A | A, ic3, I2, T, V
// toutput: t2, oc4, A | A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemma(
    Type * __restrict toutput, Type * __restrict tinput, Type *tweights, int _t2, int Tz)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD5(Type, atinput, tinput,  A, this->ic3, this->I2, Tz, V);
  MD5(Type, atoutput, toutput, A, this->oc3, this->O2, Tz, V);
  MD4(Type, atweights, tweights, A, this->oc3, this->ic3, this->O2 * this->I2 * V * V);

  iter_each (_hA, A) {
    iter_each (_oc3, this->oc3) {
      iter_each (_ic3, this->ic3 - 1) {
        int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm(*this, &md5(atoutput, _hA, _oc3, 0, 0, 0),
            &md5(atinput, _hA, _ic3, 0, 0, 0),
            &md4(atweights, _hA, _oc3, _ic3, 0),
	        nullptr, attr);
      }
      int attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
      ker_gemm_tail(*this, &md5(atoutput, _hA, _oc3, 0, 0, 0),
          &md5(atinput, _hA, this->ic3 - 1, 0, 0, 0),
          &md4(atweights, _hA, _oc3, this->ic3 - 1, 0),
	      nullptr, attr);
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_plain(
    Type * __restrict output, Type * __restrict toutput, Type * __restrict bias
    , int _t2, int Tz, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

  alignas(64) Type ain[A][A][V];
  alignas(64) Type aout[A - K + 1][A - K + 1][V];
  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _oc3, int _O2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr int scale = sizeof(Type);
          _mm<V>::i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr int scale = sizeof(Type);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    iter_each (_T, Tz) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      ker_trans_output_(
          *this, (Type *)aout, ain, (_ic4 == this->ic4 - 1 || _ic4 == -1) ?
          &md3(abias, _oc3, _O2, 0) : nullptr, 0, -1);

      if (this->Or != V)
        writeout_r(_oc3, _O2, _T, aout);
      else
        writeout_v(_oc3, _O2, _T, aout);
    }
  }}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_blocked(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz, int _ic4)
{
  auto ker_trans_output = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output_acc_ : ker_trans_output_;
  auto ker_trans_output_tail = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output0_acc_ : ker_trans_output0_;

  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, this->oh, this->ow, V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

  alignas(64) Type ain[A][A][V];

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n_off = res.quot;
  auto _t_off = res.rem;

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    output_tile_iter<A, K> t2spato_o(_n_off, _t_off, this->ht, this->wt, this->oh, this->ow);
  iter_each (_T, Tz) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
#pragma omp simd
    iter_each (_V, V) {
      ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
    }}}

    auto _n = t2spato_o.n_;
    auto _oh = t2spato_o.t_;
    auto _ow = t2spato_o.l_;
    Type *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

    /*
    int n_, o_h, o_w, hOA__end, wOA__end;
    t2spato(_t2, _T, n_, o_h, o_w, hOA__end, wOA__end);

    if (n_ != _n || o_h != _oh || o_w != _ow || hOA__end != t2spato_o.d_ || wOA__end != t2spato_o.r_)
      printf("========================here===================\n");

    if (t2spato_o.is_border() != (hOA__end < A - K || wOA__end < A - K))
      printf("==========================there=================\n");
      */

    if (t2spato_o.is_border())
      ker_trans_output_tail(*this, out, ain, (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, t2spato_o.d_, t2spato_o.r_);
    else
      ker_trans_output(*this, out, ain, (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);

    ++ t2spato_o;
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz, int _ic4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _t2, Tz, _ic4);
  else
    __trans_output_plain(output, toutput, bias, _t2, Tz, _ic4);
}

// toutput:  mthr | hA/A, oc3, O2, T, V
// toutputa: t2, oc4 | oc3, O2, T, wA/A | hA/A-K+1, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_outputa_th(
    Type *toutputa, Type *toutput, int Tz)
{
  MD4(Type, atoutput, toutput, A, this->oc3 * this->O2, Tz, V);
  MD4(Type, atoutputa, toutputa, this->oc3 * this->O2, Tz, A, (A - K + 1) * V);

  iter_each (_oc, this->oc3 * this->O2) {
    iter_each (_T, Tz) {
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
  iter_each (_t2, this->t2) {
  iter_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(Type, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz, A * (A - K + 1) * V);

    iter_each (_T, Tz) {
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

  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _t2, int _oc2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD5(Type, aoutput, output, this->n, this->oc2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each (_V, V)
            md5(aoutput, _n, _oc2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(Type);
          _mm<V>::i32scatter_ps(
              &md5(aoutput, _n, _oc2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
          if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(Type);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, _oc2 * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

#pragma omp for nowait collapse(2)
  iter_each (_t2, this->t2) {
  iter_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(Type, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz, A * (A - K + 1) * V);
    alignas(64) Type aout[A - K + 1][A - K + 1][V];

    iter_each (_T, Tz) {
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
    Type *output, Type *toutput, Type *bias, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, this->oh, this->ow, V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD3(Type, abias, bias, this->oc3, this->O2, V);

  auto ker_trans_output = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output_acc_ : ker_trans_output_;
  auto ker_trans_output_tail = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output0_acc_ : ker_trans_output0_;

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atoutput, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    alignas(64) Type ain[A][A][V];

    iter_each (_T, Tz) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      Type *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_output_tail(
            *this, out, ain, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, _hOA_end, _wOA_end);
      else
        ker_trans_output(
            *this, out, ain, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__trans_output_plain(
    Type * __restrict output, Type * __restrict toutput, Type *bias, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD3(Type, abias, bias, this->oc3, this->O2, V);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);

  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _t2, int _oc3, int _O2, int _T,
                      Type aout[A - K + 1][A - K + 1][V]) {
    MD7(Type, aoutput, output, this->n, this->oc4, this->oc3, this->O2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(Type);
          _mm<V>::i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
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
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<Type, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(Type);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(Type, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    alignas(64) Type ain[A][A][V];
    alignas(64) Type aout[A - K + 1][A - K + 1][V];

    iter_each (_T, Tz) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        ain[_wA][_hA][_V] = md6(atoutput6, _wA, _hA, _oc3, _O2, _T, _V);
      }}}

      ker_trans_output_(
          *this, (Type *)aout, ain, (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, 0, -1);

      if (this->Or != V)
        writeout_r(_t2, _oc3, _O2, _T, aout);
      else
        writeout_v(_t2, _oc3, _O2, _T, aout);
    }
  }}}
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _ic4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _ic4);
  else
    __trans_output_plain(output, toutput, bias, _ic4);
}

} // namespace euler
