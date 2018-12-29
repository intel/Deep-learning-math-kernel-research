#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elx_conv_direct.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::elx_conv_direct_t(eld_conv_t<UserTypes> &dc)
    : elx_conv_t<UserTypes>(dc)
{
  // user input
  xopt_ = this->execution_mode;

  this->Vx = 1;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = this->ic2;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  no_pad_ = this->lp == 0 && this->rp == 0 && this->tp == 0 && this->bp == 0;

  // t3, t2, (T, Tr)
  if (xopt_ == 0xa060 || xopt_ == 0xd060) {
    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = (this->ow + this->T - 1)/ this->T;
    this->Tr = this->ow % this->T ? this->ow % this->T : this->T;
    this->nt = this->oh * this->ow;
    this->t2 = this->nt / this->T;
    this->t = this->nt * this->n;

    if (this->T <= this->lp || this->Tr <= this->rp) {
      el_error("Unimplemented T: (T,Tr) must greater than (lp,rp)");
    }
    if (no_pad_ && (this->ht * this->hs != this->ih
        || this->wt * this->ws * this->T != this->iw)) {
      el_error("Unimplemented non-unitride shape or blocking");
    }
  }

  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  // oc4, (oc3, oc3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  if (this->O2r != this->O2 || this->oc3r != this->oc3) {
    el_error("No oc tailing support");
  }

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC) {
    el_error("IC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
  attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n",
      this->V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}

Template_elx_conv_direct_t
int Instance_elx_conv_direct_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING) : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING) : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING) : false;

  input_is_bfmt_ = this->input_fmt == nchw ? false : true;
  weights_is_bfmt_ = this->weights_fmt == oihw ? false : true;
  output_is_bfmt_ = this->output_fmt == nchw ? false : true;
  input_as_bfmt_ = !input_is_bfmt_ && this->input_as_blocked;
  weights_as_bfmt_ = !weights_is_bfmt_ && this->weights_as_blocked;
  output_as_bfmt_ = !output_is_bfmt_ && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->with_ip_sum && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (this->ic4 > 1 && this->Ir != V) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  if (xopt_ == 0xa060 || xopt_ == 0xd060) {
    if ((this->input_fmt == nchw || input_is_bfmt_) && weights_is_bfmt_ && output_is_bfmt_)
      ;
    else
      el_error("Unimplemented: ixa060|0xd060 support only nchw|blocked + blocked => blocked\n");
  } else if (!is_bfmt_ && (xopt_ != 0xa061 && xopt_ != 0xf061)) {
    el_error("Unimplemented: only a061, f061 mode support plain format\n");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw;
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC;
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow;

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  tinput_msk_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case 0xa060:
  case 0xd060:
    tweights_size = this->kh * this->kw * this->IC * this->OC;
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

  const size_t align = PAGE_SIZE / sizeof(TarrayType);
#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;

  scratch_ = nullptr;
  size_t workspace_size = tweights_size_;
  size_t scratch_size = tinput_size_ + tweights_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_;
  // TODO: user provided buffer
  if (scratch_size != 0)
    scratch_ = (TarrayType *)galloc::acquire(scratch_size * sizeof(TarrayType));

  set_trans_buffers();

  // dbg
  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  return 0;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::set_trans_buffers()
{
  tinput_ = (TarrayType *)galloc::get();
  tweights_ = tinput_ + tinput_size_;
  toutput_ = tweights_ + tweights_size_;
  binput_ = toutput_ + toutput_size_;
  bweights_ = binput_ + binput_size_;
  boutput_ = bweights_ + bweights_size_;
}

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::~elx_conv_direct_t()
{
  if (tinput_msk_ != nullptr) {
    free(tinput_msk_);
    tinput_msk_ = nullptr;
  }

  galloc::release();
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::trans_weights_blocked_to_compact(
    TarrayType *tweights, WeightsType *weights)
{
  MD11(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2, this->kh, this->kw, V, V);
  MD11(TarrayType, atweights, tweights, this->ic4, this->oc4, this->kh,
      this->kw, this->oc3, this->ic3, this->O1, this->I2, V, this->O, V);

  // weights: oc2, ic2, kh, kw, V, V
  // tweights: ic4, oc4, kh, kw, oc3, _ic3, O1, I2, V, O, V
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(6)
  iter_each (_oc4, this->oc4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_O1, this->O1) {
  iter_each (_O, this->O) {
  iter_each (_ic4, this->ic4) {
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_kh, this->kh) {
  iter_each (_kw, this->kw) {
  iter_each (_iV, V) {
#pragma omp simd
    iter_each (_oV, V) {
      md11(atweights, _ic4, _oc4, _kh, _kw, _oc3, _ic3, _O1, _I2, _iV, _O, _oV)
        = md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, _kh, _kw, _iV, _oV);
    }
  }}}}}}}}}}
}


Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::conv_a060_blocked_input(OutputType *output,
    InputType *input, WeightsType *weights, BiasType *bias, int _ic4, int _oc4,
    int _ht, int _wt)
{
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD2(InputType, ainput, input, this->ic3, this->I2 * this->ih * this->iw * V);
  MD2(OutputType, aoutput, output, this->oc3,
      this->O2 * this->ht * this->ow * V);
  MD4(WeightsType, aweights, weights, this->kh * this->kw, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);

  int khs = _ht == 0 ? 1 : 0;
  int khe = _ht == this->ht - 1 ? this->kh - 1 : this->kh;
  int kws = _wt == 0 ? 1 : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - 1 : this->kw;
  assert(this->T > this->lp);
  assert(this->Tr > this->rp);

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
    int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? set_attr(attr, relu_idx)
        : attr;

    ker_conv_(*this, &md2(aoutput, _oc3, 0),
        &md2(ainput, _ic3, 0), &md4(aweights, 0, _oc3, _ic3, 0),
        &md2(abias, _oc3, 0), _wt, khs, khe, kws, kwe, attr);
  }}
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::gemm_d060_blocked_input(OutputType *output, InputType *input,
    WeightsType *weights, BiasType *bias, int _ic4, int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD5(InputType, ainput, input, this->ic3, this->I2, this->ih, this->iw, V);
  MD5(OutputType, aoutput, output, this->oc3, this->O2, this->ht, this->ow, V);
  MD5(WeightsType, aweights, weights, this->kh, this->kw, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  const int AKH = this->kh / 2;
  const int AKW = this->kw / 2;

  int khs = _ht == 0 ? 1 : 0;
  int khe = _ht == this->ht - 1 ? this->kh - 1 : this->kh;
  int kws = _wt == 0 ? 1 : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - 1 : this->kw;
  assert(this->T > this->lp);
  assert(this->Tr > this->rp);

  auto ker_gemm_I = _wt == this->wt - 1 ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;
  auto ker_gemm_Ir = _wt == this->wt - 1 ? ker_gemm_IrO_Tr_ : ker_gemm_IrO_T_;
  auto ker_gemm_left_I = _wt == this->wt - 1 ? ker_gemm_left_I_O_Tr_ : ker_gemm_left_I_O_T_;
  auto ker_gemm_left_Ir = _wt == this->wt - 1 ? ker_gemm_left_IrO_Tr_ : ker_gemm_left_IrO_T_;
  auto ker_gemm_right_I = _wt == this->wt - 1 ? ker_gemm_right_I_O_Tr_ : ker_gemm_right_I_O_T_;
  auto ker_gemm_right_Ir = _wt == this->wt - 1 ? ker_gemm_right_IrO_Tr_ : ker_gemm_right_IrO_T_;

  iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr =
          (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
                 ? set_attr(attr, relu_idx)
                 : attr;
      int attr_bk = attr;

      auto ker_gemm = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ? ker_gemm_Ir : ker_gemm_I;
      auto ker_gemm_left = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ? ker_gemm_left_Ir : ker_gemm_left_I;
      auto ker_gemm_right = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 ? ker_gemm_right_Ir : ker_gemm_right_I;

      for (int _kh = khs; _kh < khe; ++_kh) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          ker_gemm(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
              &md5(ainput, _ic3, 0, _kh - AKH, _kw - AKW, 0),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
          attr &= ~r_output_idx;
        }
        // left
        if (_wt == 0) {
          int _kw = 0;
          ker_gemm_left(*this, &md5(aoutput, _oc3, 0, 0, 1, 0),
              &md5(ainput, _ic3, 0, _kh - AKH, 1 + _kw - AKW, 0),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
          attr &= ~r_output_idx;
        }
        // right
        if (_wt == this->wt - 1) {
          int _kw = 2;
          ker_gemm_right(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
              &md5(ainput, _ic3, 0, _kh - AKH, _kw - AKW, 0),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
        }
      }
      attr = attr_bk;
    }
  }
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::gemm_d060_nchw_input(OutputType *output, InputType *input,
    WeightsType *weights, BiasType *bias, int _ic4, int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  oc3*, O2, ht*, wt*, T, V
  MD5(InputType, ainput, input, this->ic3, this->I2, V, this->ih, this->iw);
  MD5(OutputType, aoutput, output, this->oc3, this->O2, this->ht, this->ow, V);
  MD5(WeightsType, aweights, weights, this->kh, this->kw, this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  const int AKH = this->kh / 2;
  const int AKW = this->kw / 2;

  int khs = _ht == 0 ? 1 : 0;
  int khe = _ht == this->ht - 1 ? this->kh - 1 : this->kh;
  int kws = _wt == 0 ? 1 : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - 1 : this->kw;
  assert(this->T > this->lp);
  assert(this->Tr > this->rp);

  auto ker_gemm_I = _wt == this->wt - 1 ? ker_gemm_I_O_Tr_ : ker_gemm_I_O_T_;
  auto ker_gemm_Ir = _wt == this->wt - 1 ? ker_gemm_IrO_Tr_ : ker_gemm_IrO_T_;
  auto ker_gemm_left_I = _wt == this->wt - 1 ? ker_gemm_left_I_O_Tr_ : ker_gemm_left_I_O_T_;
  auto ker_gemm_left_Ir = _wt == this->wt - 1 ? ker_gemm_left_IrO_Tr_ : ker_gemm_left_IrO_T_;
  auto ker_gemm_right_I = _wt == this->wt - 1 ? ker_gemm_right_I_O_Tr_ : ker_gemm_right_I_O_T_;
  auto ker_gemm_right_Ir = _wt == this->wt - 1 ? ker_gemm_right_IrO_Tr_ : ker_gemm_right_IrO_T_;

  iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr =
          (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
                 ? set_attr(attr, relu_idx)
                 : attr;
      int attr_bk = attr;

      auto ker_gemm = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? ker_gemm_Ir : ker_gemm_I;
      auto ker_gemm_left = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? ker_gemm_left_Ir : ker_gemm_left_I;
      auto ker_gemm_right = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? ker_gemm_right_Ir : ker_gemm_right_I;

      for (int _kh = khs; _kh < khe; ++_kh) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          ker_gemm(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
              &md5(ainput, _ic3, 0, 0, _kh - AKH, _kw - AKW),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
          attr &= ~r_output_idx;
        }
        // left
        if (_wt == 0) {
          int _kw = 0;
          ker_gemm_left(*this, &md5(aoutput, _oc3, 0, 0, 1, 0),
              &md5(ainput, _ic3, 0, 0, _kh - AKH, 1 + _kw - AKW),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
          attr &= ~r_output_idx;
        }
        // right
        if (_wt == this->wt - 1) {
          int _kw = 2;
          ker_gemm_right(*this, &md5(aoutput, _oc3, 0, 0, 0, 0),
              &md5(ainput, _ic3, 0, 0, _kh - AKH, _kw - AKW),
              &md5(aweights, _kh, _kw, _oc3, _ic3, 0), &md3(abias, _oc3, 0, 0),
              attr, 0, nullptr, nullptr, nullptr);
        }
      }
      attr = attr_bk;
    }
  }
}

} // namespace euler
