#include "elx_conv_direct_1x1_lp.hpp"
#include "el_parallel.hpp"

namespace euler {

static constexpr float INT8GEMM_TWT_QTSCALE = 127.0;

Template_elx_conv_direct_1x1_lp_t
Instance_elx_conv_direct_1x1_lp_t::elx_conv_direct_1x1_lp_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = this->execution_mode;
  attr_ = 0x0;

  this->Vx = 4;
  this->V1 = V / this->Vx;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = this->ic2;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  no_pad_ = this->lp == 0 && this->rp == 0 && this->tp == 0 && this->bp == 0;
  if (!no_pad_)
    el_error("no support for padding in 1x1 u8s8 conv");

  // t3, t2, (T, Tr)
  bool shape_ok = this->hs < 3 && this->ws < 3 && no_pad_;
  if (!shape_ok)
    el_error("direct_1x1_lp: Shape not supported");

  if (xopt_ == 0xc160) {
    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = this->ow;
    this->nt = this->ht * this->wt;
    this->t = this->nt * this->n;
    this->t2 = (this->nt + this->T - 1) / this->T;
    this->Tr = this->nt % this->T ? this->nt % this->T : this->T;
  } else if (xopt_ == 0xb161){
    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = this->ow / this->T;
    this->nt = this->oh * this->ow;
    this->t2 = this->nt / this->T;
    this->Tr = this->T;
    this->t = this->nt * this->n;
    if (this->ow % T != 0) {
      el_error("direct_1x1_lp: b161: No Tr support");
    }
  } else {
    el_error("direct_1x1_lp: xopt not supported");
  }

  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  if (this->Ir != V)
    el_error("ic / 16 != 0 is not implement while doing int8 gemm");

  // oc4, (oc3, oc3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  if (this->O2r != this->O2 || this->oc3r != this->oc3) {
    el_error("No oc tailing for 0xa061, 0xb061, 0xe060, 0xf061");
  }

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC)
    el_error("IC blocking error");

  if (this->Ir != V) {
    el_error("direct_1x1: int8: Ir not support");
  }
  if ((this->output_fmt != nChw16c || this->weights_fmt != OIhw16i16o) &&
      this->Or != V) {
    el_error("direct_1x1: int8: Or not support");
  }

  attr_ = set_attr(attr_, fma_opt_idx);
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
  attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;
  toutput_opt_ = false;
  prepare_quant_calibration(dc);

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Vx=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      V, this->Vx, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}

Template_elx_conv_direct_1x1_lp_t
int Instance_elx_conv_direct_1x1_lp_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t tweights_s8_size = 0, input_scale_size = 0, weights_scale_size = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING) : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING) : false;

  input_is_bfmt_ = this->input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = this->weights_fmt == OIhw16i16o;
  output_is_bfmt_ = this->output_fmt == nChw16c;
  input_as_bfmt_ = this->input_fmt == nchw && this->input_as_blocked;
  weights_as_bfmt_ = this->input_fmt == oihw && this->weights_as_blocked;
  output_as_bfmt_ = this->output_fmt == nchw && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->with_ip_sum && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;
  if (this->n * this->oc4 >= mthr_)
    toutput_opt_ = true;

  switch (xopt_) {
  case 0xc160:
  case 0xb161:
    input_scale_size = this->T * 2 * sizeof(TscaleType);
    tweights_s8_size = this->IC * this->OC * sizeof(int8_t);
    weights_scale_size = this->OC * 2 * sizeof(TscaleType);
    toutput_size = (this->OC / this->oc4) * this->oh * this->ow *
                   sizeof(ToutputType);
    toutput_size *= toutput_opt_ ? mthr_ : this->n * this->oc4;
    break;
  default:
      el_error("Unknown xopt!");
      return -1;
    break;
  }

  const size_t align = PAGE_SIZE;
#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;
  tweights_s8_size_ = tweights_s8_size > 0 ? alignup(tweights_s8_size, align) : 0;
  input_scale_size_ = input_scale_size > 0 ? alignup(input_scale_size, align) : 0;
  weights_scale_size_ = weights_scale_size > 0 ? alignup(weights_scale_size, align) : 0;

  scratch_ = nullptr;
  workspace_ = nullptr;
  size_t workspace_size = tweights_size_ + tweights_s8_size_
      + weights_scale_size_ + input_scale_size_;
  size_t scratch_size = tinput_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_;
  // TODO: user provided buffer
  if (scratch_size != 0)
    scratch_ = galloc::acquire(scratch_size);
  if (workspace_size != 0)
    MEMALIGN64(&workspace_, workspace_size);

  // dbg
  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  printf("sampling_kind = %d\n", this->sampling_kind);
  printf("input_quant_S = %f\n", this->input_quant_S);
  printf("input_quant_z = %f\n", this->input_quant_z);
  printf("output_quant_S = %f\n", this->output_quant_S);
  printf("output_quant_z = %f\n", this->output_quant_z);
  printf("sum_quant_S = %f\n", this->sum_quant_S);
  printf("sum_quant_z = %f\n", this->sum_quant_z);
  return 0;
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::set_trans_buffers()
{
  if (workspace_ != nullptr) {
    tweights_ = (TweightsType *)workspace_;
    input_scale_ = (TscaleType *)((char *)tweights_ + tweights_size_);
    weights_scale_ = (TscaleType *)((char *)input_scale_ + input_scale_size_);
    tweights_s8_ = (int8_t *)((char *)weights_scale_ + weights_scale_size_);
  }
  tinput_ = (TinputType *)galloc::get();
  toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
  binput_ = (InputType *)((char *)toutput_ + toutput_size_);
  bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
  boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::prepare_quant_calibration(eld_conv_t &dc)
{
  this->input_quant_S = dc.input_quant.scale;
  this->input_quant_repS = 1 / dc.input_quant.scale;
  this->input_quant_z = dc.input_quant.z;
  this->output_quant_S = dc.output_quant.scale;
  this->output_quant_repS = 1 / dc.output_quant.scale;
  this->output_quant_z = dc.output_quant.z;
  this->sum_quant_S = dc.sum_quant.scale;
  this->sum_quant_z = dc.sum_quant.z;

  if (this->sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_conv_direct_1x1_lp_t
Instance_elx_conv_direct_1x1_lp_t::~elx_conv_direct_1x1_lp_t()
{
  if (workspace_ != nullptr) {
    ::free(workspace_);
    workspace_ = nullptr;
  }

  galloc::release();
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::__trans_weights_s8_blocked_oc(
    TscaleType *weights_scale, int8_t *tweights_s8, WeightsType *weights,
    BiasType *bias)
{
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);

  // abs max
  parallel_for<3>(mthr_, [&](int _oc4, int _oc3, int _O2) {
    MD5(TscaleType, aweights_scale, weights_scale,
        this->oc4, this->oc3, 2, this->O2, V);
    __m<V> mmabs_max = _mm<V>::set1_ps(0.0);
    iter_each (_ic4, this->ic4) {
    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_iV1, this->V1) {
    iter_each (_iVx, this->Vx) {
      MD9(WeightsType, aweights, weights, this->oc4, this->oc3, this->O2,
          this->ic4, this->ic3, this->I2, this->V1, this->Vx, V);
      mmabs_max = _mm<V>::max_ps(mmabs_max, _mm512_abs_ps(*(__m<V> *)
          &md9(aweights, _oc4, _oc3, _O2, _ic4, _ic3, _I2, _iV1, _iVx, 0)));
    }}}}}
    _mm<V>::store_ps(
        &md5(aweights_scale, _oc4, _oc3, 0, _O2, 0), mmabs_max);
  }, this->oc4, this->oc3, this->O2);

  // oc4, (oc3, oc3r), (O2, O2r), ic4, ic3, I2, V1, Vx, V ->
  // oc4, ic4, (oc3, oc3r), ic3, I2, V1, (O2, O2r), V, Vx
  // quantization
  parallel_for<9>(mthr_, [&](int _oc4, int _ic4, int _oc3,
    int _ic3, int _O1, int _I2, int _iV1, int _O, int _iVx) {
    MD10(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O,
        this->ic4, this->ic3, this->I2, this->V1, this->Vx, V);
    MD10(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4,
        this->oc3, this->ic3, this->O1, this->I2, this->V1, this->O, V, this->Vx);
    MD6(TscaleType, aweights_scale, weights_scale,
        this->oc4, this->oc3, 2, this->O1, this->O, V);

    auto mmresf32 = _mm<V>::mul_ps(
        *(__m<V> *)&md10(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, _iV1, _iVx, 0),
        mmscale);
    mmresf32 = _mm<V>::div_ps(mmresf32,
        *(__m<V> *)&md6(aweights_scale, _oc4, _oc3, 0, _O1, _O, 0));
    mmresf32 = _mm<V>::roundscale_ps(
        mmresf32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    TscaleType *resf32 = (TscaleType *)&mmresf32;
#pragma omp simd
    iter_each (_oV, V) {
      md10(atweights_s8, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _iV1, _O, _oV, _iVx) =
          (int8_t)resf32[_oV];
    }
  }, this->oc4, this->ic4, this->oc3, this->ic3, this->O1, this->I2, this->V1, this->O, this->Vx);

  // accumulation
  parallel_for<5>(mthr_, [&](int _oc4, int _oc3, int _O1, int _O, int _oV) {
    MD10(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4,
        this->oc3, this->ic3, this->O1, this->I2, this->V1, this->O, V, this->Vx);
    MD6(TscaleType, aweights_scale, weights_scale,
        this->oc4, this->oc3, 2, this->O1, this->O, V);
    TscaleType acc = 0;
    iter_each (_ic4, this->ic4) {
    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_iV1, this->V1) {
    iter_each (_iVx, this->Vx) {
      acc += (TscaleType)md10(atweights_s8,
          _oc4, _ic4, _oc3, _ic3, _O1, _I2, _iV1, _O, _oV, _iVx);
    }}}}}
    md6(aweights_scale, _oc4, _oc3, 1, _O1, _O, _oV) = acc;
  }, this->oc4, this->oc3, this->O1, this->O, V);

  // scale
  parallel_for<3>(mthr_, [&](int _oc4, int _oc3, int _O2) {
    MD5(TscaleType, aweights_scale, weights_scale,
        this->oc4, this->oc3, 2, this->O2, V);
    __m<V> &mmqs = *(__m<V> *)&md5(
        aweights_scale, _oc4, _oc3, 0, _O2, 0);
    mmqs = mmqs / mmscale;
  }, this->oc4, this->oc3, this->O2);

  // combine
  __m<V> mmorepS = _mm<V>::set1_ps(this->output_quant_repS);
  __m<V> mmoz = _mm<V>::set1_ps(this->output_quant_z);
  __m<V> mmiS = _mm<V>::set1_ps(this->input_quant_S);
  __m<V> mmiz = _mm<V>::set1_ps(this->input_quant_z);
  parallel_for<3>(mthr_, [&](int _oc4, int _oc3, int _O2) {
    MD5(TscaleType, aweights_scale, weights_scale,
        this->oc4, this->oc3, 2, this->O2, V);
    MD4(BiasType, abias, bias, this->oc4, this->oc3, this->O2, V);
    __m<V> &mmqs = *(__m<V> *)&md5(
        aweights_scale, _oc4, _oc3, 0, _O2, 0);
    __m<V> &mmqf = *(__m<V> *)&md5(
        aweights_scale, _oc4, _oc3, 1, _O2, 0);
    __m<V> &mmbias = *(__m<V> *)&md4(abias, _oc4, _oc3, _O2, 0);

    if (std::is_same<OutputType, float>::value) {
      mmqs = mmiS * mmqs;
      mmqf = mmbias - mmiz * mmqf * mmqs;
    } else {
      mmqs = mmiS * mmqs * mmorepS;
      mmqf = mmbias * mmorepS + mmoz - mmiz * mmqf * mmqs;
    }

    if (this->with_ip_sum) {
      __m<V> sum_S = _mm<V>::set1_ps(this->sum_quant_S);
      __m<V> sum_z = _mm<V>::set1_ps(this->sum_quant_z);
      mmqf -= sum_z * sum_S;
    }
  }, this->oc4, this->oc3, this->O2);
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::trans_weights_s8_oc(
    TscaleType *weights_scale, int8_t *tweights, WeightsType *weights,
    BiasType *bias)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_s8_blocked_oc(weights_scale, tweights, weights, bias);
  else
    el_error("Unimplement format");
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::requant_output(
    OutputType *output, ToutputType *toutput)
{
  __m<V> mmorepS = _mm<V>::set1_ps(this->output_quant_repS);
  __m<V> mmoz = _mm<V>::set1_ps(this->output_quant_z);

  parallel_for<4>(mthr_, [&](int _t3, int _o, int _oh, int _ow) {
    MD5(ToutputType, atoutput, toutput,
        this->t3, this->OC / V, this->oh, this->ow, V);
    MD5(OutputType, aoutput, output,
        this->t3, this->OC / V, this->oh, this->ow, V);
    __m<V> mmres = *(__m<V> *)&md5(atoutput, _t3, _o, _oh, _ow, 0);
    __m<V> mmresf32 = mmres * mmorepS + mmoz;
    __i<V> mmress32 = _mm<V>::cvt_roundps_epi32(
        mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
    __m128i mmresx8;
    if (std::is_same<OutputType, int8_t>::value)
      mmresx8 = _mm<V>::cvtsepi32_epi8(mmress32);
    else if (std::is_same<OutputType, uint8_t>::value)
      mmresx8 = _mm<V>::cvtusepi32_epi8(mmress32);
    else {
      mmresx8 = _mm_setzero_si128();
      el_error("Unsupported output type for int8 direct 1x1");
    }
    _mm_store_si128((__m128i *)&md5(aoutput, _t3, _o, _oh, _ow, 0), mmresx8);
  }, this->t3, this->OC / V, this->oh, this->ow);
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::gemm_b161(ToutputType *toutput,
    OutputType *output, uint8_t *input_u8, int8_t *weights_s8, TscaleType *input_scale,
    TscaleType *weights_scale, BiasType *bias, int _ic4)
{
  MD2(uint8_t, ainput_u8, input_u8, this->ic3, this->I2 * this->ih * this->iw * V);
  MD5(ToutputType, atoutput, toutput,
      this->oc3, this->O2, this->ht, this->wt,  this->T * V);
  MD5(OutputType, aoutput, output,
      this->oc3, this->O2, this->ht, this->wt,  this->T * V);
  MD3(int8_t, aweights_s8, weights_s8,
      this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);
  MD2(TscaleType, ainput_scale, input_scale, 2, this->T);
  MD4(TscaleType, aweights_scale, weights_scale,
      this->oc3, 2, this->O2, V);

  auto ker_gemm = ker_u8s8_gemm_I_O_T_;

  iter_each (_ic3, this->ic3) {
    int attr = _ic4 == 0 && _ic3 == 0
        ? set_attr(attr_, r_output_idx)
        : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? set_attr(attr, relu_idx)
        : attr;
    attr = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? set_attr(attr, c_output_idx)
        : attr;

    iter_each (_oc3, this->oc3) {
      ker_gemm(
          *this,
          &md5(atoutput, _oc3, 0, 0, 0, 0),
          &md5(aoutput, _oc3, 0, 0, 0, 0),
          &md2(ainput_u8, _ic3, 0),
          &md3(aweights_s8, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0),
          attr,
          &md2(ainput_scale, 0, 0),
          &md2(ainput_scale, 1, 0),
          &md4(aweights_scale, _oc3, 0, 0, 0),
          &md4(aweights_scale, _oc3, 1, 0, 0));
    }
  }
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::gemm_c160(ToutputType *toutput,
    OutputType *output, uint8_t *input_u8, int8_t *weights_s8, TscaleType *input_scale,
    TscaleType *weights_scale, BiasType *bias, int _ic4, int _oc4, int _t2)
{
  // weights: oc3, O2, ic4*, ic3, I2, V1, V, Vx
  // input:   ic3, I2, t2*, T(Tr), V1, Vx
  // output:  oc3, O2, t2*, T(Tr), V
  MD2(uint8_t, ainput_u8, input_u8,
      this->ic3, this->I2 * this->ih * this->iw * V);
  MD2(ToutputType, atoutput, toutput,
      this->oc3, this->O2 * this->oh * this->ow * V);
  MD2(OutputType, aoutput, output,
      this->oc3, this->O2 * this->oh * this->ow * V);
  MD3(int8_t, aweights_s8, weights_s8,
      this->oc3, this->ic3, this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);
  MD2(TscaleType, ainput_scale, input_scale, 2, this->T);
  MD4(TscaleType, aweights_scale, weights_scale,
      this->oc3, 2, this->O2, V);

  auto ker_gemm = (_t2 == this->t2 - 1)
      ? ker_u8s8_gemm_I_O_Tr_
      : ker_u8s8_gemm_I_O_T_;

  iter_each (_ic3, this->ic3) {
    MD2(uint8_t, ainput2_u8, &md2(ainput_u8, _ic3, 0), this->t2, this->T * V);
    int attr = _ic4 == 0 && _ic3 == 0
        ? set_attr(attr_, r_output_idx)
        : attr_;
    attr = this->with_relu && _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? set_attr(attr, relu_idx)
        : attr;
    attr = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
        ? set_attr(attr, c_output_idx)
        : attr;

    iter_each (_oc3, this->oc3) {
      MD2(OutputType, aoutput2, &md2(aoutput, _oc3, 0), this->t2, this->T * V);
      MD2(ToutputType, atoutput2, &md2(atoutput, _oc3, 0), this->t2, this->T * V);
      ker_gemm(
          *this,
          &md2(atoutput2, _t2, 0),
          &md2(aoutput2, _t2, 0),
          &md2(ainput2_u8, _t2, 0),
          &md3(aweights_s8, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0),
          attr,
          &md2(ainput_scale, 0, 0),
          &md2(ainput_scale, 1, 0),
          &md4(aweights_scale, _oc3, 0, 0, 0),
          &md4(aweights_scale, _oc3, 1, 0, 0));
    }
  }
}

} // namespace euler
