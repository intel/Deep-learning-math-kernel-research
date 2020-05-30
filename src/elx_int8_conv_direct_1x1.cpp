#include "elx_int8_conv_direct_1x1.hpp"
#include "elx_int8_conv_direct_1x1_bind.hpp"
#include "elx_int8_conv_direct_1x1_xopt.hpp"
#include "el_parallel.hpp"

namespace euler {

Template_elx_int8_conv_direct_1x1_t
Instance_elx_int8_conv_direct_1x1_t::elx_int8_conv_direct_1x1_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  //xopt_ = ep.execution_mode;
  xopt_ = 0xa160;
  attr_ = 0x0;

  ep.Vx = 4;
  ep.V1 = V / ep.Vx;
  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  if (ep.I2 == 0) ep.I2 = ep.ic2;
  if (ep.T == 0)  ep.T = 1;
  if (ep.O == 0)  ep.O = 1;
  if (ep.O1 == 0) ep.O1 = 1;
  ep.O2 = ep.O * ep.O1;

  ep.O4 = ep.O4 == 0 ? 1 : ep.O4;
  ep.I4 = ep.I4 == 0 ? 1 : ep.I4;

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;

  no_pad_ = ep.lp == 0 && ep.rp == 0 && ep.tp == 0 && ep.bp == 0;
  if (!no_pad_)
    el_error("no support for padding in 1x1 u8s8 conv");

  // n, t2, (T, Tr)
  bool shape_ok = ep.hs < 3 && ep.ws < 3 && no_pad_;
  if (!shape_ok)
    el_error("direct_1x1: int8: Shape not supported");

  if (ep.ws == 1) {
    ep.ht = ep.oh;
    ep.wt = ep.ow;
    ep.nt = ep.ht * ep.wt;
    ep.t = ep.nt * ep.n;
    ep.t2 = (ep.nt + ep.T - 1) / ep.T;
    ep.Tr = ep.nt % ep.T ? ep.nt % ep.T : ep.T;
  } else if (ep.ws == 2) {
    ep.ht = ep.oh;
    ep.wt = ep.ow / ep.T;
    ep.nt = ep.oh * ep.ow;
    ep.t2 = ep.nt / ep.T;
    ep.Tr = ep.T;
    ep.t = ep.nt * ep.n;
    if (ep.ow % ep.T != 0) {
      el_error("direct_1x1: int8: No Tr support for ws == 2");
    }
  } else {
    el_error("direct_1x1: int8: ws > 2 not supported");
  }

  int V1r = ALIGNUP(ep.ic % V, 4) / ep.Vx;
  ep.Ir = V1r % ep.V1 ? V1r % ep.V1 : ep.V1;
  ep.Or = ep.oc % V ? ep.oc % V : V;

  // O4, (O3, O3r), (O2, O2r)
  ep.oc34 = (ep.oc2 + ep.O2 - 1) / ep.O2;
  ep.O2r = ep.oc2 % ep.O2;
  if (ep.O2r == 0) ep.O2r = ep.O2;
  ep.O3 = ep.O4; // FIXME, swap order
  ep.O4 = (ep.oc34 + ep.O3 - 1) / ep.O3;
  ep.O3r = ep.oc34 % ep.O3;
  if (ep.O3r == 0) ep.O3r = ep.O3;

  if (ep.O2r != ep.O2 || ep.O3r != ep.O3) {
    el_error("direct_1x1: int8: No oc tailing");
  }

  // I4, I3, I3
  ep.ic34 = ep.ic2 / ep.I2;
  ep.I3 = ep.ic34 / ep.I4;
  if (ep.I4 * ep.I3 * ep.I2 * V != ep.IC)
    el_error("IC blocking error");

  if ((ep.output_fmt != nChw16c || ep.weights_fmt != OIhw16i16o) &&
      ep.Or != V) {
    el_error("direct_1x1: int8: Or not support");
  }

  attr_ = set_bit(attr_, AT_FMAOPT_MASK);
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = estl::max_concurrency();
  inference_acc_ = ep.prop_kind == forward_inference;

  attr_ = ep.with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  attr_ = ep.with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;
  toutput_opt_ = false;
  prepare_quant_calibration(dc);

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Vx=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d",
         V, ep.Vx, ep.Ir, ep.I2, ep.I3, ep.I4, ep.IC);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC);
}

Template_elx_int8_conv_direct_1x1_t
int Instance_elx_int8_conv_direct_1x1_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t tweights_s8_size = 0, input_scale_size = 0, weights_scale_size = 0;

  stream_in_ = ep.streaming_input
      ? (ep.streaming_input == STORE_STREAMING) : false;
  stream_out_ = ep.streaming_output
      ? (ep.streaming_output == STORE_STREAMING) : false;

  input_is_bfmt_ = ep.input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = ep.weights_fmt == OIhw16i16o;
  output_is_bfmt_ = ep.output_fmt == nChw16c;
  input_as_bfmt_ = ep.input_fmt == nchw && ep.input_as_blocked;
  weights_as_bfmt_ = ep.input_fmt == oihw && ep.weights_as_blocked;
  output_as_bfmt_ = ep.output_fmt == nchw && ep.output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (input_as_bfmt_)
    binput_size = ep.n * ep.IC * ep.ih * ep.iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = ep.OC * ep.IC * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = ep.n * ep.OC * ep.oh * ep.ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;
  if (ep.n * ep.O4 >= mthr_ && ep.output_fmt == nChw16c)
    toutput_opt_ = true;

  switch (xopt_) {
  case 0xa160:
    input_scale_size = ep.T * 2 * sizeof(float);
    tweights_s8_size = ep.IC * ep.OC * sizeof(int8_t);
    weights_scale_size = ep.OC * 2 * sizeof(float);
    toutput_size = (ep.OC / ep.O4) * ep.oh * ep.ow *
                   sizeof(ToutputType);
    toutput_size *= toutput_opt_ ? mthr_ : ep.n * ep.O4;
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

  workspace_size_ = tweights_size_ + tweights_s8_size_
      + weights_scale_size_ + input_scale_size_;
  scratch_size_ = tinput_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_;

  return 0;
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::set_scratch_buffers(void *base)
{
  if (base != nullptr) {
    tinput_ = (TinputType *)base;
    toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
    binput_ = (InputType *)((char *)toutput_ + toutput_size_);
    bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
    boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
  }
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    tweights_ = (TweightsType *)base;
    input_scale_ = (float *)((char *)tweights_ + tweights_size_);
    weights_scale_ = (float *)((char *)input_scale_ + input_scale_size_);
    tweights_s8_ = (int8_t *)((char *)weights_scale_ + weights_scale_size_);
  }
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::prepare_quant_calibration(eld_conv_t &dc)
{
  ep.input_quant_S = dc.input_quant.scale;
  ep.input_quant_repS = 1 / dc.input_quant.scale;
  ep.input_quant_z = dc.input_quant.z;
  ep.output_quant_S = dc.output_quant.scale;
  ep.output_quant_repS = 1 / dc.output_quant.scale;
  ep.output_quant_z = dc.output_quant.z;
  ep.sum_quant_S = dc.sum_quant.scale;
  ep.sum_quant_z = dc.sum_quant.z;

  if (ep.sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_int8_conv_direct_1x1_t
Instance_elx_int8_conv_direct_1x1_t::~elx_int8_conv_direct_1x1_t()
{
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::trans_weights_s8_blocked_oc(
    float *weights_scale, int8_t *tweights_s8, WeightsType *weights,
    BiasType *bias)
{
  __m<V> mmscale = _mm<V>::set1_ps(EL_INT8_MAX);

  // abs max
  int Vr = ep.ic % V;
  if (Vr == 0) Vr = V;

  estl::parallel_for<3>([&](int _O4, int _O3, int _O2) {
    MD5(float, aweights_scale, weights_scale,
        ep.O4, ep.O3, 2, ep.O2, V);
    __m<V> mmabs_max = _mm<V>::set1_ps(0.0);
    iter_each (_I4, ep.I4) {
    iter_each (_I3, ep.I3) {
    iter_each (_I2, ep.I2) {
      auto r = last_I2(_I2, _I3, _I4) ? Vr : V;
      iter_each (_iV, r) {
        MD8(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O2,
            ep.I4, ep.I3, ep.I2, V, V);
        mmabs_max = _mm<V>::max_ps(mmabs_max, _mm512_abs_ps(*(__m<V> *)
            &md8(aweights, _O4, _O3, _O2, _I4, _I3, _I2, _iV, 0)));
      }
    }}}
    _mm<V>::store_ps(
        &md5(aweights_scale, _O4, _O3, 0, _O2, 0), mmabs_max);
  }, ep.O4, ep.O3, ep.O2);

  // O4, (O3, O3r), (O2, O2r), I4, I3, I2, V1, Vx, V ->
  // O4, I4, (O3, O3r), I3, I2, V1, (O2, O2r), V, Vx
  // quantization
  estl::parallel_for<9>([&](int _O4, int _I4, int _O3,
    int _I3, int _O1, int _I2, int _iV1, int _O, int _iVx) {
    MD10(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O1, ep.O,
        ep.I4, ep.I3, ep.I2, ep.V1, ep.Vx, V);
    MD10(int8_t, atweights_s8, tweights_s8, ep.O4, ep.I4,
        ep.O3, ep.I3, ep.O1, ep.I2, ep.V1, ep.O, V, ep.Vx);
    MD6(float, aweights_scale, weights_scale,
        ep.O4, ep.O3, 2, ep.O1, ep.O, V);

    if (last_I2(_I2, _I3, _I4) && _iV1 * ep.Vx + _iVx >= Vr) {
#pragma omp simd
      iter_each (_oV, V)
        md10(atweights_s8,
             _O4, _I4, _O3, _I3, _O1, _I2, _iV1, _O, _oV, _iVx) = 0;
    } else {
      auto mmresf32 = _mm<V>::mul_ps(*(__m<V> *)&md10(aweights,
          _O4, _O3, _O1, _O, _I4, _I3, _I2, _iV1, _iVx, 0), mmscale);
      mmresf32 = _mm<V>::div_ps(mmresf32,
          *(__m<V> *)&md6(aweights_scale, _O4, _O3, 0, _O1, _O, 0));
      mmresf32 = _mm<V>::roundscale_ps(
          mmresf32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      float *resf32 = (float *)&mmresf32;
#pragma omp simd
      iter_each (_oV, V) {
        md10(atweights_s8, _O4, _I4, _O3, _I3, _O1, _I2, _iV1, _O, _oV, _iVx) =
            (int8_t)resf32[_oV];
      }
    }
  }, ep.O4, ep.I4, ep.O3, ep.I3, ep.O1, ep.I2, ep.V1, ep.O, ep.Vx);

  // accumulation
  estl::parallel_for<5>([&](int _O4, int _O3, int _O1, int _O, int _oV) {
    MD10(int8_t, atweights_s8, tweights_s8, ep.O4, ep.I4,
        ep.O3, ep.I3, ep.O1, ep.I2, ep.V1, ep.O, V, ep.Vx);
    MD6(float, aweights_scale, weights_scale,
        ep.O4, ep.O3, 2, ep.O1, ep.O, V);
    float acc = 0;
    iter_each (_I4, ep.I4) {
    iter_each (_I3, ep.I3) {
    iter_each (_I2, ep.I2) {
    iter_each (_iV1, ep.V1) {
    iter_each (_iVx, ep.Vx) {
      acc += (float)md10(atweights_s8,
          _O4, _I4, _O3, _I3, _O1, _I2, _iV1, _O, _oV, _iVx);
    }}}}}
    md6(aweights_scale, _O4, _O3, 1, _O1, _O, _oV) = acc;
  }, ep.O4, ep.O3, ep.O1, ep.O, V);

  // scale
  estl::parallel_for<3>([&](int _O4, int _O3, int _O2) {
    MD5(float, aweights_scale, weights_scale,
        ep.O4, ep.O3, 2, ep.O2, V);
    __m<V> &mmqs = *(__m<V> *)&md5(
        aweights_scale, _O4, _O3, 0, _O2, 0);
    mmqs = mmqs / mmscale;
  }, ep.O4, ep.O3, ep.O2);

  // combine
  __m<V> mmorepS = _mm<V>::set1_ps(ep.output_quant_repS);
  __m<V> mmoz = _mm<V>::set1_ps(ep.output_quant_z);
  __m<V> mmiS = _mm<V>::set1_ps(ep.input_quant_S);
  __m<V> mmiz = _mm<V>::set1_ps(ep.input_quant_z);
  estl::parallel_for<3>([&](int _O4, int _O3, int _O2) {
    MD5(float, aweights_scale, weights_scale,
        ep.O4, ep.O3, 2, ep.O2, V);
    MD4(BiasType, abias, bias, ep.O4, ep.O3, ep.O2, V);
    __m<V> &mmqs = *(__m<V> *)&md5(
        aweights_scale, _O4, _O3, 0, _O2, 0);
    __m<V> &mmqf = *(__m<V> *)&md5(
        aweights_scale, _O4, _O3, 1, _O2, 0);
    __m<V> mmbias = ep.with_bias
                  ? *(__m<V> *)&md4(abias, _O4, _O3, _O2, 0)
                  : _mm<V>::setzero_ps();

    if (std::is_same<OutputType, float>::value) {
      mmqs = mmiS * mmqs;
      mmqf = mmbias - mmiz * mmqf * mmqs;
    } else {
      mmqs = mmiS * mmqs * mmorepS;
      mmqf = mmbias * mmorepS + mmoz - mmiz * mmqf * mmqs;
    }

    if (ep.with_ip_sum) {
      __m<V> sum_S = _mm<V>::set1_ps(ep.sum_quant_S);
      __m<V> sum_z = _mm<V>::set1_ps(ep.sum_quant_z);
      mmqf -= sum_z * sum_S;
    }
  }, ep.O4, ep.O3, ep.O2);
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::gemm_a160_s2(ToutputType *toutput,
    OutputType *output, uint8_t *input, int8_t *weights, float *input_scale,
    float *weights_scale, BiasType *bias, int _I4)
{
  MD3(int8_t, aweights, weights,
      ep.O3, ep.I3, ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);
  MD2(float, ainput_scale, input_scale, 2, ep.T);
  MD4(float, aweights_scale, weights_scale, ep.O3, 2, ep.O2, V);
  // blocked
  MD2(uint8_t, ainput_blocked, input,
      ep.I3, ep.I2 * ep.ih * ep.iw * V);
  MD2(OutputType, aoutput_blocked, output,
      ep.O3, ep.O2 * ep.ht * ep.wt *  ep.T * V);
  MD2(ToutputType, atoutput_blocked, toutput,
      ep.O3, ep.O2 * ep.ht * ep.wt *  ep.T * V);
  // nhwc
  MD2(uint8_t, ainput_nhwc, input, ep.I3, ep.I2 * V);
  MD2(OutputType, aoutput_nhwc, output, ep.O3, ep.O2 * V);
  MD2(ToutputType, atoutput_nhwc, toutput, ep.O3, ep.O2 * V);

  auto ker_gemm = ker_u8s8_gemm_I_O_T_;

  iter_each (_I3, ep.I3) {
    int attr = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;
    if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
      attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
      if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      if (ep.Ir != ep.V1) attr = set_bit(attr, AT_Ir_MASK);
    }
    auto ain = ep.input_fmt == nhwc
        ? &md2(ainput_nhwc, _I3, 0) : &md2(ainput_blocked, _I3, 0);
    iter_each (_O3, ep.O3) {
      auto aout = ep.output_fmt == nhwc
                ? &md2(aoutput_nhwc, _O3, 0) : &md2(aoutput_blocked, _O3, 0);
      auto atout = ep.output_fmt == nhwc
                ? &md2(atoutput_nhwc, _O3, 0) : &md2(atoutput_blocked, _O3, 0);
      ker_gemm(ep, atout, aout, ain,
          &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0),
          attr,
          &md2(ainput_scale, 0, 0),
          &md2(ainput_scale, 1, 0),
          &md4(aweights_scale, _O3, 0, 0, 0),
          &md4(aweights_scale, _O3, 1, 0, 0));
    }
  }
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::gemm_a160_s1(ToutputType *toutput,
    OutputType *output, uint8_t *input, int8_t *weights_s8, float *input_scale,
    float *weights_scale, BiasType *bias, int _I4, int _O4, int _t2)
{
  // input
  MD3(uint8_t, ainput_blocked, input,
      ep.I4, ep.I3, ep.I2 * ep.ih * ep.iw * V);
  MD3(uint8_t, ainput_nhwc, input, ep.t2, ep.T, ep.ic);
  // output
  MD3(OutputType, aoutput_blocked, output,
      ep.O4, ep.O3, ep.O2 * ep.oh * ep.ow * V);
  MD3(OutputType, aoutput_nhwc, output, ep.t2, ep.T, ep.oc);
  // toutput
  MD3(ToutputType, atoutput_blocked, toutput,
      ep.O4, ep.O3, ep.O2 * ep.oh * ep.ow * V);
  MD3(ToutputType, atoutput_nhwc, toutput, ep.t2, ep.T, ep.oc);

  MD3(int8_t, aweights_s8, weights_s8,
      ep.O3, ep.I3, ep.O2 * ep.I2 * V * V);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);
  MD2(float, ainput_scale, input_scale, 2, ep.T);
  MD4(float, aweights_scale, weights_scale,
      ep.O3, 2, ep.O2, V);

  auto ker_gemm = (_t2 == ep.t2 - 1)
      ? ker_u8s8_gemm_I_O_Tr_
      : ker_u8s8_gemm_I_O_T_;

  iter_each (_I3, ep.I3) {
    MD2(uint8_t, ainput2_blocked, &md3(ainput_blocked, _I4, _I3, 0), ep.t2, ep.T * V);
    MD3(uint8_t, ainput2_nhwc, &md3(ainput_nhwc, _t2, 0, 0), ep.I4, ep.I3, ep.I2 * V);
    int attr = _I4 == 0 && _I3 == 0
        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
        : attr_;

    if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
      attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
      if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      if (ep.Ir != ep.V1) attr = set_bit(attr, AT_Ir_MASK);
    }

    auto ain = ep.input_fmt == nhwc
             ? &md3(ainput2_nhwc, _I4, _I3, 0)
             : &md2(ainput2_blocked, _t2, 0);
    iter_each (_O3, ep.O3) {
      int _O4_tout = toutput_opt_ ? 0 : _O4;
      MD2(OutputType, aoutput2_blocked, &md3(aoutput_blocked, _O4, _O3, 0),
          ep.t2, ep.T * V);
      MD3(OutputType, aoutput2_nhwc, &md3(aoutput_nhwc, _t2, 0, 0),
          ep.O4, ep.O3, ep.O2 * V);
      MD2(ToutputType, atoutput2_blocked,
          &md3(atoutput_blocked, _O4_tout, _O3, 0), ep.t2, ep.T * V);
      MD3(ToutputType, atoutput2_nhwc, &md3(atoutput_nhwc, _t2, 0, 0),
          ep.O4, ep.O3, ep.O2 * V);
      auto aout = ep.output_fmt == nhwc
                ? &md3(aoutput2_nhwc, _O4, _O3, 0)
                : &md2(aoutput2_blocked, _t2, 0);
      auto atout = ep.output_fmt == nhwc
                ? &md3(atoutput2_nhwc, _O4, _O3, 0)
                : &md2(atoutput2_blocked, _t2, 0);
      ker_gemm(ep, atout, aout, ain,
          &md3(aweights_s8, _O3, _I3, 0),
          &md2(abias, _O3, 0),
          attr,
          &md2(ainput_scale, 0, 0),
          &md2(ainput_scale, 1, 0),
          &md4(aweights_scale, _O3, 0, 0, 0),
          &md4(aweights_scale, _O3, 1, 0, 0));
    }
  }
}

//u8f32u8f32-u8s8f32
template class elx_int8_conv_direct_1x1_t<conv::U8F32U8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;
//u8f32s8f32-u8s8f32
template class elx_int8_conv_direct_1x1_t<conv::U8F32S8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;

} // namespace euler
