#pragma once

#include "el_parallel.hpp"
#include "elx_int8_conv_wino.hpp"

namespace euler {

//
// -------------+------------+--------------+-------------
//  execute-opt | gemm dtype | fusion-along | duplication
// -------------+------------+--------------+-------------
//     A133     |   INT8     |    i + o     |  I + O
// -------------+------------+--------------+-------------
//     A161     |   INT8     |    t + o     |    I
// -------------+------------+--------------+-------------
//     A173     |   INT8     |  i + t + o   |  I + O
// -------------+------------+--------------+-------------
//


Template_elx_int8_conv_wino_t
void Instance_elx_int8_conv_wino_t::__execute_a133(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_scale_, tweights_shift_,
                       tweights_s8_, tweights_, weights, ep.O4);
    });
  }

  THREAD_PARALLEL()
  {
    int last_I4 = -1;
    MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    MD3(int8_t, atweights_s8, tweights_s8_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD3(float, atweights_scale, tweights_scale_,
        ep.O4, ep.I4, ep.O3 * ep.O2 * V * A * A);
    MD3(float, aweights_shift, tweights_shift_,
        ep.O4, ep.I4, ep.O3 * ep.O2 * V * A * A);

    iter_each (_I4, ep.I4) {
    iter_each (_O4, ep.O4) {
      if (_I4 != last_I4) {
        trans_input_u8(
            tinput_scale_, tinput_u8_, tinput_, input, _I4);
        last_I4 = _I4;
      }
      THREAD_BARRIER()
      u8s8_gemm.execute_na(toutput_, tinput_u8_,
          &md3(atweights_s8, _O4, _I4, 0),
          tinput_scale_, nullptr,
          &md3(atweights_scale, _O4, _I4, 0),
          &md3(aweights_shift, _O4, _I4, 0),
          _I4);
      THREAD_BARRIER()
      trans_output(output, toutput_, &md2(abias, _O4, 0), _O4, _I4);
    }}
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_wino_t
void Instance_elx_int8_conv_wino_t::__execute_a161(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_scale_, tweights_shift_,
                       tweights_s8_, tweights_, weights, ep.O4);
    });
  }

  auto t2_history = -1;
  estl::parallel_for<2>([&, t2_history](int _t2, int _O4) mutable {
    int ithr = estl::current_thread_index();
    MD2(TinputType, atinput2, tinput_, mthr_, ep.sampling_kind == COARSE ?
        A * A * ep.IC * ep.T : A * A * ep.I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * ep.T * ep.O3 * ep.O2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
    MD2(float, atinput_scale, tinput_scale_, mthr_,
        ep.sampling_kind == CALIBRATED ? 2 * ep.T
                                          : ep.I3 * A * A * 2 * ep.T);
    MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
        A * A * ep.T * ep.IC);
    MD2(int8_t, atweights_s8, tweights_s8_, ep.O4,
        A * A * ep.IC * ep.O3 * ep.O2 * V);
    MD2(float, atweights_scale, tweights_scale_,
        ep.O4, ep.O3 * ep.O2 * V * A * A);
    MD2(float, aweights_shift, tweights_shift_,
        ep.O4, ep.O3 * ep.O2 * V * A * A);

    int Tz = _t2 == (ep.t2 - 1) ? ep.Tr : ep.T;

    if (t2_history != _t2) {
      trans_input_u8(&md2(atinput_scale, ithr, 0),
          &md2(atinput2_u8, ithr, 0), &md2(atinput2, ithr, 0), input, _t2, Tz);
      t2_history = _t2;
    }
    u8s8_gemm.execute(&md2(atoutput2, ithr, 0),
        &md2(atinput2_u8, ithr, 0),
        &md2(atweights_s8, _O4, 0),
        &md2(atinput_scale, ithr, 0),
        &md2(atweights_scale, _O4, 0),
        &md2(aweights_shift, _O4, 0), _t2, Tz);
    trans_output(output, &md2(atoutput2, ithr, 0),
                 &md2(abias, _O4, 0), Tz, _t2, _O4, 0);
  }, ep.t2, ep.O4);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_wino_t
void Instance_elx_int8_conv_wino_t::__execute_a173(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_scale_, tweights_shift_,
                       tweights_s8_, tweights_, weights, ep.O4);
    });
  }

  int last_I4 = -1, last_t2 = -1;
  estl::parallel_for<3, 1>(
      mthr_, [&, last_I4, last_t2](int _t2, int _I4, int _O4) mutable {
    int Tz = _t2 == (ep.t2 - 1) ? ep.Tr : ep.T;
    size_t ithr = estl::current_thread_index();
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * ep.I3 * ep.I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * ep.T * ep.O3 * ep.O2 * V);
    MD3(InputType, ainput, input, ep.n, ep.I4,
        ep.ih * ep.iw * ep.I3 * ep.I2 * V);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
    MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
        A * A * ep.T * ep.I3 * ep.I2 * V);
    MD3(int8_t, atweights_s8, tweights_s8_, ep.O4, ep.I4,
        A * A * ep.I3 * ep.I2 * V * ep.O3 * ep.O2 * V);
    MD2(float, atinput_scale, tinput_scale_, mthr_,
        ep.sampling_kind == CALIBRATED ? 2 * ep.T
                                          : ep.I3 * A * A * 2 * ep.T);
    MD3(float, atweights_scale, tweights_scale_,
        ep.O4, ep.I4, ep.O3 * ep.O2 * V * A * A);
    MD3(float, aweights_shift, tweights_shift_,
        ep.O4, ep.I4, ep.O3 * ep.O2 * V * A * A);

    if (last_I4 != _I4 || last_t2 != _t2) {
      trans_input_u8(
          &md2(atinput_scale, ithr, 0),
          &md2(atinput2_u8, ithr, 0), &md2(atinput2, ithr, 0),
          &md3(ainput, 0, _I4, 0), _t2, Tz);
      last_t2 = _t2; last_I4 = _I4;
    }
    u8s8_gemm.execute_na(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2_u8, ithr, 0),
        &md3(atweights_s8, _O4, _I4, 0),
        &md2(atinput_scale, ithr, 0),
        &md3(atweights_scale, _O4, _I4, 0),
        &md3(aweights_shift, _O4, _I4, 0), _t2, Tz, _I4);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _O4, 0), Tz, _t2, _O4, _I4);
  }, ep.t2, ep.I4, ep.O4);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_wino_t
void Instance_elx_int8_conv_wino_t::execute(
    void * __restrict output, void * __restrict input,
    void * __restrict weights, void * __restrict bias)
{
  if (is_bfmt_)
    return (this->*execute_opt_)((OutputType *)output,
        (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
  else {
    InputType *in = (InputType *)input;
    WeightsType *wei = (WeightsType *)weights;
    OutputType *out = output_as_bfmt_ ? boutput_ : (OutputType *)output;

    if (input_as_bfmt_) {
      estl::parallel_for<3>([&](int _n, int _ic2, int _ih) {
        int v = _ic2 == ep.ic2 - 1 ? ep.Ir : V;
        MD5(InputType, abinput, binput_, ep.n, ep.ic2, ep.ih, ep.iw, V);
        MD4(InputType, ainput, input, ep.n, ep.ic, ep.ih, ep.iw);
        iter_each (_iw, ep.iw) {
#pragma omp simd
          iter_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }, ep.n, ep.ic2, ep.ih);
      in = binput_;
    }

    if (weights_as_bfmt_) {
      estl::parallel_for<3>([&](int _oc2, int _ic2, int _kh) {
        MD6(WeightsType, abweights, bweights_, ep.oc2, ep.ic2,
            ep.kh, ep.kw, V, V);
        MD4(WeightsType, aweights, weights, ep.oc, ep.ic, ep.kh, ep.kw);
        int iv = _ic2 == ep.ic2 - 1 ? ep.Ir : V;
        int ov = _oc2 == ep.oc2 - 1 ? ep.Or : V;
        iter_each (_kw, ep.kw) {
        iter_each (_iv, iv) {
#pragma omp simd
        iter_each (_ov, ov) {
          md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
            = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
        }}}
      }, ep.oc2, ep.ic2, ep.kh);
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)((OutputType *)out,
        (InputType *)in, (WeightsType *)wei, (BiasType *)bias);

    if (output_as_bfmt_) {
      estl::parallel_for<3>([&](int _n, int _oc2, int _oh) {
        MD5(OutputType, aboutput, boutput_, ep.n, ep.oc2, ep.oh, ep.ow, V);
        MD4(OutputType, aoutput, output, ep.n, ep.oc, ep.oh, ep.ow);
        int v = _oc2 == ep.oc2 - 1 ? ep.Or : V;

        if (ep.with_ip_sum)
          iter_each (_V, v) {
          iter_each (_ow, ep.ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              += md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        else
          iter_each (_V, v) {
          iter_each (_ow, ep.ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
      }, ep.n, ep.oc2, ep.oh);
    }
  }
}

} // namespace euler
