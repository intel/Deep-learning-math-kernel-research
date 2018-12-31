#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "euler.hpp"

namespace euler {

//
// -------------+------------+--------------+-------------
//  execute-opt | gemm dtype | fusion-along | duplication
// -------------+------------+--------------+-------------
//     A000     |   FP32     |      _       |    _
// -------------+------------+--------------+-------------
//     A033     |   FP32     |    i + o     |  I + O
// -------------+------------+--------------+-------------
//     A061     |   FP32     |    t + o     |    I
// -------------+------------+--------------+-------------
//     A071     |   FP32     |  i + t + o   |    I
// -------------+------------+--------------+-------------
//     A073     |   FP32     |  i + t + o   |  I + O
// -------------+------------+--------------+-------------
//     A079     |   FP32     |  i + t + o   |  I + W
// -------------+------------+--------------+-------------
//     A07b     |   FP32     |  i + t + o   |  I + W + O
// -------------+------------+--------------+-------------
//     A0e0     |   FP32     |  t + o + wA  |    _
// -------------+------------+--------------+-------------
//     A0e1     |   FP32     |  t + o + wA  |    I
// -------------+------------+--------------+-------------
//     A133     |   INT8     |    i + o     |  I + O
// -------------+------------+--------------+-------------
//     A161     |   INT8     |    t + o     |    I
// -------------+------------+--------------+-------------
//     A173     |   INT8     |  i + t + o   |  I + O
// -------------+------------+--------------+-------------
//


// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2      | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a061(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->T * this->IC);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * A * this->T * this->oc3 * this->O2 * V);
  MD2(TweightsType, atweights2, tweights_, this->oc4,
      A * A * this->IC * this->oc3 * this->O2 * V);

  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(tweights_, weights, this->oc4);
#pragma omp barrier
    }

    auto t2_history = -1;

#pragma omp for nowait collapse(2)
    iter_each (_t2, this->t2) {
    iter_each (_oc4, this->oc4) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      if (t2_history != _t2) {
        trans_input(&md2(atinput2, ithr, 0), input, _t2, Tz);
        t2_history = _t2;
      }
      gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
          &md2(atweights2, _oc4, 0), _t2, Tz);
      trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
          &md2(abias, _oc4, 0), _t2, Tz);
    }}
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, wA | hA, oc3, ic3, O2, I2, V, V
// tinputa:  t2,      wA | hA, ic3, I2, T, V
// toutput:  t2, oc4, wA | hA, oc3, O2, T, V
// toutputa: t2, oc4, oc3, O2, T, wA, hA, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a0e1(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinputa2, tinput_, mthr_,
      A * this->T * this->IC);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * this->T * this->oc3 * this->O2 * V);
  MD2(TrOpType, atoutputa2, toutputa_, this->t2,
      this->OC * A * (A - K + 1) * this->T);
  MD3(TweightsType, atweights3, tweights_, this->oc4, A,
      A * this->IC * this->oc3 * this->O2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weightsa(tweights_, weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(3)
    iter_each (_t2, this->t2) {
    iter_each (_oc4, this->oc4) {
    iter_each (_wA, A) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      MD6(TrOpType, atoutputa6, &md2(atoutputa2, _t2, 0),
          this->oc4, this->oc3, this->O2, Tz, A, (A - K + 1) * V);
      trans_inputa(&md2(atinputa2, ithr, 0), input, _t2, _wA, Tz);
      gemma(&md2(atoutput2, ithr, 0), &md2(atinputa2, ithr, 0),
          &md3(atweights3, _oc4, _wA, 0), _t2, Tz);
      trans_outputa_th(&md6(atoutputa6, _oc4, 0, 0, 0, _wA, 0),
          &md2(atoutput2, ithr, 0), Tz);
    }}}
#pragma omp barrier
    trans_outputa_bh(output, toutputa_, bias);
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, wA | hA, oc3, ic3, O2, I2, V, V
// tinputa:  t2,      wA | hA, ic3, I2, T, V
// toutput:  t2, oc4, wA | hA, oc3, O2, T, V
// toutputa: t2, oc4, oc3, O2, T, wA, hA, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a0e0(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, this->t2,
      A * A * this->T * this->IC);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * this->T * this->oc3 * this->O2 * V);
  MD2(TrOpType, atoutputa2, toutputa_, this->t2,
      this->OC * A * (A - K + 1) * this->T);
  MD3(TweightsType, atweights3, tweights_, this->oc4, A,
      A * this->IC * this->oc3 * this->O2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weightsa(tweights_, weights);
    }
    trans_input(tinput_, input);
#pragma omp barrier

#pragma omp for nowait collapse(3)
    iter_each (_t2, this->t2) {
    iter_each (_oc4, this->oc4) {
    iter_each (_wA, A) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      MD6(TrOpType, atoutputa6, &md2(atoutputa2, _t2, 0),
          this->oc4, this->oc3, this->O2, Tz, A, (A - K + 1) * V);
      MD2(TinputType, atinputa2, &md2(atinput2, _t2, 0), A, A * Tz * this->IC);
      gemma(&md2(atoutput2, ithr, 0), &md2(atinputa2, _wA, 0),
          &md3(atweights3, _oc4, _wA, 0), _t2, Tz);
      trans_outputa_th(&md6(atoutputa6, _oc4, 0, 0, 0, _wA, 0),
          &md2(atoutput2, ithr, 0), Tz);
    }}}
#pragma omp barrier
    trans_outputa_bh(output, toutputa_, bias);
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, ic4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2,      ic4 | A, A, ic3, I2, T, V
// toutput:  t2, oc4      | A, A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a071(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput_, this->t2,
      this->oc4 * A * A * this->T * this->oc3 * this->O2 * V);
  MD3(TweightsType, atweights3, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    trans_weights(tweights_, weights, this->oc4);
  }

  int last_ic4 = -1, last_t2 = -1;
#pragma omp parallel num_threads(mthr_) proc_bind(close) firstprivate(last_ic4, last_t2)
  iter_each(_ic4, this->ic4) {
#pragma omp for nowait collapse(2)
    iter_each(_t2, this->t2) {
    iter_each(_oc4, this->oc4) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();
      MD2(ToutputType, atoutput3, &md2(atoutput2, _t2, 0),
          this->oc4, A * A * Tz * this->oc3 * this->O2 * V);

      if (last_ic4 != _ic4 || last_t2 != _t2) {
        trans_input(
            &md2(atinput2, ithr, 0), &md3(ainput, 0, _ic4, 0), _t2, Tz);
        last_t2 = _t2;
        last_ic4 = _ic4;
      }
      gemm(&md2(atoutput3, _oc4, 0), &md2(atinput2, ithr, 0),
          &md3(atweights3, _oc4, _ic4, 0), _t2, Tz, _ic4);
      if (_ic4 == this->ic4 - 1)
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput3, _oc4, 0),
            &md2(abias, _oc4, 0), _t2, Tz);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a073(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * A * this->T * this->oc3 * this->O2 * V);
  MD3(TweightsType, atweights3, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    trans_weights(tweights_, weights, this->oc4);
  }

  int last_ic4 = -1, last_t2 = -1;
#pragma omp parallel num_threads(mthr_) proc_bind(close) firstprivate(last_ic4, last_t2)
  iter_each(_ic4, this->ic4) {
#pragma omp for nowait collapse(2)
    iter_each(_t2, this->t2) {
    iter_each(_oc4, this->oc4) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      if (last_ic4 != _ic4 || last_t2 != _t2) {
        trans_input(
            &md2(atinput2, ithr, 0), &md3(ainput, 0, _ic4, 0), _t2, Tz);
        last_t2 = _t2;
        last_ic4 = _ic4;
      }
      gemm_non_acc(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
          &md3(atweights3, _oc4, _ic4, 0), _t2, Tz, _ic4);
      trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
          &md2(abias, _oc4, 0), _t2, Tz, _ic4);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a07b(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * A * this->T * this->oc3 * this->O2 * V);
  MD2(TweightsType, atweights2, tweights_, mthr_,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  int last_ic4 = -1, last_t2 = -1, last_oc4 = -1;
#pragma omp parallel num_threads(mthr_) proc_bind(close) firstprivate(last_t2, last_ic4, last_oc4)
  iter_each(_ic4, this->ic4) {
#pragma omp for nowait collapse(2)
    iter_each(_oc4, this->oc4) {
    iter_each(_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      if (last_ic4 != _ic4 || last_oc4 != _oc4) {
        trans_weightsf(&md2(atweights2, ithr, 0), weights, _ic4, _oc4);
      }
      if (last_ic4 != _ic4 || last_t2 != _t2) {
        trans_input(&md2(atinput2, ithr, 0), &md3(ainput, 0, _ic4, 0), _t2, Tz);
      }
      gemm_non_acc(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
                   &md2(atweights2, ithr, 0), _t2, Tz, _ic4);
      trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
                   &md2(abias, _oc4, 0), _t2, Tz, _ic4);

      last_oc4 = _oc4; last_ic4 = _ic4; last_t2 = _t2;
    }}
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a079(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput_, this->t2,
      this->oc4 * A * A * this->T * this->oc3 * this->O2 * V);
  MD2(TweightsType, atweights2, tweights_, mthr_,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  int last_ic4 = -1, last_t2 = -1, last_oc4 = -1;
#pragma omp parallel num_threads(mthr_) proc_bind(close) firstprivate(last_t2, last_ic4, last_oc4)
  iter_each(_ic4, this->ic4) {
#pragma omp for nowait collapse(2)
    iter_each(_oc4, this->oc4) {
    iter_each(_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();
      MD2(ToutputType, atoutput3, &md2(atoutput2, _t2, 0),
          this->oc4, A * A * Tz * this->oc3 * this->O2 * V);

      if (last_ic4 != _ic4 || last_oc4 != _oc4) {
        trans_weightsf(&md2(atweights2, ithr, 0), weights, _ic4, _oc4);
      }
      if (last_ic4 != _ic4 || last_t2 != _t2) {
        trans_input(&md2(atinput2, ithr, 0), &md3(ainput, 0, _ic4, 0), _t2, Tz);
      }
      gemm(&md2(atoutput3, _oc4, 0), &md2(atinput2, ithr, 0),
           &md2(atweights2, ithr, 0), _t2, Tz, _ic4);
      if (_ic4 == this->ic4 - 1)
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput3, _oc4, 0),
                     &md2(abias, _oc4, 0), _t2, Tz, _ic4);

      last_oc4 = _oc4; last_ic4 = _ic4; last_t2 = _t2;
    }}
  }
}

// Flat mode
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a000(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_)
      trans_weights(tweights_, weights);
    trans_input(tinput_, input);
#pragma omp barrier
    gemm(toutput_, tinput_, tweights_);
#pragma omp barrier
    trans_output(output, toutput_, bias);
  }
  if (inference_acc_) is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a033(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * V);
  MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    trans_weights(tweights_, weights, this->oc4);
  }

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    int last_ic4 = -1;
    iter_each(_ic4, this->ic4) {
    iter_each(_oc4, this->oc4) {
      if (_ic4 != last_ic4) {
        trans_input(tinput_, &md3(ainput, 0, _ic4, 0));
        last_ic4 = _ic4;
      }
#pragma omp barrier
      gemm_non_acc(toutput_, tinput_, &md3(atweights, _oc4, _ic4, 0), _ic4);
#pragma omp barrier
      trans_output(&md3(aoutput, 0, _oc4, 0), toutput_, &md2(abias, _oc4, 0), _ic4);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a133(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * this->Vx * V);
  MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * this->Vx * V * this->oc3 * this->O2 * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * this->Vx * V * this->oc3 * this->O2 * V);

  MD3(TscaleType, atweights_qt_scale, tweights_qt_scale_,
      this->oc4, this->ic4, this->oc3 * this->ic3 * this->O2 * V * A * A);
  MD3(TscaleType, aweights_qt_factor, tweights_qt_factor_,
      this->oc4, this->ic4, this->oc3 * this->ic3 * this->O2 * V * A * A);

  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    trans_weights_s8(tweights_qt_scale_, tweights_qt_factor_,
        tweights_s8_, tweights_, weights, this->oc4);
  }

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    int last_ic4 = -1;
    iter_each(_ic4, this->ic4) {
    iter_each(_oc4, this->oc4) {
      if (_ic4 != last_ic4) {
        trans_input(tinput_, &md3(ainput, 0, _ic4, 0));
#pragma omp barrier
        trans_input_quantization(tinput_u8_, tinput_qt_scale_,
            tinput_qt_factor_, tinput_max_abs_, tinput_);
        last_ic4 = _ic4;
      }
#pragma omp barrier
      gemm_non_acc(toutput_, tinput_u8_, &md3(atweights_s8, _oc4, _ic4, 0),
          tinput_qt_scale_, tinput_qt_factor_, &md3(atweights_qt_scale, _ic4, _oc4, 0),
          &md3(aweights_qt_factor, _ic4, _oc4, 0), _ic4);
#pragma omp barrier
      trans_output(&md3(aoutput, 0, _oc4, 0), toutput_, &md2(abias, _oc4, 0), _ic4);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a161(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_, A * A * this->I2 * this->Vx * V);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * A * this->T * this->oc3 * this->O2 * V);

  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
      A * A * this->T * this->IC);
  MD2(int8_t, atweights_s8, tweights_s8_, this->oc4,
      A * A * this->IC * this->oc3 * this->O2 * V);
  MD2(TscaleType, atinput_qt_scale, tinput_qt_scale_,
      mthr_, this->ic3 * this->A * this->A * 2 * this->T);
  MD2(TscaleType, atweights_qt_scale, tweights_qt_scale_,
      this->oc4, this->oc3 * this->ic3 * this->O2 * V * A * A);
  MD2(TscaleType, aweights_qt_factor, tweights_qt_factor_,
      this->oc4, this->oc3 * this->ic3 * this->O2 * V * A * A);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights_s8(tweights_qt_scale_, tweights_qt_factor_,
          tweights_s8_, tweights_, weights, this->oc4);
#pragma omp barrier
    }

    auto t2_history = -1;

#pragma omp for nowait collapse(2)
    iter_each (_t2, this->t2) {
    iter_each (_oc4, this->oc4) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      ToutputType *tbuf = (toutput_size_ >= tinput_size_)
                              ? &md2(atoutput2, ithr, 0)
                              : (ToutputType *)&md2(atinput2, ithr, 0);

      if (t2_history != _t2) {
        trans_input_u8(&md2(atinput_qt_scale, ithr, 0),
            &md2(atinput2_u8, ithr, 0), (TinputType *)tbuf, input, _t2, Tz);
        t2_history = _t2;
      }
      gemm(tbuf, &md2(atinput2_u8, ithr, 0),
          &md2(atweights_s8, _oc4, 0), _t2, Tz,
          &md2(atinput_qt_scale, ithr, 0),
          &md2(atweights_qt_scale, _oc4, 0), &md2(aweights_qt_factor, _oc4, 0));
      trans_output(&md3(aoutput, 0, _oc4, 0), tbuf,
          &md2(abias, _oc4, 0), _t2, Tz);
    }}
  }
  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__execute_a173(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  MD2(TinputType, atinput2, tinput_, mthr_,
      A * A * this->ic3 * this->I2 * V * this->Vx);
  MD2(ToutputType, atoutput2, toutput_, mthr_,
      A * A * this->T * this->oc3 * this->O2 * V);

  MD3(InputType, ainput, input, this->n, this->ic4,
      this->ih * this->iw * this->ic3 * this->I2 * this->Vx * V);
  MD3(OutputType, aoutput, output, this->n, this->oc4,
      this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
      A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * this->Vx * V * this->oc3 * this->O2 * V);

  MD2(TscaleType, atinput_qt_scale, tinput_qt_scale_,
      mthr_, this->ic3 * this->A * this->A * 2 * this->T);
  MD3(TscaleType, atweights_qt_scale, tweights_qt_scale_, this->oc4,
      this->ic4, this->oc3 * this->ic3 * this->O2 * V * A * A);
  MD3(TscaleType, aweights_qt_factor, tweights_qt_factor_,
      this->oc4, this->ic4, this->oc3 * this->ic3 * this->O2 * V * A * A);

  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    trans_weights_s8(tweights_qt_scale_, tweights_qt_factor_,
        tweights_s8_, tweights_, weights, this->oc4);
  }

  int last_ic4 = -1, last_t2 = -1;
#pragma omp parallel num_threads(mthr_) proc_bind(close) firstprivate(last_ic4, last_t2)
  iter_each(_ic4, this->ic4) {
#pragma omp for nowait collapse(2)
    iter_each(_t2, this->t2) {
    iter_each(_oc4, this->oc4) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      if (last_ic4 != _ic4 || last_t2 != _t2) {
        trans_input_u8(
            &md2(atinput_qt_scale, ithr, 0),
            &md2(atinput2_u8, ithr, 0), &md2(atinput2, ithr, 0),
            &md3(ainput, 0, _ic4, 0), _t2, Tz);
        last_t2 = _t2;
        last_ic4 = _ic4;
      }
      gemm_non_acc(&md2(atoutput2, ithr, 0), &md2(atinput2_u8, ithr, 0),
          &md3(atweights_s8, _oc4, _ic4, 0), _t2, Tz,
          &md2(atinput_qt_scale, ithr, 0),
          &md3(atweights_qt_scale, _oc4, _ic4, 0),
          &md3(aweights_qt_factor, _oc4, _ic4, 0),
          _ic4);
      trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
          &md2(abias, _oc4, 0), _t2, Tz, _ic4);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::execute(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  set_trans_buffers();

  if (is_bfmt_)
    return (this->*execute_opt_)(output, input, weights, bias);
  else {
    InputType *in = input;
    WeightsType *wei = weights;
    OutputType *out = output_as_bfmt_ ? boutput_ : output;

    if (input_as_bfmt_) {
      MD5(InputType, abinput, binput_, this->n, this->ic2, this->ih, this->iw, V);
      MD4(InputType, ainput, input, this->n, this->ic, this->ih, this->iw);

#pragma omp parallel for collapse(3)
      iter_each (_n, this->n) {
      iter_each (_ic2, this->ic2) {
      iter_each (_ih, this->ih) {
        int v = _ic2 == this->ic2 - 1 ? this->Ir : V;
        iter_each (_iw, this->iw) {
#pragma omp simd
          iter_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }}}
      in = binput_;
    }

    if (weights_as_bfmt_) {
      MD6(WeightsType, abweights, bweights_, this->oc2, this->ic2,
          this->kh, this->kw, V, V);
      MD4(WeightsType, aweights, weights, this->oc, this->ic, this->kh, this->kw);

#pragma omp parallel for collapse(3)
      iter_each (_oc2, this->oc2) {
      iter_each (_ic2, this->ic2) {
      iter_each (_kh, this->kh) {
        int iv = _ic2 == this->ic2 - 1 ? this->Ir : V;
        int ov = _oc2 == this->oc2 - 1 ? this->Or : V;
        iter_each (_kw, this->kw) {
        iter_each (_iv, iv) {
#pragma omp simd
        iter_each (_ov, ov) {
          md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
            = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
        }}}
      }}}
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)(out, in, wei, bias);

    if (output_as_bfmt_) {
      MD5(OutputType, aboutput, boutput_, this->n, this->oc2, this->oh, this->ow, V);
      MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);

#pragma omp parallel for collapse(3)
      iter_each (_n, this->n) {
      iter_each (_oc2, this->oc2) {
      iter_each (_oh, this->oh) {
        int v = _oc2 == this->oc2 - 1 ? this->Or : V;
        if (this->with_ip_sum)
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              += md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        else
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
      }}}
    }
  }
}

} // namespace euler
