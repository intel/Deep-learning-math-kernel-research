#pragma once

#include "elx_int8_conv_direct.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  c160 |conv |   t+o  |  -  | blocked<->nhwc, Ir, Tr, K=3,5,7 S=1,2
// ------+-----+--------+-----+------------------------------------------------
//  a160 |gemm |   t+o  |  -  | blocked<->nhwc, Ir, Tr
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::__execute_c160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ht*, S, wt*, T, S, V(V1, Vx)
  // weights: O4*, O3, O2, I4*, I3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  n*, O4*, O3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights(weights_scale_, weights_shift_, tweights_s8_,
                    weights, bias);
    });
  }

  auto V1 = compact_ir_weights_ ? ep.Ir : ep.V1;

  INIT_LOOP_ORDER(5);
  CREATE_LOOP_ORDER(5, n, I4, O4, ht, wt);
  CREATE_LOOP_ORDER(5, I4, O4, n, ht, wt);

  auto loop_for = [&](int a0, int a1, int a2, int a3, int a4) {
    int _n, _I4, _O4, _ht, _wt;
    if (CHECK_LOOP_ORDER(5, n, I4, O4, ht, wt)) {
      _n = a0, _I4 = a1, _O4 = a2, _ht = a3, _wt = a4;
    } else {
      _I4 = a0, _O4 = a1, _n = a2, _ht = a3, _wt = a4;
    }
    MD3(int8_t, atweights_s8, tweights_s8_, ep.O4, ep.I4, ep.V1 * ep.Vx * V
        * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2 * ep.O2);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
    MD2(float, atweights_scale, weights_scale_, ep.O4,
        ep.O3 * ep.O2 * V);
    MD2(float, aweights_shift, weights_shift_, ep.O4,
        ep.O3 * ep.O2 * ep.T * V);
    // nhwc input
    MD4(InputType, ainput0_nhwc, input, ep.n, ep.ih, ep.iw,
        ep.g * ep.ic);
    MD2(InputType, ainput1_nhwc, &md4(ainput0_nhwc, _n, 0, 0, 0), ep.I4,
        ep.I3 * ep.I2 * V);
    // blocked input
    MD4(InputType, ainput_blocked, input, ep.n, ep.I4,
        ep.I3 * ep.I2, ep.ih * ep.iw * V);
    // nhwc output
    MD4(OutputType, aoutput0_nhwc, output, ep.n, ep.ht, ep.ow, ep.oc);
    MD3(OutputType, aoutput1_nhwc, &md4(aoutput0_nhwc, _n, _ht, 0, 0), ep.wt,
        ep.T, ep.oc);
    MD2(OutputType, aoutput2_nhwc, &md3(aoutput1_nhwc, _wt, 0, 0), ep.O4,
        ep.O3 * ep.O2 * V);
    // blocked output
    MD5(OutputType, aoutput0_blocked, output, ep.n, ep.O4,
        ep.O3 * ep.O2, ep.ht, ep.ow * V);
    MD3(OutputType, aoutput1_blocked, &md5(aoutput0_blocked, _n, _O4, 0, _ht, 0),
        ep.wt, ep.T, V);
    // nhwc toutput
    MD4(ToutputType, atoutput0_nhwc, toutput_, ep.n, ep.ht, ep.ow, ep.oc);
    MD3(ToutputType, atoutput1_nhwc, &md4(atoutput0_nhwc, _n, _ht, 0, 0),
        ep.wt, ep.T, ep.oc);
    MD2(ToutputType, atoutput2_nhwc, &md3(atoutput1_nhwc, _wt, 0, 0), ep.O4,
        ep.O3 * ep.O2 * V);
    // blocked toutput
    MD5(ToutputType, atoutput0_blocked, toutput_, ep.n, ep.O4,
        ep.O3 * ep.O2, ep.ht, ep.ow * V);
    MD3(ToutputType, atoutput1_blocked, &md5(atoutput0_blocked, _n, _O4, 0, _ht, 0),
        ep.wt, ep.T, V);

    auto ainput = ep.input_fmt == nhwc
                       ? &md2(ainput1_nhwc, _I4, 0)
                       : &md4(ainput_blocked, _n, _I4, 0, 0);
    auto aoutput = ep.output_fmt == nhwc
                       ? &md2(aoutput2_nhwc, _O4, 0)
                       : &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = ep.output_fmt == nhwc
                       ? &md2(atoutput2_nhwc, _O4, 0)
                       : &md3(atoutput1_blocked, _wt, 0, 0);
    conv_c160(aoutput, atoutput, ainput,
              &md3(atweights_s8, _O4, _I4, 0),
              &md2(abias, _O4, 0), input_scale_, &md2(atweights_scale, _O4, 0),
              &md2(aweights_shift, _O4, 0), _I4, _O4, _ht, _wt);
  };

  if (ep.oh <= 7 && ep.ow <= 7) {
    SET_LOOP_ORDER(5, I4, O4, n, ht, wt);
    estl::parallel_for<5, 0>(loop_for, ep.I4, ep.O4, ep.n, ep.ht, ep.wt);
  } else {
    SET_LOOP_ORDER(5, n, I4, O4, ht, wt);
    estl::parallel_for<5, 1>(loop_for, ep.n, ep.I4, ep.O4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::__execute_a160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ih, iw, V(V1, Vx)
  // weights: O4*, O3, O2, I4*, I3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  n*, O4*, O3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights(weights_scale_, weights_shift_, tweights_s8_,
                    weights, bias);
    });
  }

  estl::parallel_for<5, 1>([&](int _n, int _I4, int _O4, int _ht, int _wt) {
    MD3(int8_t, atweights_s8, tweights_s8_,
        ep.O4, ep.I4, V * V * ep.kh * ep.kw
        * ep.I3 * ep.O3 * ep.I2 * ep.O2);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
    MD2(float, atweights_scale, weights_scale_, ep.O4, ep.O3 * ep.O2 * V);
    MD2(float, aweights_shift, weights_shift_,
        ep.O4, ep.O3 * ep.O2 * V);

    // input
    MD4(InputType, ainput0_nhwc, input, ep.n, ep.ih, ep.iw, ep.ic);
    MD2(InputType, ainput1_nhwc, &md4(ainput0_nhwc, _n, 0, 0, 0),
        ep.I4, ep.I3 * ep.I2 * V);
    MD3(InputType, ainput_blocked, input,
        ep.n, ep.I4, ep.I3 * ep.I2 * ep.ih * ep.iw * V);

    // output
    MD4(OutputType, aoutput0_nhwc, output, ep.n, ep.ht, ep.ow, ep.oc);
    MD2(OutputType, aoutput1_nhwc, &md4(aoutput0_nhwc, _n, 0, 0, 0),
        ep.O4, ep.O3 * ep.O2 * V);
    MD3(OutputType, aoutput_blocked, output,
        ep.n, ep.O4, ep.O3 * ep.O2 * ep.ht * ep.ow * V);

    // toutput
    MD4(ToutputType, atoutput0_nhwc, toutput_,
        ep.n, ep.ht, ep.ow, ep.OC);
    MD2(ToutputType, atoutput1_nhwc, &md4(atoutput0_nhwc, _n, 0, 0, 0),
        ep.O4, ep.O3 * ep.O2 * V);
    MD3(ToutputType, atoutput_blocked, toutput_,
        ep.n, ep.O4, ep.O3 * ep.O2 * ep.ht * ep.ow * V);

    auto ain = ep.input_fmt == nhwc
           ? &md2(ainput1_nhwc, _I4, 0) : &md3(ainput_blocked, _n, _I4, 0);
    auto aout = ep.output_fmt == nhwc
           ? &md2(aoutput1_nhwc, _O4, 0) : &md3(aoutput_blocked, _n, _O4, 0);
    auto atout = ep.output_fmt == nhwc
           ? &md2(atoutput1_nhwc, _O4, 0) : &md3(atoutput_blocked, _n, _O4, 0);

      gemm_a160(aout, atout, ain,
                &md3(atweights_s8, _O4, _I4, 0), &md2(abias, _O4, 0),
                input_scale_, &md2(atweights_scale, _O4, 0),
                &md2(aweights_shift, _O4, 0), _I4, _O4, _ht, _wt);
  }, ep.n, ep.I4, ep.O4, ep.ht, ep.wt);

  int oc2 = ep.Or ? ep.oc2 - 1 : ep.oc2;
  if (ep.with_argmax) {
    estl::parallel_for<3>([&](int _n, int _oh, int _ow) {
      constexpr int V8 = 8;
      MD6(float, atoutput_blocked, toutput_, ep.n, ep.oc2, ep.oh, ep.ow, 2, V8);
      MD6(float, atoutput_nhwc, toutput_, ep.n, ep.oh, ep.ow, ep.oc2, 2, V8);
      MD3(int, aoutput, output, ep.n, ep.oh, ep.ow);

      auto aout = (ep.input_fmt == nhwc)
            ? &md6(atoutput_nhwc, _n, _oh, _ow, 0, 0, 0)
            : &md6(atoutput_blocked, _n, 0, _oh, _ow, 0, 0);
      __m<V/2> vmax = _mm256_load_ps(aout);
      __i<V/2> kmax = _mm256_setzero_si256();

      iter_each(_oc2, oc2) {
        iter_each(_V2, 2) {
          int index = _oc2 * 2 + _V2;
          assert(index < (1 << 15));
          if (index > 0) {
            aout = (ep.input_fmt == nhwc)
                 ? &md6(atoutput_nhwc, _n, _oh, _ow, _oc2, _V2, 0)
                 : &md6(atoutput_blocked, _n, _oc2, _oh, _ow, _V2, 0);
            __m<V/2> vcur = _mm256_load_ps(aout);
            __i<V/2> kcur = _mm256_castps_si256(_mm256_cmp_ps(vmax, vcur, _CMP_GE_OQ));
            kcur = _mm256_add_epi32(_mm256_set1_epi32(1), kcur);
            kcur = _mm256_mullo_epi16(kcur, _mm256_set1_epi32(index));
            vmax = _mm256_max_ps(vmax, vcur);
            kmax = _mm256_max_epi32(kmax, kcur);
          }
        }
      }
      float vmaxbuf[V8]; int kmaxbuf[V8];
      _mm256_store_ps(vmaxbuf, vmax);
      _mm256_store_si256((__m256i *)kmaxbuf, kmax);
      float gmax = vmaxbuf[0]; int pos = 0;
      for(int i = 1; i < V8; ++i) {
        if (vmaxbuf[i] > gmax) {
          gmax = vmaxbuf[i];
          pos = i;
        }
      }
      md3(aoutput, _n, _oh, _ow) = kmaxbuf[pos] * V8 + pos;

      int tail_start = oc2 * V;
      for (int _V = 0; _V < ep.Or; ++_V) {
        MD5(float, atout_blocked, toutput_, ep.n, ep.oc2, ep.oh, ep.ow, V);
        MD5(float, atout_nhwc, toutput_, ep.n, ep.oh, ep.ow, ep.oc2, V);
        float atout = (ep.input_fmt == nhwc)
          ? md5(atout_nhwc, _n, _oh, _ow, ep.oc2 - 1, _V)
          : md5(atout_blocked, _n, ep.oc2 - 1, _oh, _ow, _V);
        if (atout > gmax) {
          gmax = atout;
          md3(aoutput, _n, _oh, _ow) = tail_start + _V;
        }
      }
    }, ep.n, ep.oh, ep.ow);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
