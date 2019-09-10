#include "elx_conv_direct_lp.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a160 |conv |   t+o  |  -  | blocked<->nhwc, Ir, Tr, K=3,5,7 S=1,2
// ------+-----+--------+-----+------------------------------------------------
//  d160 |gemm |   t+o  |  -  | blocked<->nhwc, Ir, Tr
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::__execute_a160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(V1, Vx)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  t3*, oc4*, oc3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights(weights_scale_, weights_factor_, tweights_s8_,
                    weights, bias);
    });
  }

  auto V1 = compact_ir_weights_ ? this->Ir : this->V1;

  INIT_LOOP_ORDER(5);
  CREATE_LOOP_ORDER(5, t3, ic4, oc4, ht, wt);
  CREATE_LOOP_ORDER(5, ic4, oc4, t3, ht, wt);

  auto loop_for = [&](int a0, int a1, int a2, int a3, int a4) {
    int _t3, _ic4, _oc4, _ht, _wt;
    if (CHECK_LOOP_ORDER(5, t3, ic4, oc4, ht, wt)) {
      _t3 = a0, _ic4 = a1, _oc4 = a2, _ht = a3, _wt = a4;
    } else {
      _ic4 = a0, _oc4 = a1, _t3 = a2, _ht = a3, _wt = a4;
    }
    MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4, V1 * Vx * V
        * this->kh * this->kw * this->ic3 * this->oc3 * this->I2 * this->O2);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
    MD2(TscaleType, atweights_scale, weights_scale_, this->oc4,
        this->oc3 * this->O2 * V);
    MD2(TscaleType, aweights_factor, weights_factor_, this->oc4,
        this->oc3 * this->O2 * T * V);
    // nhwc input
    MD4(InputType, ainput0_nhwc, input, this->t3, this->ih, this->iw,
        this->g * this->ic);
    MD2(InputType, ainput1_nhwc, &md4(ainput0_nhwc, _t3, 0, 0, 0), this->ic4,
        this->ic3 * this->I2 * V);
    // blocked input
    MD4(InputType, ainput_blocked, input, this->t3, this->ic4,
        this->ic3 * this->I2, this->ih * this->iw * V);
    // nhwc output
    MD4(OutputType, aoutput0_nhwc, output, this->t3, this->ht, this->ow, this->oc);
    MD3(OutputType, aoutput1_nhwc, &md4(aoutput0_nhwc, _t3, _ht, 0, 0), this->wt,
        this->T, this->oc);
    MD2(OutputType, aoutput2_nhwc, &md3(aoutput1_nhwc, _wt, 0, 0), this->oc4,
        this->oc3 * this->O2 * V);
    // blocked output
    MD5(OutputType, aoutput0_blocked, output, this->t3, this->oc4,
        this->oc3 * this->O2, this->ht, this->ow * V);
    MD3(OutputType, aoutput1_blocked, &md5(aoutput0_blocked, _t3, _oc4, 0, _ht, 0),
        this->wt, this->T, V);
    // nhwc toutput
    MD4(ToutputType, atoutput0_nhwc, toutput_, this->t3, this->ht, this->ow, this->oc);
    MD3(ToutputType, atoutput1_nhwc, &md4(atoutput0_nhwc, _t3, _ht, 0, 0),
        this->wt, this->T, this->oc);
    MD2(ToutputType, atoutput2_nhwc, &md3(atoutput1_nhwc, _wt, 0, 0), this->oc4,
        this->oc3 * this->O2 * V);
    // blocked toutput
    MD5(ToutputType, atoutput0_blocked, toutput_, this->t3, this->oc4,
        this->oc3 * this->O2, this->ht, this->ow * V);
    MD3(ToutputType, atoutput1_blocked, &md5(atoutput0_blocked, _t3, _oc4, 0, _ht, 0),
        this->wt, this->T, V);

    auto ainput = this->input_fmt == nhwc
                       ? &md2(ainput1_nhwc, _ic4, 0)
                       : &md4(ainput_blocked, _t3, _ic4, 0, 0);
    auto aoutput = this->output_fmt == nhwc
                       ? &md2(aoutput2_nhwc, _oc4, 0)
                       : &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = this->output_fmt == nhwc
                       ? &md2(atoutput2_nhwc, _oc4, 0)
                       : &md3(atoutput1_blocked, _wt, 0, 0);
    conv_a160(aoutput, atoutput, ainput,
              &md3(atweights_s8, _oc4, _ic4, 0),
              &md2(abias, _oc4, 0), input_scale_, &md2(atweights_scale, _oc4, 0),
              &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
  };

  if (this->oh <= 7 && this->ow <= 7) {
    SET_LOOP_ORDER(5, ic4, oc4, t3, ht, wt);
    parallel_for<5, 0>(mthr_, loop_for,
                       this->ic4, this->oc4, this->t3, this->ht, this->wt);
  } else {
    SET_LOOP_ORDER(5, t3, ic4, oc4, ht, wt);
    parallel_for<5, 1>(mthr_, loop_for,
                       this->t3, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::__execute_d160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ih, iw, V(V1, Vx)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  t3*, oc4*, oc3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights(weights_scale_, weights_factor_, tweights_s8_,
                    weights, bias);
    });
  }

  parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
    MD3(int8_t, atweights_s8, tweights_s8_,
        this->oc4, this->ic4, V * V * this->kh * this->kw
        * this->ic3 * this->oc3 * this->I2 * this->O2);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
    MD2(TscaleType, atweights_scale, weights_scale_,
        this->oc4, this->oc3 * this->O2 * V);
    MD2(TscaleType, aweights_factor, weights_factor_,
        this->oc4, this->oc3 * this->O2 * V);

    // input
    MD4(InputType, ainput0_nhwc, input, this->t3, this->ih, this->iw, this->ic);
    MD2(InputType, ainput1_nhwc, &md4(ainput0_nhwc, _t3, 0, 0, 0),
        this->ic4, this->ic3 * this->I2 * V);
    MD3(InputType, ainput_blocked, input,
        this->t3, this->ic4, this->ic3 * this->I2 * this->ih * this->iw * V);

    // output
    MD4(OutputType, aoutput0_nhwc, output, this->t3, this->ht, this->ow, this->oc);
    MD2(OutputType, aoutput1_nhwc, &md4(aoutput0_nhwc, _t3, 0, 0, 0),
        this->oc4, this->oc3 * this->O2 * V);
    MD3(OutputType, aoutput_blocked, output,
        this->t3, this->oc4, this->oc3 * this->O2 * this->ht * this->ow * V);

    // toutput
    MD4(ToutputType, atoutput0_nhwc, toutput_,
        this->t3, this->ht, this->ow, this->OC);
    MD2(ToutputType, atoutput1_nhwc, &md4(atoutput0_nhwc, _t3, 0, 0, 0),
        this->oc4, this->oc3 * this->O2 * V);
    MD3(ToutputType, atoutput_blocked, toutput_,
        this->t3, this->oc4, this->oc3 * this->O2 * this->ht * this->ow * V);

    auto ain = this->input_fmt == nhwc
           ? &md2(ainput1_nhwc, _ic4, 0) : &md3(ainput_blocked, _t3, _ic4, 0);
    auto aout = this->output_fmt == nhwc
           ? &md2(aoutput1_nhwc, _oc4, 0) : &md3(aoutput_blocked, _t3, _oc4, 0);
    auto atout = this->output_fmt == nhwc
           ? &md2(atoutput1_nhwc, _oc4, 0) : &md3(atoutput_blocked, _t3, _oc4, 0);

      gemm_d160(aout, atout, ain,
                &md3(atweights_s8, _oc4, _ic4, 0), &md2(abias, _oc4, 0),
                input_scale_, &md2(atweights_scale, _oc4, 0),
                &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
  }, this->t3, this->ic4, this->oc4, this->ht, this->wt);

  int oc2 = this->Or ? this->oc2 - 1 : this->oc2;
  if (this->with_argmax) {
    parallel_for<3>(mthr_, [&](int _n, int _oh, int _ow) {
      constexpr int V8 = 8;
      MD6(float, atoutput_blocked, toutput_,
          this->n, this->oc2, this->oh, this->ow, 2, V8);
      MD6(float, atoutput_nhwc, toutput_,
          this->n, this->oh, this->ow, this->oc2, 2, V8);
      MD3(int, aoutput, output, this->n, this->oh, this->ow);

      auto aout = (this->input_fmt == nhwc)
            ? &md6(atoutput_nhwc, _n, _oh, _ow, 0, 0, 0)
            : &md6(atoutput_blocked, _n, 0, _oh, _ow, 0, 0);
      __m<V/2> vmax = _mm256_load_ps(aout);
      __i<V/2> kmax = _mm256_setzero_si256();

      iter_each(_oc2, oc2) {
        iter_each(_V2, 2) {
          int index = _oc2 * 2 + _V2;
          assert(index < (1 << 15));
          if (index > 0) {
            aout = (this->input_fmt == nhwc)
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
      for (int _V = 0; _V < this->Or; ++_V) {
        MD5(float, atoutput_blocked, toutput_,
            this->n, this->oc2, this->oh, this->ow, V);
        MD5(float, atoutput_nhwc, toutput_,
            this->n, this->oh, this->ow, this->oc2, V);
        float aout = (this->input_fmt == nhwc)
          ? md5(atoutput_nhwc, _n, _oh, _ow, this->oc2 - 1, _V)
          : md5(atoutput_blocked, _n, this->oc2 - 1, _oh, _ow, _V);
        if (aout > gmax) {
          gmax = aout;
          md3(aoutput, _n, _oh, _ow) = tail_start + _V;
        }
      }
    }, this->n, this->oh, this->ow);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
