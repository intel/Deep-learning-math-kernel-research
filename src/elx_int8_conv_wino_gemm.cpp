#include "elx_int8_conv_wino_gemm.hpp"
#include "el_parallel.hpp"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_int8_conv_wino_gemm_t<GarrayTypes, A, V, I>::setup(elx_param_t *conv_ep)
{
  attr_ = A != 6 ? set_bit(0x0, AT_FMAOPT_MASK) : 0x0;
  ep    = conv_ep;
  mthr_ = ep->nthreads;

  bind_kernel_functions();
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_int8_conv_wino_gemm_t<GarrayTypes, A, V, I>::bind_kernel_functions()
{
  u8s8_gemm_kernel_binder::bind<1, GKF_CCC>(ep->O, ep->T, &ker_u8s8_gemm_);
  u8s8_gemm_kernel_binder::bind<1, GKF_CCC>(ep->O, ep->Tr, &ker_u8s8_gemm0_);
}

// tweights:      O4 | O3, I3, A, A, O2, I2, V1, V, Vx
// tinputs:        t2 | A, A, I3, I2, T, V1, Vx
// toutput:   t2, O4 | A, A, O3, O2, T, V
// weights_scale  O4 | O3, O2, V
// facotr:        O4 | O3, A, A, O2, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_int8_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    float *src_scale, float *weights_scale, float *weights_shift,
    int _t2, int Tz, int _I4)
{
  auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

  MD6(uint8_t, atinput, tinput, A, A, ep->I3, ep->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD5(int8_t, atweights, tweights, ep->O3, ep->I3, A, A,
      ep->O2 * ep->I2 * V * V);
  MD5(float, aweights_scale, weights_scale, ep->O3, A, A, ep->O2, V);
  MD5(float, aweights_shift, weights_shift, ep->O3, A, A, ep->O2, V);
  MD5(float, asrc_scale, src_scale, ep->I3,  A, A, 2, ep->T);

  iter_each (_hA, A) {
  iter_each (_wA, A) {
  iter_each (_I3, ep->I3) {
  iter_each (_O3, ep->O3) {
    int attr = _I3 == 0 && _I4 == 0 ?  set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (_I4 == ep->I4 - 1 && _I3 == ep->I3 - 1) {
      attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
      if (ep->Ir != V)
        attr = set_bit(attr, AT_Ir_MASK);
    }

    float *asrc_s = nullptr, *asrc_z = nullptr;
    if (ep->sampling_kind == COARSE) {
      asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
      asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
    } else if (ep->sampling_kind == FINE) {
      asrc_s = &md5(asrc_scale, _I3, _hA, _wA, 0, 0);
      asrc_z = &md5(asrc_scale, _I3, _hA, _wA, 1, 0);
    } else { // CALIBRATED
      // nothing to do.
      // asrc_s/asrc_z are folded into weights_scale/shift
    }

    ker_gemm(*(elx_param_t *)ep,
        &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
        nullptr,
        &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
        &md5(atweights, _O3, _I3, _hA, _wA, 0),
        nullptr, attr, asrc_s, asrc_z,
        &md5(aweights_scale, _O3, _hA, _wA, 0, 0),
        &md5(aweights_shift, _O3, _hA, _wA, 0, 0));
  }}}}
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_int8_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    float *src_scale, float *weights_scale, float *weights_shift,
    int _t2, int Tz, int _I4)
{
  auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

  MD6(uint8_t, atinput, tinput, A, A, ep->I3, ep->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD5(int8_t, atweights, tweights, ep->O3, ep->I3, A, A,
      ep->O2 * ep->I2 * V * V);
  MD5(float, aweights_scale, weights_scale, ep->O3, A, A, ep->O2, V);
  MD5(float, aweights_shift, weights_shift, ep->O3, A, A, ep->O2, V);
  MD5(float, asrc_scale, src_scale, ep->I3, A, A, 2, Tz);

  bool scramble = (ep->T == ep->Tr) || (ep->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = estl::current_thread_index();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_I3, ep->I3) {
      iter_each(_O3, ep->O3) {
        auto attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
        if (_I4 == ep->I4 - 1 && _I3 == ep->I3 - 1) {
          attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
          if (ep->Ir != V)
            attr = set_bit(attr, AT_Ir_MASK);
        }

        float *asrc_s = nullptr, *asrc_z = nullptr;
        if (ep->sampling_kind == COARSE) {
          asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
          asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
        } else if (ep->sampling_kind == FINE) {
          asrc_s = &md5(asrc_scale, _I3, _hA, _wA, 0, 0);
          asrc_z = &md5(asrc_scale, _I3, _hA, _wA, 1, 0);
        } else { // CALIBRATED
          // nothing to do.
          // asrc_s/asrc_z are folded into weights_scale/shift
        }

        ker_gemm(*(elx_param_t *)ep,
            &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
            nullptr,
            &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
            &md5(atweights, _O3, _I3, _hA, _wA, 0),
            nullptr, attr, asrc_s, asrc_z,
            &md5(aweights_scale, _O3, _hA, _wA, 0, 0),
            &md5(aweights_shift, _O3, _hA, _wA, 0, 0));
      }}
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_I3, ep->I3) {
    iter_each(_O3, ep->O3) {
      auto attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep->I4 - 1 && _I3 == ep->I3 - 1) {
        attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
        if (ep->Ir != V)
          attr = set_bit(attr, AT_Ir_MASK);
      }

      ker_gemm(*(elx_param_t *)ep,
          &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
          nullptr,
          &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
          &md5(atweights, _O3, _I3, _hA, _wA, 0),
          nullptr, attr,
          &md5(asrc_scale, _I3, _hA, _wA, 0, 0),
          &md5(asrc_scale, _I3, _hA, _wA, 1, 0),
          &md5(aweights_scale, _O3, _hA, _wA, 0, 0),
          &md5(aweights_shift, _O3, _hA, _wA, 0, 0));
    }}}}
  }
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_int8_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    float *src_scale, float *src_shift,
    float *weights_scale, float *weights_shift, int _I4)
{
  int ithr = estl::current_thread_index();
  THREAD_FOR2(5, 2, mthr_, ithr,
              [&](int _hA, int _wA, int _I3, int _O3, int _t2) {
    MD2(uint8_t, atinput2, tinput, ep->t2, A * A * ep->I3 * ep->I2 * ep->T * V);
    MD2(ToutputType, atoutput2, toutput, ep->t2, A * A * ep->O3 * ep->O2 * ep->T * V);
    MD5(int8_t, atweights, tweights, ep->O3, ep->I3, A, A, ep->O2 * ep->I2 * V * V);
    MD5(float, aweights_scale, weights_scale, ep->O3, A, A, ep->O2, V);
    MD5(float, aweights_shift, weights_shift, ep->O3, A, A, ep->O2, V);
    MD6(float, asrc_scale, src_scale, ep->t2, A, A, ep->I3, 2, ep->T);
    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    MD6(uint8_t, atinput6, &md2(atinput2, _t2, 0), A, A, ep->I3, ep->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, ep->O3, ep->O2, Tz, V);
    auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_u8s8_gemm0_ : ker_u8s8_gemm_;

    int attr = _I3 == 0 ?  set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
    if (_I3 == ep->I3 - 1) {
      attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
      if (_I4 == ep->I4 - 1 && ep->Ir != V)
        attr = set_bit(attr, AT_Ir_MASK);
    }

    float *asrc_s = nullptr, *asrc_z = nullptr;
    if (ep->sampling_kind == COARSE) {
      asrc_s = &md6(asrc_scale, 0, 0, 0, 0, 0, 0);
      asrc_z = &md6(asrc_scale, 0, 0, 0, 0, 1, 0);
    } else if (ep->sampling_kind == FINE) {
      asrc_s = &md6(asrc_scale, _t2, _hA, _wA, _I3, 0, 0);
      asrc_z = &md6(asrc_scale, _t2, _hA, _wA, _I3, 1, 0);
    } else { // CALIBRATED
      // nothing to do.
      // asrc_s/asrc_z are folded into weights_scale/shift
    }

    ker_gemm(*ep, &md6(atoutput6, _hA, _wA, _O3, 0, 0, 0),
        nullptr,
        &md6(atinput6, _hA, _wA, _I3, 0, 0, 0),
        &md5(atweights, _O3, _I3, _hA, _wA, 0),
        nullptr, attr, asrc_s, asrc_z,
        &md5(aweights_scale, _O3, _hA, _wA, 0, 0),
        &md5(aweights_shift, _O3, _hA, _wA, 0, 0));
  }, A, A, ep->I3, ep->O3, ep->t2);
}

// GarrayTypes IN8_FP32
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F32, 4, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F32, 5, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F32, 6, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F32, 7, 16, ISA_AVX512>;

// GarrayTypes IN8_F16o
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16o, 4, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16o, 5, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16o, 6, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// GarrayTypes IN8_F16b
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16b, 4, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16b, 5, 16, ISA_AVX512>;
template class elx_int8_conv_wino_gemm_t<conv_impl::INT8_F16b, 6, 16, ISA_AVX512>;
#endif

} // namespace euler
