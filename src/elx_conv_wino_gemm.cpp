#include "el_parallel.hpp"
#include "elx_conv_wino_gemm.hpp"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::setup(elx_param_t *conv_ep)
{
  attr_ = 0x0;
  ep    = conv_ep;
  mthr_ = ep->nthreads;

  bind_kernel_functions();
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::bind_kernel_functions()
{
  gemm_kernel_binder::bind<1, GKF_CCC>(ep->O, ep->T, &ker_gemm_);
  gemm_kernel_binder::bind<1, GKF_CCC>(ep->O, ep->Tr, &ker_gemm0_);
}

// tweights:     O4 | O3, I3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, I3, I2, T, V
// toutput:  t2, O4 | A, A, O3, O2, T, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights,
    int _t2, int Tz, int _I4)
{
  auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  MD6(TinputType, atinput, tinput, A, A, ep->I3, ep->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, ep->O3, ep->I3, A, A, ep->O2 * ep->I2 * V * V);

  bool scramble = (ep->T == ep->Tr) || (ep->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = estl::current_thread_index();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_O3, ep->O3) {
        bool last_I4 = _I4 == ep->I4 - 1;
        int I3 = last_I4 ? ep->I3 - 1 : ep->I3;
        iter_each(_I3, I3) {
          int attr = _I3 == 0 && _I4 == 0
              ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
          ker_gemm(*ep,
              &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
              &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
              &md5(atweights, _O3, _I3, _hA, _wA, 0),
              nullptr, attr);
        }
        if (last_I4) {
          auto attr = ep->I3 == 1 && ep->I4 == 1
                          ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
                          : attr_;
         if (ep->Ir != V)
           attr = set_bit(attr, AT_Ir_MASK);
         ker_gemm(*ep,
             &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
             &md6(atinput, _hA, _wA, ep->I3 - 1, 0, 0, 0),
             &md5(atweights, _O3, ep->I3 - 1, _hA, _wA, 0),
             nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_O3, ep->O3) {
      bool last_I4 = _I4 == ep->I4 - 1;
      int I3 = last_I4 ? ep->I3 - 1 : ep->I3;
      iter_each(_I3, I3) {
        int attr =
            _I3 == 0 && _I4 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
        ker_gemm(*ep,
            &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
            &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
            &md5(atweights, _O3, _I3, _hA, _wA, 0),
            nullptr, attr);
      }
      if (last_I4) {
        auto attr = ep->I3 == 1 && ep->I4 == 1
                        ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK)
                        : attr_;
        if (ep->Ir != V)
          attr = set_bit(attr, AT_Ir_MASK);
        ker_gemm(*ep,
            &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
            &md6(atinput, _hA, _wA, ep->I3 - 1, 0, 0, 0),
            &md5(atweights, _O3, ep->I3 - 1, _hA, _wA, 0),
            nullptr, attr);
      }
    }}}
  }
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights,
    int _t2, int Tz, int _I4)
{
  auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  MD6(TinputType, atinput, tinput, A, A, ep->I3, ep->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, ep->O3, ep->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, ep->O3, ep->I3, A, A,
      ep->O2 * ep->I2 * V * V);

  bool scramble = (ep->T == ep->Tr) || (ep->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = estl::current_thread_index();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _wA = n % A;
      int _hA = n / A;
      iter_each(_O3, ep->O3) {
        bool last_I4 = _I4 == ep->I4 - 1;
        int I3 = last_I4 ? ep->I3 - 1 : ep->I3;
        iter_each(_I3, I3) {
          int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
          ker_gemm(*ep,
              &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
              &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
              &md5(atweights, _O3, _I3, _hA, _wA, 0),
              nullptr, attr);
        }
        if (last_I4) {
          auto attr = ep->I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
          if (ep->Ir != V)
            attr = set_bit(attr, AT_Ir_MASK);
          ker_gemm(*ep,
              &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
              &md6(atinput, _hA, _wA, ep->I3 - 1, 0, 0, 0),
              &md5(atweights, _O3, ep->I3 - 1, _hA, _wA, 0),
              nullptr, attr);
        }
      }
    }
  } else {
    iter_each(_hA, A) {
    iter_each(_wA, A) {
    iter_each(_O3, ep->O3) {
      bool last_I4 = _I4 == ep->I4 - 1;
      int I3 = last_I4 ? ep->I3 - 1 : ep->I3;
      iter_each(_I3, I3) {
        int attr = _I3 == 0 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
        ker_gemm(*ep,
            &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
            &md6(atinput, _hA, _wA, _I3, 0, 0, 0),
            &md5(atweights, _O3, _I3, _hA, _wA, 0),
            nullptr, attr);
      }
      if (last_I4) {
        auto attr = ep->I3 == 1 ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
        if (ep->Ir != V)
          attr = set_bit(attr, AT_Ir_MASK);
        ker_gemm(*ep,
            &md6(atoutput, _hA, _wA, _O3, 0, 0, 0),
            &md6(atinput, _hA, _wA, ep->I3 - 1, 0, 0, 0),
            &md5(atweights, _O3, ep->I3 - 1, _hA, _wA, 0),
            nullptr, attr);
      }
    }}}
  }
}

// tweights: O3, I3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, I3, I2, T, V
// toutput:  t2, A, A, O3, O2, T, V
template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _I4)
{
  int ithr = estl::current_thread_index();
  THREAD_FOR2(5, 4, mthr_, ithr, [&](int _hA, int _wA,
                                     int _O3, int _t2, int _I3) {
    MD2(TinputType, atinput2, tinput, ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    MD2(ToutputType, atoutput2, toutput, ep->t2, A * A * ep->T * ep->O3 * ep->O2 * V);
    MD5(TweightsType, atweights, tweights, ep->O3, ep->I3, A, A, ep->O2 * ep->I2 * V * V);

    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, ep->I3, ep->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, ep->O3, ep->O2, Tz, V);

    int attr = attr_;
    if (_I3 == 0 && _I4 == 0)
      attr = set_bit(attr, AT_CLEAR_OUTPUT_MASK);
    if (ep->Ir != V && _I4 == ep->I4 - 1 && _I3 == ep->I3 - 1)
      attr = set_bit(attr, AT_Ir_MASK);

    ker_gemm(*ep, &md6(atoutput6, _hA, _wA, _O3, 0, 0, 0),
             &md6(atinput6, _hA, _wA, _I3, 0, 0, 0),
             &md5(atweights, _O3, _I3, _hA, _wA, 0), nullptr, attr);
  }, A, A, ep->O3, ep->t2, ep->I3);
}

template <typename GarrayTypes, const int A, const int V, const int I>
void elx_conv_wino_gemm_t<GarrayTypes, A, V, I>::execute_na(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _I4)
{
  int ithr = estl::current_thread_index();
  THREAD_FOR2(5, 4, mthr_, ithr, [&](int _hA, int _wA,
                                     int _O3, int _t2, int _I3) {
    MD2(TinputType, atinput2, tinput, ep->t2, A * A * ep->T * ep->I3 * ep->I2 * V);
    MD2(ToutputType, atoutput2, toutput, ep->t2, A * A * ep->T * ep->O3 * ep->O2 * V);
    MD5(TweightsType, atweights, tweights, ep->O3, ep->I3, A, A, ep->O2 * ep->I2 * V * V);

    int Tz = _t2 == (ep->t2 - 1) ? ep->Tr : ep->T;
    auto ker_gemm = (_t2 == ep->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, ep->I3, ep->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, ep->O3, ep->O2, Tz, V);

    int attr = attr_;
    if (_I3 == 0)
      attr = set_bit(attr, AT_CLEAR_OUTPUT_MASK);
    if (ep->Ir != V && _I4 == ep->I4 - 1 && _I3 == ep->I3 - 1)
      attr = set_bit(attr, AT_Ir_MASK);

    ker_gemm(*ep, &md6(atoutput6, _hA, _wA, _O3, 0, 0, 0),
             &md6(atinput6, _hA, _wA, _I3, 0, 0, 0),
             &md5(atweights, _O3, _I3, _hA, _wA, 0), nullptr, attr);
  }, A, A, ep->O3, ep->t2, ep->I3);
}

// f32f32f32f32
template class elx_conv_wino_gemm_t<conv_impl::FP32, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 7, 16, ISA_AVX512>;

// f16f16f16f32
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 7, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// f32f16f16f16
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 7, 16, ISA_AVX512>;
#endif

} // namespace euler
