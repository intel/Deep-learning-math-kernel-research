#include <string.h>
#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_wino.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD6(Type, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);
  MD6(Type, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD8(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);

  for_each (_wA, A) {
    for_each (_hA, A) {
      for_each (_oc3, this->oc3) {
        for_each (_ic3, this->ic3 - 1) {
          ker_gemm(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
              &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
              &md8(atweights, _oc3, _ic3, _wA, _hA, 0, 0, 0, 0), _ic3 == 0);
        }
        ker_gemm_tail(*this, &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
            &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
            &md8(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0, 0, 0, 0),
            this->ic3 == 1);
      }
    }
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::gemm(
    Type * __restrict toutput, Type * __restrict tinput, Type * __restrict tweights)
{
  MD2(Type, atinput2, tinput, this->t2, A * A * this->T * this->IC);
  MD2(Type, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD8(Type, atweights, tweights, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);

#pragma omp for nowait collapse(4)
  for_each (_t2, this->t2) {
    for_each (_wA, A) {
      for_each (_hA, A) {
        for_each (_oc3, this->oc3) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
          auto ker_gemm_tail
              = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;
          MD6(Type, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
          MD6(Type, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);

          for_each (_ic3, this->ic3 - 1) {
            ker_gemm(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
                &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
                &md8(atweights, _oc3, _ic3, _wA, _hA, 0, 0, 0, 0), _ic3 == 0);
          }
          ker_gemm_tail(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
              &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
              &md8(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0, 0, 0, 0),
              this->ic3 == 1);
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
  MD7(Type, atweights, tweights, A, this->oc3, this->ic3, this->O2, this->I2, V, V);

  for_each (_hA, A) {
    for_each (_oc3, this->oc3) {
      for_each (_ic3, this->ic3 - 1) {
        ker_gemm(*this, &md5(atoutput, _hA, _oc3, 0, 0, 0),
            &md5(atinput, _hA, _ic3, 0, 0, 0),
            &md7(atweights, _hA, _oc3, _ic3, 0, 0, 0, 0), _ic3 == 0);
      }
      ker_gemm_tail(*this, &md5(atoutput, _hA, _oc3, 0, 0, 0),
          &md5(atinput, _hA, this->ic3 - 1, 0, 0, 0),
          &md7(atweights, _hA, _oc3, this->ic3 - 1, 0, 0, 0, 0),
          this->ic3 == 1);
    }
  }
}
} // namespace euler
