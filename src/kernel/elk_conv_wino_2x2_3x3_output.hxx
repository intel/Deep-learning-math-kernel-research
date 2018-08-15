#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

#ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
#error "Don't include this file directly"
#endif

namespace euler {

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (_wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, 16);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#undef OP
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3;
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
  MATRIX_DEF(4, 4);

  c0 = ADD(ADD(t10, t11), t12);
  c1 = ADD(ADD(t20, t21), t22);
  c2 = SUB(ADD(t12, t13), t11);
  c3 = SUB(ADD(t22, t23), t21);

  __m512 p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
  if (with_bias_) p00 = ADD(p00, *(__m512*)bias);
  __m512 p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
  if (with_bias_) p10 = ADD(p10, *(__m512*)bias);
  __m512 p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
  if (with_bias_) p01 = ADD(p01, *(__m512*)bias);
  __m512 p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);
  if (with_bias_) p11 = ADD(p11, *(__m512*)bias);

#undef OP
#define p_(m, n) p##m##n
#define OP(m,n) _mm512_store_ps(P(m, n), p_(m, n))
  MATRIX_DEF(2, 2);
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 4, 3, 16, ISA_SKX_AVX512)
{
  // TODO
  el_error("Unimplemented");
}

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 4, 3, 16, ISA_SKX_AVX512)
{
  // TODO
  el_error("Unimplemented");
}

TRANS_OUPUT(float, 4, 3, 16, ISA_SKX_AVX512);
TRANS_OUTPUTA_TH(float, 4, 3, 16, ISA_SKX_AVX512);
TRANS_OUTPUTA_BH(float, 4, 3, 16, ISA_SKX_AVX512);

} // namespace euler
