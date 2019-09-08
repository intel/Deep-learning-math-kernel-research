#ifndef __ELK_DEF_HPP__
#define __ELK_DEF_HPP__

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#define USE_AVX512 512
#define USE_AVX2   256

#define M1   (0)
#define M2   M1(1)
#define M3   M2(2)
#define M4   M3(3)
#define M5   M4(4)
#define M6   M5(5)
#define M7   M6(6)
#define M8   M7(7)
#define M9   M8(8)
#define M10  M9(9)
#define M11  M10(10)
#define M12  M11(11)
#define M13  M12(12)
#define M14  M13(13)
#define M15  M14(14)
#define M16  M15(15)
#define M17  M16(16)
#define M18  M17(17)
#define M19  M18(18)
#define M20  M19(19)
#define M21  M20(20)
#define M22  M21(21)
#define M23  M22(22)
#define M24  M23(23)
#define M25  M24(24)
#define M26  M25(25)
#define M27  M26(26)
#define M28  M27(27)
#define M29  M28(28)
#define M30  M29(29)
#define M31  M30(30)
#define M32  M31(31)
#define M33  M32(32)
#define M34  M33(33)
#define M35  M34(34)
#define M36  M35(35)

#define ME1     (0)
#define ME2  ME1(2)
#define ME3  ME2(4)
#define ME4  ME3(6)
#define ME5  ME4(8)

#define MO1     (1)
#define MO2  MO1(3)
#define MO3  MO2(5)
#define MO4  MO3(7)
#define MO5  MO4(9)

#define TUPLE(r, product) (BOOST_PP_SEQ_TO_TUPLE(product))

#define EXPAND_2D(_, d, seq)                                                   \
  OP(BOOST_PP_TUPLE_ELEM(2, 0, seq), BOOST_PP_TUPLE_ELEM(2, 1, seq));

#define MATRIX_DEF(m, n)                                                       \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
      EXPAND_2D, _, BOOST_PP_SEQ_FOR_EACH_PRODUCT(TUPLE, (M##m)(M##n)))

#define VECTOR_DEF(s1, s2)                                                     \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
      EXPAND_2D, _, BOOST_PP_SEQ_FOR_EACH_PRODUCT(TUPLE, (s1)(s2)))

#define EXPAND_2D_OP(_, op, seq)                                               \
  op(BOOST_PP_TUPLE_ELEM(2, 0, seq), BOOST_PP_TUPLE_ELEM(2, 1, seq));

#define MATRIX_OP(op, m, n)                                                    \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
      EXPAND_2D_OP, op, BOOST_PP_SEQ_FOR_EACH_PRODUCT(TUPLE, (M##m)(M##n)))

#define IMM_BCAST16(x) x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x

namespace euler {

// Transform kernel format
// C: compact
//    Input: A * A * V
//    Output: (A - K + 1) * (A - K + 1) * V
// D: blocked
//    Input: I2, ih, iw, V, Vx
//    Output: O1, O, oh, ow, V
// E: nchw
//    Input: I2, V, ih, iw
//    Output: O2, V, ih, iw
// F: nhwc
//    Input: ih, iw, I2, V
//    Output: oh, ow, O1, O, V
const int TKF_COMPACT = 0xC;
const int TKF_BLOCKED = 0xD;
const int TKF_NCHW = 0xE;
const int TKF_NHWC = 0xF;

// GEMM kernel format
// Input - weights - output
// B: compact b
//    Weights: O1, Ir, O, oV
// C: compact c
//    Input: I2, T, S, V, Vx
//    Weights: O1, I2, V, O, V, Vx
//    Output: O1, O, T, V
//    factor: O1, O, V
//    weights_scale: O1, O, V
// D: blocked
//    Input: I2, ih, iw, V, Vx
//    Weights: O1, O, ic2, V, V, Vx
//    Output: O1, O, oh, ow, V
//    factor: O1, O, V
//    weights_scale: O1, O, V
// E: nchw
//    Input: I2, V, ih, iw
// F: nhwc
//    Input: ih, iw, I2, V
//    Output: oh, ow, O1, O, V
const int GKF_CCC = 0xccc;
const int GKF_CCD = 0xccd;
const int GKF_DCD = 0xdcd;
const int GKF_DDD = 0xddd;
const int GKF_EBD = 0xebd;
const int GKF_FCF = 0xfcf;
const int GKF_FBD = 0xfbd;
const int GKF_FBF = 0xfbf;
const int GKF_DCF = 0xdcf;
const int GKF_FCD = 0xfcd;

}  // namespace euler

#endif // __ELK_DEF_HPP__
