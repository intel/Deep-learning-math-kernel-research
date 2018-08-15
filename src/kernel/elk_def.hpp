#ifndef __ELK_DEF_HPP__
#define __ELK_DEF_HPP__

#define USE_AVX512 512
#define USE_AVX2   256

#define M1    (0)
#define M2  M1(1)
#define M3  M2(2)
#define M4  M3(3)
#define M5  M4(4)
#define M6  M5(5)
#define M7  M6(6)
#define M8  M7(7)
#define M9  M8(8)
#define M10 M9(9)

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

#define IMM_BCAST16(x) x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x

// TODO: avx2/sse
#define ADD     _mm512_add_ps
#define SUB     _mm512_sub_ps
#define MUL     _mm512_mul_ps
#define FMADD   _mm512_fmadd_ps
#define FMSUB   _mm512_fmsub_ps
#define FNMADD  _mm512_fnmadd_ps
#define FNMSUB  _mm512_fnmsub_ps
#define MAX     _mm512_max_ps
#define XOR     _mm512_xor_ps

#endif // __ELK_DEF_HPP__
