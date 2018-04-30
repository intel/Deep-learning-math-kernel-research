#ifndef __ELK_CONV_HPP__
#define __ELK_CONV_HPP__

#include "el_def.hpp"

#define __E()
#define __DEFER(x) x __E()
#define __EXPAND(...) __VA_ARGS__
#define __OP(x) __EXPAND(__DEFER(OP)(x))

#define OP_0_to_1()  __OP(0);      __OP(1)
#define OP_0_to_2()  OP_0_to_1();  __OP(2)
#define OP_0_to_3()  OP_0_to_2();  __OP(3)
#define OP_0_to_4()  OP_0_to_3();  __OP(4)
#define OP_0_to_5()  OP_0_to_4();  __OP(5)
#define OP_0_to_6()  OP_0_to_5();  __OP(6)
#define OP_0_to_7()  OP_0_to_6();  __OP(7)
#define OP_0_to_8()  OP_0_to_7();  __OP(8)
#define OP_0_to_9()  OP_0_to_8();  __OP(9)
#define OP_0_to_10() OP_0_to_9();  __OP(10)
#define OP_0_to_11() OP_0_to_10(); __OP(11)
#define OP_0_to_12() OP_0_to_11(); __OP(12)
#define OP_0_to_13() OP_0_to_12(); __OP(13)
#define OP_0_to_14() OP_0_to_13(); __OP(14)
#define OP_0_to_15() OP_0_to_14(); __OP(15)
#define OP_0_to_16() OP_0_to_15(); __OP(16)
#define OP_0_to_17() OP_0_to_16(); __OP(17)
#define OP_0_to_18() OP_0_to_17(); __OP(18)
#define OP_0_to_19() OP_0_to_18(); __OP(19)
#define OP_0_to_20() OP_0_to_19(); __OP(20)
#define OP_0_to_21() OP_0_to_20(); __OP(21)
#define OP_0_to_22() OP_0_to_21(); __OP(22)
#define OP_0_to_23() OP_0_to_22(); __OP(23)
#define OP_0_to_24() OP_0_to_23(); __OP(24)
#define OP_0_to_25() OP_0_to_24(); __OP(25)
#define OP_0_to_26() OP_0_to_25(); __OP(26)
#define OP_0_to_27() OP_0_to_26(); __OP(27)
#define OP_0_to_28() OP_0_to_27(); __OP(28)
#define OP_0_to_29() OP_0_to_28(); __OP(29)
#define OP_0_to_30() OP_0_to_29(); __OP(30)
#define OP_0_to_31() OP_0_to_30(); __OP(31)
#define OP_0_to_32() OP_0_to_31(); __OP(32)

namespace euler {

template <typename T, const int A, const int K, const int V, const int I>
void elk_trans_weights(T atweights[A][A][V][V], T aweights[K][K][V][V]);

template <typename T, const int A, const int K, const int V, const int I>
void elk_trans_input(elx_conv_t<T> &xc, T atinput[A][A][V], T *input,
                     bool margin);

template <typename Type, int A, const int K, int V, int I>
void elk_trans_output(elx_conv_t<Type> &xc, Type *output,
                      Type atoutput[A][A][V], bool margin);

template <typename T, const int A, const int K, const int V, const int I>
void elk_product_trans_output(elx_conv_t<T> &xc, T *tinput, T *tweights,
                              T *output, int _ih2, int _iw2);

// Type: data type;
// T: tile blocking unit;
// V: vector size
// I: ISA

template <typename Type, int T, int V, int I>
void elk_gemm(elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
              bool zero_out);

}  // namespace euler

#endif // __ELK_CONV_HPP__
