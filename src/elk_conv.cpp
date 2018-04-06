#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

namespace euler {

#define IMM_BCAST16(x) x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x

template<typename F, const int T, const int K, const int V, const int I> void
elk_trans_weights(F atweights[T][T][V][V], F aweights[K][K][V][V]) { }

template<typename F, const int T, const int K, const int V, const int I> void
elk_trans_input(elx_conv_t<F> &xc, F atinput[T][T][V], F *input, int _oh2, int _ow2)
{ }

template<typename F, const int T, const int K, const int V, const int I> void
elk_product_trans_output(elx_conv_t<F> &xc, F *tinput, F *tweights, F *output,
                         int _ih2, int _iw2)
{ }

template<> void elk_trans_weights<float, 5, 3, 16, ISA_GENERIC>
(float atweights[5][5][16][16], float aweights[3][3][16][16])
{
    const float r12 = 1.0f / 12.0f;
    const float r6  = 1.0f / 6.0f;
    const float r3  = 1.0f / 3.0f;
    const float r4  = 1.0f / 4.0f;
    const float r2  = 1.0f / 2.0f;
    const float r2_3 = 2.0f / 3.0f;

    float C10[16], C11[16], C12[16],
          C20[16], C21[16], C22[16],
          C30[16], C31[16], C32[16];
#undef F
#undef T
#define F(h,w) aweights[h][w][_IV][_OV]
#define T(h,w) atweights[h][w][_IV][_OV]
#define C(c,n) C##c##n[_OV]
    for (int _IV = 0; _IV < 16; ++_IV) {
#pragma omp simd
        for (int _OV = 0; _OV < 16; ++_OV) {
            T(0,0) = r4 * F(0,0);
            T(1,0) = -r12 * (F(0,0) - F(1,0) + F(2,0));
            T(2,0) = -r4 * (F(0,0) + F(1,0) + F(2,0));
            T(3,0) = r12 * F(0,0) + r6 * F(1,0) + r3 * F(2,0);
            T(4,0) = r2 * F(2,0);

            C(1,0) = -r6 * (F(0,0) - F(0,1) + F(0,2));
            C(1,1) = -r6 * (F(1,0) - F(1,1) + F(1,2));
            C(1,2) = -r6 * (F(2,0) - F(2,1) + F(2,2));

            T(0,1) = r2 * C(1,0);
            T(1,1) = -r6 * (C(1,0) - C(1,1) + C(1,2));
            T(2,1) = -r2 * (C(1,0) + C(1,1) + C(1,2));
            T(3,1) = r6 * C(1,0) + r3 * C(1,1) + r2_3 * C(1,2);
            T(4,1) = C(1,2);

            C(2,0) = -r2 * (F(0,0) + F(0,1) + F(0,2));
            C(2,1) = -r2 * (F(1,0) + F(1,1) + F(1,2));
            C(2,2) = -r2 * (F(2,0) + F(2,1) + F(2,2));

            T(0,2) = r2 * C(2,0);
            T(1,2) = -r6 * (C(2,0) - C(2,1) + C(2,2));
            T(2,2) = -r2 * (C(2,0) + C(2,1) + C(2,2));
            T(3,2) = r6 * C(2,0) + r3 * C(2,1) + r2_3 * C(2,2);
            T(4,2) = C(2,2);

            C(3,0) = r6 * F(0,0) + r3 * F(0,1) + r2_3 * F(0,2);
            C(3,1) = r6 * F(1,0) + r3 * F(1,1) + r2_3 * F(1,2);
            C(3,2) = r6 * F(2,0) + r3 * F(2,1) + r2_3 * F(2,2);

            T(0,3) = r2 * C(3,0);
            T(1,3) = -r6 * (C(3,0) - C(3,1) + C(3,2));
            T(2,3) = -r2 * (C(3,0) + C(3,1) + C(3,2));
            T(3,3) = r6 * C(3,0) + r3 * C(3,1) + r2_3 * C(3,2);
            T(4,3) = C(3,2);

            T(0,4) = r2 * F(0,2);
            T(1,4) = -r6 * (F(0,2) - F(1,2) + F(2,2));
            T(2,4) = -r2 * (F(0,2) + F(1,2) + F(2,2));
            T(3,4) = r6 * F(0,2) + r3 * F(1,2) + r2_3 * F(2,2);
            T(4,4) = F(2,2);
        }
    }
}

template<> void elk_trans_weights<float, 5, 3, 16, ISA_SKX_AVX512>
(float atweights[5][5][16][16], float aweights[3][3][16][16])
{
    _allow_cpu_features(_FEATURE_AVX512F);

    // Constants
    __m512 r12  = _mm512_set_ps(IMM_BCAST16(1.0f  / 12.0f));
    __m512 r_12 = _mm512_set_ps(IMM_BCAST16(-1.0f / 12.0f));
    __m512 r6   = _mm512_set_ps(IMM_BCAST16(1.0f  / 6.0f));
    __m512 r_6  = _mm512_set_ps(IMM_BCAST16(-1.0f / 6.0f));
    __m512 r4   = _mm512_set_ps(IMM_BCAST16(1.0f  / 4.0f));
    __m512 r_4  = _mm512_set_ps(IMM_BCAST16(-1.0f / 4.0f));
    __m512 r3   = _mm512_set_ps(IMM_BCAST16(1.0f  / 3.0f));
    __m512 r2   = _mm512_set_ps(IMM_BCAST16(1.0f  / 2.0f));
    __m512 r_2  = _mm512_set_ps(IMM_BCAST16(-1.0f / 2.0f));
    __m512 r2_3 = _mm512_set_ps(IMM_BCAST16(2.0f  / 3.0f));

    // Inputs
    __m512 f00, f10, f20, f01, f11, f21, f02, f12, f22;
    // Cache
    __m512 c10, c11, c12, c20, c21, c22, c30, c31, c32;
    // Outputs
    __m512 t00, t10, t20, t30, t40,
           t01, t11, t21, t31 /*, t41 */,
           t02, t12, t22, t32 /*, t42 */,
           t03, t13, t23, t33 /*, t43 */,
           t04, t14, t24, t34 /*, t44 */;
#undef F
#undef T
#define F(h,w) aweights[h][w][_V]
#define T(h,w) atweights[h][w][_V]
    for (int _V = 0; _V < 16; ++_V) {
        f00 = _mm512_load_ps(F(0,0));
        f01 = _mm512_load_ps(F(0,1));
        f02 = _mm512_load_ps(F(0,2));
        f10 = _mm512_load_ps(F(1,0));
        f11 = _mm512_load_ps(F(1,1));
        f12 = _mm512_load_ps(F(1,2));
        f20 = _mm512_load_ps(F(2,0));
        f21 = _mm512_load_ps(F(2,1));
        f22 = _mm512_load_ps(F(2,2));

        c10 = _mm512_add_ps(f00, f02);
        c20 = _mm512_add_ps(c10, f01);
        c20 = _mm512_mul_ps(r_2, c20);
        c10 = _mm512_sub_ps(c10, f01);
        c10 = _mm512_mul_ps(r_6, c10);

        c11 = _mm512_add_ps(f10, f12);
        c21 = _mm512_add_ps(c11, f11);
        c21 = _mm512_mul_ps(r_2, c21);
        c11 = _mm512_sub_ps(c11, f11);
        c11 = _mm512_mul_ps(r_6, c11);

        c12 = _mm512_add_ps(f20, f22);
        c22 = _mm512_add_ps(c12, f21);
        c22 = _mm512_mul_ps(r_2, c22);
        c12 = _mm512_sub_ps(c12, f21);
        c12 = _mm512_mul_ps(r_6, c12);

        t00 = _mm512_mul_ps(r4, f00);
        _mm512_store_ps(T(0,0), t00);
        t10 = _mm512_add_ps(f00, f20);
        t20 = _mm512_add_ps(t10, f10);
        t10 = _mm512_sub_ps(t10, f10);
        t10 = _mm512_mul_ps(r_12, t10);
        _mm512_store_ps(T(1,0), t10);
        t20 = _mm512_mul_ps(r_4, t20);
        _mm512_store_ps(T(2,0), t20);
        t30 = _mm512_mul_ps(r12, f00);
        t30 = _mm512_fmadd_ps(r6, f10, t30);
        t30 = _mm512_fmadd_ps(r3, f20, t30);
        _mm512_store_ps(T(3,0), t30);
        t40 = _mm512_mul_ps(r2, f20);
        _mm512_store_ps(T(4,0), t40);

        t01 = _mm512_mul_ps(r2, c10);
        _mm512_store_ps(T(0,1), t01);
        t11 = _mm512_add_ps(c10, c12);
        t21 = _mm512_add_ps(t11, c11);
        t11 = _mm512_sub_ps(t11, c11);
        t11 = _mm512_mul_ps(r_6, t11);
        _mm512_store_ps(T(1,1), t11);
        t21 = _mm512_mul_ps(r_2, t21);
        _mm512_store_ps(T(2,1), t21);
        t31 = _mm512_mul_ps(r6, c10);
        t31 = _mm512_fmadd_ps(r3, c11, t31);
        t31 = _mm512_fmadd_ps(r2_3, c12, t31);
        _mm512_store_ps(T(3,1), t31);
        _mm512_store_ps(T(4,1), c12);

        t02 = _mm512_mul_ps(r2, c20);
        _mm512_store_ps(T(0,2), t02);
        t12 = _mm512_add_ps(c20, c22);
        t22 = _mm512_add_ps(t12, c21);
        t12 = _mm512_sub_ps(t12, c21);
        t12 = _mm512_mul_ps(r_6, t12);
        _mm512_store_ps(T(1,2), t12);
        t22 = _mm512_mul_ps(r_2, t22);
        _mm512_store_ps(T(2,2), t22);
        t32 = _mm512_mul_ps(r6, c20);
        t32 = _mm512_fmadd_ps(r3, c21, t32);
        t32 = _mm512_fmadd_ps(r2_3, c22, t32);
        _mm512_store_ps(T(3,2), t32);
        _mm512_store_ps(T(4,2), c22);

        c30 = _mm512_mul_ps(r6, f00);
        c30 = _mm512_fmadd_ps(r3, f01, c30);
        c30 = _mm512_fmadd_ps(r2_3, f02, c30);
        c31 = _mm512_mul_ps(r6, f10);
        c31 = _mm512_fmadd_ps(r3, f11, c31);
        c31 = _mm512_fmadd_ps(r2_3, f12, c31);
        c32 = _mm512_mul_ps(r6, f20);
        c32 = _mm512_fmadd_ps(r3, f21, c32);
        c32 = _mm512_fmadd_ps(r2_3, f22, c32);

        t03 = _mm512_mul_ps(r2, c30);
        _mm512_store_ps(T(0,3), t03);
        t13 = _mm512_add_ps(c30, c32);
        t23 = _mm512_add_ps(t13, c31);
        t13 = _mm512_sub_ps(t13, c31);
        t13 = _mm512_mul_ps(r_6, t13);
        _mm512_store_ps(T(1,3), t13);
        t23 = _mm512_mul_ps(r_2, t23);
        _mm512_store_ps(T(2,3), t23);
        t33 = _mm512_mul_ps(r6, c30);
        t33 = _mm512_fmadd_ps(r3, c31, t33);
        t33 = _mm512_fmadd_ps(r2_3, c32, t33);
        _mm512_store_ps(T(3,3), t33);
        _mm512_store_ps(T(4,3), c32);

        t04 = _mm512_mul_ps(r2, f02);
        _mm512_store_ps(T(0,4), t04);
        t14 = _mm512_add_ps(f02, f22);
        t24 = _mm512_add_ps(t14, f12);
        t14 = _mm512_sub_ps(t14, f12);
        t14 = _mm512_mul_ps(r_6, t14);
        _mm512_store_ps(T(1,4), t14);
        t24 = _mm512_mul_ps(r_2, t24);
        _mm512_store_ps(T(2,4), t24);
        t34 = _mm512_mul_ps(r6, f02);
        t34 = _mm512_fmadd_ps(r3, f12, t34);
        t34 = _mm512_fmadd_ps(r2_3, f22, t34);
        _mm512_store_ps(T(3,4), t34);
        _mm512_store_ps(T(4,4), f22);
    }
}

template<> void
elk_trans_input<float, 5, 3, 16, ISA_GENERIC>
(elx_conv_t<float> &xc, float atinput[5][5][16], float *input, int _oh2, int _ow2)
{
    const float z2  = 2.0f;
    const float z3  = 3.0f;
    const float z4  = 4.0f;
    const float z6  = 6.0f;

    auto f = [&](int _hT, int _wT, int _V) {
        int _ih = _oh2 * 3 - xc.lp + _hT;
        int _iw = _ow2 * 3 - xc.tp + _wT;
        int _i = _ih * xc.input_strides[2] + _iw * 16 + _V;
        if (_ih < 0 || _iw < 0 || _ih >= xc.ih || _iw >= xc.iw)
            return 0.0f;
        else
            return *(input + _i);
    };

#undef F
#undef C
#undef T
#define F(_hT, _wT) f(_hT, _wT, _V)
#define C(n) C##n[_V]
#define T(_hT, _wT) atinput[_hT][_wT][_V]

    float C1[16], C2[16], C3[16];
#pragma omp simd
    for (int _V = 0; _V < 16; ++_V) {
        C(1) = F(1,1) + z2 * F(1,2) - z2 * F(1,0) - F(1,3);
        C(2) = F(2,1) + z2 * F(2,2) - z2 * F(2,0) - F(2,3);
        C(3) = F(3,1) + z2 * F(3,2) - z2 * F(3,0) - F(3,3);
        T(0,0) = z4 * F(0,0) - z2 * F(0,1) - z4 * F(0,2) + z2 * F(0,3) + C(1) + z2 * C(2) - C(3);
        T(1,0) = z3 * C(2) - z2 * C(1) - C(3);
        T(2,0) = z2 * C(1) + C(2) - C(3);
        T(3,0) = C(1) - C(3);
        T(4,0) = z2 * F(4,0) - F(4,1) - z2 * F(4,2) + F(4,3) - z2 * C(1) + C(2) + z2 * C(3);

        C(1) = z3 * F(1,2) - z2 * F(1,1) - F(1,3);
        C(2) = z3 * F(2,2) - z2 * F(2,1) - F(2,3);
        C(3) = z3 * F(3,2) - z2 * F(3,1) - F(3,3);
        T(0,1) = z4 * F(0,1) - z6 * F(0,2) + z2 * F(0,3) + C(1) + z2 * C(2) - C(3);
        T(1,1) = z3 * C(2) - z2 * C(1) - C(3);
        T(2,1) = z2 * C(1) + C(2) - C(3);
        T(3,1) = C(1) - C(3);
        T(4,1) = z2 * F(4,1) - z3 * F(4,2) + F(4,3) - z2 * C(1) + C(2) + z2 * C(3);

        C(1) = z2 * F(1,1) + F(1,2) - F(1,3);
        C(2) = z2 * F(2,1) + F(2,2) - F(2,3);
        C(3) = z2 * F(3,1) + F(3,2) - F(3,3);
        T(0,2) = z2 * F(0,3) - z2 * F(0,2) - z4 * F(0,1) + C(1) + z2 * C(2) - C(3);
        T(1,2) = z3 * C(2) - z2 * C(1) - C(3);
        T(2,2) = z2 * C(1) + C(2) - C(3);
        T(3,2) = C(1) - C(3);
        T(4,2) = F(4,3) - z2 * F(4,1) - F(4,2) - z2 * C(1) + C(2) + z2 * C(3);

        C(1) = F(1,1) - F(1,3);
        C(2) = F(2,1) - F(2,3);
        C(3) = F(3,1) - F(3,3);
        T(0,3) = z2 * F(0,3) - z2 * F(0,1) + C(1) + z2 * C(2) - C(3);
        T(1,3) = z3 * C(2) - z2 * C(1) - C(3);
        T(2,3) = z2 * C(1) + C(2) - C(3);
        T(3,3) = C(1) - C(3);
        T(4,3) = F(4,3) - F(4,1) - z2 * C(1) + C(2) + z2 * C(3);

        C(1) = F(1,2) + z2 * F(1,3) - z2 * F(1,1) - F(1,4);
        C(2) = F(2,2) + z2 * F(2,3) - z2 * F(2,1) - F(2,4);
        C(3) = F(3,2) + z2 * F(3,3) - z2 * F(3,1) - F(3,4);
        T(0,4) = z4 * F(0,1) - z2 * F(0,2) - z4 * F(0,3)  + z2 * F(0,4) + C(1) + z2 * C(2) - C(3);
        T(1,4) = z3 * C(2) - z2 * C(1) - C(3);
        T(2,4) = z2 * C(1) + C(2) - C(3);
        T(3,4) = C(1) - C(3);
        T(4,4) = z2 * F(4,1) - F(4,2) - z2 * F(4,3) + F(4,4) - z2 * C(1) + C(2) + z2 * C(3);
    }
}

template<> void elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>
(elx_conv_t<float> &xc, float atinput[5][5][16], float *input, int _oh2, int _ow2)
{
    _allow_cpu_features(_FEATURE_AVX512F);
    // Inputs
    __m512 f00, f01, f02, f03, f04,
           f10, f11, f12, f13, f14,
           f20, f21, f22, f23, f24,
           f30, f31, f32, f33, f34,
           f40, f41, f42, f43, f44;
    // Cache
    __m512 c1, c2, c3;
    // Outputs
    __m512 t00, t01, t02, t03, t04,
           t10, t11, t12, t13, t14,
           t20, t21, t22, t23, t24,
           t30, t31, t32, t33, t34,
           t40, t41, t42, t43, t44;
    
    __m512 z0 = _mm512_setzero_ps();
    __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
    __m512 z3 = _mm512_set_ps(IMM_BCAST16(3.0f));
    __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
    __m512 z6 = _mm512_set_ps(IMM_BCAST16(6.0f));

    auto f0 = [&](int _hT, int _wT) {
        int _ih = _oh2 * 3 - 1 + _hT;
        int _iw = _ow2 * 3 - 1 + _wT;
        int _i = _ih * xc.input_strides[2] + _iw * 16;
        if (_ih < 0 || _iw < 0 || _ih >= xc.ih || _iw >= xc.iw)
            return z0;
        else
            return _mm512_load_ps(input + _i);
    };
    auto f1 = [&](int _hT, int _wT) {
        int _ih = _oh2 * 3 - 1 + _hT;
        int _iw = _ow2 * 3 - 1 + _wT;
        int _i = _ih * xc.input_strides[2] + _iw * 16;
        return _mm512_load_ps(input + _i);
    };

#undef F
#undef C
#undef T
#define F0(_hT, _wT) f0(_hT, _wT)
#define F1(_hT, _wT) f1(_hT, _wT)
#define T(h,w) atinput[h][w]

    if (_oh2 == 0 || _ow2 == 0 || _oh2 == xc.oh2 - 1 || _ow2 == xc.ow2 - 1) {
        f00 = F0(0,0);
        f01 = F0(0,1);
        f02 = F0(0,2);
        f03 = F0(0,3);
        f10 = F0(1,0);
        f11 = F0(1,1);
        f12 = F0(1,2);
        f13 = F0(1,3);
        f20 = F0(2,0);
        f21 = F0(2,1);
        f22 = F0(2,2);
        f23 = F0(2,3);
        f30 = F0(3,0);
        f31 = F0(3,1);
        f32 = F0(3,2);
        f33 = F0(3,3);
        f40 = F0(4,0);
        f41 = F0(4,1);
        f42 = F0(4,2);
        f43 = F0(4,3);
    } else {
        f00 = F1(0,0);
        f01 = F1(0,1);
        f02 = F1(0,2);
        f03 = F1(0,3);
        f10 = F1(1,0);
        f11 = F1(1,1);
        f12 = F1(1,2);
        f13 = F1(1,3);
        f20 = F1(2,0);
        f21 = F1(2,1);
        f22 = F1(2,2);
        f23 = F1(2,3);
        f30 = F1(3,0);
        f31 = F1(3,1);
        f32 = F1(3,2);
        f33 = F1(3,3);
        f40 = F1(4,0);
        f41 = F1(4,1);
        f42 = F1(4,2);
        f43 = F1(4,3);
    }

    c1 = _mm512_sub_ps(f11, f13);
    c1 = _mm512_fmadd_ps(z2, f12, c1);
    c1 = _mm512_fmsub_ps(z2, f10, c1);
    c2 = _mm512_sub_ps(f21, f23);
    c2 = _mm512_fmadd_ps(z2, f22, c2);
    c2 = _mm512_fmsub_ps(z2, f20, c2);
    c3 = _mm512_sub_ps(f31, f33);
    c3 = _mm512_fmadd_ps(z2, f32, c3);
    c3 = _mm512_fmsub_ps(z2, f30, c3);
    t00 = _mm512_sub_ps(c1, c3);
    t00 = _mm512_fmadd_ps(z4, f00, t00);
    t00 = _mm512_fmsub_ps(z2, f01, t00);
    t00 = _mm512_fmsub_ps(z4, f02, t00);
    t00 = _mm512_fmadd_ps(z2, f03, t00);
    t00 = _mm512_fmadd_ps(z2, c2, t00);
    _mm512_store_ps(T(0,0), t00);
    t10 = _mm512_mul_ps(z3, c2);
    t10 = _mm512_fmsub_ps(z2, c1, t10);
    t10 = _mm512_sub_ps(t10, c3);
    _mm512_store_ps(T(1,0), t10);
    t20 = _mm512_sub_ps(c2, c3);
    t20 = _mm512_fmadd_ps(z2, c1, t20);
    _mm512_store_ps(T(2,0), t20);
    t30 = _mm512_sub_ps(c1, c3);
    _mm512_store_ps(T(3,0), t30);
    t40 = _mm512_sub_ps(f43, f41);
    t40 = _mm512_fmadd_ps(z2, f40, t40);
    t40 = _mm512_fmsub_ps(z2, f42, t40);
    t40 = _mm512_fmsub_ps(z2, c1, t40);
    t40 = _mm512_fmadd_ps(z2, c3, t40);
    t40 = _mm512_add_ps(t40, c2);
    _mm512_store_ps(T(4,0), t40);

    c1 = _mm512_mul_ps(z3, f12);
    c1 = _mm512_fmsub_ps(z2, f11, c1);
    c1 = _mm512_sub_ps(c1, f13);
    c2 = _mm512_mul_ps(z3, f22);
    c2 = _mm512_fmsub_ps(z2, f21, c2);
    c2 = _mm512_sub_ps(c2, f23);
    c3 = _mm512_mul_ps(z3, f32);
    c3 = _mm512_fmsub_ps(z2, f31, c3);
    c3 = _mm512_sub_ps(c3, f33);
    t01 = _mm512_sub_ps(c1, c3);
    t01 = _mm512_fmadd_ps(z4, f01, t01);
    t01 = _mm512_fmsub_ps(z6, f02, t01);
    t01 = _mm512_fmadd_ps(z2, f03, t01);
    t01 = _mm512_fmadd_ps(z2, c2, t01);
    _mm512_store_ps(T(0,1), t01);
    t11 = _mm512_mul_ps(z3, c2);
    t11 = _mm512_fmsub_ps(z2, c1, t11);
    t11 = _mm512_sub_ps(t11, c3);
    _mm512_store_ps(T(1,1), t11);
    t21 = _mm512_sub_ps(c2, c3);
    t21 = _mm512_fmadd_ps(z2, c1, t21);
    _mm512_store_ps(T(2,1), t21);
    t31 = _mm512_sub_ps(c1, c3);
    _mm512_store_ps(T(3,1), t31);
    t41 = _mm512_add_ps(f43, c2);
    t41 = _mm512_fmadd_ps(z2, f41, t41);
    t41 = _mm512_fmsub_ps(z3, f42, t41);
    t41 = _mm512_fmsub_ps(z2, c1, t41);
    t41 = _mm512_fmadd_ps(z2, c3, t41);
    _mm512_store_ps(T(4,1), t41);

    c1 = _mm512_sub_ps(f12, f13);
    c1 = _mm512_fmadd_ps(z2, f11, c1);
    c2 = _mm512_sub_ps(f22, f23);
    c2 = _mm512_fmadd_ps(z2, f21, c2);
    c3 = _mm512_sub_ps(f32, f33);
    c3 = _mm512_fmadd_ps(z2, f31, c3);
    t02 = _mm512_sub_ps(c1, c3);
    t02 = _mm512_fmadd_ps(z2, f03, t02);
    t02 = _mm512_fmsub_ps(z2, f02, t02);
    t02 = _mm512_fmsub_ps(z4, f01, t02);
    t02 = _mm512_fmadd_ps(z2, c2, t02);
    _mm512_store_ps(T(0,2), t02);
    t12 = _mm512_mul_ps(z3, c2);
    t12 = _mm512_fmsub_ps(z2, c1, t12);
    t12 = _mm512_sub_ps(t12, c3);
    _mm512_store_ps(T(1,2), t12);
    t22 = _mm512_sub_ps(c2, c3);
    t22 = _mm512_fmadd_ps(z2, c1, t22);
    _mm512_store_ps(T(2,2), t22);
    t32 = _mm512_sub_ps(c1, c3);
    _mm512_store_ps(T(3,2), t32);
    t42 = _mm512_sub_ps(f43, f42);
    t42 = _mm512_fmsub_ps(z2, f41, t42);
    t42 = _mm512_fmsub_ps(z2, c1, t42);
    t42 = _mm512_fmadd_ps(z2, c3, t42);
    t42 = _mm512_add_ps(t42, c2);
    _mm512_store_ps(T(4,2), t42);

    c1 = _mm512_sub_ps(f11, f13);
    c2 = _mm512_sub_ps(f21, f23);
    c3 = _mm512_sub_ps(f31, f33);
    t03 = _mm512_sub_ps(c1, c3);
    t03 = _mm512_fmadd_ps(z2, f03, t03);
    t03 = _mm512_fmsub_ps(z2, f01, t03);
    t03 = _mm512_fmadd_ps(z2, c2, t03);
    _mm512_store_ps(T(0,3), t03);
    t13 = _mm512_mul_ps(z3, c2);
    t13 = _mm512_fmsub_ps(z2, c1, t13);
    t13 = _mm512_sub_ps(t13, c3);
    _mm512_store_ps(T(1,3), t13);
    t23 = _mm512_sub_ps(c2, c3);
    t23 = _mm512_fmadd_ps(z2, c1, t23);
    _mm512_store_ps(T(2,3), t23);
    t33 = _mm512_sub_ps(c1, c3);
    _mm512_store_ps(T(3,3), t33);
    t43 = _mm512_sub_ps(f43, f41);
    t33 = _mm512_mul_ps(z2, t33);
    t43 = _mm512_add_ps(t43, t33);
    t43 = _mm512_add_ps(t43, c2);
    _mm512_store_ps(T(4,3), t43);
    
    if (_oh2 == 0 || _ow2 == 0 || _oh2 == xc.oh2 - 1 || _ow2 == xc.ow2 - 1) {
        f04 = F0(0,4);
        f14 = F0(1,4);
        f24 = F0(2,4);
        f34 = F0(3,4);
        f44 = F0(4,4);
    } else {
        f04 = F1(0,4);
        f14 = F1(1,4);
        f24 = F1(2,4);
        f34 = F1(3,4);
        f44 = F1(4,4);
    }
    c1 = _mm512_sub_ps(f12, f14);
    c1 = _mm512_fmadd_ps(z2, f13, c1);
    c1 = _mm512_fmsub_ps(z2, f11, c1);
    c2 = _mm512_sub_ps(f22, f24);
    c2 = _mm512_fmadd_ps(z2, f23, c2);
    c2 = _mm512_fmsub_ps(z2, f21, c2);
    c3 = _mm512_sub_ps(f32, f34);
    c3 = _mm512_fmadd_ps(z2, f33, c3);
    c3 = _mm512_fmsub_ps(z2, f31, c3);
    t34 = _mm512_sub_ps(c1, c3);
    _mm512_store_ps(T(3,4), t34);
    t04 = _mm512_sub_ps(f01, f03);
    t04 = _mm512_mul_ps(z4, t04);
    t04 = _mm512_fmsub_ps(z2, f02, t04);
    t04 = _mm512_fmadd_ps(z2, f04, t04);
    t04 = _mm512_fmadd_ps(z2, c2, t04);
    t04 = _mm512_add_ps(t04, t34);
    _mm512_store_ps(T(0,4), t04);
    t44 = _mm512_sub_ps(f41, f43);
    t44 = _mm512_mul_ps(z2, t44);
    t44 = _mm512_sub_ps(t44, f42);
    t44 = _mm512_add_ps(t44, f44);
    t34 = _mm512_mul_ps(z2, t34);
    t44 = _mm512_sub_ps(t44, t34);
    t44 = _mm512_add_ps(t44, c2);
    _mm512_store_ps(T(4,4), t44);
    t14 = _mm512_mul_ps(z3, c2);
    t14 = _mm512_fmsub_ps(z2, c1, t14);
    t14 = _mm512_sub_ps(t14, c3);
    _mm512_store_ps(T(1,4), t14);
    t24 = _mm512_sub_ps(c2, c3);
    t24 = _mm512_fmadd_ps(z2, c1, t24);
    _mm512_store_ps(T(2,4), t24);

}

template<> void elk_product_trans_output<float, 5, 3, 16, ISA_GENERIC>
(elx_conv_t<float> &xc, float *tinput, float *tweights, float *output,
 int _ih2, int _iw2)
{
#undef T
#undef F
#undef W
#define T(h,w) t##h##w[_OV]
#define F(h,w) atinput[_ic2][_ih2][_iw2][h][w][_IV]
#define W(h,w) atweights[_ic2][h][w][_IV][_OV]

    float t00[16], t01[16], t02[16], t03[16], t04[16],
          t10[16], t11[16], t12[16], t13[16], t14[16],
          t20[16], t21[16], t22[16], t23[16], t24[16],
          t30[16], t31[16], t32[16], t33[16], t34[16],
          t40[16], t41[16], t42[16], t43[16], t44[16];

    MD(float, atinput,   [xc.ic2][xc.oh2][xc.ow2][5][5][16], tinput);
    MD(float, atweights, [xc.ic2][5][5][16][16], tweights);
    MD(float, aoutput,   [xc.oh][xc.ow][16], output);

#pragma omp simd
    for (int _OV = 0; _OV < 16; ++_OV) {
        T(0,0) = 0.0f;
        T(0,1) = 0.0f;
        T(0,2) = 0.0f;
        T(0,3) = 0.0f;
        T(0,4) = 0.0f;
        T(1,0) = 0.0f;
        T(1,1) = 0.0f;
        T(1,2) = 0.0f;
        T(1,3) = 0.0f;
        T(1,4) = 0.0f;
        T(2,0) = 0.0f;
        T(2,1) = 0.0f;
        T(2,2) = 0.0f;
        T(2,3) = 0.0f;
        T(2,4) = 0.0f;
        T(3,0) = 0.0f;
        T(3,1) = 0.0f;
        T(3,2) = 0.0f;
        T(3,3) = 0.0f;
        T(3,4) = 0.0f;
        T(4,0) = 0.0f;
        T(4,1) = 0.0f;
        T(4,2) = 0.0f;
        T(4,3) = 0.0f;
        T(4,4) = 0.0f;

    }
    for (int _ic2 = 0; _ic2 < xc.ic2; ++_ic2) {
        for (int _IV = 0; _IV < 16; ++_IV) {
#pragma omp simd
            for (int _OV = 0; _OV < 16; ++_OV) {
                T(0,0) += W(0,0) * F(0,0);
                T(0,1) += W(0,1) * F(0,1);
                T(0,2) += W(0,2) * F(0,2);
                T(0,3) += W(0,3) * F(0,3);
                T(0,4) += W(0,4) * F(0,4);
                T(1,0) += W(1,0) * F(1,0);
                T(1,1) += W(1,1) * F(1,1);
                T(1,2) += W(1,2) * F(1,2);
                T(1,3) += W(1,3) * F(1,3);
                T(1,4) += W(1,4) * F(1,4);
                T(2,0) += W(2,0) * F(2,0);
                T(2,1) += W(2,1) * F(2,1);
                T(2,2) += W(2,2) * F(2,2);
                T(2,3) += W(2,3) * F(2,3);
                T(2,4) += W(2,4) * F(2,4);
                T(3,0) += W(3,0) * F(3,0);
                T(3,1) += W(3,1) * F(3,1);
                T(3,2) += W(3,2) * F(3,2);
                T(3,3) += W(3,3) * F(3,3);
                T(3,4) += W(3,4) * F(3,4);
                T(4,0) += W(4,0) * F(4,0);
                T(4,1) += W(4,1) * F(4,1);
                T(4,2) += W(4,2) * F(4,2);
                T(4,3) += W(4,3) * F(4,3);
                T(4,4) += W(4,4) * F(4,4);
            }
        } 
    }

#undef C
#define C(n) c##n[_OV]
#define P(_hT, _wT) aoutput[xc.oh2 * 3 + _hT][xc.ow2 * 3 + _wT][_OV] // TODO: overflow
    float c0[16], c1[16], c2[16], c3[16], c4[16];

#pragma omp simd
    for (int _OV = 0; _OV < 16; ++_OV) {
        C(0) = T(0,0) + T(0,1) + T(0,2) + T(0,3);
        C(1) = T(1,0) + T(1,1) + T(1,2) + T(1,3);
        C(2) = T(2,0) + T(2,1) + T(2,2) + T(2,3);
        C(3) = T(3,0) + T(3,1) + T(3,2) + T(3,3);
        C(4) = T(4,0) + T(4,1) + T(4,2) + T(4,3);
        P(0,0) = C(0) + C(1) + C(2) + C(3);
        P(1,0) = C(2) - C(1) + 2 * C(3);
        P(2,0) = C(1) + C(2) + 4 * C(3) + C(4);

        C(0) = T(0,2) - T(0,1) + 2 * T(0,3);
        C(1) = T(1,2) - T(1,1) + 2 * T(1,3);
        C(2) = T(2,2) - T(2,1) + 2 * T(2,3);
        C(3) = T(3,2) - T(3,1) + 2 * T(3,3);
        C(4) = T(4,2) - T(4,1) + 2 * T(4,3);
        P(0,1) = C(0) + C(1) + C(2) + C(3);
        P(1,1) = C(2) - C(1) + 2 * C(3);
        P(2,1) = C(1) + C(2) + 4 * C(3) + C(4);

        C(0) = T(0,1) + T(0,2) + 4 * T(0,3) + T(0,4);
        C(1) = T(1,1) + T(1,2) + 4 * T(1,3) + T(1,4);
        C(2) = T(2,1) + T(2,2) + 4 * T(2,3) + T(2,4);
        C(3) = T(3,1) + T(3,2) + 4 * T(3,3) + T(3,4);
        C(4) = T(4,1) + T(4,2) + 4 * T(4,3) + T(4,4);
        P(0,2) = C(0) + C(1) + C(2) + C(3);
        P(1,2) = C(2) - C(1) + 2 * C(3);
        P(2,2) = C(1) + C(2) + 4 * C(3) + C(4);
    }
}

template<> void elk_product_trans_output<float, 5, 3, 16, ISA_SKX_AVX512>
(elx_conv_t<float> &xc, float *tinput, float *tweights, float *output,
 int _ih2, int _iw2)
{
    _allow_cpu_features(_FEATURE_AVX512F);

    __m512 t00, t10, t20, t30, t40,
           t01, t11, t21, t31, t41,
           t02, t12, t22, t32, t42,
           t03, t13, t23, t33, t43,
           t04, t14, t24, t34, t44;

    MD(float, atinput,   [xc.ic2][xc.oh2][xc.ow2][5][5][16], tinput);
    MD(float, atweights, [xc.ic2][5][5][16][16], tweights);
    MD(float, aoutput,   [xc.oh][xc.ow][16], output);

#undef T
#undef F
#undef W
#define T(_h,_w) t##_h##_w
#define F(_h,_w) atinput[_ic2][_ih2][_iw2][_h][_w][_V]
#define W(_h,_w) atweights[_ic2][_h][_w][_V]

#define FMA(_h,_w) do { \
    __m512 f##_h##_w = _mm512_set1_ps(F(_h,_w)); \
    __m512 w##_h##_w = _mm512_load_ps(W(_h,_w)); \
    t##_h##_w = _mm512_fmadd_ps(w##_h##_w, f##_h##_w, t##_h##_w); \
} while(0)
#define STORE(_h,_w) do { \
    _mm512_store_ps(T(_h,_w), t##_h##_w); \
} while(0)

    t00 = _mm512_setzero_ps();
    t01 = _mm512_setzero_ps();
    t02 = _mm512_setzero_ps();
    t03 = _mm512_setzero_ps();
    t04 = _mm512_setzero_ps();
    t10 = _mm512_setzero_ps();
    t11 = _mm512_setzero_ps();
    t12 = _mm512_setzero_ps();
    t13 = _mm512_setzero_ps();
    t14 = _mm512_setzero_ps();
    t20 = _mm512_setzero_ps();
    t21 = _mm512_setzero_ps();
    t22 = _mm512_setzero_ps();
    t23 = _mm512_setzero_ps();
    t24 = _mm512_setzero_ps();
    t30 = _mm512_setzero_ps();
    t31 = _mm512_setzero_ps();
    t32 = _mm512_setzero_ps();
    t33 = _mm512_setzero_ps();
    t34 = _mm512_setzero_ps();
    t40 = _mm512_setzero_ps();
    t41 = _mm512_setzero_ps();
    t42 = _mm512_setzero_ps();
    t43 = _mm512_setzero_ps();
    t44 = _mm512_setzero_ps();

    for (int _ic2 = 0; _ic2 < xc.ic2; ++_ic2) {
        for (int _V = 0; _V < 16; ++_V) {
            //__m512 w00 = _mm512_load_ps(W(0,0));
            //__m512 f00 = _mm512_set1_ps(F(0,0));
            //t00 = _mm512_fmadd_ps(w00, f00, t00);
            FMA(0,0);
            FMA(0,1);
            FMA(0,2);
            FMA(0,3);
            FMA(0,4);
            FMA(1,0);
            FMA(1,1);
            FMA(1,2);
            FMA(1,3);
            FMA(1,4);
            FMA(2,0);
            FMA(2,1);
            FMA(2,2);
            FMA(2,3);
            FMA(2,4);
            FMA(3,0);
            FMA(3,1);
            FMA(3,2);
            FMA(3,3);
            FMA(3,4);
            FMA(4,0);
            FMA(4,1);
            FMA(4,2);
            FMA(4,3);
            FMA(4,4);
        }
    }

#undef P
#define P(_hT, _wT) (aoutput[xc.oh2 * 3 + _hT][xc.ow2 * 3 + _wT]) // TODO: overflow
    __m512 c0, c1, c2, c3, c4;
    __m512 p00, p01, p02, p10, p11, p12, p20, p21, p22;
    __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
    __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

    c0 = _mm512_add_ps(t00, t01);
    c0 = _mm512_add_ps(c0, t02);
    c0 = _mm512_add_ps(c0, t03);
    c1 = _mm512_add_ps(t10, t11);
    c1 = _mm512_add_ps(c1, t12);
    c1 = _mm512_add_ps(c1, t13);
    c2 = _mm512_add_ps(t20, t21);
    c2 = _mm512_add_ps(c2, t22);
    c2 = _mm512_add_ps(c2, t23);
    c3 = _mm512_add_ps(t30, t31);
    c3 = _mm512_add_ps(c3, t32);
    c3 = _mm512_add_ps(c3, t33);
    c4 = _mm512_add_ps(t40, t41);
    c4 = _mm512_add_ps(c4, t42);
    c4 = _mm512_add_ps(c4, t43);

    p00 = _mm512_add_ps(c0, c1);
    p00 = _mm512_add_ps(p00, c2);
    p00 = _mm512_add_ps(p00, c3);
    _mm512_store_ps(P(0,0), p00); //
    p10 = _mm512_sub_ps(c2, c1);
    p10 = _mm512_fmadd_ps(z2, c3, p10);
    _mm512_store_ps(P(1,0), p10); //
    p20 = _mm512_add_ps(c0, c1);
    p20 = _mm512_add_ps(p20, c4);
    p20 = _mm512_fmadd_ps(z4, c3, p20);
    _mm512_store_ps(P(2,0), p20); //

    c0 = _mm512_sub_ps(t01, t02);
    c0 = _mm512_fmadd_ps(z2, t03, c0);
    c1 = _mm512_sub_ps(t11, t12);
    c1 = _mm512_fmadd_ps(z2, t13, c1);
    c2 = _mm512_sub_ps(t21, t22);
    c2 = _mm512_fmadd_ps(z2, t23, c2);
    c3 = _mm512_sub_ps(t31, t32);
    c3 = _mm512_fmadd_ps(z2, t33, c3);
    c4 = _mm512_sub_ps(t41, t42);
    c4 = _mm512_fmadd_ps(z2, t43, c4);
    p01 = _mm512_add_ps(c0, c1);
    p01 = _mm512_add_ps(p01, c2);
    p01 = _mm512_add_ps(p01, c3);
    _mm512_store_ps(P(0,1), p01); //
    p11 = _mm512_sub_ps(c2, c1);
    p11 = _mm512_fmadd_ps(z2, c3, p11);
    _mm512_store_ps(P(1,1), p11); //
    p21 = _mm512_add_ps(c1, c2);
    p21 = _mm512_add_ps(p21, c4);
    p21 = _mm512_fmadd_ps(z4, c3, p21);
    _mm512_store_ps(P(2,1), p21); //

    c0 = _mm512_add_ps(t01, t02);
    c0 = _mm512_add_ps(c0, t04);
    c0 = _mm512_fmadd_ps(z4, t03, c0);
    c1 = _mm512_add_ps(t11, t12);
    c1 = _mm512_add_ps(c1, t14);
    c1 = _mm512_fmadd_ps(z4, t13, c1);
    c2 = _mm512_add_ps(t21, t22);
    c2 = _mm512_add_ps(c2, t24);
    c2 = _mm512_fmadd_ps(z4, t23, c2);
    c3 = _mm512_add_ps(t31, t32);
    c3 = _mm512_add_ps(c3, t34);
    c3 = _mm512_fmadd_ps(z4, t33, c3);
    c4 = _mm512_add_ps(t41, t42);
    c4 = _mm512_add_ps(c4, t44);
    c4 = _mm512_fmadd_ps(z4, t43, c4);
    p02 = _mm512_add_ps(c0, c1);
    p02 = _mm512_add_ps(p02, c2);
    p02 = _mm512_add_ps(p02, c3);
    _mm512_store_ps(P(0,2), p02); //
    p12 = _mm512_sub_ps(c2, c1);
    p12 = _mm512_fmadd_ps(z2, c3, p12);
    _mm512_store_ps(P(1,2), p12); //
    p22 = _mm512_add_ps(c1, c2);
    p22 = _mm512_add_ps(p22, c4);
    p22 = _mm512_fmadd_ps(z4, c3, p22);
    _mm512_store_ps(P(2,2), p22); //
}



}
