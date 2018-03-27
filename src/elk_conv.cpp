#include <stdio.h>
#include <assert.h>
#include <x86intrin.h>
#include "euler.hpp"
#include "el_utils.hpp"

namespace euler {


#define pragma_unroll _Pragma("unroll")

void elk_trans_weights_ref(float Fw_[5][5][16][16], float F[3][3][16][16]) // F(3x3, 3x3)
{
    const float r1_2 = 1.0f / 2.0f;
    const float r1_3 = 1.0f / 3.0f;
    const float r1_6 = 1.0f / 6.0f;
    const float r2_3 = 2.0f / 3.0f;

    float Fw[5][16];
    float T[5][3][16];
    float t0[16];
    float t1[16];

#undef E
#undef R
#define E(x) F[x][i][j][k]
#define R(x) T[x][i][k]
    for (int j = 0; j < 16; j++) {
pragma_unroll
        for (int i = 0; i < 3; i++) {
#pragma omp simd
            for (int k = 0; k < 16; k++) {
                R(0) = r1_2 * E(0);
                R(1) = -r1_6 * E(0) + r1_6 * E(1)-r1_6 * E(2);
                R(2) = -r1_2 * E(0)-r1_2 * E(1)-r1_2 * E(2);
                R(3) = r1_6 * E(0) + r1_3 * E(1) + r2_3 * E(2);
                R(4) = E(2);
            }
        }
#undef E
#undef R
#define E(x) T[i][x][k]
#define R(x) Fw[x][k]
pragma_unroll
        for (int i = 0; i < 5; i++) {
#pragma omp simd
            for (int k = 0; k < 16; k++) {
                R(0) = r1_2 * E(0);
                R(1) = -r1_6 * E(0) + r1_6 * E(1)-r1_6 * E(2);
                R(2) = -r1_2 * E(0)-r1_2 * E(1)-r1_2 * E(2);
                R(3) = r1_6 * E(0) + r1_3 * E(1) + r2_3 * E(2);
                R(4) = E(2);

pragma_unroll
                for (int l = 0; l < 5; l++) {
                    Fw_[i][l][j][k] = Fw[l][k];
                }
            }
        }
    }
}

__declspec(cpu_dispatch(generic, skylake_avx512))
void elk_trans_weights(float to[5][5][16][16], float from[3][3][16][16]) { }

#define IMM_BCAST16(x) x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x
__declspec(cpu_specific(skylake_avx512))
void elk_trans_weights(float to[5][5][16][16], float from[3][3][16][16])
{
    // Constants
    __m512 r12  = _mm512_set_ps(IMM_BCAST16(1.0f  / 12.0f));
    __m512 r_12 = _mm512_set_ps(IMM_BCAST16(-1.0f / 12.0f));
    __m512 r6   = _mm512_set_ps(IMM_BCAST16(1.0f  / 6.0f));
    __m512 r_6  = _mm512_set_ps(IMM_BCAST16(-1.0f / 6.0f));
    __m512 r4   = _mm512_set_ps(IMM_BCAST16(1.0f  / 4.0f));
    __m512 r_4  = _mm512_set_ps(IMM_BCAST16(-1.0f / 4.0f));
    __m512 r3   = _mm512_set_ps(IMM_BCAST16(1.0f  / 3.0f));
    __m512 r_3  = _mm512_set_ps(IMM_BCAST16(-1.0f / 3.0f));
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
#define F(h,w) from[h][w][ic]
#define T(h,w) to[h][w][ic]
    for (int ic = 0; ic < 16; ++ic) {
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

__declspec(cpu_specific(generic))
void elk_trans_weights(float to[5][5][16][16], float from[3][3][16][16])
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
#define F(h,w) from[h][w][ic][oc]
#define T(h,w) to[h][w][ic][oc]
#define C(c,n) C##c##n[oc]
    for (int ic = 0; ic < 16; ++ic) {
#pragma omp simd
        for (int oc = 0; oc < 16; ++oc) {
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


template<typename F, const int T> int
elk_trans_input(elx_conv_t &xc, F tr_input[T][T][16], F *input, int oh, int ow)
{
    return 0;
}

template<> int
elk_trans_input<float, 5>(elx_conv_t &xc, float tr_input[5][5][16], float *input, int oh, int ow)
{
    return 0;
}

}

