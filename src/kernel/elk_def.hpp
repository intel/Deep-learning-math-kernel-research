#ifndef __ELK_DEF_HPP__
#define __ELK_DEF_HPP__

#define USE_AVX512 512
#define USE_AVX2   256

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
