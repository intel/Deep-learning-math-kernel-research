#ifndef _ELT_UNITESTS_HPP_
#define _ELT_UNITESTS_HPP_

extern int iterations;

template <typename Type, const int T, const int V, const int I>
int test_elk_gemm(bool, bool);

template <typename Type, const int A, const int K, const int V, const int I>
int test_elk_trans_weights(bool perf, bool show_diff);

template <typename Type, const int A, const int K, const int V, const int I>
int test_elk_trans_input(bool perf, bool show_diff);

#endif
