#ifndef _ELT_UNITESTS_HPP_
#define _ELT_UNITESTS_HPP_

extern int iterations;

template <typename Type, const int T, const int A, const int V, const int I>
void test_elk_gemm(bool perf, bool show_diff, int execution_mode,
                   int input_format, int weights_format, int output_format,
                   bool with_bias, bool with_relu, int mb);

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_weights(bool perf, bool show_diff, int execution_mode,
                            int input_format, int weights_format,
                            int output_format, bool with_bias, bool with_relu,
                            int mb);

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_input(bool perf, bool show_diff, int execution_mode);

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_output(bool perf, bool show_diff, int execution_mode,
                           int input_format, int weights_format,
                           int output_format, bool with_bias, bool with_relu,
                           int mb);

#endif
