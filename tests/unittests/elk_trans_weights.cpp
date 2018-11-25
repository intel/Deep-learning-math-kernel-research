#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include "gtest/gtest.h"
#include "euler.hpp"
#include "elt_unitests.hpp"
#include "tests/elt_utils.hpp"
#include "src/elk_conv_wino.hpp"
#include "src/elx_conv.hpp"
#include "src/elx_conv_wino.hpp"

int iterations = 10;
using namespace euler;
using ::testing::TestWithParam;
using ::testing::Values;

Template_elx_conv_wino_t
void test_elk_trans_weights(bool perf, bool show_diff) {

  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;
  using TarrayType = typename InnerTypes::TarrayType;

  alignas(64) WeightsType aweights[K][K][V][V];
  alignas(64) TarrayType atweights[A][A][V][V];
  alignas(64) TarrayType ref_atweights[A][A][V][V];

  for (int _hK = 0; _hK < K; ++_hK) {
    for (int _wK = 0; _wK < K; ++_wK) {
      for (int _iV = 0; _iV < V; ++_iV) {
        for (int _oV = 0; _oV < V; ++_oV) {
          aweights[_hK][_wK][_iV][_oV] = _hK * _wK * _iV * _oV % 32;
        }
      }
    }
  }

  memset(atweights, 0, sizeof(atweights));
  TT(elk_trans_weights, iterations, perf,
     (Instance_convolution_winograd_kernel::trans_weights(atweights, aweights)));

  memset(ref_atweights, 0, sizeof(ref_atweights));
  TT(ref_elk_trans_weights, iterations, perf,
      (convolution_winograd_kernel<
          euler::ConvTypes<InputType, WeightsType, OutputType, BiasType>,
          TarrayType, ISA_GENERIC, V, A, K>::trans_weights(ref_atweights,
          aweights)));

  for (int _hK = 0; _hK < K; ++_hK) {
    for (int _wK = 0; _wK < K; ++_wK) {
      for (int _iV = 0; _iV < V; ++_iV) {
        for (int _oV = 0; _oV < V; ++_oV) {
          EXPECT_NEAR(ref_atweights[_hK][_wK][_iV][_oV],
                      atweights[_hK][_wK][_iV][_oV], 1e-6);
          if (show_diff) {
            printf("actual atweights: [%d][%d][%d][%d]: %f vs reference: %f "
                   "(ref)\n",
                   _hK, _wK, _iV, _oV, atweights[_hK][_wK][_iV][_oV],
                   ref_atweights[_hK][_wK][_iV][_oV]);
          }
        }
      }
    }
  }
}

class elkTransWeightsTest : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_CASE_P(elk_elk_trans_test, elkTransWeightsTest,
                        testing::Values(4, 5, 6, 7));
bool test_perf = false;
bool show_diff = false;
TEST_P(elkTransWeightsTest, combineTest) {
  int test_tile_size = GetParam();
  switch (test_tile_size) {
  case 4:
    test_elk_trans_weights<conv::FP32, conv_impl::FP32, float, 4, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff);
    break;

  case 5:
    test_elk_trans_weights<conv::FP32, conv_impl::FP32, float, 5, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff);
    break;
  case 6:
//     test_elk_trans_weights<conv::FP32, conv_impl::FP32, float, 6, 3, 16, ISA_SKX_AVX512>(test_perf,
//                                                             show_diff);
    break;
  case 7:
    test_elk_trans_weights<conv::FP32, conv_impl::FP32, float, 7, 3, 16, ISA_SKX_AVX512>(
        test_perf, show_diff);
    break;
  default:
    el_error("Unimplemented tile size");
    break;
  }
}
