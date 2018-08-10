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

template <typename Type, const int A, const int K, const int V, const int I>
void test_elk_trans_weights(bool perf, bool show_diff) {
  alignas(64) Type aweights[K][K][V][V];
  alignas(64) Type atweights[A][A][V][V];
  alignas(64) Type ref_atweights[A][A][V][V];

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
     (convolution_winograd_kernel<Type, I, V, A, K>::trans_weights(
         atweights, aweights)));

  memset(ref_atweights, 0, sizeof(ref_atweights));
  TT(ref_elk_trans_weights, iterations, perf,
     (convolution_winograd_kernel<Type, ISA_GENERIC, V, A, K>::trans_weights(ref_atweights,
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

bool test_perf = false;
bool show_diff = false;
TEST(elkTransWeightsTest, A5) {
  test_elk_trans_weights<float, 5, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff);
}

TEST(elkTransWeightsTest, A7) {
  test_elk_trans_weights<float, 7, 3, 16, ISA_SKX_AVX512>(test_perf, show_diff);
}
