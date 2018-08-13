#include "gtest/gtest.h"
#include "euler.hpp"
#include "el_def.hpp"
#include "elx_conv_wino.hpp"

class tuner_test_parameters {
public:
  std::size_t l1_sz_, l2_sz_, l3_sz_;
  int core_num_;
  int sockets_;

  int i, h, o, H, n;
  // comparable facts
  int mode, blk_t, pat_o, blk_i, blk_o;
};

template <typename Type, int ...configs>
class tuner_test : public ::testing::TestWithParam<tuner_test_parameters> {
  constexpr static int I = euler::winograd_traits<configs...>::I;
  constexpr static int V = euler::winograd_traits<configs...>::V;
  // We need further eliminate A parameters
  constexpr static int A = euler::winograd_traits<configs...>::A;
  constexpr static int K = euler::winograd_traits<configs...>::K;
public:
  virtual void SetUp() {
    auto p = testing::TestWithParam<tuner_test_parameters>::GetParam();

    desc_.dims = {
      {p.n, p.i, p.h, p.h}, {p.o, p.i, K, K}, {p.n, p.o, p.H, p.H}, {p.o} };
    desc_.formats = { euler::nChw16c, euler::OIhw16i16o, euler::nChw16c };
    desc_.pads = {1, 1, 1, 1};
    desc_.with_bias = false;
    desc_.algorithm = euler::CONV_WINOGRAD;
    desc_.with_relu = false;
    desc_.with_sum = false;
    desc_.execution_mode = 0xa061;
    desc_.prop_kind = euler::forward_inference;
    desc_.tile_size = A;

    p_xc_.reset(new euler::elx_conv_wino_t<Type, A, K, V, I> (desc_));
  }
  euler::eld_conv_t<Type> desc_;
  std::unique_ptr<euler::elx_conv_wino_t<Type, A, K, V, I>> p_xc_;
};

using tuner_test_tile_5 = tuner_test<float, euler::ISA_SKX_AVX512, 16, 5, 3>;
TEST_P(tuner_test_tile_5, tile_5) {
  auto p = testing::TestWithParam<tuner_test_parameters>::GetParam();
  auto tile_oc4 = p_xc_->tile_blocking_oc4(p.core_num_ * p.sockets_);
  auto blk_t = tile_oc4.first;
  auto pat_o = tile_oc4.second;
  auto blk_i = p_xc_->I2_num(p.l1_sz_, blk_t);
  auto blk_o = p_xc_->O2_num(p.l2_sz_, blk_i, tile_oc4);

  EXPECT_EQ(blk_t, p.blk_t);
  EXPECT_EQ(pat_o, p.pat_o);
  EXPECT_EQ(blk_i, p.blk_i);
  EXPECT_EQ(blk_o, p.blk_o);
}

#define l1 (32 * 1024)
#define l2 (1024 * 1024)
#define l3 39424
#define skx_8180  28
using ttp = tuner_test_parameters;

INSTANTIATE_TEST_CASE_P(resnet_n1, tuner_test_tile_5,
  ::testing::Values(
    /* resnet-n1-8180-1s-wino-f3_3 */
    ttp {l1, l2, l3, skx_8180, 1, 64, 56, 64, 56, 1, 0xa061, 26, 2, 4, 2}
    , ttp {l1, l2, l3, skx_8180, 1, 128, 28, 128, 28, 1, 0xa061, 15, 4, 8, 2}
    , ttp {l1, l2, l3, skx_8180, 1, 256, 14, 256, 14, 1, 0xa000, 25, 1, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 1, 512, 7, 512, 7, 1, 0xa000, 9, 1, 8, 4}
    /* resnet-n1-8180-2s-wino-f3_3 */
    , ttp {l1, l2, l3, skx_8180, 2, 64, 56, 64, 56, 1, 0xa061, 13, 2, 4, 2}
    , ttp {l1, l2, l3, skx_8180, 2, 128, 28, 128, 28, 1, 0xa061, 15, 8, 8, 2}
    , ttp {l1, l2, l3, skx_8180, 2, 256, 14, 256, 14, 1, 0xa000, 9, 4, 8, 1}
    , ttp {l1, l2, l3, skx_8180, 2, 512, 7, 512, 7, 1, 0xa000, 9, 4, 8, 4}
));

INSTANTIATE_TEST_CASE_P(vgg_n1, tuner_test_tile_5,
  ::testing::Values(
    /* vgg-n1-8180-1s-wino-f3_3 */
    ttp {l1, l2, l3, skx_8180, 1, 64, 224, 64, 224, 1, 0xa040, 29, 1, 4, 4}
    , ttp {l1, l2, l3, skx_8180, 1, 64, 112, 128, 112, 1, 0xa040, 26, 1, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 1, 128, 112, 128, 112, 1, 0xa040, 26, 1, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 1, 128, 56, 512, 56, 1, 0xa061, 26, 2, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 1, 256, 56, 256, 56, 1, 0xa061, 26, 2, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 1, 256, 28, 512, 56, 1, 0xa000, 25, 2, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 1, 512, 28, 512, 56, 1, 0xa000, 26, 2, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 1, 512, 14, 512, 14, 1, 0xa000, 25, 2, 8, 4}
    /* vgg-n1-8180-2s-wino-f3_3 */
    , ttp {l1, l2, l3, skx_8180, 2, 64, 224, 64, 224, 1, 0xa040, 29, 1, 4, 4}
    , ttp {l1, l2, l3, skx_8180, 2, 64, 112, 128, 112, 1, 0xa040, 26, 1, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 2, 128, 112, 128, 112, 1, 0xa040, 26, 1, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 2, 128, 56, 512, 56, 1, 0xa061, 26, 2, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 2, 256, 56, 256, 56, 1, 0xa061, 26, 2, 8, 8}
    , ttp {l1, l2, l3, skx_8180, 2, 256, 28, 512, 56, 1, 0xa000, 25, 2, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 2, 512, 28, 512, 56, 1, 0xa000, 26, 2, 8, 4}
    , ttp {l1, l2, l3, skx_8180, 2, 512, 14, 512, 14, 1, 0xa000, 25, 2, 8, 4}
));
