#include "gtest/gtest.h"
#include "euler.hpp"
#include "el_def.hpp"
#include "elx_conv_wino.hpp"

class tuner_test_parameters {
public:
  std::size_t l1_sz_, l2_sz_, l3_sz_;
  int core_num_;
  int sockets;

  int i, h, w, o, H, W, n;
  int pad_tl, pad_dr;
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

    desc_.dims = { {p.n, p.o, p.h, p.w}, {p.o, p.i, p.K, p.K}, {p.n, p.o, p.H, p.W} };
    desc_.formats = { euler::nChw16c, euler::OIhw16i16o, euler::nChw16c };
    desc_.pads = {p.pad_tl, p.pad_tl, p.pad_dr, p.pad_dr};
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
  auto tile_oc4 = p_xc_->tile_blocking_oc4(p.core_num_);
  auto blk_t = tile_oc4.first;
  auto pat_o = tile_oc4.second;
  auto blk_i = p_xc_->I2_num(p.l1_sz_, blk_t);
  auto blk_o = p_xc_->O2_num(p.l2_sz_, blk_i, tile_oc4);

  EXPECT_EQ(blk_t, p.blk_t);
  EXPECT_EQ(pat_o, p.pat_o);
  EXPECT_EQ(blk_i, p.blk_i);
  EXPECT_EQ(blk_o, p.blk_o);
}
