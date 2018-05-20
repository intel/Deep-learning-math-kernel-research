#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "elt_unitests.hpp"
#include "lest.hpp"
#include "../elt_utils.hpp"
#include "../../src/elx_conv.hpp"
#include "../../src/elk_conv.hpp"

using namespace std;
using namespace euler;

bool test_perf = false;
bool show_diff = true;
int iterations = 10;

#define TEST_CASE(name) lest_CASE(specification, name)
static lest::tests specification;

TEST_CASE("test_elk_gemm")
{
  EXPECT((0
      == test_elk_gemm<float, 25, 16, ISA_SKX_AVX512>(test_perf, show_diff)));
}

TEST_CASE("test_elk_trans_weights")
{
  EXPECT((0
      == test_elk_trans_weights<float, 5, 3, 16, ISA_SKX_AVX512>(
             test_perf, show_diff)));
}

TEST_CASE("test_elk_trans_input")
{
  EXPECT((0
      == test_elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(
             test_perf, show_diff)));
}

TEST_CASE("test_elk_trans_output")
{
  EXPECT((0
      == test_elk_trans_output<float, 5, 3, 16, ISA_SKX_AVX512>(
             test_perf, show_diff)));
}

int main(int argc, char **argv) {
  return lest::run(specification, argc, argv, std::cout);
}
