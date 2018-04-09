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

#define CASE(name) lest_CASE(specification, name)
static lest::tests specification;

CASE("test_gemm") {
    EXPECT(0 == test_gemm<float>(test_perf, show_diff));
}


int main(int argc, char **argv) {
    return lest::run(specification, argc, argv, std::cout);
}
