#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "elt_unitests.hpp"
#include "../elt_utils.hpp"
#include "../../src/elx_conv.hpp"
#include "../../src/elk_conv.hpp"

using namespace euler;

int main() {
    int iterations = 1;
    bool test_perf = false;

    test_gemm<float>(iterations, test_perf);
}
