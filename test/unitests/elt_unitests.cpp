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
    bool test_perf = false;
    bool show_diff = true;

    printf("test-gemm: ");
    if (test_gemm<float>(test_perf, show_diff) == 0)
        printf("OK\n");
    else
        printf("FAIL!\n");
}
