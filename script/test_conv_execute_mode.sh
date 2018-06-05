#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

# correctness
function test_conv() {
  echo ====== Test execution-mode: $1 $2 $3 ======
  $ROOT_DIR/script/run.sh -c -v1 -n 3 -i 128 -o 256 -h 56 -w 56 -H 56 -W 56 \
    --execution-mode=$1 \
    --nteams=$2 --nthreads=$3
}

test_conv 0x241 0 0
test_conv 0x241 1 18
test_conv 0x241 2 9

test_conv 0x281 0 0
test_conv 0x281 2 9
test_conv 0x281 1 18

test_conv 0x442 0 0
test_conv 0x442 1 18
test_conv 0x442 2 9

test_conv 0x848 0 0
test_conv 0x848 1 18
test_conv 0x848 2 9

test_conv 0x888 0 0
test_conv 0x888 1 18
test_conv 0x888 2 9
