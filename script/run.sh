#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

echo Root dir: $ROOT_DIR
echo
echo Release Build...
#cd "$ROOT_DIR" && make distclean && make -j all && cd -

echo Run convolution test...
OMP_NUM_THREADS=18 \
KMP_HW_SUBSET=1s,18c,1t \
KMP_AFFINITY=compact,granularity=fine \
KMP_BLOCKTIME=infinite \
$ROOT_DIR/build/release/bin/elt_conv 

echo Run unit tests...
OMP_NUM_THREADS=18 \
KMP_HW_SUBSET=1s,18c,1t \
KMP_AFFINITY=compact,granularity=fine \
KMP_BLOCKTIME=infinite \
$ROOT_DIR/build/release/bin/elt_unitests -p -t

