#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

echo Root dir: $ROOT_DIR
echo
echo Release Build...
#cd "$ROOT_DIR" && make distclean && make -j all && cd -

sockets=$( lscpu | grep 'Socket(s)' | cut -d: -f2 )
cores_per_socket=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 )
cores=$(( sockets  * cores_per_socket ))

echo Run convolution test...
OMP_NUM_THREADS=$(( cores ))                              \
KMP_HW_SUBSET=$(( sockets ))s,$(( cores_per_socket ))c,1t \
KMP_AFFINITY=compact,granularity=fine                     \
KMP_BLOCKTIME=infinite                                    \
$ROOT_DIR/build/release/bin/elt_conv 

echo Run unit tests...
OMP_NUM_THREADS=$(( cores ))                              \
KMP_HW_SUBSET=$(( sockets ))s,$(( cores_per_socket ))c,1t \
KMP_AFFINITY=compact,granularity=fine                     \
KMP_BLOCKTIME=infinite                                    \
$ROOT_DIR/build/release/bin/elt_unitests -p -t

