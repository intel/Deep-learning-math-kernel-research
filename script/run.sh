#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

echo Root dir: $ROOT_DIR
echo

sockets=$( lscpu | grep 'Socket(s)' | cut -d: -f2 )
cores_per_socket=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 )
cores=$(( sockets  * cores_per_socket ))
OMP_ENV="OMP_NUM_THREADS=$(( cores )) \
  KMP_HW_SUBSET=$(( sockets ))s,$(( cores_per_socket ))c,1t \
  KMP_AFFINITY=compact,granularity=fine \
  KMP_BLOCKTIME=infinite"
echo OMP Environment: $OMP_ENV

function build() {
  cd "$ROOT_DIR" && make distclean && make -j all && cd -
}

function conv_test() {
  eval $OMP_ENV $ROOT_DIR/build/release/bin/elt_conv \
    -n1 -i64 -o64 -h224 -w224 -H224 -W224 -k3 -K3 -p1 -P1 -s1 -S1 \
    -b1 -r0 -v1 -awino
}

function unit_test() {
  eval $OMP_ENV $ROOT_DIR/build/release/bin/elt_unitests -t
}

function show_help() {
cat <<@

Euler test script:
  -h        display this help and exit.
  -b        rebuild
  -c        convolution test
  -u        unit test

@
}

OPTIND=1
while getopts "hbuc" opt; do
  case "$opt" in
    h)
      show_help
      exit 0
      ;;
    b)
      echo Release Build...
      build
      ;;
    c)
      echo Run convolution test...
      conv_test
      ;;
    u)
      echo Run unit tests...
      unit_test
      ;;
    *)
      show_help
      ;;
  esac
done
shift $((OPTIND-1))
