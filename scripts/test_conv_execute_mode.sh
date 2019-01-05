#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

function usage() {
cat <<!
  -p   Use plain format.
  -s   Inplace sum
  -t   Tile size
  -h   This page.
!
}

input_format=nChw16c
weights_format=OIhw16i16o
output_format=nChw16c
tile_size=5
with_ip_sum=0
with_relu=0

OPTIND=1
while getopts "prst:h" opt; do
  case $opt in
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    r)
      with_relu=1
      ;;
    s)
      with_ip_sum=1
      ;;
    t)
      tile_size=$OPTARG
      ;;
    h)
      usage
      ;;
    ?)
      usage
      ;;
  esac
done
shift $((OPTIND-1))


# correctness
function __val_conv() {
  echo ====== Test execution-mode: $1 $2 $3 ======
  $ROOT_DIR/scripts/run.sh -c -v1 -r$with_relu -n 3 -i 128 -o 256 -h 56 -w 56 -H 56 -W 56 \
    --execution-mode=$1 \
    --nthreads=$2 \
    --input-format=$input_format \
    --weights-format=$weights_format \
    --output-format=$output_format \
    --with-ip-sum=$with_ip_sum    \
    --tile-size=$tile_size

  if [ $? != 0 ]; then
    echo "XXXXX FAILURE: XXXXX"
    exit -1
  fi
}

function val_conv() {
  __val_conv 0xa000 0
  __val_conv 0xa000 18
  __val_conv 0xa061 0
  __val_conv 0xa061 18
  __val_conv 0xa071 0
  __val_conv 0xa071 18
  __val_conv 0xa073 0
  __val_conv 0xa073 18
  __val_conv 0xa079 0
  __val_conv 0xa079 18
  __val_conv 0xa07b 0
  __val_conv 0xa07b 18
}

set -x
val_conv
set +x
