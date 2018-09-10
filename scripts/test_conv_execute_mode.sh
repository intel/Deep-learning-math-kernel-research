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

OPTIND=1
while getopts "pst:h" opt; do
  case $opt in
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
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
  $ROOT_DIR/scripts/run.sh -c -v1 -n 3 -i 128 -o 256 -h 56 -w 56 -H 56 -W 56 \
    --execution-mode=$1 \
    --nteams=$2 --nthreads=$3 \
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

function bench_conv() {
  $ROOT_DIR/scripts/run.sh -c -v1 -n 3 -i 128 -o 256 -h 56 -w 56 -H 56 -W 56 \
    --execution-mode=$1 \
    --nteams=$2 --nthreads=$3 \
    --input-format=$input_format \
    --weights-format=$weights_format \
    --output-format=$output_format \
    --tile-size=$tile_size
}

function val_conv() {
  __val_conv 0xa040 0 0
  __val_conv 0xa040 1 18
  __val_conv 0xa040 2 9
  __val_conv 0xa061 0 0
  __val_conv 0xa061 1 18
  __val_conv 0xa061 2 9
  __val_conv 0xa448 0 0
  __val_conv 0xa448 1 18
  __val_conv 0xa448 2 9
  __val_conv 0xa241 0 0
  __val_conv 0xa241 1 18
  __val_conv 0xa241 2 9
  __val_conv 0xa000 0 0
  __val_conv 0xa000 1 18
  __val_conv 0xa000 2 9
  __val_conv 0xa201 0 0
  __val_conv 0xa201 2 9
  __val_conv 0xa201 1 18
  __val_conv 0xa0e0 1 18
  __val_conv 0xa0e0 2 9
  __val_conv 0xa0e1 1 18
  __val_conv 0xa0e1 2 9
}

vgg19_conv1_2=" -i64  -h224 -o64  -H224 "
vgg19_conv2_1=" -i64  -h112 -o128 -H112 "
vgg19_conv2_2=" -i128 -h112 -o128 -H112 "
vgg19_conv3_1=" -i128 -h56  -o256 -H56  "
vgg19_conv3_2=" -i256 -h56  -o256 -H56  "
vgg19_conv4_1=" -i256 -h28  -o512 -H28  "
vgg19_conv4_2=" -i512 -h28  -o512 -H28  "
vgg19_conv5_1=" -i512 -h14  -o512 -H14  "

resnet50_res2a_branch2b=" -i64  -h56 -o64  -H56 "
resnet50_res3a_branch2b=" -i128 -h28 -o128 -H28 "
resnet50_res4a_branch2b=" -i256 -h14 -o256 -H14 "
resnet50_res5a_branch2b=" -i512 -h7  -o512 -H7  "

set -x

val_conv

$ROOT_DIR/scripts/run.sh -c $resnet50_res5a_branch2b -v1 -n1 \
  --execution-mode=0xa000 \
  --nteams=1 --nthreads=18 \
  --blk-i=2 --blk-o=2 --flt-t=18 --tile-size=4

set +x
