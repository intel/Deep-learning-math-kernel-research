#!/bin/bash

function usage() {
cat <<!
  -v   Validation on.
  -p   Use nchw format.
  -P   Use nhwc format.
  -r   ReLU fusion.
  -l   Repeated layer.
  -t   UserTypes.
  -B   Double buffering.
  -A   Output buffer bas input.
  -s   Inplace sum.
  -H   Half precision opt.
  -F   Use fine sampling for int8-gemm.
  -h   This page.
!
}

v=0; r=0; s=0; l=1; B=0; A=0; T=0; H=0
input_format=nChw16c
weights_format=OIhw16i16o
output_format=nChw16c
tile_size=5
sampling_kind=0
data_type_cfg=FP32

OPTIND=1
while getopts "vpPrst:l:BATF:Hh" opt; do
  case $opt in
    v)
      v=1
      ;;
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    P)
      input_format=nhwc
      weights_format=hwio
      output_format=nhwc
      ;;
    r)
      r=1
      ;;
    s)
      s=1
      ;;
    t)
      data_type_cfg=$OPTARG
      ;;
    l)
      l=$OPTARG
      ;;
    B)
      B=1
      ;;
    A)
      A=1
      ;;
    H)
      H=1
      ;;
    F)
      sampling_kind=$OPTARG
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

COMMON="-v$v --input-format=$input_format --weights-format=$weights_format --output-format=$output_format -r$r --with-ip-sum=$s -l$l -B$B -A$A --f16c-opt=$H --data-type-cfg=$data_type_cfg --sampling-kind=$sampling_kind"

