#!/bin/bash

function usage() {
cat <<!
  -v   Validation on.
  -p   Use plain format.
  -r   ReLU fusion.
  -l   Repeated layer.
  -f   fp16 UserTypes
  -B   Double buffering.
  -A   Output buffer bas input
  -T   Preprocess tweights.
  -s   Inplace sum.
  -H   Half precision opt.
  -h   This page.
!
}

v=0; r=0; s=0; l=1; B=0; A=0; T=0; f=0; H=0
input_format=nChw16c
weights_format=OIhw16i16o
output_format=nChw16c
tile_size=5

OPTIND=1
while getopts "vprsfl:BATHh" opt; do
  case $opt in
    v)
      v=1
      ;;
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    r)
      r=1
      ;;
    s)
      s=1
      ;;
    f)
      f=1
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
    T)
      T=1
      ;;
    H)
      H=1
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

COMMON="-v$v --input-format=$input_format --weights-format=$weights_format --output-format=$output_format -r$r --with-ip-sum=$s -l$l -B$B -A$A -T$T --f16c-opt=$H --fp16-mode=$f"

echo "Common option:" $COMMON
echo

