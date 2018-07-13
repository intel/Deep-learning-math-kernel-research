#!/bin/bash

function usage() {
cat <<!
  -v   Validation on|off.
  -p   Plain format or blocked format.
  -h   This page.
!
}

v=0
input_format=nChw16c
weights_format=OIhw16i16o
output_format=nchw

while getopts "vph" opt; do
  case $opt in
    v)
      v=1
      ;;
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    h)
      usage
      ;;
    ?)
      usage
      ;;
  esac
done

COMMON="-v$v --input-format=$input_format --weights-format=$weights_format --output-format=$output_format "

echo "Common option:" $COMMON
echo

