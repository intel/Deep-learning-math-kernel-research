#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON --f16c-opt=1"

# direct: a060
NSOCKETS=1 ./scripts/run.sh -c -g32 -i128 -h56 -o128 -H56 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 -p1 -P1 -adirect --execution-mode=0xa060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nhwc --output-format=nhwc -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i256 -h28 -o256 -H28 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=28 -p1 -P1 -adirect --execution-mode=0xa060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nhwc --output-format=nhwc -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i512 -h14 -o512 -H14 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect --execution-mode=0xa060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i1024 -h7 -o1024 -H7 -n1 --blk-i=2 --blk-o=1 --flt-o=2 --flt-t=7 -p1 -P1 -adirect --execution-mode=0xa060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1


# direct: c060
NSOCKETS=1 ./scripts/run.sh -c -g32 -i128 -h56 -o128 -H56 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nhwc --output-format=nhwc -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i256 -h28 -o256 -H28 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nhwc --output-format=nhwc -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i512 -h14 -o512 -H14 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i1024 -h7 -o1024 -H7 -n1 --blk-i=2 --blk-o=1 --flt-o=2 --flt-t=7 -p1 -P1 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1

# direct_vmg
NSOCKETS=1 ./scripts/run.sh -c -g32 -i128 -h56 -o128 -H56 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect_vmg --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1

NSOCKETS=1 ./scripts/run.sh -c -g32 -i256 -h28 -o256 -H28 -n1 --blk-i=1 --blk-o=1 --flt-o=1 --flt-t=14 -p1 -P1 -adirect_vmg --execution-mode=0xc060 --pat-o=1 $COMMON --weights-format=ghwio --input-format=nChw16c --output-format=nChw16c -b1


