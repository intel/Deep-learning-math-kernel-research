#!/bin/bash

# asymmetric padding, k3 s2
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o32 -H112 -k3 -K3 -s2 -S2 -p1 -P1 -n1 -adirect --execution-mode=0xc060 --blk-i=1 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 --input-format=nchw --output-format=nChw16c --weights-format=hwio -v1
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o32 -H112 -k3 -K3 -s2 -S2 -p0 -P0 -n1 -adirect --execution-mode=0xc060 --blk-i=1 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 --input-format=nchw --output-format=nChw16c --weights-format=hwio -v1

# asymmetric padding, k7 s2
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o64 -H112 -k7 -K7 -s2 -S2 -p3 -P3 -n1 -adirect --execution-mode=0xc060 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 --input-format=nchw --output-format=nChw16c --weights-format=hwio -v1
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o64 -H112 -k7 -K7 -s2 -S2 -p2 -P2 -n1 -adirect --execution-mode=0xc060 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 --input-format=nchw --output-format=nChw16c --weights-format=hwio -v1
