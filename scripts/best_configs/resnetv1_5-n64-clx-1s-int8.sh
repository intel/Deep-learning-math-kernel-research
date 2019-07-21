#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON "

echo "Compute conv: 7x7"
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o64 -H112 -k7 -K7 -s2 -S2 -p3 -P3 -n64 -adirect --execution-mode=0xa160 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32z --sampling-kind=2 --input-format=nhwc --output-format=nChw16c --weights-format=OIhw16i16o

echo "Compute conv: 3x3 s1"
# resnet50_res2a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 --tile-size=6 --execution-mode=0xa161 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 --output-as-blocked=true $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# resnet50_res3a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 --tile-size=6 --execution-mode=0xa161 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=2 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# resnet50_res4a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 --tile-size=6 --execution-mode=0xa161 --blk-i=16 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=4 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# resnet50_res5a_branch2b
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=6 --execution-mode=0xa133 --blk-i=32 --blk-o=1 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 3x3 s2"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h58 -o128 -H28 -k3 -K3 -s2 -S2 -p0 -P0 -n64 -adirect --execution-mode=0xa160 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i256 -h30 -o256 -H14 -k3 -K3 -s2 -S2 -p0 -P0 -n64 -adirect --execution-mode=0xa160 --blk-i=16 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i512 -h16 -o512 -H7 -k3 -K3 -s2 -S2 -p0 -P0 -n64 -adirect --execution-mode=0xa160 --blk-i=32 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 1x1 s2"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o512 -H28 -k1 -K1 -s2 -S2 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xb161 --blk-i=16 --blk-o=4 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o1024 -H14 -k1 -K1 -s2 -S2 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xb161 --blk-i=32 --blk-o=4 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h14 -o2048 -H7 -k1 -K1 -s2 -S2 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xb161 --blk-i=64 --blk-o=4 --flt-o=2 --flt-t=7 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 1x1 s1 h56"
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o256 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=4 --blk-o=8 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o64 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=16 --blk-o=1 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o128 -H56 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=16 --blk-o=1 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 1x1 s1 h28"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o512 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=8 --blk-o=4 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# xxxxx
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o128 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=32 --blk-o=2 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# xxxxx
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o256 -H28 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=32 --blk-o=2 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 1x1 s1 h14"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o1024 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=16 --blk-o=4 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
# xxxxx
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h14 -o256 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=64 --blk-o=4 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h14 -o512 -H14 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=64 --blk-o=4 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute conv: 1x1 s1 h7"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o2048 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=32 --blk-o=4 --flt-o=4 --flt-t=6 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
NSOCKETS=1 ./scripts/run.sh -c -i2048 -h7 -o512 -H7 -k1 -K1 -s1 -S1 -p0 -P0 -n64 -adirect_1x1 --execution-mode=0xc160 --blk-i=128 --blk-o=2 --flt-o=4 --flt-t=6 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2

echo "Compute gemm: 1x2048, 2048x1001"
NSOCKETS=1 ./scripts/run.sh -c -i2048 -h1 -o1001 -H1 -k1 -K1 -s1 -S1 -p0 -P0 -n1 -adirect --execution-mode=0xd160 --blk-i=128 --blk-o=1 --flt-o=1 --flt-t=1 --pat-i=1 --pat-o=1 $COMMON --data-type-cfg=U8F32S8F32 --sampling-kind=2
