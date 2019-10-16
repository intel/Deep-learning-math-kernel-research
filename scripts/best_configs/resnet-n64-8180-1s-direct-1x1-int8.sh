#/bin/bash

source ./scripts/best_configs/common.sh $@

# bs=64, stide=1, blocked
# resnet_50_sparse:res3a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa160 --pat-o=4 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
# resnet_50_sparse:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa160 --pat-o=4 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
# resnet_50_sparse:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --execution-mode=0xa160 --pat-o=4 --pat-i=1 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
# resnet_50_sparse:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --execution-mode=0xa160 --pat-o=4 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
# resnet_50_sparse:res5a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=13 --execution-mode=0xa160 --pat-o=1 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
# resnet_50_sparse:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=13 --execution-mode=0xa160 --pat-o=1 $COMMON --sampling-kind=2 --data-type-cfg=U8F32S8F32
