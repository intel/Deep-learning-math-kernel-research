#/bin/bash

source ./scripts/best_configs/common.sh $@

# bs=1, stride=1, blocked
# ssd_300_voc0712:fc7
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=2 --blk-t=13 --execution-mode=0xe000 --pat-o=4 $COMMON
# ssd_300_voc0712:conv6_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o256  -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=2 --blk-t=13 --execution-mode=0xe000 --pat-o=2 $COMMON
# ssd_300_voc0712:conv7_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512  -o128  -h10 -k1 -K1 -H10 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=4 --blk-t=4 --execution-mode=0xe000 $COMMON
# ssd_300_voc0712:conv8_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -o128  -h5  -k1 -K1 -H5  -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=3 --blk-t=3 --execution-mode=0xe000 $COMMON
# ssd_300_voc0712:conv9_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -o128  -h3  -k1 -K1 -H3  -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=3 --blk-t=3 --execution-mode=0xe000 $COMMON

# bs=64, stride=1, blocked
# ssd_300_voc0712:fc7
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o1024 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=2 --blk-t=13 --execution-mode=0xe000 --pat-o=4 --pat-i=2 $COMMON
# ssd_300_voc0712:conv6_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o256  -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=2 --blk-t=13 --execution-mode=0xe000 --pat-o=8 --pat-i=2 $COMMON
# ssd_300_voc0712:conv7_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512  -o128  -h10 -k1 -K1 -H10 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON
# ssd_300_voc0712:conv8_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -o128  -h5  -k1 -K1 -H5  -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=4 --blk-t=5 --execution-mode=0xe000 --pat-o=1 $COMMON
# ssd_300_voc0712:conv9_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -o128  -h3  -k1 -K1 -H3  -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=2 --blk-t=9 --execution-mode=0xe000 --pat-o=1 $COMMON

# bs=1, stide=1, blocked
# resnet_50_sparse:res3a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=1 $COMMON
# resnet_50_sparse:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=1 $COMMON
# resnet_50_sparse:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=1 $COMMON
# resnet_50_sparse:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=2 $COMMON
# resnet_50_sparse:res5a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=64 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON
# resnet_50_sparse:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=64 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=1 $COMMON


# bs=64, stride=1, blocked
# resnet_50_sparse:res3a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=8 --blk-t=7 --execution-mode=0xe000 --pat-o=1 $COMMON
# resnet_50_sparse:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=16 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON
# resnet_50_sparse:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON
# resnet_50_sparse:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --blk-o=3 --blk-t=7 --execution-mode=0xe000 --pat-o=2 $COMMON
# resnet_50_sparse:res5a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=64 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON
# resnet_50_sparse:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -r0 -v0 -adirect_1x1 --blk-i=64 --blk-o=3 --blk-t=7 --execution-mode=0xe000 $COMMON


