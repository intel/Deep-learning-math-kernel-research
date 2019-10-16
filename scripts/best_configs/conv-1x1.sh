#/bin/bash

source ./scripts/best_configs/common.sh $@

#n=1, stride=2, blocked
#resnet_50:res3a_branch1, 3.53
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -h56 -o512  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=16 --execution-mode=0xa061 $COMMON
#resnet_50:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -h56 -o128  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=4 --execution-mode=0xa061 $COMMON
#resnet_50:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512  -h28 -o1024 -H14 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=4 --execution-mode=0xa061 $COMMON
#resnet_50:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512  -h28 -o256  -H14 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=7 --pat-i=1 --pat-o=2 --execution-mode=0xa061 $COMMON
#resnet_50:res5a_branch1, 3.05
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -h14 -o2048 -H7  -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=7 --pat-i=1 --pat-o=1 --execution-mode=0xa061 $COMMON
#resnet_50:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -h14 -o512  -H7  -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=7 --pat-i=2 --pat-o=1 --execution-mode=0xa061 $COMMON

#n=64, stride=2, blocked
#resnet_50:res3a_branch1, 3.06
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -h56 -o512  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=16 --execution-mode=0xa061 $COMMON
#resnet_50:res3a_branch2a, 2.3
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -h56 -o128  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=4 --execution-mode=0xa061 $COMMON
#resnet_50:res4a_branch1, 3.1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512  -h28 -o1024 -H14 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 --execution-mode=0xa061 $COMMON
#resnet_50:res4a_branch2a, 2.96
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512  -h28 -o256  -H14 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=8 --execution-mode=0xa061 $COMMON
#resnet_50:res5a_branch1, 3.1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -h14 -o2048 -H7  -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=7 --pat-i=1 --pat-o=2 --execution-mode=0xa061 $COMMON
#resnet_50:res5a_branch2a, 2.9
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -h14 -o512  -H7  -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=7 --pat-i=1 --pat-o=1 --execution-mode=0xa061 $COMMON

# bs=1, stride=1, blocked
# ssd_300_voc0712:fc7
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=13 --execution-mode=0xa060 --pat-o=1 --pat-i=4 $COMMON
# ssd_300_voc0712:conv6_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o256  -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=13 --execution-mode=0xa060 --pat-o=4 --pat-i=4 $COMMON
# ssd_300_voc0712:conv7_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512  -o128  -h10 -k1 -K1 -H10 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=32 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-o=2 $COMMON
# ssd_300_voc0712:conv8_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -o128  -h5  -k1 -K1 -H5  -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=4 --execution-mode=0xa060 $COMMON
# ssd_300_voc0712:conv9_1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256  -o128  -h3  -k1 -K1 -H3  -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=3 --execution-mode=0xa060 $COMMON

# bs=64, stride=1, blocked
# ssd_300_voc0712:fc7
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o1024 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=13 --execution-mode=0xa060 --pat-o=4 --pat-i=4 $COMMON
# ssd_300_voc0712:conv6_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o256  -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=8 --flt-o=2 --flt-t=13 --execution-mode=0xa060 --pat-o=8 --pat-i=8 $COMMON
# ssd_300_voc0712:conv7_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512  -o128  -h10 -k1 -K1 -H10 -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=8 --flt-o=2 --flt-t=10 --execution-mode=0xa060 --pat-o=4 --pat-i=4 $COMMON
# ssd_300_voc0712:conv8_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -o128  -h5  -k1 -K1 -H5  -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-o=1 $COMMON
# ssd_300_voc0712:conv9_1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256  -o128  -h3  -k1 -K1 -H3  -p0 -P0 -s1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=9 --execution-mode=0xa060 --pat-o=1 $COMMON

# bs=1, stide=1, blocked
# resnet_50_sparse:res3a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON
# resnet_50_sparse:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=4 $COMMON
# resnet_50_sparse:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=1 --pat-i=2 $COMMON
# resnet_50_sparse:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-o=4 $COMMON
# resnet_50_sparse:res5a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-o=1 --pat-i=16 $COMMON
# resnet_50_sparse:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-i=16 $COMMON


# bs=64, stride=1, blocked
# resnet_50_sparse:res3a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON
# resnet_50_sparse:res3a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i256 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=4 $COMMON
# resnet_50_sparse:res4a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=4 $COMMON
# resnet_50_sparse:res4a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i512 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=4 --pat-i=2 $COMMON
# resnet_50_sparse:res5a_branch1
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=64 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=2 $COMMON
# resnet_50_sparse:res5a_branch2a
NSOCKETS=1 ./scripts/run.sh -c -n64 -i1024 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=2 --pat-i=2 $COMMON


