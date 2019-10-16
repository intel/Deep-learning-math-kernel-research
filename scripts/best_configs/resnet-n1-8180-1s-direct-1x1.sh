#/bin/bash

source ./scripts/best_configs/common.sh $@

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


## missing resnet 1x1 cases without tuning
# #resnet_50_sparse:res2a_branch2a
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i64 -o64 -h56 -w56 -H56 -W56 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-i=16 $COMMON
# 
# #resnet_50_sparse:res2a_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i64 -o256 -h56 -w56 -H56 -W56 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-i=16 $COMMON
# 
# #resnet_50_sparse:res2c_branch2a
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o64 -h56 -w56 -H56 -W56 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=46 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-i=4 $COMMON
# 
# #resnet_50_sparse:res2c_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i64 -o256 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-i=1 $COMMON
# 
# #resnet_50_sparse:res3b_branch2a
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o128 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=32 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-i=4 $COMMON
# 
# #resnet_50_sparse:res3b_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i128 -o512 -h28 -w28 -H28 -W28 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-i=1 $COMMON
# 
# #resnet_50_sparse:res3d_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i128 -o512 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-i=1 $COMMON
# 
# #resnet_50_sparse:res4c_branch2a
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o256 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-i=16 $COMMON
# 
# #resnet_50_sparse:res4d_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i256 -o1024 -h14 -w14 -H14 -W14 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-i=2 $COMMON
# 
# #resnet_50_sparse:res5a_branch2c
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i512 -o2048 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=16 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-i=2 $COMMON
# 
# #resnet_50_sparse:res5b_branch2a
# NSOCKETS=1 ./scripts/run.sh -c -n1 -i2048 -o512 -h7 -w7 -H7 -W7 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect_1x1 --blk-i=4 --flt-o=1 --flt-t=25 --execution-mode=0xa060 --pat-i=16 $COMMON
