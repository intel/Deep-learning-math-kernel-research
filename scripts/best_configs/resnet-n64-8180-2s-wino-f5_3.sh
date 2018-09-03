#/bin/bash

#/bin/bash

# Resnet50
# batch size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@

# resnet50_res2a_branch2b, 11.8T
NSOCKETS=2 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 --tile-size=7 --execution-mode=0xa040 --blk-i=4 --blk-o=4 --flt-t=15 $COMMON
sleep 1
# resnet50_res3a_branch2b, 11.9T
NSOCKETS=2 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 --tile-size=7 --execution-mode=0xa040 --blk-i=8 --blk-o=8 --flt-t=21 $COMMON
sleep 1
# resnet50_res4a_branch2b, 10.0T
NSOCKETS=2 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 --tile-size=7 --execution-mode=0xa000 --blk-i=8 --blk-o=8 --flt-t=29 $COMMON
sleep 1
# resnet50_res5a_branch2b, 6.7T
NSOCKETS=2 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=7 --execution-mode=0xa000 --blk-i=8 --blk-o=8 --flt-t=29 $COMMON
