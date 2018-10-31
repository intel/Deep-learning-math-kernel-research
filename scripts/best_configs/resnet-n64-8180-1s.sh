#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

# resnet50_res2a_branch2b, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 --tile-size=6 --execution-mode=0xa061 --blk-i=4 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=2 --output-as-blocked=true $COMMON
sleep 1
# resnet50_res3a_branch2b, 7.0 - 7.5T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 --tile-size=6 --execution-mode=0xa061 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=13 --pat-o=4 $COMMON
sleep 1
# resnet50_res4a_branch2b, 4.2T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 --tile-size=7 --execution-mode=0xa073 --blk-i=4 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=4 --pat-o=8 $COMMON
sleep 1
# resnet50_res5a_branch2b, 2.5T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 --tile-size=6 --execution-mode=0xa033 --blk-i=8 --blk-o=2 --flt-t=29 --pat-i=1 --pat-o=2 $COMMON
