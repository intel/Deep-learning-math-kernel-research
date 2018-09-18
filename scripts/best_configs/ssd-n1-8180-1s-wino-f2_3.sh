#/bin/bash

# SSD300
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

#"ssd_300_voc0712:conv1_1":ic3 ih300 oc64 oh300 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i3 -h300 -o64 -H300 -n1 --tile-size=4 --execution-mode=0xa040 --blk-i=1 --blk-o=4 --flt-t=30 $COMMON 
sleep 1
#"ssd_300_voc0712:conv1_2":ic64 ih300 oc64 oh300 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i64 -h300 -o64 -H300 -n1 --tile-size=4 --execution-mode=0xa040 --blk-i=4 --blk-o=4 --flt-t=30 $COMMON 
sleep 1
#"ssd_300_voc0712:conv2_1":ic64 ih150 oc128 oh150 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i64 -h150 -o128 -H150 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=4 --blk-o=4 --pat-o=2 --flt-t=31 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:conv2_2":ic128 ih150 oc128 oh150 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h150 -o128 -H150 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=4 --blk-o=1 --flt-t=31 --pat-o=8 $COMMON 
sleep 1
#"ssd_300_voc0712:conv3_1":ic128 ih75 oc256 oh75 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h75 -o256 -H75 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=4 --blk-o=2 --pat-o=8 --flt-t=26 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:conv3_2":ic256 ih75 oc256 oh75 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h75 -o256 -H75 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=4 --blk-o=4 --pat-o=4 --flt-t=26 --output-as-blocked=true $COMMON  
sleep 1
#"ssd_300_voc0712:conv3_3":ic256 ih75 oc256 oh75 kh3 ph1n"
#"ssd_300_voc0712:conv4_1":ic256 ih38 oc512 oh38 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h38 -o512 -H38 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=16 --blk-o=4 --flt-t=26 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:conv4_2":ic512 ih38 oc512 oh38 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h38 -o512 -H38 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=4 --blk-o=4 --flt-t=26 $COMMON 
sleep 1
#"ssd_300_voc0712:conv4_3":ic512 ih38 oc512 oh38 kh3 ph1n"
#"ssd_300_voc0712:conv5_1":ic512 ih19 oc512 oh19 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h19 -o512 -H19 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=2 --flt-t=25 $COMMON 
sleep 1
#"ssd_300_voc0712:conv5_2":ic512 ih19 oc512 oh19 kh3 ph1n"
#"ssd_300_voc0712:conv5_3":ic512 ih19 oc512 oh19 kh3 ph1n"
#"ssd_300_voc0712:conv8_2": ic128 ih5 oc256 oh3 kh3"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h5 -o256 -H3 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=4 --flt-t=4 -p0 -P0 $COMMON 
sleep 1
#"ssd_300_voc0712:conv9_2": ic128 ih3 oc256 oh1 kh3"
NSOCKETS=1 ./scripts/run.sh -c -i128 -h3 -o256 -H1 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=2 --flt-t=1 -p0 -P0 $COMMON 
sleep 1
#"ssd_300_voc0712:conv4_3_norm_mbox_l oc": ic512 ih38 oc16 oh38 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h38 -o16 -H38 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=8 --blk-o=1 --flt-t=13 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:conv4_3_norm_mbox_conf": ic512 ih38 oc84 oh38 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h38 -o84 -H38 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=8 --blk-o=3 --flt-t=13 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:fc7_mbox_loc": ic1024 ih19 oc24 oh19 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i1024 -h19 -o24 -H19 -n1 --tile-size=4 --execution-mode=0xa040 --blk-i=16 --blk-o=2 --flt-t=4 $COMMON 
sleep 1
#"ssd_300_voc0712:fc7_mbox_conf": ic1024 ih19 oc126 oh19 kh3 ph1n"
 NSOCKETS=1 ./scripts/run.sh -c -i1024 -h19 -o126 -H19 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=8 --flt-t=25 $COMMON 
sleep 1
#"ssd_300_voc0712:conv6_2_mbox_l oc": ic512 ih10 oc24 oh10 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h10 -o24 -H10 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=2 --flt-t=3 --output-as-blocked=true $COMMON 
sleep 1
#"ssd_300_voc0712:conv6_2_mbox_conf": ic512 ih10 oc126 oh10 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i512 -h10 -o126 -H10 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=16 --blk-o=2 --flt-t=9 $COMMON 
sleep 1
#"ssd_300_voc0712:conv7_2_mbox_l oc": ic256 ih5 oc24 oh5 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h5 -o24 -H5 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=2 --flt-t=9 $COMMON 
sleep 1
#"ssd_300_voc0712:conv7_2_mbox_conf": ic256 ih5 oc126 oh5 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h5 -o126 -H5 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=4 --flt-t=9 $COMMON 
sleep 1
#"ssd_300_voc0712:conv8_2_mbox_l oc": ic256 ih3 oc16 oh3 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h3 -o16 -H3 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=16 --blk-o=1 --flt-t=4 $COMMON 
sleep 1
#"ssd_300_voc0712:conv8_2_mbox_conf": ic256 ih3 oc84 oh3 kh3 ph1n"
NSOCKETS=1 ./scripts/run.sh -c -i256 -h3 -o84 -H3 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=8 --blk-o=2 --flt-t=1 $COMMON 
sleep 1
#"ssd_300_voc0712:conv9_2_mbox_l oc": ic256 ih1 oc16 oh1 kh3 ph1n"
#NSOCKETS=1 ./scripts/run.sh -c -i256 -h1 -o16 -H1 -n1
#sleep 1
#"ssd_300_voc0712:conv9_2_mbox_conf": ic256 ih1 oc84 oh1 kh3 ph1n"
#NSOCKETS=1 ./scripts/run.sh -c -i256 -h1 -o84 -H1 -n1
#sleep 1
