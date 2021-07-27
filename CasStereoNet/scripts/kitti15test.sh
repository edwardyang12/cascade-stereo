#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path=$2

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="./linked_sim_v9"

python -m torch.distributed.launch --nproc_per_node=$1 main.py --dataset custom_dataset_test \
    --datapath $DATAPATH --trainlist ./filenames/custom_test_sim.txt --testlist ./filenames/custom_test_sim.txt \
    --test_datapath $DATAPATH --test_dataset custom_dataset_test \
    --epochs 10 --lrepochs "200:10" \
    --crop_width 512  --crop_height 256 --test_crop_width 960  --test_crop_height 544 \
    --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
    --model gwcnet-c --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt
