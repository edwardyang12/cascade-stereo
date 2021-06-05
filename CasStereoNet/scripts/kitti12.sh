#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path="/cephfs/jianyu/eval/mt_sim_output"

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="/cephfs/datasets/iccv_pnp/real_data_v9/"
pip install --upgrade pip setuptools wheel
pip install -r /cephfs/jianyu/mt_real_eval/requirements.txt
python -m torch.distributed.launch --nproc_per_node=1 /cephfs/jianyu/mt_real_eval/CasStereoNet/main.py --dataset kitti \
    --datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training --trainlist /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training_lists/all.txt --testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training_lists/all.txt \
    --test_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training --test_dataset kitti \
    --epochs 300 --lrepochs "200:10" \
    --crop_width 512  --crop_height 256 --test_crop_width 1248  --test_crop_height 768 \
    --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
    --model gwcnet-c --logdir "/cephfs/jianyu/eval/mt_sim_output"  --ndisps "48,24" \
    --disp_inter_r "4,1"   --batch_size 2 --mode test \
    --loadckpt "/cephfs/jianyu/mtcaseval_train_output/checkpoint_best.ckpt"
