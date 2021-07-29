export PYTHONPATH=.
export PYTHONWARNINGS="ignore"

# Command example
# ./CasStereoNet/scripts/messytable_local.sh 4 ./train_7_28/gaussian
# First argument - number of gpus in your nautilus job
# Second argument - saving path

save_path=$2

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

python -m torch.distributed.launch --nproc_per_node=$1 CasStereoNet/main.py \
  --debug \
  --gaussian-blur \
  --config-file /CasStereoNet/configs/remote_train_config.yaml \
  --warp_op \
  --logdir $save_path | tee -a  $save_path/log.txt


# Test
  # python CasStereoNet/test_on_sim_real.py --config-file /isabella-fast/Cascade-Stereo/CasStereoNet/utils/messytable_dataset_config.py --model /isabella-fast/Cascade-Stereo/outputs/7_19_dataaug-0/checkpoint_000006.ckpt --debug --annotate local_test_real --exclude-bg --onreal