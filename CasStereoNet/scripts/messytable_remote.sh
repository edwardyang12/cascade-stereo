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
  --config-file ./CasStereoNet/configs/remote_train_config.yaml \
  --warp-op \
  --logdir $save_path | tee -a  $save_path/log.txt
