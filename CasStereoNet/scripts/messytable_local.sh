export PYTHONPATH=.
export PYTHONWARNINGS="ignore"

# first argument as the number of gpus
# second argument to be the saving path
save_path=$1

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATASET="messytable"

python -m torch.distributed.launch --nproc_per_node=$2 CasStereoNet/main.py \
  --dataset $DATASET --test_dataset $DATASET \
  --gaussian-blur --color-jitter \
  --epochs 2 --lrepochs "10,20,30:2" \
  --log_freq 2000 \
  --using_ns --ns_size 3 \
  --model gwcnet-c \
  --logdir $save_path | tee -a  $save_path/log.txt