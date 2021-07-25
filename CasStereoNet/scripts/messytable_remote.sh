export PYTHONPATH=.
export PYTHONWARNINGS="ignore"

# first argument as the number of gpus
# second argument to be the saving path
save_path=$1

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATASET="messytable"

python CasStereoNet/main.py \
  --debug \
  --dataset $DATASET --test_dataset $DATASET \
  --epochs 35 --lrepochs "10,20,30:2" \
  --log_freq 2000 \
  --using_ns --ns_size 3 \
  --model gwcnet-c \
  --logdir $save_path | tee -a  $save_path/log.txt


# Test
  # python CasStereoNet/test_on_sim_real.py --config-file /isabella-fast/Cascade-Stereo/CasStereoNet/utils/messytable_dataset_config.py --model /isabella-fast/Cascade-Stereo/outputs/7_19_dataaug-0/checkpoint_000006.ckpt --debug --annotate local_test_real --exclude-bg --onreal