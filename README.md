# Casacde-Stereo on Messytable Dataset

## Installation
### Docker Image
```markdown
isabella98/cascade-stereo-image:latest
```

### Optional: Apex (not tested)
Install apex to enable synchronized batch normalization:
```shell
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Train
```shell
./CasStereoNet/scripts/messytable_remote.sh $NUM_OF_GPUS $OUTPUT_PATH
```
Example: Use `./CasStereoNet/scripts/messytable_remote.sh 2 ./train_7_28/debug` to train your model with 2 GPUs.

I have simplified the `argparser` part in `main.py`, it now only contains some basic args. You can use `--gaussian-blur`
and `--color-jitter` in your shell file ( e.g. `./CasStereoNet/scripts/messytable_remote.sh`) to enable gaussian blur 
and color jitter during training. 

If you want to change some other args such as model parameters, (e.g. lr, batch_size), please go to `CasStereoNet/configs/remote_train_config.yaml`,
it contains some parameters that we don't need to change often. If your dataset folder is not mounted as `/cephfs`, 
please go to the this file to change the path to your dataset. The parameters of data augmentation are also included in
this file.

## Test
```shell
python ./CasStereoNet/test_on_sim_real.py --config-file ./CasStereoNet/configs/remote_train_config.yaml --model $PATH_TO_YOUR_MODEL --annotate $EXPERIMENT_ANNOTATION --exclude-bg
```
Use `--exlude-bg` if you want to exclude background when calculating the error metric.

Use `--onreal` if you are testing on real dataset, omit if you want to test on sim dataset.


## Reference
```
@inproceedings{gu2019cas,
  title={Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching},
  author={Gu, Xiaodong and Fan, Zhiwen and Zhu, Siyu and Dai, Zuozhuo and Tan, Feitong and Tan, Ping},
  journal={arxiv preprint arXiv:1912.06378},
  year={2019}
}
```
