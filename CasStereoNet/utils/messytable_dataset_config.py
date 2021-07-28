"""
Author: Isabella Liu 7/18/21
Feature: Config file for messy-table-dataset
"""

from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

# Directories
_C.DIR = CN()
_C.DIR.DATASET = '/cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training'
_C.DIR.OUTPUT = '/isabella-fast/Cascade-Stereo/outputs_sim_real'

# Split files
_C.SPLIT = CN()
_C.SPLIT.TRAIN = '/cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_train.txt'
_C.SPLIT.VAL = '/cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_val.txt'
_C.SPLIT.OBJ_NUM = 18  # Note: table + ground - 17th

_C.SPLIT.LEFT = '0128_irL_denoised_half.png'
_C.SPLIT.RIGHT = '0128_irR_denoised_half.png'
_C.SPLIT.DEPTHL = 'depthL.png'
_C.SPLIT.DEPTHR = 'depthR.png'
_C.SPLIT.META = 'meta.pkl'
_C.SPLIT.LABEL = 'irL_label_image.png'

# Configuration for testing on real dataset
_C.REAL = CN()
_C.REAL.DATASET = '/code/cascade-stereo/real_dataset_local_v9'
_C.REAL.LEFT = '1024_irL_real_1080.png'
_C.REAL.RIGHT = '1024_irR_real_1080.png'
_C.REAL.PAD_WIDTH = 960
_C.REAL.PAD_HEIGHT = 544

# Model Args
_C.ARGS = CN()
_C.ARGS.MAX_DISP = 192  # maximum disparity
_C.ARGS.MODEL = 'gwcnet-c'
_C.ARGS.NDISP = '48,24'
_C.ARGS.DISP_INTER_R = '4,1'
_C.ARGS.CR_BASE_CHS = '32,32,16'
_C.ARGS.GRAD_METHOD = 'detach'
_C.ARGS.USING_NS = True
_C.ARGS.NS_SIZE = 3

# Data Augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2
_C.DATA_AUG.GAUSSIAN_KERNEL = 9
