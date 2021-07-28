"""
Author: Isabella Liu 7/18/21
Feature: Some util functions for messy-table-dataset
"""

import os, sys
import logging
import pickle
import numpy as np
from datetime import datetime
from utils.messytable_dataset_config import cfg


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_split_files(split_file, debug=False, sub=100, isTest=False, onReal=False):
    """
    :param split_file: Path to the split .txt file, e.g. train.txt
    :param debug: Debug mode, load less data
    :param sub: If debug mode is enabled, sub will be the number of data loaded
    :param isTest: Whether on test, if test no random shuffle
    :param onReal: Whether test on real dataset, folder and file names are different
    :return: Lists of paths to the entries listed in split file
    """
    dataset = cfg.REAL.DATASET if onReal else cfg.DIR.DATASET
    img_left_name = cfg.REAL.LEFT if onReal else cfg.SPLIT.LEFT
    img_right_name = cfg.REAL.RIGHT if onReal else cfg.SPLIT.RIGHT

    with open(split_file, 'r') as f:
        prefix = [line.strip() for line in f]
        if isTest is False:
            np.random.shuffle(prefix)

        img_L = [os.path.join(os.path.join(dataset, p), img_left_name) for p in prefix]
        img_R = [os.path.join(os.path.join(dataset, p), img_right_name) for p in prefix]
        img_depth_l = [os.path.join(os.path.join(cfg.DIR.DATASET, p), cfg.SPLIT.DEPTHL) for p in prefix]
        img_depth_r = [os.path.join(os.path.join(cfg.DIR.DATASET, p), cfg.SPLIT.DEPTHR) for p in prefix]
        img_meta = [os.path.join(os.path.join(cfg.DIR.DATASET, p), cfg.SPLIT.META) for p in prefix]
        img_label = [os.path.join(os.path.join(cfg.REAL.DATASET, p), cfg.SPLIT.LABEL) for p in prefix]

        if debug is True:
            img_L = img_L[:sub]
            img_R = img_R[:sub]
            img_depth_l = img_depth_l[:sub]
            img_depth_r = img_depth_r[:sub]
            img_meta = img_meta[:sub]
            img_label = img_label[:sub]
    return img_L, img_R, img_depth_l, img_depth_r, img_meta, img_label


def get_time_string():
    """
    :return: Datetime in '%d_%m_%Y_%H_%M_%S' format
    """
    now = datetime.now()
    dt_string = now.strftime('%m_%d_%Y_%H_%M_%S')
    return dt_string


def setup_logger(name, distributed_rank, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    # Img_L, Img_R, Img_depth_l, Img_depth_r, Img_meta, Img_label = get_split_files(cfg.SPLIT.VAL,
    #                                                                               isTest=True, onReal=True)
    Img_L, Img_R, Img_depth_l, Img_depth_r, Img_meta, Img_label = get_split_files(cfg.SPLIT.TRAIN,
                                                                                  isTest=False, onReal=False)
    print(Img_L)
    print(Img_R)
    print(Img_depth_l)
    print(Img_depth_r)
    print(Img_meta)
    print(Img_label)