"""
Author: Isabella Liu 7/18/21
Feature: Some util functions for messy-table-dataset
"""

import os, sys
import logging
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
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


def gen_error_colormap_depth():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 2000./(2**10) , 49, 54, 149],
         [2000./(2**10) , 2000./(2**9) , 69, 117, 180],
         [2000./(2**9) , 2000./(2**8) , 116, 173, 209],
         [2000./(2**8), 2000./(2**7), 171, 217, 233],
         [2000./(2**7), 2000./(2**6), 224, 243, 248],
         [2000./(2**6), 2000./(2**5), 254, 224, 144],
         [2000./(2**5), 2000./(2**4), 253, 174, 97],
         [2000./(2**4), 2000./(2**3), 244, 109, 67],
         [2000./(2**3), 2000./(2**2), 215, 48, 39],
         [2000./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def gen_error_colormap_disp():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1., dilate_radius=1):
    D_gt_np = D_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = D_est_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]


def disp_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    D_gt_np = D_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = D_est_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap_disp()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]


def save_img(log_dir, prefix,
             pred_disp_np, gt_disp_np, pred_disp_err_np,
             pred_depth_np, gt_depth_np, pred_depth_err_np):
    disp_path = os.path.join('pred_disp', prefix) + '.png'
    disp_gt_path = os.path.join('gt_disp', prefix) + '.png'
    disp_abs_err_cm_path = os.path.join('pred_disp_abs_err_cmap', prefix) + '.png'
    depth_path = os.path.join('pred_depth', prefix) + '.png'
    depth_gt_path = os.path.join('gt_depth', prefix) + '.png'
    depth_abs_err_cm_path = os.path.join('pred_depth_abs_err_cmap', prefix) + '.png'

    # Save predicted images
    masked_pred_disp_np = np.ma.masked_where(pred_disp_np == -1, pred_disp_np)  # mark background as red
    custom_cmap = plt.get_cmap('viridis').copy()
    custom_cmap.set_bad(color='red')
    plt.imsave(os.path.join(log_dir, disp_path), masked_pred_disp_np, cmap=custom_cmap, vmin=0, vmax=cfg.ARGS.MAX_DISP)

    masked_pred_depth_np = np.ma.masked_where(pred_depth_np == -1, pred_depth_np)  # mark background as red
    plt.imsave(os.path.join(log_dir, depth_path), masked_pred_depth_np, cmap=custom_cmap, vmin=0, vmax=1.25)

    # Save ground truth images
    masked_gt_disp_np = np.ma.masked_where(gt_disp_np == -1, gt_disp_np)  # mark background as red
    plt.imsave(os.path.join(log_dir, disp_gt_path), masked_gt_disp_np, cmap=custom_cmap, vmin=0, vmax=cfg.ARGS.MAX_DISP)
    masked_gt_depth_np = np.ma.masked_where(gt_depth_np == -1, gt_depth_np)  # mark background as red
    plt.imsave(os.path.join(log_dir, depth_gt_path), masked_gt_depth_np, cmap=custom_cmap, vmin=0, vmax=1.25)

    # Save error images
    plt.imsave(os.path.join(log_dir, disp_abs_err_cm_path), pred_disp_err_np)
    plt.imsave(os.path.join(log_dir, depth_abs_err_cm_path), pred_depth_err_np)


def save_obj_err_file(total_obj_disp_err, total_obj_depth_err, log_dir):
    result = np.append(total_obj_disp_err[None], total_obj_depth_err[None], axis=0).T
    result = np.append(np.arange(cfg.SPLIT.OBJ_NUM)[:, None].astype(int), result, axis=-1)
    result = result.astype('str').tolist()
    head = [['     ', 'disp_err', 'depth_err']]
    result = head + result

    err_file = open(os.path.join(log_dir, 'obj_err.txt'), 'w')
    for line in result:
        content = ' '.join(line)
        err_file.write(content + '\n')
    err_file.close()


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