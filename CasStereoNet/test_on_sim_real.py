"""
Author: Isabella Liu 7/19/21
Feature: Test cascade-stereo model on sim-real dataset
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models.psmnet import PSMNet
from datasets.messytable_test_dataset import get_test_loader
from utils.metrics import compute_err_metric, compute_obj_err
from utils.messytable_dataset_config import cfg
from utils.messytable_util import get_time_string, setup_logger
from utils.warp_ops import apply_disparity_cu

parser = argparse.ArgumentParser(description='Testing for Cascade-Stereo on messy-table-dataset')
parser.add_argument('--config-file', type=str, default='./CasStereoNet/configs/remote_test_config.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--model', type=str, default='', metavar='FILE', help='Path to test model')
parser.add_argument('--output', type=str, default='./testing_output', help='Path to output folder')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
parser.add_argument('--annotate', type=str, default='', help='Annotation to the experiment')
parser.add_argument('--onreal', action='store_true', default=False, help='Test on real dataset')
parser.add_argument('--analyze-objects', action='store_true', default=True, help='Analyze on different objects')
parser.add_argument('--exclude-bg', action='store_true', default=False, help='Exclude background when testing')
parser.add_argument('--warp_op', action='store_true', default=False, help='whether use warp_op function to get disparity')
args = parser.parse_args()
cfg.merge_from_file(args.config_file)


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


def test(model, val_loader, logger, log_dir):
    model.eval()
    total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                         'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}
    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, 'pred_disp'))
    os.mkdir(os.path.join(log_dir, 'gt_disp'))
    os.mkdir(os.path.join(log_dir, 'pred_disp_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'pred_depth'))
    os.mkdir(os.path.join(log_dir, 'gt_depth'))
    os.mkdir(os.path.join(log_dir, 'pred_depth_abs_err_cmap'))

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data['img_L'].cuda()
        img_R = data['img_R'].cuda()

        img_disp_l = data['img_disp_l'].cuda()
        img_depth_l = data['img_depth_l'].cuda()
        img_label = data['img_label'].cuda()
        img_focal_length = data['focal_length'].cuda()
        img_baseline = data['baseline'].cuda()
        prefix = data['prefix'][0]

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data['img_disp_r'].cuda()
            img_depth_r = data['img_depth_r'].cuda()
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(img_depth_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)
        if args.onreal:
            img_L = F.interpolate(img_L, (540, 960))
            img_R = F.interpolate(img_R, (540, 960))

        img_disp_l = F.interpolate(img_disp_l, (540, 960))
        img_depth_l = F.interpolate(img_depth_l, (540, 960))
        img_label = F.interpolate(img_label, (540, 960)).type(torch.int)

        # Pad the imput image and depth disp image to 960 * 544
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540
        img_L = F.pad(img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        img_R = F.pad(img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)

        if args.exclude_bg:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * img_ground_mask
        else:
            img_ground_mask = torch.ones_like(img_depth_l).type(torch.bool)
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0)
        mask = mask.type(torch.bool)
        mask.detach_()  # [bs, 1, H, W]

        ground_mask = torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(img_L, img_R)
        pred_disp = outputs['stage2']['pred']  # [bs, H, W]
        pred_disp = pred_disp.unsqueeze(1)  # [bs, 1, H, W]
        pred_disp = pred_disp[:, :, top_pad:, :]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m

        # Get loss metric
        err_metrics = compute_err_metric(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                         img_baseline, mask)
        for k in total_err_metrics.keys():
            total_err_metrics[k] += err_metrics[k]
        logger.info(f'Test instance {prefix} - {err_metrics}')

        # Get object error
        obj_disp_err, obj_depth_err, obj_count = compute_obj_err(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                                                 img_baseline, img_label, mask, cfg.SPLIT.OBJ_NUM)
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_count += obj_count

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        pred_disp_np[ground_mask] = -1

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_disp_np[ground_mask] = -1

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, img_ground_mask)

        # Get depth image
        pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        pred_depth_np[ground_mask] = -1

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_depth_np[ground_mask] = -1

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, img_ground_mask)

        del pred_disp, pred_depth, outputs, img_L, img_R

        # Save images
        save_img(log_dir, prefix, pred_disp_np, gt_disp_np, pred_disp_err_np,
                 pred_depth_np, gt_depth_np, pred_depth_err_np)

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f'\nTest on {len(val_loader)} instances\n {total_err_metrics}')

    # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    save_obj_err_file(total_obj_disp_err, total_obj_depth_err, log_dir)

    logger.info(f'Successfully saved object error to obj_err.txt')


def main():
    # Obtain the dataloader
    val_loader = get_test_loader(cfg.SPLIT.VAL, args.debug, sub=10, isTest=True, onReal=args.onreal)

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f'{get_time_string()}_{args.annotate}')
    os.mkdir(log_dir)
    logger = setup_logger("CascadeStereo Testing", distributed_rank=0, save_dir=log_dir)
    logger.info(f'Annotation: {args.annotate}')
    logger.info(f'Input args {args}')
    logger.info(f'Loaded config file \'{args.config_file}\'')
    logger.info(f'Running with configs:\n{cfg}')

    # Get the model
    logger.info(f'Loaded the checkpoint: {args.model}')
    model = PSMNet(
        maxdisp=cfg.ARGS.MAXDISP,
        ndisps=[int(nd) for nd in cfg.ARGS.NDISP.split(",") if nd],
        disp_interval_pixel=[float(d_i) for d_i in cfg.ARGS.DISP_INTER_R.split(",") if d_i],
        cr_base_chs=[int(ch) for ch in cfg.ARGS.CR_BASE_CHS.split(",") if ch],
        grad_method=cfg.ARGS.GRAD_METHOD,
        using_ns=cfg.ARGS.USING_NS,
        ns_size=cfg.ARGS.NS_SIZE
    )
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict['model'])
    model.cuda()
    test(model, val_loader, logger, log_dir)


if __name__ == '__main__':
    main()
