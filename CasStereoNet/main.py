from __future__ import print_function, division
import argparse
import os, sys, shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, __loss__
from utils import *
from utils.metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
import gc

cudnn.benchmark = True
assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=False, help='data path')
parser.add_argument('--test_dataset', required=False, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--test_datapath', required=False, help='data path')
parser.add_argument('--trainlist', required=False, help='training list')
parser.add_argument('--testlist', required=False, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--log_freq', type=int, default=50, help='log freq')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--mode', type=str, default="train", help='train or test mode')


parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')


parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

parser.add_argument('--crop_height', type=int, default=256, required=False, help="crop height")
parser.add_argument('--crop_width', type=int, default=512, required=False, help="crop width")
parser.add_argument('--test_crop_height', type=int, default=480, required=False, help="crop height")
parser.add_argument('--test_crop_width', type=int, default=720, required=False, help="crop width")

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

parser.add_argument('--debug', action='store_true', help='whether run in debug mode')
parser.add_argument('--warp_op', action='store_true', help='whether use warp_op function to get disparity')


# parse arguments
args = parser.parse_args()
os.makedirs(args.logdir, exist_ok=True)

#using sync_bn by using nvidia-apex, need to install apex.
if args.sync_bn:
    assert args.using_apex, "must set using apex and install nvidia-apex"
if args.using_apex:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

#dis
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
# is_distributed = num_gpus > 1
is_distributed = False
args.is_distributed = is_distributed

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

#set seed
set_random_seed(args.seed)

if (not is_distributed) or (dist.get_rank() == 0):
    # create summary logger
    print("argv:", sys.argv[1:])
    print_args(args)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

# model
model = __models__[args.model](
                            maxdisp=args.maxdisp,
                            ndisps=[int(nd) for nd in args.ndisps.split(",") if nd],
                            disp_interval_pixel=[float(d_i) for d_i in args.disp_inter_r.split(",") if d_i],
                            cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                            grad_method=args.grad_method,
                            using_ns=args.using_ns,
                            ns_size=args.ns_size
                           )
if args.sync_bn:
    import apex
    print("using apex synced BN")
    model = apex.parallel.convert_syncbn_model(model)

model_loss = __loss__[args.model]
model.cuda()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if (fn.endswith(".ckpt") and not fn.endswith("best.ckpt"))]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

if args.using_apex:
    # Initialize Amp
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

#conver model to dist
if is_distributed:
    print("Dist Train, Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        # find_unused_parameters=False,
        # this should be removed if we update BatchNorm stats
        # broadcast_buffers=False,
    )
else:
    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)


# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
Test_StereoDataset = __datasets__[args.test_dataset]
if args.dataset == 'messytable':
    from utils.messytable_dataset_config import cfg
    train_dataset = StereoDataset(cfg.SPLIT.TRAIN, args.debug, sub=100)
    test_dataset = Test_StereoDataset(cfg.SPLIT.VAL, args.debug, sub=10)
else:
    train_dataset = StereoDataset(args.datapath, args.trainlist, True,
                                  crop_height=args.crop_height, crop_width=args.crop_width,
                                  test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)
    test_dataset = Test_StereoDataset(args.test_datapath, args.testlist, False,
                                 crop_height=args.crop_height, crop_width=args.crop_width,
                                 test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)
if is_distributed:
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                        rank=dist.get_rank())
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                       rank=dist.get_rank())

    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                                 drop_last=True, pin_memory=True)
    TestImgLoader = torch.utils.data.DataLoader(test_dataset, args.test_batch_size, sampler=test_sampler, num_workers=1,
                                                drop_last=False, pin_memory=True)

else:
    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=1, drop_last=True)

    TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=1, drop_last=False)


num_stage = len([int(nd) for nd in args.ndisps.split(",") if nd])

def train():
    avg_test_scalars = None
    Cur_D1 = 1
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        total_loss = 0
        total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                             'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # do_summary = global_step % args.summary_freq == 0
            do_summary = True
            # loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            loss, err_metrics = train_sample(sample, compute_metrics=do_summary)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    # save_scalars(logger, 'train', scalar_outputs, global_step)
                    # save_images(logger, 'train', image_outputs, global_step)
                    for k in total_err_metrics.keys():
                        total_err_metrics[k] += err_metrics[k]
                    if isinstance(loss, (list, tuple)):
                        loss = loss[0]
                    total_loss += loss
                # del scalar_outputs, image_outputs
                if batch_idx % args.log_freq == 0:
                    if isinstance(loss, (list, tuple)):
                        loss = loss[0]
                    print('Epoch {}/{}, Iter {}/{}, lr {:.5f}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx,
                                                                                           args.epochs,
                                                                                           batch_idx,
                                                                                           len(TrainImgLoader),
                                                                                           optimizer.param_groups[0]["lr"],
                                                                                           loss, time.time() - start_time))
        for k in total_err_metrics.keys():
            total_err_metrics[k] /= len(TrainImgLoader)
        total_loss /= len(TrainImgLoader)
        print(f'Epoch {epoch_idx} train_loss: {total_loss} total_err_metrics: {total_err_metrics}')
        logger.add_scalar('loss/train', total_loss, global_step=epoch_idx)
        logger.add_scalar('disp_epe/train', total_err_metrics['epe'], global_step=epoch_idx)
        logger.add_scalar('disp_bad1/train', total_err_metrics['bad1'], global_step=epoch_idx)
        logger.add_scalar('disp_bad2/train', total_err_metrics['bad2'], global_step=epoch_idx)
        logger.add_scalar('depth_abs_err/train', total_err_metrics['depth_abs_err'], global_step=epoch_idx)
        logger.add_scalar('depth_abs_err_2mm/train', total_err_metrics['depth_err2'], global_step=epoch_idx)
        logger.add_scalar('depth_abs_err_4mm/train', total_err_metrics['depth_err4'], global_step=epoch_idx)
        logger.add_scalar('depth_abs_err_8mm/train', total_err_metrics['depth_err8'], global_step=epoch_idx)
        logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch_idx)

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            if (not is_distributed) or (dist.get_rank() == 0):
                checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                save_filename = "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx)
                torch.save(checkpoint_data, save_filename)
        gc.collect()

        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            # testing
            total_loss = 0
            total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                                 'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}

            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                # do_summary = global_step % args.summary_freq == 0
                do_summary = True
                loss, err_metrics = test_sample(sample, compute_metrics=do_summary)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        # save_scalars(logger, 'train', scalar_outputs, global_step)
                        # save_images(logger, 'train', image_outputs, global_step)
                        for k in total_err_metrics.keys():
                            total_err_metrics[k] += err_metrics[k]
                        if isinstance(loss, (list, tuple)):
                            loss = loss[0]
                        total_loss += loss
                    # avg_test_scalars.update(scalar_outputs)
                    # del scalar_outputs, image_outputs
                    # if batch_idx % args.log_freq == 0:
                    #     if isinstance(loss, (list, tuple)):
                    #         loss = loss[0]
                    #     print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                    #                                                                          batch_idx,
                    #                                                                          len(TestImgLoader), loss,
                    #                                                                          time.time() - start_time))
            # if (not is_distributed) or (dist.get_rank() == 0):
            #     avg_test_scalars = avg_test_scalars.mean()
            #     save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            #     print("avg_test_scalars", avg_test_scalars)

            for k in total_err_metrics.keys():
                total_err_metrics[k] /= len(TestImgLoader)
            total_loss /= len(TestImgLoader)
            print(f'Epoch {epoch_idx} total_err_metrics: {total_err_metrics}')
            logger.add_scalar('disp_epe/val', total_err_metrics['epe'], global_step=epoch_idx)
            logger.add_scalar('disp_bad1/val', total_err_metrics['bad1'], global_step=epoch_idx)
            logger.add_scalar('disp_bad2/val', total_err_metrics['bad2'], global_step=epoch_idx)
            logger.add_scalar('depth_abs_err/val', total_err_metrics['depth_abs_err'], global_step=epoch_idx)
            logger.add_scalar('depth_abs_err_2mm/val', total_err_metrics['depth_err2'], global_step=epoch_idx)
            logger.add_scalar('depth_abs_err_4mm/val', total_err_metrics['depth_err4'], global_step=epoch_idx)
            logger.add_scalar('depth_abs_err_8mm/val', total_err_metrics['depth_err8'], global_step=epoch_idx)

            # # saving bset checkpoints
            # if (not is_distributed) or (dist.get_rank() == 0):
            #     if avg_test_scalars is not None:
            #         New_D1 = avg_test_scalars["D1"][0]
            #         if New_D1 < Cur_D1:
            #             Cur_D1 = New_D1
            #             #save
            #             checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(),
            #                                'optimizer': optimizer.state_dict()}
            #             save_filename = "{}/checkpoint_best.ckpt".format(args.logdir)
            #             torch.save(checkpoint_data, save_filename)
            #             print("Best Checkpoint epoch_idx:{}".format(epoch_idx))

            gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    if args.dataset == 'messytable':
        imgL, imgR, disp_gt = sample['img_L'], sample['img_R'], sample['img_disp_l']
        depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
        img_focal_length = sample['focal_length'].cuda()
        img_baseline = sample['baseline'].cuda()
    else:
        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].cuda()
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r
    if args.dataset == 'messytable':
        disp_gt = F.interpolate(disp_gt, (256, 512)).squeeze(1) # [bs, H, W]
        depth_gt = F.interpolate(depth_gt, (256, 512))  # [bs, 1, H, W]

    optimizer.zero_grad()

    outputs = model(imgL, imgR)
    # mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    img_ground_mask = (depth_gt > 0) & (depth_gt < 1.25)
    img_ground_mask = img_ground_mask.squeeze(1)  # [bs, H, W]
    mask = (disp_gt < args.maxdisp) * (disp_gt > 0)
    loss = model_loss(outputs, disp_gt, mask, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    outputs_stage = outputs["stage{}".format(num_stage)]
    # disp_ests = [outputs_stage["pred1"], outputs_stage["pred2"], outputs_stage["pred3"]]
    disp_pred = outputs_stage['pred']  # [bs, H, W]
    del outputs

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        err_metrics = compute_err_metric(disp_gt.unsqueeze(1),
                        depth_gt,
                        disp_pred.unsqueeze(1),
                        img_focal_length,
                        img_baseline,
                        mask.unsqueeze(1))
        # with torch.no_grad():
        #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
        #     scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        #     scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        #     scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        #     scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        #     scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    # return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), image_outputs
    return tensor2float(scalar_outputs["loss"]), err_metrics

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    if args.dataset == 'messytable':
        imgL, imgR, disp_gt = sample['img_L'], sample['img_R'], sample['img_disp_l']
        depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
        img_focal_length = sample['focal_length'].cuda()
        img_baseline = sample['baseline'].cuda()
    else:
        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].cuda()
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r
    if args.dataset == 'messytable':
        disp_gt = F.interpolate(disp_gt, (256, 512)).squeeze(1) # [bs, H, W]
        depth_gt = F.interpolate(depth_gt, (256, 512))

    outputs = model_eval(imgL, imgR)

    # mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    img_ground_mask = (depth_gt > 0) & (depth_gt < 1.25)
    img_ground_mask = img_ground_mask.squeeze(1)  # [bs, H, W]
    mask = (disp_gt < args.maxdisp) * (disp_gt > 0)
    loss = torch.tensor(0, dtype=imgL.dtype, device=imgL.device, requires_grad=False)
    # loss = model_loss(outputs, disp_gt, mask, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    outputs_stage = outputs["stage{}".format(num_stage)]
    # disp_pred = [outputs_stage["pred"]]
    disp_pred = outputs_stage["pred"]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    err_metrics = compute_err_metric(disp_gt.unsqueeze(1),
                                     depth_gt,
                                     disp_pred.unsqueeze(1),
                                     img_focal_length,
                                     img_baseline,
                                     mask.unsqueeze(1))

    # scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)
        err_metrics = reduce_scalar_outputs(err_metrics)

    # return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), image_outputs
    return tensor2float(scalar_outputs["loss"]), err_metrics


def test_all():
    # testing
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        do_summary = batch_idx % args.summary_freq == 0
        loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=False)
        if (not is_distributed) or (dist.get_rank() == 0):
            avg_test_scalars.update(scalar_outputs)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, batch_idx)
                save_images(logger, 'test', image_outputs, batch_idx)
            del scalar_outputs, image_outputs
            if batch_idx % args.log_freq == 0:
                if isinstance(loss, (list, tuple)):
                    loss = loss[0]
                print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
                                                                             batch_idx,
                                                                             len(TestImgLoader), loss,
                                                                             time.time() - start_time))
    if (not is_distributed) or (dist.get_rank() == 0):
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TestImgLoader))
        print("avg_test_scalars", avg_test_scalars)

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_all()

