from __future__ import print_function, division
import argparse
import os, sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from datasets.messytable_dataset import MessytableDataset
from models import __models__, __loss__
from utils import *
from utils.metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
from utils.messytable_dataset_config import cfg
import gc

cudnn.benchmark = True
assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')

# Model parameters
parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys())
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"],
                    help='predicted disp detach, undetach')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Apex and distributed training configuration
parser.add_argument("--local_rank", type=int, default=0, help='rank of device in distributed training')
parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

# Messytable dataset configuration
parser.add_argument('--config-file', type=str, default='./CasStereoNet/configs/local_train_config.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--color-jitter', action='store_true', help='whether apply color jitter in data augmentation')
parser.add_argument('--gaussian-blur', action='store_true', help='whether apply gaussian blur in data augmentation')
parser.add_argument('--debug', action='store_true', help='whether run in debug mode')
parser.add_argument('--warp-op', action='store_true', help='whether use warp_op function to get disparity')

args = parser.parse_args()
cfg.merge_from_file(args.config_file)
os.makedirs(args.logdir, exist_ok=True)

# Use sync_bn by using nvidia-apex, need to install apex.
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

# Distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

# Set seed
set_random_seed(args.seed)

# Create summary logger and print args
if (not is_distributed) or (dist.get_rank() == 0):
    print("argv:", sys.argv[1:])
    print_args(args)
    print(f'Runing with configs : \n {cfg}')
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

# Create model and model_loss
model = __models__[args.model](
                            maxdisp=cfg.ARGS.MAX_DISP,
                            ndisps=[int(nd) for nd in cfg.ARGS.NDISP],
                            disp_interval_pixel=[float(d_i) for d_i in cfg.ARGS.DISP_INTER_R],
                            cr_base_chs=[int(ch) for ch in cfg.ARGS.CR_BASE_CHS],
                            grad_method=args.grad_method,
                            using_ns=cfg.ARGS.USING_NS,
                            ns_size=cfg.ARGS.NS_SIZE
                           )
if args.sync_bn:
    import apex
    print("using apex synced BN")
    model = apex.parallel.convert_syncbn_model(model)
model_loss = __loss__[args.model]
model.cuda()
if dist.get_rank() == 0:
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999))

# Load parameters if ckpt is provided
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
if dist.get_rank() == 0:
    print("start at epoch {}".format(start_epoch))

# Initialize Amp
if args.using_apex:
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
# Enable Multiprocess training
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


# Dataset, dataloader
train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, args.gaussian_blur, args.color_jitter, args.debug, sub=100)
val_dataset = MessytableDataset(cfg.SPLIT.VAL, args.gaussian_blur, args.color_jitter, args.debug, sub=100)

if is_distributed:
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                        rank=dist.get_rank())
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                       rank=dist.get_rank())

    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, cfg.SOLVER.BATCH_SIZE, sampler=train_sampler,
                                                 num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True, pin_memory=True)
    ValImgLoader = torch.utils.data.DataLoader(val_dataset, cfg.SOLVER.BATCH_SIZE, sampler=val_sampler,
                                                num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False, pin_memory=True)

else:
    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                 shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

    ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)


num_stage = len([int(nd) for nd in cfg.ARGS.NDISP])


def train():
    Cur_err = np.inf
    for epoch_idx in range(start_epoch, cfg.SOLVER.EPOCHS):
        adjust_learning_rate(optimizer, epoch_idx, cfg.SOLVER.LR, cfg.SOLVER.LR_EPOCHS)

        # Training
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            loss, scalar_outputs = train_sample(sample)
            if (not is_distributed) or (dist.get_rank() == 0):
                avg_train_scalars.update(scalar_outputs)

        # Calculate average error in the main process
        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metrics = avg_train_scalars.mean()
            print(f'Epoch {epoch_idx} train total_err_metrics: {total_err_metrics}')

            # Add lr to dict and save results to tensorboard
            total_err_metrics.update({'lr': optimizer.param_groups[0]['lr']})
            save_scalars(logger, 'train', total_err_metrics, epoch_idx)

            # Save checkpoints
            if (epoch_idx + 1) % args.save_freq == 0:
                if (not is_distributed) or (dist.get_rank() == 0):
                    checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                    save_filename = "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx)
                    torch.save(checkpoint_data, save_filename)
        gc.collect()

        # Validation
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            loss, scalar_outputs = test_sample(sample)
            if (not is_distributed) or (dist.get_rank() == 0):
                avg_test_scalars.update(scalar_outputs)

        # Calculate average error and save checkpoint in the main process
        if (not is_distributed) or (dist.get_rank() == 0):
            # Get average results among all batches
            total_err_metrics = avg_test_scalars.mean()
            print(f'Epoch {epoch_idx} val   total_err_metrics: {total_err_metrics}')
            save_scalars(logger, 'val', total_err_metrics, epoch_idx)

            # Save best checkpoints
            if (not is_distributed) or (dist.get_rank() == 0):
                New_err = total_err_metrics["depth_abs_err"][0]
                if New_err < Cur_err:
                    Cur_err = New_err
                    checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(),
                                       'optimizer': optimizer.state_dict()}
                    save_filename = "{}/checkpoint_best.ckpt".format(args.logdir)
                    torch.save(checkpoint_data, save_filename)
                    print("Best Checkpoint epoch_idx:{}".format(epoch_idx))
        gc.collect()


# train one sample
def train_sample(sample):
    model.train()

    # Load data
    imgL = sample['img_L'].cuda()
    imgR = sample['img_R'].cuda()
    disp_gt = sample['img_disp_l'].cuda()
    depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].cuda()
    img_baseline = sample['baseline'].cuda()

    disp_gt = F.interpolate(disp_gt, (256, 512))  # [bs, H, W]
    depth_gt = F.interpolate(depth_gt, (256, 512))  # [bs, 1, H, W]

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].cuda()
        img_disp_r = F.interpolate(img_disp_r, (256, 512))  # [bs, H, W]
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r

    disp_gt = disp_gt.squeeze(1)
    optimizer.zero_grad()

    outputs = model(imgL, imgR)
    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)  # Note in training we do not exclude bg
    loss = model_loss(outputs, disp_gt, mask, dlossw=[float(e) for e in cfg.ARGS.DLOSSW])

    outputs_stage = outputs["stage{}".format(num_stage)]
    disp_pred = outputs_stage['pred']  # [bs, H, W]
    del outputs

    # Compute error metrics
    scalar_outputs = {"loss": loss}
    err_metrics = compute_err_metric(disp_gt.unsqueeze(1),
                    depth_gt,
                    disp_pred.unsqueeze(1),
                    img_focal_length,
                    img_baseline,
                    mask.unsqueeze(1))
    scalar_outputs.update(err_metrics)

    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    imgL = sample['img_L'].cuda()
    imgR = sample['img_R'].cuda()
    disp_gt = sample['img_disp_l'].cuda()
    depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].cuda()
    img_baseline = sample['baseline'].cuda()

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].cuda()
        disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r

    disp_gt = F.interpolate(disp_gt, (256, 512)).squeeze(1) # [bs, H, W]
    depth_gt = F.interpolate(depth_gt, (256, 512))

    outputs = model_eval(imgL, imgR)
    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)
    loss = torch.tensor(0, dtype=imgL.dtype, device=imgL.device, requires_grad=False)
    # loss = model_loss(outputs, disp_gt, mask, dlossw=[float(e) for e in cfg.ARGS.DLOSSW])

    outputs_stage = outputs["stage{}".format(num_stage)]
    disp_pred = outputs_stage["pred"]

    # Compute error metrics
    scalar_outputs = {"loss": loss}
    err_metrics = compute_err_metric(disp_gt.unsqueeze(1),
                                     depth_gt,
                                     disp_pred.unsqueeze(1),
                                     img_focal_length,
                                     img_baseline,
                                     mask.unsqueeze(1))
    scalar_outputs.update(err_metrics)

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
