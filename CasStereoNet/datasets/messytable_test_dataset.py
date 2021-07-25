"""
Author: Isabella Liu 7/19/21
Feature: Load data from messy-table-dataset
"""

import os
import random
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset, DataLoader
from utils.messytable_dataset_config import cfg
from utils.messytable_util import get_split_files, load_pickle


class MessytableTestDataset(Dataset):
    def __init__(self, split_file, debug=False, sub=100, isTest=False, onReal=False):
        self.img_L, self.img_R, self.img_depth_l, self.img_depth_r, self.img_meta, self.img_label = \
            get_split_files(split_file, debug, sub, isTest, onReal)
        self.isTest = isTest
        self.onReal = onReal
        self.normalize_transform = Transforms.Compose([
            Transforms.ToTensor(),
            Transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        if self.onReal:
            img_L_rgb = np.array(Image.open(self.img_L[idx]).convert(mode='RGB'))
            img_R_rgb = np.array(Image.open(self.img_R[idx]).convert(mode='RGB'))
        else:
            img_L_rgb = np.array(Image.open(self.img_L[idx]))[:, :, :-1]
            img_R_rgb = np.array(Image.open(self.img_R[idx]))[:, :, :-1]

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000  # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000  # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])
        img_label = np.array(Image.open(self.img_label[idx]))

        # Convert depth map to disparity map
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        if not self.isTest:
            # random crop the image to a fixed size
            h, w = img_L_rgb.shape[:2]
            # th, tw = h//2, w//2
            th, tw = 256, 512
            x = random.randint(0, h - th)
            y = random.randint(0, w - tw)
            img_L_rgb = img_L_rgb[x:(x+th), y:(y+tw)]
            img_R_rgb = img_R_rgb[x:(x+th), y:(y+tw)]
            img_disp_l = img_disp_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]
            img_depth_l = img_depth_l[2*x: 2*(x+th), 2*y: 2*(y+tw)]
            img_disp_r = img_disp_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
            img_depth_r = img_depth_r[2*x: 2*(x+th), 2*y: 2*(y+tw)]
            img_label = img_label[2*x: 2*(x+th), 2*y: 2*(y+tw)]
        # else:
        #     if self.onReal: # real input image is 1080 * 1920
        #         img_L_rgb = img_L_rgb[28:-28]
        #         img_R_rgb = img_R_rgb[28:-28]
        #     else:
        #         img_L_rgb = img_L_rgb[14:-14]
        #         img_R_rgb = img_R_rgb[14:-14]
        #     img_disp_l = img_disp_l[28:-28]
        #     img_depth_l = img_depth_l[28:-28]
        #     img_disp_r = img_disp_r[28:-28]
        #     img_depth_r = img_depth_r[28:-28]
        #     img_label = img_label[28:-28]

        item = {}
        item['img_L'] = self.normalize_transform(img_L_rgb)
        item['img_R'] = self.normalize_transform(img_R_rgb)
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_label'] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = self.img_L[idx].split('/')[-2]
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return item



def get_test_loader(split_file, debug=False, sub=100, isTest=False, onReal=False):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableTestDataset(split_file, debug, sub, isTest=isTest, onReal=onReal)
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == '__main__':
    cdataset = MessytableTestDataset(cfg.SPLIT.VAL, isTest=True, onReal=True)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
