import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import pandas as pd
from math import inf
import cv2

class CustomDatasetTest(Dataset):
    def __init__(self, datapath, list_filename, training, crop_width, crop_height, test_crop_width, test_crop_height):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.meta_filenames, self.label_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.test_crop_width = test_crop_width
        self.test_crop_height = test_crop_height

        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        disp_images = [x[2] for x in splits]
        meta = [x[3] for x in splits]
        label = [x[4] for x in splits]

        return left_images, right_images, disp_images, meta, label

    def load_image(self, filename):
        img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
        # img = cv2.GaussianBlur(img,(9, 9),0.1,2)
        return Image.fromarray(img.astype(np.uint8))

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        else:
            disparity = None

        if self.meta_filenames:
            temp = pd.read_pickle(os.path.join(self.datapath, self.meta_filenames[index]))
            intrinsic = temp['intrinsic']
            baseline = abs((temp['extrinsic_l']-temp['extrinsic_r'])[0][3])

            temp = disparity*256.

            temp = (baseline*1000*intrinsic[0][0]/2)/(temp)
            temp[temp==inf] = 0
            disparity = temp

        if self.label_filenames:
            temp = os.path.join(self.datapath, self.label_filenames[index])
            label = np.array(Image.open(temp).resize((960,540), resample=Image.NEAREST))

        w, h = left_img.size

        # normalize
        processed = get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()


        top_pad = self.test_crop_height - h
        right_pad = self.test_crop_width - w
        # assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
        # pad disparity gt
        if disparity is not None:
            assert len(disparity.shape) == 2
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            label = np.lib.pad(label, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        if disparity is not None:
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "label": label,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "intrinsic": intrinsic,
                    "baseline": baseline}
        else:
            return {"left": left_img,
                    "right": right_img,
                    "label": label,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index],
                    "intrinsic": intrinsic,
                    "baseline": baseline}
