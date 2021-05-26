import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import pickle
from datasets.warp_ops import *
import torch
Image.LOAD_TRUNCATED_IMAGES = True

class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training, crop_width, crop_height, test_crop_width, test_crop_height):
        self.datapath = datapath
        self.training = training
        self.left_filenames, self.right_filenames, self.disp_filenames_L, self.disp_filenames_R, self.meta_filenames = self.load_path(list_filename)

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.test_crop_width = test_crop_width
        self.test_crop_height = test_crop_height


        if self.training:
            assert self.disp_filenames_L is not None
            assert self.disp_filenames_R is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        left_images = [os.path.join(x,"1024_irL_real_1080.png") for x in lines]
        right_images = [os.path.join(x,"1024_irR_real_1080.png") for x in lines]

        disp_images_L = [os.path.join(x,"depthL.png") for x in lines]
        disp_images_R = [os.path.join(x,"depthR.png") for x in lines]
        meta = [os.path.join(x,"meta.pkl") for x in lines]
        return left_images, right_images, disp_images_L, disp_images_R, meta


    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def load_image(self, filename):
        img = Image.open(filename).convert('RGB')
        img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
        return img

    def load_disp(self, filename_L, filename_R, metafile):
        img_L = Image.open(filename_L)
        img_R = Image.open(filename_R)
        meta = self.load_pickle(metafile)

        img_L = img_L.resize((int(img_L.size[0]/2),int(img_L.size[1]/2)))
        img_R = img_R.resize((int(img_R.size[0]/2),int(img_R.size[1]/2)))
        data_L = np.asarray(img_L,dtype=np.float32)
        data_R = np.asarray(img_R,dtype=np.float32)

        #print(meta)
        el = meta['extrinsic_l'][:3,3]
        er = meta['extrinsic_r'][:3,3]

        b = np.linalg.norm(el-er)*1000
        f = meta['intrinsic_r'][0][0]/2

        mask_l = (data_L == 0)
        mask_r = (data_R == 0)
        dis_L = b*f/data_L
        dis_L[mask_l] = 0
        dis_R = b*f/data_R
        dis_R[mask_r] = 0
        return b, f, data_L, data_R, dis_L, dis_R

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames_L:  # has disparity ground truth
            b, f, depthL, depthR, disparity_L, disparity_R = self.load_disp(os.path.join("/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training", self.disp_filenames_L[index]), \
                                                    os.path.join("/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training", self.disp_filenames_R[index]), \
                                                    os.path.join("/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training", self.meta_filenames[index]))
            #print(type(disparity_R), disparity_R.shape)
            #disparity_R_t = torch.tensor(disparity_R)
            #disparity_R_ti = torch.tensor(disparity_R, dtype=torch.int)
            #disparity_R_t = disparity_R_t.reshape((1,1,disparity_R_t.shape[0],disparity_R_t.shape[1]))
            #disparity_R_ti = disparity_R_ti.reshape((1,1,disparity_R_ti.shape[0],disparity_R_ti.shape[1]))
            #disparity_L_from_R = apply_disparity_cu(disparity_R_t, disparity_R_ti)

        else:
            disparity_L = None
            disparity_R = None
            #disparity_L_from_R = None

        if self.training:
            #print("left_img: ", left_img.size, " right_img: ", right_img.size, " dis_gt: ", disparity.size)
            w, h = left_img.size
            crop_w, crop_h = self.crop_width, self.crop_height

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity_L = disparity_L[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_R = disparity_R[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity_R}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = self.test_crop_height - h
            right_pad = self.test_crop_width - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity_R is not None:
                assert len(disparity_R.shape) == 2
                disparity_R = np.lib.pad(disparity_R, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity_R is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity_R,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "depth": depthL,
                        "baseline": b,
                        "f": f}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
