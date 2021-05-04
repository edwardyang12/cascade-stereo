import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training, crop_width, crop_height, test_crop_width, test_crop_height):
        self.datapath = datapath
        self.training = training
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.test_crop_width = test_crop_width
        self.test_crop_height = test_crop_height


        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        left_images = [os.path.join(x,"0128_irL_denoised_half.png") for x in lines]
        right_images = [os.path.join(x,"0128_irR_denoised_half.png") for x in lines]

        if self.training:
            disp_images = [os.path.join(x,"depthL.png") for x in lines]
            return left_images, right_images, disp_images
        else:
            return left_images, right_images, None

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        img = Image.open(filename)
        img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
        data = np.asarray(img,dtype=np.float32)
        dis = 55*1387.095/data
        dis = dis / 256.
        return dis

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            #print("left_img: ", left_img.size, " right_img: ", right_img.size, " dis_gt: ", disparity.size)
            w, h = left_img.size
            crop_w, crop_h = self.crop_width, self.crop_height

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
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
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
