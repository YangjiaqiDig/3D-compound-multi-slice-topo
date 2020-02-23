import torch
import numpy as np
from pre_processing import *
import torch.nn as nn
from random import randint
from PIL import Image, ImageSequence
import glob
from torch.utils.data.dataset import Dataset

BATCH_SIZE = 2
IN_SIZE = 1250
OUT_SIZE = 1250
TRAIN_VALID_RATIO = 0.8


class CREMIDataTrain(Dataset):
    def __init__(self, image_path, mask_path, in_size=IN_SIZE, out_size=OUT_SIZE):
        self.image_arr = Image.open(str(image_path))
        self.mask_arr = Image.open(str(mask_path))
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self):
        img_as_np = []
        for i, img_as_img in enumerate(ImageSequence.Iterator(self.image_arr)):
            singleImage_as_np = np.asarray(img_as_img)
            img_as_np.append(singleImage_as_np)

        msk_as_np = []
        for j, label_as_img in enumerate(ImageSequence.Iterator(self.mask_arr)):
            singleLabel_as_np = np.asarray(label_as_img)
            msk_as_np.append(singleLabel_as_np)

        img_as_np = np.stack(img_as_np, axis=0)
        msk_as_np = np.stack(msk_as_np, axis=0)

        train_size = int(img_as_np.shape[0] * TRAIN_VALID_RATIO)
        img_as_np, msk_as_np = img_as_np[:train_size], msk_as_np[:train_size]

        # img1 = Image.fromarray(img_as_np[0])
        # img1.show()

        # flip {0: vertical, 1: horizontal, 2: both, 3: none}

        img_as_np, msk_as_np = flip(img_as_np, msk_as_np)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        if randint(0, 1):
            gaus_sd, gaus_mean = randint(0, 20), 0
            img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        else:
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

        # change brightness
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)

        # img1_after_process = Image.fromarray(img_as_np[0])
        # img1_after_process.show()
        # lab1_after_process = Image.fromarray(msk_as_np[0])
        # lab1_after_process.show()

        # Elastic distort {0: distort, 1:no distort}
        # sigma = randint(6, 12)
        # img_as_np, seed = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20) # 1250 -> 512
        # img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
        # pad_size = int((self.in_size - self.out_size) / 2)
        # img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
        # y_loc, x_loc = randint(0, img_height - self.out_size), randint(0, img_width - self.out_size)
        # img_as_np = cropping(img_as_np, crop_size=self.in_size, dim1=y_loc, dim2=x_loc)

        # Normalize
        img_as_np = normalization2(img_as_np.astype(float), max=1, min=0)
        msk_as_np = msk_as_np / 255

        img_as_tensor = torch.from_numpy(img_as_np).float()
        msk_as_tensor = torch.from_numpy(msk_as_np).long()

        # lab1 = Image.fromarray(msk_as_img[0])
        # lab1.show()

        # msk_as_np, _ = add_elastic_transform(msk_as_np, alpha=34, sigma=sigma, seed=seed, pad_size=20)
        # msk_as_np = approximate_image(msk_as_np)

        # msk_as_np = cropping(msk_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)

        # msk_as_np = np.expand_dims(msk_as_np, axis=0)
        return (img_as_tensor, msk_as_tensor)


class ComDataset(Dataset):
    def __init__(self, data_1, data_2, data_3):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3

    def __getitem__(self, index):
        x1, x2, x3 = self.data_1[index], self.data_2[index], self.data_3[index]

        return x1, x2, x3

    def __len__(self):
        return len(self.data_1)


class CREMIDataVal(Dataset):
    def __init__(self, image_path, mask_path, in_size=IN_SIZE, out_size=OUT_SIZE):
        self.image_arr = Image.open(str(image_path))
        self.mask_arr = Image.open(str(mask_path))
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self):
        img_as_np = []
        for i, img_as_img in enumerate(ImageSequence.Iterator(self.image_arr)):
            singleImage_as_np = np.asarray(img_as_img)
            img_as_np.append(singleImage_as_np)

        msk_as_np = []
        for j, label_as_img in enumerate(ImageSequence.Iterator(self.mask_arr)):
            singleLabel_as_np = np.asarray(label_as_img)
            msk_as_np.append(singleLabel_as_np)

        img_as_np = np.stack(img_as_np, axis=0)
        msk_as_np = np.stack(msk_as_np, axis=0)

        train_size = int(img_as_np.shape[0] * TRAIN_VALID_RATIO)
        img_as_np, msk_as_np = img_as_np[train_size:], msk_as_np[train_size:]

        # Normalize
        img_as_np = normalization2(img_as_np.astype(float), max=1, min=0)
        msk_as_np = msk_as_np / 255

        img_as_tensor = torch.from_numpy(img_as_np).float()
        msk_as_tensor = torch.from_numpy(msk_as_np).long()

        return (img_as_tensor, msk_as_tensor)


class CREMIDataTest(Dataset):

    def __init__(self, image_path, in_size=IN_SIZE, out_size=OUT_SIZE):
        self.image_arr = glob.glob(str(image_path) + "/*")
        self.in_size, self.out_size = in_size, out_size
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)

        pad_size = int((self.in_size - self.out_size) / 2)
        img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
        img_as_np = multi_cropping(img_as_np, crop_size=self.in_size, crop_num1=2, crop_num2=2)

        img1 = Image.fromarray(img_as_np)
        # img1.show()

        processed_list = []
        # Normalize
        for array in img_as_np:
            img_to_add = normalization2(array, max=1, min=0)
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)

        return img_as_tensor


if __name__ == "__main__":
    train = CREMIDataTrain('train/train-volume.tif', 'train/train-labels.tif')
    imgs, msk = train.__getitem__()
