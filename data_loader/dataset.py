import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
import random
import os
from glob import glob
import re


import numpy as np
import h5py
import cv2

import sys
sys.path.append('.')

# from datasets.den_dataset import DensityMapDataset
from utils.misc import random_crop, get_padding

class ShangHaiTechDataset(Dataset):
    def __init__(self, root, crop_size, downsample, method, is_grey=False, unit_size=0, pre_resize=1):
        super(ShangHaiTechDataset, self).__init__()
        self.root = root
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.downsample = downsample
        self.method = method
        self.is_grey = is_grey
        self.unit_size = unit_size
        self.pre_resize = pre_resize

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),# 与下文存在重复
        ])

        self.more_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if self.method not in ['train', 'val', 'test']:
            raise ValueError('method must be train, val or test')
        
        self.img_paths = self.get_img_paths(os.path.join(self.root, method)) 
        
        if self.method in ['val', 'test']:
            self.img_paths = sorted( self.img_paths)
            
    def collate(self, batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0)
        images2 = torch.stack(transposed_batch[1], 0)
        points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[3], 0)
        bmaps = torch.stack(transposed_batch[4], 0)
        return images1, images2, (points, dmaps, bmaps)


    def get_img_paths(self, directory):
        img_paths = list()
        pattern = ".*\d+\.jpg"
        for root, dirs, files in os.walk(directory):
            # 对于当前目录中的文件，检查是否匹配模式
            for filename in files:
                if re.match(pattern, filename):
                    img_paths.append(os.path.join(root, filename))
        img_paths.sort()
        return img_paths

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img, dir_path, img_name, img_ext = self._load_img(img_path)

        gt_path = os.path.join(dir_path, f"{img_name}_points.npy")
        gt = self._load_gt(gt_path)

        if self.method == 'train':
            dmap_path = os.path.join(dir_path, f"{img_name}_density.npy")
            dmap = self._load_dmap(dmap_path)
            img1, img2, gt, dmap = self._train_transform(img, gt, dmap)
            bmap_orig = dmap.clone().reshape(1, dmap.shape[1]//16, 16, dmap.shape[2]//16, 16).sum(dim=(2, 4))
            bmap = (bmap_orig > 0).float()
            return img1, img2, gt, dmap, bmap
        elif self.method in ['val', 'test']:
            return self._val_transform(img, gt, img_name)
    
    def _load_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        # 使用os.path.basename获取路径中的最后一个部分（文件名）
        base_name = os.path.basename(img_path)
        dir_path = os.path.dirname(img_path)

        # 使用os.path.splitext分离文件名和后缀
        img_name, img_ext = os.path.splitext(base_name)
        
        return img, dir_path, img_name, img_ext
    
    def _load_gt(self, gt_path):
        gt = np.load(gt_path)
        return gt
    
    def _load_dmap(self, dmap_path):
        dmap = np.load(dmap_path)
        return dmap
    

        
    def _rotate_gt(self, gt, w, h, angle):
        gt = np.array(gt)
        gt = gt - [w/2, h/2]
        theta = angle / 180.0 * np.pi
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        gt = np.dot(gt, rot_mat)
        gt = gt + [w/2, h/2]
        return gt

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0

        dmap = torch.from_numpy(dmap).unsqueeze(0)

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < min(self.crop_size[0], self.crop_size[1]):
            padding, h, w = get_padding(h, w, self.crop_size[0], self.crop_size[1])
            left, top, _, _ = padding

            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            if len(gt) > 0:
                gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size[0], self.crop_size[1])
        h, w = self.crop_size[0], self.crop_size[1]
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]
        dmap = F.crop(dmap, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Downsampling
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmap = dmap.reshape([1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(2, 4))

        if len(gt) > 0:
            gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img1 = self.transform(img)
        img2 = self.more_transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()

        return img1, img2, gt, dmap

    def _val_transform(self, img, gt, name):
        if self.pre_resize != 1:
            img = img.resize((int(img.size[0] * self.pre_resize), int(img.size[1] * self.pre_resize)))

        if self.unit_size is not None and self.unit_size > 0:
            # Padding
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h

            padding, h, w = get_padding(h, w, new_h, new_w)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            if len(gt) > 0:
                gt = gt + [left, top]
        else:
            padding = (0, 0, 0, 0)

        # Downsampling
        gt = gt / self.downsample

        # Post-processing
        img1 = self.transform(img)
        img2 = self.more_transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img1, img2, gt, name, padding
    
    def __len__(self):
        return len(self.img_paths)

# FIDTM
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if self.args['preload_data'] == True:
            fname = self.lines[index]['fname']
            img = self.lines[index]['img']
            kpoint = self.lines[index]['kpoint']
            fidt_map = self.lines[index]['fidt_map']

        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, fidt_map, kpoint = load_data_fidt(img_path, self.args, self.train)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                fidt_map = np.ascontiguousarray(np.fliplr(fidt_map))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.ascontiguousarray(np.fliplr(kpoint))

        fidt_map = fidt_map.copy()
        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        '''crop size'''
        if self.train == True:
            fidt_map = torch.from_numpy(fidt_map).cuda()

            width = self.args['crop_size']
            height = self.args['crop_size']

            pad_y = max(0, width - img.shape[1])
            pad_x = max(0, height - img.shape[2])
            if pad_y + pad_x > 0:
                img = F.pad(img, [0, pad_x, 0, pad_y], value=0)
                fidt_map = F.pad(fidt_map, [0, pad_x, 0, pad_y], value=0)
                kpoint = np.pad(kpoint, [(0, pad_y), (0, pad_x)], mode='constant', constant_values=0)
            # print(img.shape)
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            fidt_map = fidt_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]

        return fname, img, fidt_map, kpoint

def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map')
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit


    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
        
if __name__ == '__main__':
    dataset = ShangHaiTechDataset("data/processed/ShanghaiTechA", 320, 1, 'train', False, 1)
    print(len(dataset))
    count0 = 0
    count1 = 0
    for i in range(len(dataset)):
        img, _, gt, dmap, bmap = dataset[i]
        # print(bmap)
        # break
        count0 += (bmap == 0).sum()
        count1 += (bmap == 1).sum()

    print(count0, count1)