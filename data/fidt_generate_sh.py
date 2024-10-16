import glob
import math
import os
import torch
import cv2
import h5py
import shutil
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

'''change your path'''
root = '../data/raw/ShanghaiTechA'
save_path = '../data/processed/SH_test'

# part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
# part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
# part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
# part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

train_dir = os.path.join(root, 'train_data', 'images')
test_dir = os.path.join(root, 'test_data', 'images')
# save_path_train = os.path.join(save_path, 'train')
# save_path_test = os.path.join(save_path, 'test')

# if not os.path.exists(save_path_train):
#     os.makedirs(save_path_train)
# if not os.path.exists(save_path_test):
#     os.makedirs(save_path_test)


path_sets = [train_dir, test_dir]

# if not os.path.exists(part_A_train.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_A_train.replace('images', 'gt_fidt_map'))
#
# if not os.path.exists(part_A_test.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_A_test.replace('images', 'gt_fidt_map'))
#
# if not os.path.exists(part_A_train.replace('images', 'gt_show')):
#     os.makedirs(part_A_train.replace('images', 'gt_show'))
#
# if not os.path.exists(part_A_test.replace('images', 'gt_show')):
#     os.makedirs(part_A_test.replace('images', 'gt_show'))
#
# if not os.path.exists(part_B_train.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_B_train.replace('images', 'gt_fidt_map'))
#
# if not os.path.exists(part_B_test.replace('images', 'gt_fidt_map')):
#     os.makedirs(part_B_test.replace('images', 'gt_fidt_map'))
#
# if not os.path.exists(part_B_train.replace('images', 'gt_show')):
#     os.makedirs(part_B_train.replace('images', 'gt_show'))
#
# if not os.path.exists(part_B_test.replace('images', 'gt_show')):
#     os.makedirs(part_B_test.replace('images', 'gt_show'))

img_paths = {}
train_img_paths = []
test_img_paths = []
for i, path in enumerate(path_sets):
    if i == 0:
        for img_name in os.listdir(path):
            if img_name.endswith('.jpg'):
                train_img_paths.append({f'{img_name.split(".")[0]}': os.path.join(path, img_name)})
        img_paths.update({'train': train_img_paths})
    if i == 1:
        for img_name in os.listdir(path):
            if img_name.endswith('.jpg'):
                test_img_paths.append({f'{img_name.split(".")[0]}': os.path.join(path, img_name)})
        img_paths.update({'test': test_img_paths})


def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map


for k, v in img_paths.items():
    for dic_img in v:
        for img_name, img_path in dic_img.items():
            img_save_path = os.path.join(save_path, k, img_name)
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            print(img_path)
            Img_data = cv2.imread(img_path)

            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
            Gt_data = mat["image_info"][0][0][0][0][0]

            fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

            kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
            for i in range(0, len(Gt_data)):
                if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
                    kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

            shutil.copy(img_path, os.path.join(img_save_path, f'{img_name}.jpg'))

            with h5py.File(os.path.join(img_save_path, f'{img_name}.h5'), 'w') as hf:
                hf['fidt_map'] = fidt_map1
                hf['kpoint'] = kpoint

            fidt_map1 = fidt_map1
            fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
            fidt_map1 = fidt_map1.astype(np.uint8)
            fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

            '''for visualization'''
            cv2.imwrite(os.path.join(img_save_path, f'{img_name}_gt_show.jpg'), fidt_map1)
