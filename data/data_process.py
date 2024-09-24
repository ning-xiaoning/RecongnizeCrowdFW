import shutil
import logging
import logging.config
from pathlib import Path
import sys
sys.path.append(".")
from utils import read_json
from datetime import timedelta
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.io import loadmat
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class ShanghaiTechADataPre:
    def __init__(self,input_dir, saved_dir, name, min_size, max_size, save_dm,):
        self.input_dir = os.path.join(input_dir,name)
        self.saved_dir = os.path.join(saved_dir, name)
        self.min_size, self.max_size = min_size, max_size
        self.save_dm=save_dm

    def cal_new_size(self, im_h, im_w):
        min_size, max_size = self.min_size, self.max_size
        if im_h < im_w:
            if im_h < min_size:
                ratio = 1.0 * min_size / im_h
                im_h = min_size
                im_w = round(im_w*ratio)
            elif im_h > max_size:
                ratio = 1.0 * max_size / im_h
                im_h = max_size
                im_w = round(im_w*ratio)
            else:
                ratio = 1.0
        else:
            if im_w < min_size:
                ratio = 1.0 * min_size / im_w
                im_w = min_size
                im_h = round(im_h*ratio)
            elif im_w > max_size:
                ratio = 1.0 * max_size / im_w
                im_w = max_size
                im_h = round(im_h*ratio)
            else:
                ratio = 1.0
        return im_h, im_w, ratio
    
    def get_image_and_target(self,img_path):
        im = Image.open(img_path)
        im_w, im_h = im.size
        name = img_path.split('/')[-1].split('.')[0]
        mat_path = os.path.abspath(os.path.join(img_path, os.pardir, os.pardir, 'ground-truth', 'GT_' + name + '.mat'))
        points = loadmat(mat_path)['image_info'][0][0][0][0][0].astype(np.float32)
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
        points = points[idx_mask]
        im_h, im_w, rr = self.cal_new_size(im_h, im_w)
        im = np.array(im)
        if rr != 1.0:
            im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
            points = points * rr
        return Image.fromarray(im), points
    
    def generate_density(self, img, points):
        '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
        return:
        density: the density-map we want. Same shape as input image but only has one channel.
        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
        '''
        img_shape=[img.size[1],img.size[0]]
        #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        #print ('generate density...')
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            sigma = 4 #np.average(np.array(gt.shape))/2./2. #case: 1 point
            density += gaussian_filter(pt2d, sigma, truncate=7/sigma, mode='constant')
        return density
    
    def save_density_map(self, density, filepath):
        normalized_density= density / np.max(density)
        # 使用Matplotlib的jet颜色映射
        cmap_jet = plt.get_cmap('jet')
        density_map_colored = (cmap_jet(normalized_density)[:, :, :3] * 255).astype(np.uint8)

        # 将Matplotlib的颜色映射图像转换为OpenCV格式
        # OpenCV期望的是BGR格式，而Matplotlib是RGB
        density_map_colored = cv2.cvtColor(density_map_colored, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, density_map_colored)
    
    def process(self):
        print("Start process dataset.")
        for phase in ['train', 'test']:
            sub_dir = os.path.join(self.input_dir, phase+ "_data")
            sub_save_dir = os.path.join(self.saved_dir, phase)
            
            img_path = os.path.join(sub_dir, 'images')
            img_list = glob(os.path.join(img_path, '*jpg'))
            for img_path in tqdm(img_list):
                img_name = os.path.basename(img_path)[:-4]
                img_dir_path =  os.path.join(sub_save_dir, img_name)
                if not os.path.exists(img_dir_path):
                    os.makedirs(img_dir_path)
                else:
                    continue
                img, points = self.get_image_and_target(img_path)
                density= self.generate_density(img, points)

                img_saved_path = os.path.join(img_dir_path, f"{img_name}.jpg")
                img.save(img_saved_path)
                points_save_path =  os.path.join(img_dir_path,f"{img_name}_points.npy")
                np.save(points_save_path, points)
                density_save_path =  os.path.join(img_dir_path,f"{img_name}_density.npy")
                np.save(density_save_path,density)

                if self.save_dm:
                    density_map_save_path =  os.path.join(img_dir_path,f"{img_name}_density_map.jpg")
                    self.save_density_map(density,density_map_save_path)
        print("Finished process dataset.")


class DataPre:
    def __init__(self, pre_config="data/data_pre_config.json",default_level=logging.INFO):
        pre_config = Path(pre_config)
        if pre_config.is_file():
            self.config = read_json(pre_config)
        else:
            print("Warning: preprocess configuration file is not found in {}.".format(pre_config))
            logging.basicConfig(level=default_level)
    
    def run(self):
        dname = self.config["dataset"]["name"]
        dparams = self.config["dataset"]["params"]
        input_dir = self.config["input_dir"]
        saved_dir = self.config["saved_dir"]
        if dname == "ShanghaiTechA":
            sta = ShanghaiTechADataPre(input_dir, saved_dir,dname, **dparams)
            sta.process()
        else:
            print("The dataset has not been implemented!")
            exit(-1)
        
    
if __name__=="__main__":
    dp = DataPre("data/data_pre_config.json")
    dp.run()
    print("End")
        