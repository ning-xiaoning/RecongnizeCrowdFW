import argparse
import torch
from tqdm import tqdm
import json
import glob
import nni
import math
import os
import re
import numpy as np

import data_loader.data_loaders as module_data
import model.MPCount.MPCount as module_arch
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from model.FIDTM.seg_hrnet import get_seg_model
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_loader.dataset import *
from utils.util import *


class Inference:
    def __init__(self, model_name, model, data_loader, loss_fn=None, metric_fns=None, device=None, config=None):
        self.model_name = model_name
        self.data_loader = data_loader
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        self.device = device
        self.config = config
        
        pass
    
    def inference(self):
        if self.model_name == "MPCount":
            total_loss, total_metrics, output_dict = self.mpcount()
            return total_loss, total_metrics, output_dict
        if self.model_name == "FIDTM":
            mae, visi = self.test_FIDTM()
            return mae, visi
        else:
            print("The model has not been specified!")
            exit(-1)

    
    def test_mpcount(self):
        
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        output_dict= {}
        device = self.device
        # mean_std = self.data_loader.dataset.mean_std
        # with torch.no_grad():
        #     for batch_idx, (x_g, t_g, cam_valid, time_stamp, flight) in enumerate(self.data_loader):
        #         tmp={}
        #         x_g, t_g, cam_valid,flight = x_g.to(device), t_g.to(device), cam_valid.to(device), flight.to(device)
        #         output = self.model.decode(x_g,flight[:,:,:,:])
        #         # output = model.multi_pred(x_g,flight[:,:,:,:])

        #         #
        #         # save sample images, or do something with output here
        #         #

        #         tmp['target']=z_inverse(t_g[:,:,:,:].cpu().numpy(),mean_std[0],mean_std[1]).tolist()
        #         tmp['prediction'] = z_inverse(output.cpu().numpy(),mean_std[0],mean_std[1]).tolist()
        #         tmp['time_stamp'] = time_stamp[-12:]
        #         # tmp['flight'] = flight[:,-1:,:,:]
        #         output_dict[batch_idx]=tmp.copy()
        #         # computing loss, metrics on test set
        #         loss = self.loss_fn(output, t_g[:,:,:,:])
        #         batch_size = x_g.shape[0]
        #         total_loss += loss.item() * batch_size
        #         # for i, metric in enumerate(metric_fns):
        #         #     total_metrics[i] += metric(output, t_g) * batch_size
        #         for i,met in enumerate(self.metric_fns):
        #             total_metrics[i] += met( t_g[:,:,:,:].cpu().numpy(),output.cpu().numpy(), mean_std,
        #                                                         cam_valid[:,:,:,:].cpu().numpy())
        # return total_loss, total_metrics, output_dict
        
        with torch.no_grad():
            for batch_idx, (imgs1, imgs2, gt_data) in enumerate(self.data_loader):
                pass

    def test_FIDTM(self):
        print('begin test')


        self.model.eval()

        mae = 0.0
        mse = 0.0
        visi = []
        index = 0

        if not os.path.exists('./local_eval/point_files'):
            os.makedirs('./local_eval/point_files')

        '''output coordinates'''
        f_loc = open("./local_eval/point_files/A_localization.txt", "w+")

        for i, (fname, img, fidt_map, kpoint) in enumerate(self.data_loader):

            count = 0
            img = img.cuda()

            if len(img.shape) == 5:
                img = img.squeeze(0)
            if len(fidt_map.shape) == 5:
                fidt_map = fidt_map.squeeze(0)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            if len(fidt_map.shape) == 3:
                fidt_map = fidt_map.unsqueeze(0)

            with torch.no_grad():
                d6 = self.model(img)

                '''return counting and coordinates'''
                count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, config)
                point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

                if config['visual'] == True:
                    if not os.path.exists(config['save_path'] + '_box/'):
                        os.makedirs(config['save_path'] + '_box/')
                    ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                    show_fidt = show_map(d6.data.cpu().numpy())
                    gt_show = show_map(fidt_map.data.cpu().numpy())
                    res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                    cv2.imwrite(config['save_path'] + '_box/' + fname[0], res)

            gt_count = torch.sum(kpoint).item()
            mae += abs(gt_count - count)
            mse += abs(gt_count - count) * abs(gt_count - count)

            if i % 1 == 0:
                print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
                visi.append(
                    [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                     fname])
                index += 1

        mae = mae * 1.0 / (len(self.data_loader) * config['batch_size'])
        mse = math.sqrt(mse / (len(self.data_loader)) * config['batch_size'])

        nni.report_intermediate_result(mae)
        print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

        return mae, visi
    
        
    


def main(config):
    # logger = config.get_logger('test')
    if config['name'] == "MPCount":
        data_loader = config.init_obj('data_loader', module_data)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # get function handles of loss and metrics
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Loading checkpoint: {} ...'.format(config.resume))


        if os.path.isdir(config["model_path"]):
            model_paths = glob.glob(os.path.join(config["model_path"], "*.pth"))
            model_paths.sort()
        else:
            model_paths = [config["model_path"]]

        saved_dir = config["saved_dir"]
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        for i, model_path in enumerate(model_paths):
            checkpoint = torch.load(model_path)
            state_dict = checkpoint['state_dict']
            if config['n_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)

            # prepare model for testing
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

            infer = Inference(model.name, model, data_loader, loss_fn, metric_fns, device)
            total_loss, total_metrics, output_dict = infer.inference()
            model_index = re.findall(r'\d+\.\d+|\d+', model_path)[-1]
            filename = f"{model.name}_checkpoint_{int(model_index)}.json"
            with open(os.path.join(saved_dir, filename), 'w', encoding='utf-8') as js:
                json.dump(output_dict, js, ensure_ascii=False)

            n_samples = len(data_loader.sampler)
            log = {'loss': total_loss / n_samples}
            log.update({
                met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
            })
            print('-'*100)
            print(model_path)
            # logger.info(log)
            print('-'*100)

    if config['name'] == 'FIDTM':

        with open(config['data_dir'], 'rb') as outfile:
            val_list = np.load(outfile).tolist()

        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
        model = get_seg_model()
        model = nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

        optimizer = torch.optim.Adam(
            [
                {'params': model.parameters(), 'lr': config['lr']},
            ])

        print(config['resume'])

        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        if config['resume']:
            if os.path.isfile(config['resume']):
                print("=> loading checkpoint '{}'".format(config['resume']))
                checkpoint = torch.load(config['resume'])
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                config['start_epoch'] = checkpoint['epoch']
                config['best_pred'] = checkpoint['best_prec1']
            else:
                print("=> no checkpoint found at '{}'".format(config['resume']))

        torch.set_num_threads(config['workers'])
        print(config['best_pred'], config['start_epoch'])

        if config['preload_data'] == True:
            test_data = pre_data(val_list, config, train=False)
        else:
            test_data = val_list

        test_loader = torch.utils.data.DataLoader(
            listDataset(test_data, config['save_path'],
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),

                                ]),
                                args=config, train=False),
            batch_size=config['batch_size'])

        '''inference '''
        infer = Inference(config['name'], model, test_loader, config=config)
        prec1, visi = infer.test_FIDTM()


        is_best = prec1 < config['best_pred']
        config['best_pred'] = min(prec1, config['best_pred'])

        print('\nThe visualizations are provided in ', config['save_path'])
        save_checkpoint({
            'arch': config['resume'],
            'state_dict': model.state_dict(),
            'best_prec1': config['best_pred'],
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, config['save_path'])


if __name__ == '__main__':
    config = argparse.ArgumentParser(description='PyTorch Template')
    config.add_argument('-c', '--config', default="config/test/FIDTM.json", type=str,
                      help='config file path (default: None)')
    config.add_argument('-r', '--resume', default="ShanghaiA/model_best_57.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    config.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # args = config.parse_args()
    # config = vars(args)
    # with open(config['config'], 'r') as f:
    #     dic = json.load(f)
    # config.update(dic)
    config = ConfigParser.from_args(config)
    main(config)
