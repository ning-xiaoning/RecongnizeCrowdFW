import argparse
import torch
from tqdm import tqdm
import json
import glob
import os
import re

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from utils.misc import denormalize, divide_img_into_patches


class Inference:
    def __init__(self, model_name, model, data_loader, loss_fn, metric_fns, device):
        self.model_name = model_name
        self.data_loader = data_loader
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns= metric_fns
        self.device= device
        
        pass
    
    def inference(self):
        if self.model_name == "MPCount":
            total_loss, total_metrics, output_dict = self.test_mpcount()
        elif self.model_name == "FIDTM":
            total_loss, total_metrics, output_dict = self.test_fidtm()
        else:
            print("The model has not been specified!")
            exit(-1)
        return total_loss, total_metrics, output_dict
    
    def test_mpcount(self):
        
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        output_dict = {}
        # device = self.device
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
                tmp = {}
                gt_dmaps, gt_cmaps = gt_data[-2], gt_data[-1]
                imgs1, imgs2, gt_cmaps, gt_dmaps = imgs1.to(self.device), imgs2.to(self.device), gt_cmaps.to(
                    self.device), gt_dmaps.to(self.device)
                dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = self.model.forward_train(imgs1, imgs2,
                                                                                                       gt_cmaps)
                loss_den, loss_cls, loss_total = self.loss_fn(gt_dmaps, dmaps1, dmaps2, cmaps1, cmaps2, loss_con,
                                                                gt_cmaps)
                batch_size = imgs1.shape[0]
                total_loss += loss_total.item() * batch_size
                for i, metric in enumerate(self.metric_fns):
                    pred_count = self.predict(imgs1)

                    gt_count = list()
                    for gt_each in gt_data[-3]:
                        gt_count.append(gt_each.shape[0])

                    pred_count_sum = sum(pred_count)
                    gt_count_sum = sum(gt_count)
                    total_metrics[i] += metric(pred_count_sum, gt_count_sum) * batch_size
                tmp['recongize'] = pred_count_sum
                tmp['target'] = gt_count_sum
                output_dict[batch_idx] = tmp.copy()
            return total_loss, total_metrics, output_dict
    
    def test_fidtm(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        output_dict = {}

        with torch.no_grad():
            for batch_idx, (fname, img, fidt_map, kpoint) in enumerate(self.data_loader):
                tmp = {}
                img = img.to(self.device)

                fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).to(self.device)
                d6 = self.model(img)
                loss = self.loss_fn(d6, fidt_map)
                batch_size = img.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_fns):
                    pred_count = self.LMDS_counting(d6)

                    gt_count = list()
                    for gt_each in kpoint:
                        gt_count.append(torch.sum(gt_each).item())

                    pred_count_sum = sum(pred_count)
                    gt_count_sum = sum(gt_count)
                    total_metrics[i] += metric(pred_count_sum, gt_count_sum) * batch_size
                tmp['recongize'] = pred_count_sum
                tmp['target'] = gt_count_sum
                output_dict[batch_idx] = tmp.copy()
            return total_loss, total_metrics, output_dict
    def predict(self,imgs):
        pred_count = []
        for img in imgs:
            img = torch.unsqueeze(img, dim=0)
            h, w = img.shape[2:]
            ps = self.model.patch_size
            log_para = self.model.log_para
            if h >= ps or w >= ps:
                pred_count = 0
                img_patches, _, _ = divide_img_into_patches(img, ps)
                for patch in img_patches:
                    pred = self.model(patch)[0]
                    count += torch.sum(pred).cpu().item() / log_para
            else:
                pred_dmap = self.model(img)[0]
                count = pred_dmap.sum().cpu().item() / log_para
            pred_count.append(count)
        return pred_count
    def LMDS_counting(self, inputs):
        count_list = list()
        for input in inputs:
            if len(input.shape) == 3:
                input = input.unsqueeze(0)
            input_max = torch.max(input).item()

            ''' find local maxima'''

            keep = torch.nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
            keep = (keep == input).float()
            input = keep * input

            '''set the pixel valur of local maxima as 1 for counting'''
            input[input < 100.0 / 255.0 * input_max] = 0
            input[input > 0] = 1

            ''' negative sample'''
            if input_max < 0.1:
                input = input * 0

            count = int(torch.sum(input).item())
            count_list.append(count)

            # kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

        return count_list


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture

    if config['name'] == 'MPCount':
        import model.MPCount.MPCount as module_arch
    if config['name'] == 'FIDTM':
        import model.FIDTM.FIDTM as module_arch
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
        logger.info(log)
        print('-'*100)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/test/FIDTM.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="LSaved/models/FIDTM/1015_123545/checkpoint-epoch9.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
