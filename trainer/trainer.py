import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time
from utils.misc import denormalize, divide_img_into_patches


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        self.model_name = model.name

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
    
    
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

    def train_mpcount(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # optimizer.zero_grad()
        # dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(imgs1, imgs2, gt_cmaps)
        # loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
        # loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
        # loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 

        # loss_total.backward()
        # optimizer.step()
        
        for batch_idx, (imgs1, imgs2, gt_data) in enumerate(self.data_loader):
            gt_dmaps, gt_cmaps = gt_data[-2],gt_data[-1]
            imgs1, imgs2, gt_cmaps, gt_dmaps = imgs1.to(self.device), imgs2.to(self.device), gt_cmaps.to(self.device), gt_dmaps.to(self.device)
            # data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            # output = self.model(data)
            
            dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = self.model.forward_train(imgs1, imgs2, gt_cmaps)
            # loss_den = self.compute_count_loss(loss, dmaps1, gt_data) + self.compute_count_loss(loss, dmaps2, gt_data)
            # loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
            # loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 
            
            loss_den, loss_cls, loss_total = self.criterion(gt_dmaps, dmaps1, dmaps2, cmaps1, cmaps2, loss_con, gt_cmaps)
            loss_total.backward()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss_total.item())
            for met in self.metric_ftns:
                pred_count=self.predict(imgs1)
                
                gt_count = list()
                for gt_each in gt_data[-3]:
                    gt_count.append(gt_each.shape[0])
                
                pred_count_sum = sum(pred_count)
                gt_count_sum= sum(gt_count)
                self.train_metrics.update(met.__name__, met(pred_count_sum, gt_count_sum))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_total.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    
    def _train_epoch(self, epoch):
        if not self.model_name:
            log = self.train_default(epoch)
        elif self.model_name == "MPCount":
            start = time.time()
            log = self.train_mpcount(epoch)
            end = time.time()
            print(f"Epoch {epoch}: Time cost: {end-start}s")
        else:
            print("The training model has not specified!")
            exit(-1)
        return log
    
    # def _train_epoch(self, epoch):
    #     """
    #     Training logic for an epoch

    #     :param epoch: Integer, current training epoch.
    #     :return: A log that contains average loss and metric in this epoch.
    #     """
    #     self.model.train()
    #     self.train_metrics.reset()
        
    #     # optimizer.zero_grad()
    #     # dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(imgs1, imgs2, gt_cmaps)
    #     # loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
    #     # loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
    #     # loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 

    #     # loss_total.backward()
    #     # optimizer.step()
        
    #     for batch_idx, (imgs1, imgs2, gt_data) in enumerate(self.data_loader):
    #         gt_cmaps = gt_data[-1]
    #         data, target = data.to(self.device), target.to(self.device)

    #         self.optimizer.zero_grad()
    #         # output = self.model(data)
            
    #         dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = self.model.forward_train(imgs1, imgs2, gt_cmaps)
    #         loss_den = self.compute_count_loss(loss, dmaps1, gt_data) + self.compute_count_loss(loss, dmaps2, gt_data)
    #         loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
    #         loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 
            
    #         # loss = self.criterion(output, target)
    #         loss_total.backward()
    #         self.optimizer.step()

    #         self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
    #         self.train_metrics.update('loss', loss.item())
    #         for met in self.metric_ftns:
    #             self.train_metrics.update(met.__name__, met(output, target))

    #         if batch_idx % self.log_step == 0:
    #             self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
    #                 epoch,
    #                 self._progress(batch_idx),
    #                 loss.item()))
    #             self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    #         if batch_idx == self.len_epoch:
    #             break
    #     log = self.train_metrics.result()

    #     if self.do_validation:
    #         val_log = self._valid_epoch(epoch)
    #         log.update(**{'val_'+k : v for k, v in val_log.items()})

    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()
    #     return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
