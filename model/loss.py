import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


# def mse_loss(output, target):
#     return torch.nn.MSELoss()(output,target)

def compute_count_loss(pred_dmaps, gt_dmaps, log_para=1000, weights = None):
    # gt_dmaps = gt_dmaps.to(self.device)
    if weights is not None:
        pred_dmaps = pred_dmaps * weights
        gt_dmaps = gt_dmaps * weights
    loss_value = torch.nn.MSELoss()(pred_dmaps, gt_dmaps * log_para)
    return loss_value

def mpcount_loss(gt_dmaps, dmaps1, dmaps2, cmaps1, cmaps2, loss_con, gt_cmaps):
    loss_den = compute_count_loss(dmaps1, gt_dmaps) + compute_count_loss(dmaps2, gt_dmaps)
    loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
    loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 
    
    return loss_den, loss_cls, loss_total

def fidtm_loss(dmaps1, gt_dmaps):
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(dmaps1, gt_dmaps)
    return loss
