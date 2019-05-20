import torch.nn as nn
import torch.nn.functional as F


class unet_loss(nn.Module):
    def __init__(self, pos_scale = 3, neg_scale =1):
        super(unet_loss, self).__init__()

        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, pred, target):

        b,c,h,w = pred.size()
        smooth = 1.0
        inter = (pred * target).sum()
        dice = ((2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)).log()

        target_mask = target > 0.5
        pos_pred = pred[target_mask]
        pos_target = target[target_mask]

        loss_pos = self.bce_loss(pos_pred,pos_target)*self.pos_scale / (w*h*b)

        neg_pred = pred[target_mask==False]
        neg_target = target[target_mask==False]

        loss_neg = self.bce_loss(neg_pred, neg_target)*self.neg_scale / (w*h*b)
        loss_all = loss_pos + loss_neg - dice

        loss_info = {
            'loss_all': loss_all.data.cpu(),
            'loss_pos': loss_pos.data.cpu(),
            'loss_neg': loss_neg.data.cpu(),
        }

        return loss_all, loss_info
