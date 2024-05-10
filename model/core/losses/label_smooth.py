import torch
import torch.nn as nn

from mmdet.models.losses import weight_reduce_loss


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self,
                 neg_factor=0.1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.neg_factor = neg_factor
        self.reduction = 'mean'
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets, weight):
        logits = logits.float()
        batch_size, num_pts, num_classes = logits.size(
            0), logits.size(1), logits.size(2)
        logits = logits.reshape(-1, num_classes)
        targets = targets.reshape(-1, 1)

        with torch.no_grad():
            targets = targets.clone().detach()
            label_pos, label_neg = 1. - self.neg_factor, self.neg_factor / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(label_neg)
            lb_one_hot.scatter_(1, targets, label_pos)
            lb_one_hot = lb_one_hot.detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=batch_size * num_pts)

        return loss


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, weight):
        logits = logits.float()
        batch_size, num_pts = logits.size(
            0), logits.size(1)
        # 如果是形状为 (batch_size, num_points, 2)，则计算每个点的损失并求平均
        loss = torch.mean(torch.abs(logits - targets), dim=-1).view(-1)  # 计算每个点x与y的 L1 损失，再计算每个点坐标的平均损失

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=batch_size * num_pts)

        return loss


class HuberLoss(nn.Module):
    """

    计算Huber损失
    参数:
    y_true -- 真实值
    y_pred -- 预测值
    delta -- Huber损失中的阈值参数，用于控制从MSE到MAE的过渡点
    返回:
    Huber损失的值

    """
    def __init__(self, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, weight):
        logits = logits.float()
        batch_size, num_pts = logits.size(
            0), logits.size(1)

        error = logits - targets

        delta = 1

        is_small_error = torch.abs(error) <= delta

        squared_loss = 0.5 * error ** 2

        linear_loss = delta * (torch.abs(error) - 0.5 * delta)

        loss = torch.where(is_small_error, squared_loss, linear_loss)

        loss = torch.mean(loss, dim=-1).view(-1)  # 计算每个点x与y的 L1 损失，再计算每个点坐标的平均损失

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=batch_size * num_pts)


        return loss