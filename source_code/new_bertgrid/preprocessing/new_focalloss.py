
###############################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # pred = nn.Sigmoid()(pred)

        pred = pred.view(-1,1)
        target = target.view(-1,1)

        pred = torch.cat((1-pred,pred),dim=1)

        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        log_p = probs.log()

        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
####################################################
##### This is focal loss class for multi class #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
# os.environ['CUDA_VISIBLE_DEVICE']='3'
device0 = torch.device("cuda:0")
# device0 = torch.device("cuda")
# class_weight = torch.tensor([[0.8822], [0.0114], [0.0565], [0.0336], [0.0163]])
class_weight = torch.tensor([[0.8263], [0.0394], [0.0154], [0.1107], [0.0082]])
constant_weight = 1.04
log_weight = 1/torch.log(class_weight + constant_weight)

# class_weight = torch.tensor([[1],[72.78],[16.85],[27.92],[58.03]])
# prob0 = 0.888
# constant_weight = 1.04
# log_weight = 1/torch.log(prob0/class_weight + constant_weight)
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight = log_weight.to(device0), size_average=None, ignore_index=-100,
    # def __init__(self, gamma=2, weight = torch.tensor([[1],[1],[1],[1],[1]]).to(device0), size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        # if not weight:
        #     weight = torch.tensor([[1],[72.78],[16.85],[27.92],[58.03]]).cuda()
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        input = input.contiguous().view(input.size(0),input.size(1),-1)
        target = target.contiguous().view(input.size(0),input.size(1),-1)
        #print(input.size())
        weight = Variable(self.weight)

        # compute the negative likelyhood
        # print(input.shape,target.shape)
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        # logpt = - F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = focal_loss
        return balanced_focal_loss