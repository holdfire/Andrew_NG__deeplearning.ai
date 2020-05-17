"""
name: contrastive loss
title: Dimensionality Reduction by Learning an Invariant Mapping.
author: Yann LeCun
year: 2006
journal: CVPR
link: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
aim: 增大分类器的类间差异
reference: https://mp.weixin.qq.com/s/h0N9OR_AcUw_lXgELohS0Q
// Created by LiuXing on 2020/05/17
"""

import torch
import torch.nn.functional as F

# custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def foward(self, output1, output2, label):
        """
        :param output1: predicted result of sample 1
        :param output2: predicted result of sample 2
        :param label: the absolute difference between y1 and y2. 0 means the same, 1 means different.
        :return:
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return contrastive_loss
