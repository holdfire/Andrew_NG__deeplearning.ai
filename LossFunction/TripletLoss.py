"""
name: triplet loss
title: FaceNet: A Unified Embedding for Face Recognition and Clustering。
author: Florian Schroff
year: 2015
journal: CVPR
aim: 对contrastive loss的改进
link: https://arxiv.org/abs/1503.03832
reference: https://mp.weixin.qq.com/s/h0N9OR_AcUw_lXgELohS0Q
// Created by LiuXing on 2020/05/17
"""


def triplet_loss(y_true, y_pred):
    """
    Triplet Loss的损失函数：
    最小化anchor和positive之间的距离，最大化anchor和negative之间的距离；
    并且使得锚点a和负样本的距离 > 锚点和正样本的距离 + margin
    """
    anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]
    # 欧式距离
    pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
    neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
    basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

    loss = K.maximum(basic_loss, 0.0)
    return loss