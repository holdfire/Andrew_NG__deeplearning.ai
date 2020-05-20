"""
name: GIoU loss
title: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
author: Hamid Rezatofighi
year: 2019
journal: CVPR
link: https://arxiv.org/abs/1902.09630
github: https://github.com/generalized-iou/g-darknet
github2: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.coco-giou-12.cfg
aim: 目标检测在计算loss时使用bbox和ground truth的L1或L2范数，在评测时却使用IoU去判断是否检测到目标
如果使用IoU作为损失函数，当bbox和gt没有重叠时loss为0，不可导而无法优化。
reference: https://mp.weixin.qq.com/s/CNVgrIkv8hVyLRhMuQ40EA
reference2: https://zhuanlan.zhihu.com/p/63389116
// Created by LiuXing on 2020/05/20
"""

