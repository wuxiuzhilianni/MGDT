#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder
from mmdet.core.anchor.point_generator import MlvlPointGenerator
import numpy as np
from mmrotate.core import poly2obb_np
import cv2
import mmcv

INF = 1e8
CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 
               'ship', 'soccer-ball-field', 'basketball-court', 'ground-track-field', 'small-vehicle', 
               'baseball-diamond', 'tennis-court', 'roundabout', 'storage-tank', 'harbor')


@ROTATED_LOSSES.register_module()
class RotatedDTLossS2ANet(nn.Module):
    def __init__(self, cls_channels=len(CLASSES), loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTLossS2ANet, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.loss_type = loss_type

    def convert_shape(self, logits):
        cls_scores, bbox_preds = logits
        assert len(cls_scores) == len(bbox_preds)

        batch_size = cls_scores[0].shape[0]   
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x in bbox_preds
        ], dim=1).view(-1, 5)
        return cls_scores, bbox_preds

    def forward(self, teacher_logits, student_logits, ratio=0.01, img_metas=None, **kwargs):
        # (21824, 15)  (21824, 5)
        t_cls_scores, t_bbox_preds = self.convert_shape(teacher_logits[0])
        s_cls_scores, s_bbox_preds = self.convert_shape(student_logits[0])

        with torch.no_grad():
            # Region Selection, 根据分类置信度筛选topK
            count_num = int(t_cls_scores.size(0) * 0.03) # 21824 * 0.03 = 654
            teacher_probs = t_cls_scores.sigmoid() # (21824, 15)
            max_vals = torch.max(teacher_probs, 1)[0] # (21824,)
            sorted_vals, sorted_inds = torch.topk(max_vals, t_cls_scores.size(0)) # (21824,), (21824,) 分别表示排序后的值和索引
            mask = torch.zeros_like(max_vals) # (21824,)
            mask[sorted_inds[:count_num]] = 1. # 654 取前654个
            fg_num = sorted_vals[:count_num].sum() # 654 取前654个的和
            b_mask = mask > 0. # (21824,) 值为布尔类型

        # 对正负样本计算 cls 损失
        loss_cls = QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=mask,
            reduction="sum",
        ) / fg_num


        # 仅对正样本计算 reg loss
        if self.bbox_loss_type == 'l1':
            loss_bbox = self.bbox_loss(
                s_bbox_preds[b_mask],
                t_bbox_preds[b_mask],
            ).mean()
        else:
            all_level_points = self.prior_generator.grid_priors(
                [featmap.size()[-2:] for featmap in teacher_logits[0]],
                dtype=s_bbox_preds.dtype,
                device=s_bbox_preds.device)
            flatten_points = torch.cat(
                [points.repeat(len(teacher_logits[0][0]), 1) for points in all_level_points])
            s_bbox_preds = self.bbox_coder.decode(flatten_points, s_bbox_preds)[b_mask]
            t_bbox_preds = self.bbox_coder.decode(flatten_points, t_bbox_preds)[b_mask]
            loss_bbox = self.bbox_loss(
                s_bbox_preds,
                t_bbox_preds,
            )
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid # (21824, 15) 表示预测的概率
    zerolabel = pt.new_zeros(pt.shape)  # (21824, 15) 所有类别都为负样本
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta) 
    pos = weight > 0

    # positive goes to bbox quality 覆盖正样本的权值
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss