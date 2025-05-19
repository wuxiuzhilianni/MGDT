#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/29 23:53
# @Author : WeiHua
import torch
from mmrotate.models import S2ANet, ROTATED_DETECTORS, RotatedSingleStageDetector
from mmrotate.core import rbbox2result
import mmcv
import numpy as np


@ROTATED_DETECTORS.register_module()
class SemiS2ANet(S2ANet):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      get_data=False):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)
        if not get_data:
            outs = self.fam_head(x)

            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_base = self.fam_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            for name, value in loss_base.items():
                losses[f'fam.{name}'] = value

            rois = self.fam_head.refine_bboxes(*outs)
            # rois: list(indexed by images) of list(indexed by levels)
            align_feat = self.align_conv(x, rois)
            outs = self.odm_head(align_feat)
            
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.odm_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'odm.{name}'] = value

            return losses
        
        with torch.no_grad():
            self.eval()
            bbox_results = self.simple_test(img, img_metas, rescale=True) # 返回的是一个字典，包含 bbox 和 label
            self.train()
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)
        return outs, bbox_results