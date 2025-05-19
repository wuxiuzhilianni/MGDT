#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/30 0:14
# @Author : WeiHua

from mmrotate.models import RotatedFCOSHead, ROTATED_HEADS
import torch
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

INF = 1e8
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')

# @ROTATED_HEADS.register_module()
class SemiRotatedFCOSHeadV2(RotatedFCOSHead):
    def __init__(self, num_classes, in_channels, **kwargs):
        super(SemiRotatedFCOSHeadV2, self).__init__(
            num_classes,
            in_channels,
            **kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      get_data=False,
                      **kwargs):
        if get_data:
            return self(x)  # outs 包括每个 level 的 cls_score, bbox_pred, angle_pred, centerness
        return super(SemiRotatedFCOSHeadV2, self).forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs
        )

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # 共5个值，大小为torch.Size([W, H]) W, H in 128, 64, 32, 16, 8
        all_level_points = self.prior_generator.grid_priors( # 网格坐标就是特征图中每个像素点映射到原图像上的像素点坐标
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device) # 共5个值，大小为torch.Size([W*H, 2]) 2 表示 (coord_x, coord_y) ,比如128*128的特征图一个网格大小为8，则其中心坐标为(4,4)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, img_metas)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores) # torch.Size([43648, 15]) 43648=2*(128*128+...+8*8)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds) # torch.Size([43648, 4])
        flatten_angle_preds = torch.cat(flatten_angle_preds) # torch.Size([43648, 1])
        flatten_centerness = torch.cat(flatten_centerness) # torch.Size([43648])
        flatten_labels = torch.cat(labels) # torch.Size([43648])
        flatten_bbox_targets = torch.cat(bbox_targets) # torch.Size([43648, 4])
        flatten_angle_targets = torch.cat(angle_targets) # torch.Size([43648, 1])
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]) # torch.Size([43648, 2])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes # 15 表示背景
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1) # torch.Size([30])
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device) # 30
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds] # torch.Size([30, 4])
        pos_angle_preds = flatten_angle_preds[pos_inds] # torch.Size([30, 1])
        pos_centerness = flatten_centerness[pos_inds] # torch.Size([30])
        pos_bbox_targets = flatten_bbox_targets[pos_inds] # torch.Size([30, 4])
        pos_angle_targets = flatten_angle_targets[pos_inds] # torch.Size([30, 1])
        pos_centerness_targets = self.centerness_target(pos_bbox_targets) # torch.Size([30])
        # centerness weighted iou loss，是一个标量
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if self.separate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, img_metas):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, img_metas, points, regress_ranges,
                           num_points_per_lvl,):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask1 = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)
            num_inside = inside_gt_bbox_mask1.sum().item()
            print("Number of points inside GT bbox (after center sampling)矩形:", num_inside) # 30 

        if self.center_sampling:
            area = w * h
            max_area = area.max() if num_gts > 0 else 1.0
            a = w / 2  # 直接使用目标宽高的一半
            b = h / 2
            n = 2
            inside_center_bbox_mask = (offset_x/a).pow(n) + (offset_y/b).pow(n) <= 1
            inside_gt_bbox_mask2 = torch.logical_and(inside_center_bbox_mask, inside_gt_bbox_mask)
            num_inside = inside_gt_bbox_mask2.sum().item()
            print("Number of points inside GT bbox (after center sampling)椭圆:", num_inside) # 22

        if self.center_sampling:
            """
            类别相关椭圆采样 + 尺度感知采样区域（平滑过渡）
            n 由类别决定，采样半径 a/b 根据目标大小自适应变化
            """

            # === 1. 类别对应超椭圆指数 n ===
            n_dict = {i: 10.0 for i in range(15)}  # 默认指数 n=10.0 为近似矩形
            n_dict.update({
                0: 4.0,  # plane
                1: 4.0,  # baseball-diamond
                3: 4.0,  # ground-track-field
                9: 2.0,  # storage-tank
                11: 2.0, # roundabout
                13: 4.0, # swimming-pool
                14: 4.0, # helicopter
            })

            gt_n = gt_labels.new_zeros((num_gts,), dtype=torch.float)
            for class_id, n_val in n_dict.items():
                gt_n[gt_labels == class_id] = n_val

            # === 2. 平滑自适应采样函数 ===
            def smooth_piecewise(tensor, thresholds, weights, k):
                """
                输入 tensor，返回平滑分段缩放后的 tensor * factor
                """
                assert len(weights) == len(thresholds) + 1
                sigmoids = [1 / (1 + torch.exp(-k * (tensor - t))) for t in thresholds]

                result = weights[0] * (1 - sigmoids[0])
                for i in range(1, len(thresholds)):
                    result += weights[i] * (sigmoids[i-1] - sigmoids[i])
                result += weights[-1] * sigmoids[-1]
                return tensor * result

            # === 3. 采样超参数（可调节/可扩展） ===
            k = 0.05  # 控制平滑过渡的陡峭程度
            piecewise_weights = [1/2, 1/3, 1/5, 1/7, 1/16, 1/32]  # 分段采样半径缩放系数
            ratio_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4]  # 分段阈值占图像宽度比例

            img_h, img_w = img_metas['img_shape'][:2]
            thresholds = [r * img_w for r in ratio_thresholds]

            # === 4. 计算 a, b（每个 GT 的椭圆长短轴）===
            w_tensor = w[0]  # shape: (num_gts,)
            h_tensor = h[0]

            a = smooth_piecewise(w_tensor, thresholds, piecewise_weights, k)
            b = smooth_piecewise(h_tensor, thresholds, piecewise_weights, k)

            a = a.unsqueeze(0)  # shape: (1, num_gts)
            b = b.unsqueeze(0)

            # === 5. 超椭圆中心采样区域判断 ===
            inside_center_bbox_mask = offset.new_zeros((num_points, num_gts), dtype=torch.bool)
            for gt_idx in range(num_gts):
                n_val = gt_n[gt_idx].item()
                norm_x = (offset_x[:, gt_idx] / a[0, gt_idx]).abs()
                norm_y = (offset_y[:, gt_idx] / b[0, gt_idx]).abs()
                inside_center_bbox_mask[:, gt_idx] = (norm_x.pow(n_val) + norm_y.pow(n_val)) <= 1

            # === 6. 与原 GT 匹配掩码结合 ===
            inside_gt_bbox_mask3 = torch.logical_and(inside_center_bbox_mask, inside_gt_bbox_mask)
            print("Points after adaptive sampling:", inside_gt_bbox_mask3.sum().item())

            
        if self.center_sampling:
            # 动态超椭圆参数
            radius = self.center_sample_radius # 1.5
            stride = offset.new_zeros((num_points, num_gts, 2)) # [12,12] .... [192.192]
            
            # 计算各层级的 stride，扩展为 (num_points, num_gts, 2)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                cur_stride = self.strides[lvl_idx] * radius
                stride[lvl_begin:lvl_end, :, :] = cur_stride
                lvl_begin = lvl_end

            # 避免除以零的情况
            eps = 1e-6
            w = w.clamp(min=eps)
            h = h.clamp(min=eps)

            # 动态计算 n: 小目标 n 大（接近长方形），大目标 n 小（接近椭圆）
            area = w * h
            max_area = area.max() if num_gts > 0 else 1.0
            n = 6 - 2 * (area / max_area).clamp(min=0, max=1)  # shape: (num_points, num_gts)

            # scale_factor范围[1.0, 1.8]，大目标缩放更大
            scale_factor = 2.5 - 1.5 * (area / max_area).clamp(min=0, max=1)
            # 保持宽高比的同时动态缩放
            w_div = (w / torch.max(w, h)) * scale_factor  # 移除clamp(min=0.3)
            h_div = (h / torch.max(w, h)) * scale_factor

            # 根据目标面积动态补偿采样区域
            area_ratio = (area / max_area).clamp(min=0.1, max=1)
            area_compensation = 0.7 + 0.3 * area_ratio  # ∈[0.55,1.0]
            a = stride[..., 0] * w_div * area_compensation
            b = stride[..., 1] * h_div * area_compensation

            # 计算椭圆采样区域
            offset_x_norm = (torch.abs(offset_x) / a).clamp(min=eps)
            offset_y_norm = (torch.abs(offset_y) / b).clamp(min=eps)
            inside_center_bbox_mask = (offset_x_norm.pow(n) + offset_y_norm.pow(n)) <= 1

            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask, inside_gt_bbox_mask)
            num_inside = inside_gt_bbox_mask.sum().item()
            print("Number of points inside GT bbox (after center sampling):", num_inside) # 22
            print("======================================")

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]

        return labels, bbox_targets, angle_targets
