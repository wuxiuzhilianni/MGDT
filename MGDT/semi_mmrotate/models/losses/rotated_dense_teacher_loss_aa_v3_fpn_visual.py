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
import os
import matplotlib.pyplot as plt

INF = 1e8
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')


@ROTATED_LOSSES.register_module()
class RotatedDTLossAssignerAssistentV3Visual(nn.Module):
    def __init__(self, cls_channels=len(CLASSES), loss_type='origin', bbox_loss_type='l1'):
        super(RotatedDTLossAssignerAssistentV3Visual, self).__init__()
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

        self.image_class_prompt_path = '/workspace/animax/MCL/tools/Assinger_Assistent/image_class_prompt_from_chat_with_percent5_label_modified.pt'
        self.image_class_prompt = torch.load(self.image_class_prompt_path)
                # 预处理类别映射，将类别名转换为索引
        self.class_name_to_index = {
            filename: torch.tensor(
                [CLASSES.index(cls) for cls in exist_classes if cls in CLASSES],
                device='cpu'  # 初始化时为CPU，运行时转移到设备
            )
            for filename, exist_classes in self.image_class_prompt.items()
        }
        
        # TODO 统计
        self.stats = {cls: {"teacher_probs_sum": 0.0, "count": 0} for cls in CLASSES}
        self.save_path = '/workspace/animax/MCL/tools/Class_Assigner/cls_count.txt'
        self.save_interval = 3200

    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1).view(-1, 5)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in centernesses
        ], dim=1).view(-1, 1)
        return cls_scores, bbox_preds, centernesses

    def visualize_and_save_cls_score(self, cls_scores, save_dir="output_images"):
        """
        Visualize and save classification scores for each feature map.

        Args:
            cls_scores (list[Tensor]): List of tensors, each with shape (1, 15, W, H).
            save_dir (str): Directory to save the output images.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define a fixed list of 15 light colors for the classes
        colors = [
            (255, 200, 200), (200, 255, 200), (200, 200, 255),
            (255, 255, 200), (255, 200, 255), (200, 255, 255),
            (255, 230, 180), (230, 255, 180), (180, 230, 255),
            (255, 180, 230), (230, 180, 255), (180, 255, 230),
            (240, 240, 180), (180, 240, 240), (240, 180, 240)
        ]

        font_scale = 0.3  # Font scale for text
        font_thickness = 1  # Text thickness

        num_levels = len(cls_scores)  # Number of feature map levels
        for i, cls_score in enumerate(cls_scores):
            # cls_score shape: (1, 15, W, H)
            cls_score = cls_score.squeeze(0).cpu().numpy()  # Remove batch dimension -> (15, W, H)
            max_value = np.max(cls_score, axis=0)  # max_value: (W, H)
            max_class = np.argmax(cls_score, axis=0)  # max_class: (W, H)

            height, width = max_class.shape
            cell_size = 40  # Size of each cell in the grid
            grid_height, grid_width = height * cell_size, width * cell_size

            # Create an image for the grid
            image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            for y in range(height):
                for x in range(width):
                    class_id = max_class[y, x]
                    score = max_value[y, x]

                    # Define the cell boundaries
                    top_left = (x * cell_size, y * cell_size)
                    bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)

                    # Fill the cell with the color corresponding to the class
                    color = colors[class_id]
                    cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)

                    # Add the text with class and score
                    text = f"{class_id}:{score:.3f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = top_left[0] + (cell_size - text_size[0]) // 2
                    text_y = top_left[1] + (cell_size + text_size[1]) // 2
                    cv2.putText(
                        image, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=font_thickness
                    )

            # Save the image
            save_path = os.path.join(save_dir, f"feature_map_level_{i + 1:02d}.png")
            cv2.imwrite(save_path, image)
            print(f"Saved: {save_path}")


    def visualize_and_save_cls_score_topk(self, cls_scores, save_dir="output_images", iter=0, image_name=None, topk=600):
        """
        Visualize and save classification scores for each feature map.
        Mark the topk highest scoring cells across all feature maps.

        Args:
            cls_scores (list[Tensor]): List of tensors, each with shape (1, 15, W, H).
            save_dir (str): Directory to save the output images.
            topk (int): Number of topk positions across all feature maps.
        """
        os.makedirs(save_dir, exist_ok=True)

        colors = [
            (200, 170, 170), (170, 200, 170), (170, 170, 200),
            (200, 200, 170), (200, 170, 200), (170, 200, 200),
            (200, 180, 150), (180, 200, 150), (150, 180, 200),
            (200, 150, 180), (180, 150, 200), (150, 200, 180),
            (210, 210, 160), (160, 210, 210), (210, 160, 210)
        ]

        font_scale = 0.4
        font_thickness = 1
        
        all_scores = []
        for i, cls_score in enumerate(cls_scores):
            cls_score = cls_score.squeeze(0).cpu().numpy()
            max_value = np.max(cls_score, axis=0)
            max_class = np.argmax(cls_score, axis=0)

            for y in range(max_value.shape[0]):
                for x in range(max_value.shape[1]):
                    all_scores.append((max_value[y, x], max_class[y, x], x, y, i))

        all_scores.sort(reverse=True, key=lambda x: x[0])
        topk_positions = set((x[2], x[3], x[4]) for x in all_scores[:topk])

        for i, cls_score in enumerate(cls_scores):
            cls_score = cls_score.squeeze(0).cpu().numpy()
            max_value = np.max(cls_score, axis=0)
            max_class = np.argmax(cls_score, axis=0)

            height, width = max_class.shape
            cell_size = 40
            grid_height, grid_width = height * cell_size, width * cell_size
            image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            for y in range(height):
                for x in range(width):
                    class_id = max_class[y, x]
                    score = max_value[y, x]
                    top_left = (x * cell_size, y * cell_size)
                    bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)

                    if (x, y, i) in topk_positions:
                        overlay = image.copy()
                        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
                        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), thickness=1)
                    else:
                        cv2.rectangle(image, top_left, bottom_right, colors[class_id], thickness=-1)

                    text = f"{class_id}:{score:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = top_left[0] + (cell_size - text_size[0]) // 2
                    text_y = top_left[1] + (cell_size + text_size[1]) // 2
                    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)

            save_path = os.path.join(save_dir, f"iter_{iter}_image_{image_name}_feature_map_level_{i + 1:02d}_topk.png")
            cv2.imwrite(save_path, image)
            print(f"Saved: {save_path}")

    # def visualize_and_save_cls_score_topk(self, cls_scores, save_dir="output_images", iter=0, image_name=None, topk=600):
    #     """
    #     Visualize and save classification scores for each feature map.
    #     Mark the topk highest scoring cells across all feature maps as black.

    #     Args:
    #         cls_scores (list[Tensor]): List of tensors, each with shape (1, 15, W, H).
    #         save_dir (str): Directory to save the output images.
    #         topk (int): Number of topk positions across all feature maps to mark as black.
    #     """
    #     # Ensure the save directory exists
    #     os.makedirs(save_dir, exist_ok=True)

    #     # Define a fixed list of 15 light colors for the classes
    #     # colors = [
    #     #     (255, 200, 200), (200, 255, 200), (200, 200, 255),
    #     #     (255, 255, 200), (255, 200, 255), (200, 255, 255),
    #     #     (255, 230, 180), (230, 255, 180), (180, 230, 255),
    #     #     (255, 180, 230), (230, 180, 255), (180, 255, 230),
    #     #     (240, 240, 180), (180, 240, 240), (240, 180, 240)
    #     # ]

    #     colors = [
    #     (200, 170, 170), (170, 200, 170), (170, 170, 200),
    #     (200, 200, 170), (200, 170, 200), (170, 200, 200),
    #     (200, 180, 150), (180, 200, 150), (150, 180, 200),
    #     (200, 150, 180), (180, 150, 200), (150, 200, 180),
    #     (210, 210, 160), (160, 210, 210), (210, 160, 210)
    #     ]

    #     font_scale = 0.3  # Font scale for text
    #     font_thickness = 1  # Text thickness

    #     num_levels = len(cls_scores)  # Number of feature map levels
    #     all_scores = []  # To store all class scores for selecting topk

    #     # Collect all scores and their coordinates
    #     for i, cls_score in enumerate(cls_scores):
    #         cls_score = cls_score.squeeze(0).cpu().numpy()  # Shape: (15, W, H)
    #         max_value = np.max(cls_score, axis=0)  # (W, H)
    #         max_class = np.argmax(cls_score, axis=0)  # (W, H)

    #         # Store max_value with corresponding (x, y) positions
    #         for y in range(max_value.shape[0]):
    #             for x in range(max_value.shape[1]):
    #                 all_scores.append((max_value[y, x], max_class[y, x], x, y, i))

    #     # Sort all positions by score (descending) and select the topk
    #     all_scores.sort(reverse=True, key=lambda x: x[0])
    #     topk_positions = all_scores[:topk]


    #     # Create and save images for each feature map
    #     for i, cls_score in enumerate(cls_scores):
    #         cls_score = cls_score.squeeze(0).cpu().numpy()  # Shape: (15, W, H)
    #         max_value = np.max(cls_score, axis=0)  # (W, H)
    #         max_class = np.argmax(cls_score, axis=0)  # (W, H)

    #         height, width = max_class.shape
    #         cell_size = 40  # Size of each cell in the grid
    #         grid_height, grid_width = height * cell_size, width * cell_size

    #         # Create an image for the grid
    #         image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    #         for y in range(height):
    #             for x in range(width):
    #                 class_id = max_class[y, x]
    #                 score = max_value[y, x]

    #                 # Define the cell boundaries
    #                 top_left = (x * cell_size, y * cell_size)
    #                 bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)

    #                 # Check if the current position is in the topk positions
    #                 if (score, class_id, x, y, i) in topk_positions:
    #                     # Mark topk positions as black
    #                     color = (0, 0, 0)
    #                 else:
    #                     # Otherwise, fill the cell with the color corresponding to the class
    #                     color = colors[class_id]
                    
    #                 # Fill the cell with the selected color
    #                 cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
    #                         # 生成白色背景
    #                 # white_bg = np.ones_like(image, dtype=np.uint8) * 255

    #                 # 将填充的颜色与白色背景融合，使颜色更柔和
    #                 # cv2.addWeighted(image, 0.6, white_bg, 0.4, 0, image)
    #                 # Add the text with class and score
    #                 text = f"{class_id}:{score:.3f}"
    #                 text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    #                 text_x = top_left[0] + (cell_size - text_size[0]) // 2
    #                 text_y = top_left[1] + (cell_size + text_size[1]) // 2
    #                 cv2.putText(
    #                     image, text, (text_x, text_y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=font_thickness
    #                 )

    #         # Save the image
    #         save_path = os.path.join(
    #                 save_dir, f"iter_{iter}_image_{image_name}_feature_map_level_{i + 1:02d}_topk.png"
    #             )
    #         cv2.imwrite(save_path, image)
    #         print(f"Saved: {save_path}")


    def forward(self, teacher_logits, student_logits, ratio=0.03, img_metas=None, iter=0, **kwargs):
        # (21824, 15)  (21824, 5)  (21824, 1)
        image_name = "unknown_image"
        if img_metas and "filename" in img_metas['img_metas'][0]:
            image_name = os.path.splitext(os.path.basename(img_metas['img_metas'][0]["filename"]))[0]
        # if iter % 1 == 0:
        # P1014__1024__0___0 1360__1024__824___0  P0209__1024__2472___864 P0219__1024__1633___140
        
        if image_name in ('P0464__1024__145___0', 'P0849__1024__627___140', 'P0849__1024__627___0', 'P0868__1024__824___0',
                          'P0324__1024__0___0', 'P0737__1024__0___0'):
            cls_score, bbox_pred, angle_pred, centerness = teacher_logits
            for i in range(len(cls_score)):
                cls_score[i] = cls_score[i].sigmoid()
            # self.visualize_and_save_cls_score(cls_score,"/workspace/animax/MCL/tools/visualization")
            self.visualize_and_save_cls_score_topk(cls_score,"/workspace/animax/MCL/workdirs/visual", iter, image_name)
        
        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_logits)

        #------------------------------Assigner Assistant Start---------------------------------------
        with torch.no_grad():
            # 获取图像的类别提示
            filename = img_metas['img_metas'][0]['filename'].split('/')[-1]
            exist_classes_tensor = self.class_name_to_index[filename].to(t_cls_scores.device)

            # 确定选取的 pixel 数量
            count_num = int(t_cls_scores.size(0) * ratio)  # 总共需要选取的正样本数量
            teacher_probs = t_cls_scores.sigmoid()  # torch.Size([21824, 15])
            teacher_centernesses = t_centernesses.sigmoid()  # torch.Size([21824, 1])
            joint_confidence = teacher_probs * teacher_centernesses  # torch.Size([21824, 15])

            # 按联合置信度找到最大值及其类别
            max_vals, max_inds = torch.max(joint_confidence, dim=1)  # (21824,), (21824,)

            # TODO 统计总样本中max之后各个类别的数量以及平均置信度
            #------------------------------Statistic Start---------------------------------------
            # for cls_idx, cls_name in enumerate(CLASSES):
            #     # 获取 max_inds 中等于当前类别的掩码
            #     cls_mask = max_inds == cls_idx  # (21824,)
                
            #     # 统计该类别的数量
            #     cls_count = cls_mask.sum().item()  # 属于该类别的数量
                
            #     # 如果该类别没有出现，均值为 0
            #     if cls_count > 0:
            #         cls_vals_mean = max_vals[cls_mask].mean().item()  # 属于该类别的 max_vals 的均值
            #     else:
            #         cls_vals_mean = 0.0

            #     # 保存到统计信息中
            #     self.stats[cls_name]["teacher_probs_sum"] += cls_vals_mean * cls_count
            #     self.stats[cls_name]["count"] += cls_count

            # # 定期保存统计结果
            # if iter % self.save_interval == 0:
            #     with open(self.save_path, 'w') as f:
            #         for cls_name in CLASSES:
            #             count = self.stats[cls_name]["count"]
            #             if count > 0:
            #                 mean_prob = self.stats[cls_name]["teacher_probs_sum"] / count
            #             else:
            #                 mean_prob = 0.0
            #             f.write(f"{cls_name}: count={count}, mean_prob={mean_prob:.6f}\n")
            #------------------------------Statistic End---------------------------------------

            # ------------------------------------
            # 1. 输入 Prompt 类别最优先选取
            prompt_mask = torch.isin(max_inds, exist_classes_tensor)
            prompt_confidence_mask = max_vals > 0.02
            prompt_mask = prompt_mask & prompt_confidence_mask
            prompt_vals = max_vals[prompt_mask]
            prompt_inds = torch.where(prompt_mask)[0]

            # 选取 Prompt 类别中 TopK
            prompt_count = int(count_num * 0.03)
            if len(prompt_inds) > prompt_count:
                sorted_vals, sorted_inds = torch.topk(prompt_vals, prompt_count)
                prompt_inds = prompt_inds[sorted_inds]  # 最终选取的 Prompt 索引
            else:
                prompt_count = len(prompt_inds)  # 实际选取数量

            # ------------------------------------
            # 2. 整体 TopK
            overall_count = int(count_num * 0.03) - prompt_count  # 剩余需要选取的数量
            if overall_count > 0:
                non_prompt_mask = ~prompt_mask  # 非 Prompt 类别的掩码
                non_prompt_vals = max_vals[non_prompt_mask]
                non_prompt_inds = torch.where(non_prompt_mask)[0]

                sorted_vals, sorted_inds = torch.topk(non_prompt_vals, overall_count)
                overall_inds = non_prompt_inds[sorted_inds]  # 整体 TopK 的索引
            else:
                overall_inds = torch.tensor([], device=t_cls_scores.device, dtype=torch.long)

            # ------------------------------------
            # 3. 类别 TopK（Prompt 以外）
            non_prompt_classes = torch.arange(t_cls_scores.size(1), device=t_cls_scores.device)
            non_prompt_classes = non_prompt_classes[~torch.isin(non_prompt_classes, exist_classes_tensor)]  # 非 Prompt 类别

            class_topk_count = int(count_num * 0.001)
            class_topk_inds = []

            for cls in non_prompt_classes:
                cls_mask = (max_inds == cls)
                cls_vals = max_vals[cls_mask]
                cls_inds = torch.where(cls_mask)[0]

                if len(cls_inds) > class_topk_count:
                    sorted_vals, sorted_inds = torch.topk(cls_vals, class_topk_count)
                    cls_inds = cls_inds[sorted_inds]  # 选取 TopK
                class_topk_inds.append(cls_inds)

            class_topk_inds = torch.cat(class_topk_inds) if class_topk_inds else torch.tensor([], device=t_cls_scores.device)

            # 去重
            selected_inds = torch.cat([prompt_inds, overall_inds, class_topk_inds])
            selected_inds = torch.unique(selected_inds)  # 去除重复

            # ------------------------------------
            # 初始化掩码并标记正样本
            mask = torch.zeros_like(max_vals)
            mask[selected_inds] = 1.0  # 标记为正样本

            # 计算正样本置信度和
            fg_num = mask * max_vals  # 置信度和
            fg_num = fg_num.sum()  # fg_num 是置信度和
            fg_num = max(fg_num, torch.tensor(1e-6, device=fg_num.device))
            b_mask = mask > 0.  # 转为布尔类型

        #------------------------------Assigner Assistant End-----------------------------------------

        loss_cls = QFLv2(
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=mask,
            reduction="sum",
        ) / fg_num
        if self.bbox_loss_type == 'l1':
            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                t_bbox_preds[b_mask],
            ) * t_centernesses.sigmoid()[b_mask]).mean()
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
            ) * t_centernesses.sigmoid()[b_mask]
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        loss_centerness = F.binary_cross_entropy(
            s_centernesses[b_mask].sigmoid(),
            t_centernesses[b_mask].sigmoid(),
            reduction='mean'
        )

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness
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