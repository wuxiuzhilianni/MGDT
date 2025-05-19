import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector
import cv2

@ROTATED_DETECTORS.register_module()
class RotatedMeanTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss=None, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedMeanTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            semi_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.rcnn_configs = train_cfg.get("rcnn_configs")
        self.symmetry_aware = symmetry_aware

    def forward_train(self, imgs, img_metas, **kwargs):
        super(RotatedMeanTeacher, self).forward_train(imgs, img_metas, **kwargs)
        """
        imgs: 大小为(B,C,W,H) 如(4,3,1024,1024) 表示图像的值
        img_metas: 包含 B 个图像的基本信息,包括 filename (xxx.png)、ori_shape img_shape pad_shape(1024,1024,3)、filp(True) tag(sup_weak)等
        **kwargs: 包含两项 gt_boxed 和 gt_labeles,每项的数量都为 B,
        """
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')

        # preprocess
        # 将所有读取的参数根据tag分类，format_data包含sup、unsup_weak和unsup_strong
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        # 堆叠图像值，将N个(1,3,1024,1024)叠成(N,3,1024,1024)用于后续前向传递img
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
            # print(f"{key}: {format_data[key]['img'].shape}")
        losses = dict()

        # supervised forward，计算有监督端的损失并加权
        sup_losses = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            # if key[:4] == 'loss':
            if 'loss' in key:
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
        # 此时 losses 的 key 包含 loss_rpn_cls_sup、loss_rpn_bbox_sup、loss_cls_sup和loss_bbox_sup

        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps

            with torch.no_grad():
                # get teacher data
                pseudo_bboxes, pseudo_labels = self.teacher.forward_train(
                    get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data['unsup_weak'])
                # 取前5维，第6维表示置信度加入到 unsup_strong 用于训练 student
                format_data['unsup_strong']['gt_bboxes'] = [pseudo_bboxes_img[:, :5] for pseudo_bboxes_img in pseudo_bboxes]
                format_data['unsup_strong']['gt_labels'] = pseudo_labels

            # 使用 unlabeled data 和 gt box 训练 student，获取无监督端的损失
            unsup_losses = self.student.forward_train(**format_data['unsup_strong'])
            for key, val in unsup_losses.items():
                # if key[:4] == 'loss':
                if 'loss' in key:
                    losses[f"{key}_unsup"] = unsup_weight * val
            # 此时 losses 的 key 包含 
            # loss_rpn_cls_sup、loss_rpn_bbox_sup、loss_cls_sup和loss_bbox_sup
            # loss_rpn_cls_unsup、loss_rpn_bbox_unsup、loss_cls_unsup和loss_bbox_unsup
        self.iter_count += 1

        return losses