import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector


@ROTATED_DETECTORS.register_module()
class RotatedMeanTeacherWithPseudoLabeledData(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss=None, train_cfg=None, test_cfg=None, symmetry_aware=False):
        super(RotatedMeanTeacherWithPseudoLabeledData, self).__init__(
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
        super(RotatedMeanTeacherWithPseudoLabeledData, self).forward_train(imgs, img_metas, **kwargs)
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
        for key, val in sup_losses.items(): # TODO BUG here
            # if key[:4] == 'loss':
            if 'loss' in key:   # ReDet 中存在 s0.roi_cls_loss
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

            # TODO
            # 1.把sup那一支在预热阶段后提取伪标签
            # 2.保留sup的GT
            # 3.把双阶段分类分支的伪标签从01的one-hot label换成teacher的softmax后的预测，然后用一个kl散度监督

            with torch.no_grad():
                # 1.将 sup 与 unsup 都输入到 teacher 中用于提取伪标签
                pseudo_bboxes_unsup, pseudo_labels_unsup = self.teacher.forward_train(
                    get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data['unsup_weak'])
                pseudo_bboxes_sup, pseudo_labels_sup = self.teacher.forward_train(
                    get_boxes=True, rcnn_configs=self.rcnn_configs, **format_data['sup'])

                # 2.将 unsup 的伪标签信息加入到 format_data[unsup_strong] 中
                format_data['unsup_strong']['gt_bboxes'] = [pseudo_bboxes_img[:, :5] for pseudo_bboxes_img in pseudo_bboxes_unsup]
                format_data['unsup_strong']['gt_labels'] = pseudo_labels_unsup

                # 3.将 sup 的伪标签添加到 format_data[sup] 中与 GT 融合
                for idx in range(len(pseudo_bboxes_sup)):
                    format_data['sup']['gt_bboxes'][idx] = torch.cat((format_data['sup']['gt_bboxes'][idx], pseudo_bboxes_sup[idx][:, :5]), dim=0)  # 取前5维
                    format_data['sup']['gt_labels'][idx] = torch.cat((format_data['sup']['gt_labels'][idx], pseudo_labels_sup[idx]), dim=0)  # 添加伪标签
                
                # 4.将 sup 中的 img, img_metas, gt_bboxes 和 gt_labels 拷贝到 unsup_strong 中
                format_data['unsup_strong']['img'] = torch.cat((format_data['unsup_strong']['img'], format_data['sup']['img']), dim=0)  # 复制图像
                format_data['unsup_strong']['img_metas'] += format_data['sup']['img_metas'][:]  # 复制元数据
                format_data['unsup_strong']['gt_bboxes'] += format_data['sup']['gt_bboxes']  # 合并 gt_bboxes
                format_data['unsup_strong']['gt_labels'] += format_data['sup']['gt_labels']  # 合并 gt_labels
                

            # 5.使用 sup 的 gt 和 伪标签以及 unsup 的伪标签训练 student，获取无监督端的损失
            unsup_losses = self.student.forward_train(**format_data['unsup_strong'])
            for key, val in unsup_losses.items():
                # if key[:4] == 'loss':
                if 'loss' in key:   # ReDet 中存在 s0.loss_cls s0.loss_bbox s1.loss_cls s1.loss_bbox
                    losses[f"{key}_unsup"] = unsup_weight * val
            # 此时 losses 的 key 包含 
            # loss_rpn_cls_sup、loss_rpn_bbox_sup、loss_cls_sup和loss_bbox_sup
            # loss_rpn_cls_unsup、loss_rpn_bbox_unsup、loss_cls_unsup和loss_bbox_unsup
        self.iter_count += 1

        return losses