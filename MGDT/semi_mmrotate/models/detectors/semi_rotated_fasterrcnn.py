import torch
from mmrotate.models import RotatedFasterRCNN, ROTATED_DETECTORS, RotatedTwoStageDetector
from mmrotate.models.builder import build_loss


@ROTATED_DETECTORS.register_module()
class SemiRotatedFasterRCNN(RotatedFasterRCNN):

    def forward_train(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        soft_labels=None,
                        gt_bboxes_ignore=None,
                        get_boxes=False,
                        rcnn_configs=None,
                        loss_configs=None):
        
        super(RotatedTwoStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        # x 表示从 FPN 出来的特征图，共有五个，大小分别为 (N,256,128,128) (N,256,64,64) (N,256,32,32) (N,256,16,16) (N,256,8,8) 
        losses = dict()

        # RPN forward and loss
        if not get_boxes:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            # rpn_losses 包括 loss_rpn_cls 和 loss_rpn_bbox，每个包含5个值
            # proposal_list 包括 N 个 (2000,5)
            losses.update(rpn_losses)

            if loss_configs != None and loss_configs['type'] == 'FocalLoss':  # for unbiased teacher
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
                self.roi_head.bbox_head.loss_cls = build_loss(loss_configs)
            elif loss_configs == None:
                roi_cls_loss_orign = self.roi_head.bbox_head.loss_cls
            
            if soft_labels != None:
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore)
            else:
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore)
            # roi_losses 包括 loss_cls 、acc 和 loss_bbox 都包括1个值
            self.roi_head.bbox_head.loss_cls = roi_cls_loss_orign
            losses.update(roi_losses)
            loss_dict = {}
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_dict[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    loss_dict[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')
            # loss_dict 的 key 包含 loss_rpn_cls、loss_rpn_bbox、loss_cls、acc和loss_bbox
            return loss_dict
        
        with torch.no_grad():
            self.eval()
            assert self.with_bbox, 'Bbox head must be implemented.'
            x = self.extract_feat(img)
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas) # torch.Size([2000,5])
            det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
                                            x, img_metas, proposal_list, rcnn_configs, rescale=False)
            self.train()
            # 如果有值，则 det_bboxes 为 [M,6], det_labels 为 [M,1]
        return det_bboxes, det_labels