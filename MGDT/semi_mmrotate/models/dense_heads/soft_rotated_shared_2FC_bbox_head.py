import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import nms_rotated
from mmcv.runner import force_fp32
from mmrotate.models import RotatedConvFCBBoxHead, ROTATED_HEADS

@ROTATED_HEADS.register_module()
class SoftRotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SoftRotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = self.multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    def multiclass_nms_rotated(self,
                            multi_bboxes,
                            multi_scores,
                            score_thr,
                            nms,
                            max_num=-1,
                            score_factors=None,
                            return_inds=False):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
            multi_scores (torch.Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms (float): Config of NMS.
            max_num (int, optional): if there are more than max_num bboxes after
                NMS, only top max_num will be kept. Default to -1.
            score_factors (Tensor, optional): The factors multiplied to scores
                before applying NMS. Default to None.
            return_inds (bool, optional): Whether return the indices of kept
                bboxes. Default to False.

        Returns:
            tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
        """
        num_classes = multi_scores.size(1) - 1 # 15
        # exclude background category
        if multi_bboxes.shape[1] > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 5) # torch.Size([2000,15,5]) 将 multi_bboxes 复制 15 遍
        scores = multi_scores[:, :-1] # 不包含背景类别 torch.Size([2000,15])

        labels = torch.arange(num_classes, dtype=torch.long, device=scores.device) # 0~14
        labels = labels.view(1, -1).expand_as(scores) # torch.Size([2000,15]) 2000 个 0 到 14
        bboxes = bboxes.reshape(-1, 5) # torch.Size([30000, 5])
        scores = scores.reshape(-1) # torch.Size([30000])
        labels = labels.reshape(-1) # torch.Size([30000])

        # remove low scoring boxes
        valid_mask = scores > score_thr
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

        if bboxes.numel() == 0:
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, labels, inds
            else:
                return dets, labels

        # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
        # should be calculated by polygon coordinates.
        # But the conversion from rbbox to polygon will slow down the speed.
        # So we use max(x,y) + max(w,h) as max coordinate
        # which is larger than polygon max coordinate
        # max(x1, y1, x2, y2,x3, y3, x4, y4)
        max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
        offsets = labels.to(bboxes) * (max_coordinate + 1)
        if bboxes.size(-1) == 5:
            bboxes_for_nms = bboxes.clone()
            bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
        else:
            bboxes_for_nms = bboxes + offsets[:, None]
        _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr) # 通过 IOU 进一步筛选

        if max_num > 0: # 取前 max_num 个
            keep = keep[:max_num]

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if return_inds:
            return torch.cat([bboxes, scores[:, None]], 1), labels, keep
        else:
            return torch.cat([bboxes, scores[:, None]], 1), labels
        

    