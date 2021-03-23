import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class FastHeadLayer(nn.Module):

    def __init__(self, cfg, num_classes, scale_clamp: float = _DEFAULT_SCALE_CLAMP, weights=(10.0, 10.0, 5.0, 5.0)):
        super().__init__()
        self.cfg = cfg
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.self_attn = cfg.MODEL.SparseRCNN.FAST_SELF_ATTN

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(7*7*256, 256)
        self.norm1 = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)

        self.linear3 = nn.Linear(7*7*256, 256)
        self.norm3 = nn.LayerNorm(256)
        self.linear4 = nn.Linear(256, 256)
        self.norm4 = nn.LayerNorm(256)

        self.linear5 = nn.Linear(512, 256)
        self.norm5 = nn.LayerNorm(256)
        self.linear6 = nn.Linear(256, 256)
        self.norm6 = nn.LayerNorm(256)

        self.scale_clamp = scale_clamp
        self.weights = weights

        if self.use_focal:
            self.class_logits = nn.Linear(256, num_classes)
        else:
            self.class_logits = nn.Linear(256, num_classes + 1)
        self.bboxes_delta = nn.Linear(256, 4)

        if self.self_attn:
            self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, bboxes, tgt, pooler=None, norm=None, query_pos=None):
        """
        :param features: can be original features or memory (RoI feature).
        :param bboxes: (N, nr_boxes, 4)
        :param tgt: (nr_boxes, N, d_model)
        :param pooler:
        :return:
        """

        N, nr_boxes = bboxes.shape[:2]
        proposals = list()
        for b in range(N):
            proposals.append(Boxes(bboxes[b]))
        # roi_feature
        memory = pooler(features, proposals)
        memory = memory.view(N*nr_boxes, -1)

        feat_box = self.relu(self.norm1(self.linear1(memory)))
        feat_box = self.relu(self.norm2(self.linear2(feat_box)))
        bboxes_deltas = self.bboxes_delta(feat_box)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        feat_cls = self.relu(self.norm3(self.linear3(memory)))
        feat_cls = self.relu(self.norm4(self.linear4(feat_cls)))

        if tgt is None:
            memory_attn = feat_cls.clone()
        else:
            memory_attn = tgt.clone()

        if self.self_attn:
            memory_attn = memory_attn.view(N, nr_boxes, 256).permute(1, 0, 2)
            memory_attn = self.self_attn(memory_attn, memory_attn, value=memory_attn)[0]
            memory_attn = memory_attn.transpose(0, 1).reshape(N * nr_boxes, -1)

        feat_cls = torch.cat((feat_cls, memory_attn), dim=-1)
        feat_cls = self.relu(self.norm5(self.linear5(feat_cls)))
        feat_cls = self.relu(self.norm6(self.linear6(feat_cls)))
        class_logits = self.class_logits(feat_cls)

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), feat_cls

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def adaptive_feature_pooling(x: List[torch.Tensor], box_lists, roi_poolers):

        num_level_assignments = len(roi_poolers.level_poolers)
        assert num_level_assignments > 1

        assert isinstance(x, list) and isinstance(box_lists, torch.Tensor)
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )
        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        def fmt_box_list(box_tensor, batch_index):
            repeated_index = torch.full(
                (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
            )
            return cat((repeated_index, box_tensor), dim=1)

        pooler_fmt_boxes = cat(
            [fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = roi_poolers.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for x_level, pooler in zip(x, roi_poolers.level_poolers):
            output += pooler(x_level, pooler_fmt_boxes)

        return output

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

