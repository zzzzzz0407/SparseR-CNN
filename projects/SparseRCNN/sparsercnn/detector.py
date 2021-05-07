#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm)
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .position_encoding import build_position_encoding
from .mask.mask_head import MaskHead

__all__ = ["SparseRCNN"]


@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.fast_on = cfg.MODEL.SparseRCNN.FAST_ON
        if self.fast_on:
            self.init_proposal_features = None
        else:
            self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

        # Mask head.
        self.mask_on = cfg.MODEL.MASK_ON
        self.type_mask = cfg.MODEL.SparseRCNN.TYPE_MASK
        if self.mask_on:
            self.mask_head = MaskHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
            mask_weight = cfg.MODEL.SparseRCNN.MASK_WEIGHT
            weight_dict.update({"loss_mask": mask_weight})
            # self.mask_head = DETRsegm(cfg)
            # self.position_embedding = build_position_encoding(cfg)
            # dice_weight = cfg.MODEL.SparseRCNN.DICE_WEIGHT
            # weight_dict.update({"loss_dice": dice_weight})

        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        if self.mask_on:
            losses += ["masks"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal,
                                      type_mask=self.type_mask)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        # init mask encoding if needed.
        if self.training:
            if self.type_mask == "MASK_ENCODING":
                if not self.mask_head.flag_parameters:
                    self.mask_head.loading_parameters_for_encoding(self.mask_head.path_encoding)
                    self.mask_head.flag_parameters = True

        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        if self.fast_on:
            outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features)
        else:
            outputs_class, outputs_coord, proposal_features = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.mask_on:
            output.update({"pred_masks": self.mask_head(features, output["pred_boxes"].detach())})

        """
                if self.mask_on:
            
            # B, _, H, W = images.tensor.shape
            # masks = torch.ones((B, H, W), dtype=torch.bool, device=images.device)
            # for img_size, m in zip(images.image_sizes, masks):
            #     m[:img_size[0], : img_size[1]] = False
            # masks = F.interpolate(masks[None].float(), size=features[-1].shape[-2:]).bool()[0]
            # pos = self.position_embedding(features[-1], masks)
            
            # continue

            if self.type_mask == "MASK_RCNN":
                output.update({"pred_masks": self.mask_head(features)})
            else:
                raise NotImplementedError
        
        """

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images.tensor.shape)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            if self.type_mask == "MASK_ENCODING":
                loss_dict = self.criterion(output, targets, mask_encoding=self.mask_head.mask_encoding)
            else:
                loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.mask_on:
                mask_pred = output["pred_masks"]
            else:
                mask_pred = None
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)

            # if self.mask_on:
            #     self.mask_head.mask_inference(mask_pred, results)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets, batch_shape):
        bs, _, bh, bw = batch_shape
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            if self.mask_on:
                target["gt_masks"] = targets_per_image.gt_masks
                target["type_mask"] = self.type_mask
                if self.type_mask == "MASK_RCNN":
                    target["mask_size"] = self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
                else:
                    target["mask_size"] = self.cfg.MODEL.SparseRCNN.MASK_ENCODING.MASK_SIZE

                # target[""]
                # assert targets_per_image.has("gt_masks")
                """
                if isinstance(targets_per_image.get("gt_masks"), PolygonMasks):
                    per_im_bitmasks = []
                    polygons = targets_per_image.get("gt_masks").polygons
                    for per_polygons in polygons:
                        bitmask = polygons_to_bitmask(per_polygons, h, w)
                        bitmask = torch.from_numpy(bitmask).to(self.device).float()
                        per_im_bitmasks.append(bitmask)
                    per_im_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                else:  # RLE format bitmask
                    per_im_bitmasks = targets_per_image.get("gt_masks").tensor.to(self.device).float()
                    per_im_bitmasks = F.interpolate(per_im_bitmasks.unsqueeze(0), (h, w),
                                                    mode="bilinear", align_corners=False).squeeze(0).round()
                valid_im_bitmasks = per_im_bitmasks.new_zeros(per_im_bitmasks.shape[0], bh, bw)
                valid_im_bitmasks[:, :h, :w] = per_im_bitmasks
                target["masks"] = valid_im_bitmasks
                """

            new_targets.append(target)
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                if self.mask_on:
                    mask_pred_per_image = mask_pred[i]
                    if self.type_mask == "MASK_RCNN":
                        cls_agnostic_mask = mask_pred_per_image.size(1) == 1
                        if cls_agnostic_mask:
                            raise NotImplementedError
                            mask_probs_pred = mask_pred_per_image.sigmoid()
                        else:
                            # Select masks corresponding to the predicted classes
                            ins_ids, cls_ids = topk_indices // self.num_classes, topk_indices % self.num_classes
                            # mask_probs_pred.shape: (B, 1, Hmask, Wmask)
                            mask_probs_pred = mask_pred_per_image[ins_ids, cls_ids][:, None].sigmoid()
                    elif self.type_mask == "MASK_ENCODING":
                        # Select masks corresponding to the predicted classes
                        ins_ids, cls_ids = topk_indices // self.num_classes, topk_indices % self.num_classes
                        mask_probs_pred = mask_pred_per_image[ins_ids][:, None].sigmoid()
                    else:
                        raise NotImplementedError
                    result.pred_masks = mask_probs_pred

                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
