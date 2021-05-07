#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified by Rufeng Zhang
# Contact: cxrfzhang@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Boxes
import collections
from .MaskEncoding import AutoEncoder


class MaskHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()
        assert cfg.MODEL.MASK_ON
        self.type_mask = cfg.MODEL.SparseRCNN.TYPE_MASK

        # Build Mask RoI.
        mask_pooler, in_channels = self._init_mask_pooler(cfg, roi_input_shape)
        self.mask_pooler = mask_pooler

        # Build Mask Head.
        if self.type_mask == "MASK_RCNN":
            mask_head = MaskRCNNHead(cfg, in_channels)
            if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
                num_classes = 1
            else:
                num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        elif self.type_mask == "MASK_ENCODING":
            mask_head = MaskEncodingHead(cfg, in_channels)
            num_classes = 1
            dim_mask = cfg.MODEL.SparseRCNN.MASK_ENCODING.DIM_MASK
            self.mask_size = cfg.MODEL.SparseRCNN.MASK_ENCODING.MASK_SIZE
            self.path_encoding = cfg.MODEL.SparseRCNN.MASK_ENCODING.PATH_ENCODING
            self.flag_parameters = False
            self.mask_encoding = AutoEncoder(dim_mask=dim_mask, mask_size=self.mask_size)
        else:
            raise NotImplementedError

        self.mask_head = mask_head
        self.num_classes = num_classes

    @staticmethod
    def _init_mask_pooler(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return mask_pooler, in_channels

    def loading_parameters_for_encoding(self, path_encoding):
        assert self.flag_parameters is False

        # encoding parameters.
        params = torch.load(path_encoding)
        params = collections.OrderedDict([('.'.join(x.split('.')[1:]), y) if 'module' in x
                                          else (x, y) for x, y in params.items()])
        self.mask_encoding.load_state_dict(params)
        self.mask_encoding.freeze_params()
        self.mask_encoding.eval()

    def forward(self, features, bboxes):
        N, nr_boxes = bboxes.shape[:2]
        # mask feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        mask_features = self.mask_pooler(features, proposal_boxes)

        # mask pred.
        mask_preds = self.mask_head(mask_features)
        if self.type_mask == "MASK_RCNN":
            mask_preds = mask_preds.view(N, nr_boxes, self.num_classes, mask_preds.shape[-2], mask_preds.shape[-1])
        elif self.type_mask == "MASK_ENCODING":
            if self.training:
                mask_preds = mask_preds.view(N, nr_boxes, mask_preds.shape[-1])
            else:
                mask_preds = self.mask_encoding.decoding(mask_preds).view(N, nr_boxes, self.mask_size, self.mask_size)
        return mask_preds


class MaskRCNNHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        assert len(set(in_channels)) == 1, in_channels

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_dims = [conv_dim] * (num_conv + 1)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        else:
            num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES

        self.conv_norm_relus = []

        cur_channels = in_channels[0]
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


class MaskEncodingHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        assert len(set(in_channels)) == 1, in_channels

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_dims = [conv_dim] * (num_conv + 1)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM

        # 4 conv similar in Mask R-CNN.
        self.conv_norm_relus = []
        cur_channels = in_channels[0]
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        # MLP for mask regression.
        dim_mask = cfg.MODEL.SparseRCNN.MASK_ENCODING.DIM_MASK
        num_mlp = cfg.MODEL.SparseRCNN.MASK_ENCODING.NUM_MLP
        dim_mlp = cfg.MODEL.SparseRCNN.MASK_ENCODING.DIM_MLP
        height = width = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.cur_dim = cur_channels * height * width
        self.mlp = MLP(input_shape=self.cur_dim, num_mlp=num_mlp, dim_mlp=dim_mlp, dim_mask=dim_mask)
        # init.
        for p in self.mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = x.view(-1, self.cur_dim)
        x = self.mlp(x)
        x = F.normalize(x, p=1, dim=1)  # align with mask encoding to get sparse vector.
        return x


class MLP(nn.Module):
    def __init__(self, input_shape, num_mlp, dim_mlp, dim_mask):
        super().__init__()
        self.num_mlp = num_mlp
        self.dim_mlp = dim_mlp
        self.dim_mask = dim_mask

        mlp = list()
        dim_in = input_shape
        dim_out = self.dim_mlp
        for i in range(self.num_mlp):
            mlp.append(nn.Linear(dim_in, dim_out))
            mlp.append(nn.LayerNorm(dim_out))
            mlp.append(nn.ReLU(inplace=True))
            dim_in = dim_out
        self.mlp = nn.ModuleList(mlp)
        self.pred = nn.Linear(dim_out, dim_mask)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        x = self.pred(x)
        return x
