# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sparsercnn_config(cfg):
    """
    Add config for SparseRCNN.
    """
    cfg.MODEL.SparseRCNN = CN()
    cfg.MODEL.SparseRCNN.NUM_CLASSES = 80
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.SparseRCNN.NHEADS = 8
    cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.SparseRCNN.NUM_CLS = 1
    cfg.MODEL.SparseRCNN.NUM_REG = 3
    cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.SparseRCNN.USE_FOCAL = True
    cfg.MODEL.SparseRCNN.ALPHA = 0.25
    cfg.MODEL.SparseRCNN.GAMMA = 2.0
    cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # Modified Fast Head.
    cfg.MODEL.SparseRCNN.FAST_ON = False
    cfg.MODEL.SparseRCNN.FAST_SELF_ATTN = False

    # Mask Head.
    cfg.MODEL.SparseRCNN.MASK_WEIGHT = 1.0
    cfg.MODEL.SparseRCNN.DICE_WEIGHT = 1.0
    cfg.MODEL.SparseRCNN.TYPE_POSITION = "sine"
    cfg.MODEL.SparseRCNN.POS_REFINE = True
    cfg.MODEL.SparseRCNN.TYPE_MASK = "MASK_RCNN"  # or "MASK_ENCODING"

    # Mask Encoding.
    cfg.MODEL.SparseRCNN.MASK_ENCODING = CN()
    cfg.MODEL.SparseRCNN.MASK_ENCODING.MASK_SIZE = 96
    cfg.MODEL.SparseRCNN.MASK_ENCODING.DIM_MASK = 256
    cfg.MODEL.SparseRCNN.MASK_ENCODING.PATH_ENCODING = "datasets/coco/encoding/local/encoder36_size96_dim256.pth"
    cfg.MODEL.SparseRCNN.MASK_ENCODING.NUM_MLP = 2
    cfg.MODEL.SparseRCNN.MASK_ENCODING.DIM_MLP = 1024
    # cfg.MODEL.SparseRCNN.MASK_ENCODING.PRED_NORM = True
    # cfg.MODEL.SparseRCNN.MASK_ENCODING.TYPE_NORM = "p1"
