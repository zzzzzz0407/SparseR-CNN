#!/usr/bin/env bash

export http_proxy=10.20.47.147:3128 https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest-runner
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy

CURDIR=$(cd $(dirname $0); pwd)

# process.
######################################################
URL=tcp://127.0.0.1:50066
CONFIG_FILE=${CURDIR}/projects/SparseRCNN/configs/sparsercnn.res101.300pro.3x.yaml
GPU_NUM=4
OUTPUT_DIR=${CURDIR}/debug
MODEL_FILE=/data00/home/zhangrufeng1/pretrained/detectron2/sparse_rcnn/r101_300pro/torch1_4/model_final.pth


python3 ${CURDIR}/projects/SparseRCNN/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} \
--eval-only MODEL.WEIGHTS ${MODEL_FILE} OUTPUT_DIR ${OUTPUT_DIR}
