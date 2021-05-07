#!/usr/bin/env bash

export http_proxy=10.20.47.147:3128 https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest-runner
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

# compile.
python3 ${CURDIR}/setup.py build develop
sudo python3 ${CURDIR}/setup.py build develop
echo 'Finish Compile.'

# data.
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
mkdir -p ${CURDIR}/datasets/coco
echo 'Start Downloading Data.'
hadoop fs -get ${HDFS_ROOT}/datasets/coco/annotations ${CURDIR}/datasets/coco
hadoop fs -get ${HDFS_ROOT}/datasets/coco/val2017.zip ${CURDIR}/datasets/coco
hadoop fs -get ${HDFS_ROOT}/datasets/coco/train2017.zip ${CURDIR}/datasets/coco
unzip -qq ${CURDIR}/datasets/coco/train2017.zip -d ${CURDIR}/datasets/coco
unzip -qq ${CURDIR}/datasets/coco/val2017.zip -d ${CURDIR}/datasets/coco
echo 'Finish Downloading Data.'

# process.
######################################################
URL=tcp://127.0.0.1:50122
CONFIG_FILE=${CURDIR}/projects/SparseRCNN/configs/zhang/sparsercnn.res50.300pro.3x_mask.yaml
GPU_NUM=8
OUTPUT_DIR=${CURDIR}/sparsercnn.res50.300pro.3x_mask
# MODEL_FILE=/data00/home/zhangrufeng1/pretrained/detectron2/sparse_rcnn/r101_300pro/torch1_4/model_final.pth
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
HDFS_DIR=${HDFS_ROOT}/models/QueryRCNN/sparsercnn.res50.300pro.3x_mask


#python3 ${CURDIR}/projects/SparseRCNN/train_net.py --dist-url ${URL} \
#--config-file ${CONFIG_FILE} \
#--num-gpus ${GPU_NUM} \
#--eval-only MODEL.WEIGHTS ${MODEL_FILE} OUTPUT_DIR ${OUTPUT_DIR}

python3 ${CURDIR}/projects/SparseRCNN/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} \
OUTPUT_DIR ${OUTPUT_DIR}

python3 ${CURDIR}/pkl2pth_file.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}


#--eval-only MODEL.WEIGHTS ${MODEL_FILE} OUTPUT_DIR ${OUTPUT_DIR}