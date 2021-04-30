# coding: utf-8

import os
import torch
import pickle


if __name__ == "__main__":
    fileName = "/data00/home/zhangrufeng1/pretrained/detectron2/sparse_rcnn/r50_100pro_3x_model.pkl"
    dstDir = "/data00/home/zhangrufeng1/pretrained/detectron2/sparse_rcnn/"

    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    if ".pkl" in fileName:
        file_name = fileName
        print("Loading {}".format(file_name))
        pkl_file = open(file_name, 'rb')
        checkpoint = pickle.load(pkl_file)
        name, ext = os.path.splitext(file_name)
        out_path = os.path.join(dstDir, name + ".pth")
        torch.save(checkpoint, out_path)
