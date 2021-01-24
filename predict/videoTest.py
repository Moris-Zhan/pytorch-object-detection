from __future__ import division


import os

import argparse

import torch

from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


from dataset.Container import ContainerDataset
# from VocFormat import VOCDataset
from dataset.AsiaTraffic import AsiaTrafficDataset

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument("--dropout", type=float, default=0.2, help="interval evaluations on validation set")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="interval evaluations on validation set")
    parser.add_argument("--num_class", type=int, default=80, help="interval evaluations on validation set")
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--model", type=str, default="Yolo_v4", help="Yolo_v1/Yolo_v2/Yolo_v3/Yolo_v4")
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    multiscale_training = opt.multiscale_training
    n_cpu = opt.n_cpu
    batch_size = opt.batch_size
    epochs = opt.epochs

    # train_path = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\data\\coco\\trainvalno5k.txt'
    # valid_path = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\data\\coco\\5k.txt'


    # Get dataloader
    # grid_size(v1 = 7, v2 = 13)
    # dataset = ContainerDataset(img_size=416)
    dataset = AsiaTrafficDataset(img_size=416)
    # dataset.videoTest()
    # dataset = CoCoDataset(img_size=416)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,        
        collate_fn=dataset.collate_fn
    )   

    total_step = len(dataloader)
    total_train_step = opt.epochs * total_step

    for epoch in range(opt.epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            current_train_step = (epoch) * total_step + (batch_i + 1)
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)   
            # 1. Padding h & w to each size
            # 2. calculate new padding coordinate targets (x, y, w, h)
            # 2. Selects new scale image size every tenth batch

            # print('imgs',type(imgs),imgs.shape)
            # print('final_targets',type(targets), targets.shape)
            
            # print('--------------------------------------------------------------------------------')

            

            # break
