from __future__ import division


import os
from os import path
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
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

from dataset.CoCo5K import CoCo5KDataset
from dataset.CoCoData import CoCoDataset
from dataset.Container import ContainerDataset
from dataset.VOCData import VOCDataset
from dataset.AsiaTraffic import AsiaTrafficDataset


from tqdm import tqdm
import statistics
from utils.saver import Saver
from utils.pytorchtools import EarlyStopping


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument('--dataset', default='AsiaTrafficDataset', type=str, help='training dataset, CoCo5K, CoCo, Container, VOCDataset, AsiaTrafficDataset')
    parser.add_argument("--model", type=str, default="Yolo_v4", help="Yolo_v1/Yolo_v2/Yolo_v3/Yolo_v4")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument("--dropout", type=float, default=0.2, help="interval evaluations on validation set")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning_rate")
    parser.add_argument("--reduction", type=int, default=32)    
    parser.add_argument('--optimizer', help='training optimizer', default='adam', type=str)
    parser.add_argument('--weight_decay', help='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_decay_gamma', help='learning rate decay ratio', default=0.95, type=float)
    
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    multiscale_training = opt.multiscale_training
    n_cpu = opt.n_cpu
    batch_size = opt.batch_size
    epochs = opt.epochs 

    # Get dataset (CoCo5K, CoCo, Container, VOCDataset)
    # 1. Padding h & w to each size
    # 2. calculate new padding coordinate targets (x, y, w, h)
    # 3. Selects new scale image size every tenth batch
    dataset = None
    if opt.dataset == "CoCo5K":
        dataset = CoCo5KDataset(img_size=416)
    elif opt.dataset == "CoCo":
        dataset = CoCoDataset(img_size=416)
    elif opt.dataset == "Container":
        dataset = ContainerDataset(img_size=416)
    elif opt.dataset == "VOCDataset":
        dataset = VOCDataset(img_size=416)
    elif opt.dataset == "AsiaTrafficDataset":
        dataset = AsiaTrafficDataset(img_size=416)
    opt.num_class = len(dataset.classes)
    print("num_class: ", opt.num_class)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,        
        collate_fn=dataset.collate_fn,
    )

    net = None
    criterion = None
    if opt.model == "Yolo_v1":        
        # Yolo V1 define
        from model.yolo_v1 import YOLOv1, RegionLoss
        net = YOLOv1(dropout= opt.dropout, num_class = opt.num_class)
        criterion = RegionLoss(net.num_classes)
        # outputs_shape torch.Size([1, 7, 7, (80 + 5)])
    elif opt.model == "Yolo_v2":
        # Yolo V2 define
        from model.yolo_v2 import YOLOv2, RegionLoss
        net = YOLOv2(num_classes = opt.num_class)    
        criterion = RegionLoss(net.anchors, net.num_classes)
        # outputs_shape torch.Size([1, 5 * (80 + 5), 13, 13])   
    elif opt.model == "Yolo_v3":
        # Yolo V3 define
        from model.yolo_v3 import YOLOv3, MultiScaleRegionLoss
        net = YOLOv3(num_classes = opt.num_class)
        criterion = MultiScaleRegionLoss(net.anchors, net.num_classes)
        # outputs_0_shape torch.Size([1,  3 * (80 + 5), 13, 13])
        # outputs_1_shape torch.Size([1,  3 * (80 + 5), 26, 26])
        # outputs_2_shape torch.Size([1,  3 * (80 + 5), 52, 52])
    elif opt.model == "Yolo_v4":
        # Yolo V4 define
        from model.yolo_v4 import YOLOv4, MultiScaleRegionLoss
        net = YOLOv4(yolov4conv137weight=None, n_classes=opt.num_class, inference=False)
        criterion = MultiScaleRegionLoss(net.anchors, net.anch_masks, opt.num_class)
        # filters=(classes + 5)*<number of mask> mask = 3
        # outputs_0_shape torch.Size([1, 3 * (80 + 5), 52, 52])
        # outputs_1_shape torch.Size([1, 3 * (80 + 5), 26, 26])
        # outputs_2_shape torch.Size([1, 3 * (80 + 5), 13, 13])    

    opt.checkname = opt.model
    print("device : ", device)
    if device.type == 'cpu':
        model = torch.nn.DataParallel(net)
    else:
        num_gpus = [i for i in range(opt.n_gpu)]
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()
    # print(model)

    # Define Saver
    saver = Saver(opt)
    saver.save_experiment_config(model.module)

    # Train the model
    optimizer = None
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay_gamma)

    total_step = len(dataloader)
    total_train_step = opt.epochs * total_step

    # initialize the early_stopping object
    early_stopping = EarlyStopping(saver, patience=3, verbose=True)

    loss = 0
    best_pred = 0.0
    print("Ready to Training...")
    for epoch in range(opt.epochs):
        epoch_loss = []
        with tqdm(total=len(dataloader)) as pbar:
            for iteration, (_, imgs, targets) in enumerate(dataloader):                
                if type(targets) != torch.Tensor: continue
                current_train_step = (epoch) * total_step + (iteration + 1)
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)  
                
                outputs = model(imgs)         # Forward pass       
                if type(outputs) == list:      
                    shape = outputs[0].shape
                else:          
                    shape = outputs.shape

                # Calc Yolo V1/V2/V3/V4 Loss            
                loss, \
                loss_box_reg, \
                loss_box_size, \
                loss_conf, \
                loss_classifier, \
                loss_objectness, \
                loss_noobjness = criterion(outputs, targets)  

                pbar.set_description("Model: {}, lr: {}, loss: {:.4f}, loss_classifier: {:.4f}, loss_box_reg: {:.4f}, loss_box_size: {:.4f}, loss_noobjness: {:.4f}, loss_objectness: {:.4f}".format(  
                    opt.model, optimizer.param_groups[0]["lr"], loss.item(), loss_classifier, loss_box_reg, loss_box_size, loss_noobjness, loss_objectness))

                # Report Progress
                if (((current_train_step) % 100) == 0) or (current_train_step % 10 == 0 and current_train_step < 100):
                    print("\nepoch: [{}/{}], total step: [{}/{}], batch step [{}/{}]".format(epoch + 1, opt.epochs, current_train_step, total_train_step, iteration + 1, total_step))
               
                epoch_loss.append(loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                if torch.isnan(loss).item() != 1:
                    # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
                    loss.backward()
                optimizer.step()
                pbar.update(1)
                # break

        early_stopping(model, optimizer, epoch, statistics.mean(epoch_loss)) # update patience
        if early_stopping.early_stop:
            print("Early stopping epoch %s"%(epoch))
            break