from __future__ import division


import os
from os import path

import cv2
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

import pickle
import copy
from utils.mark import mark_pred, mark_target
from utils.func import *
from utils.suppression import non_max_suppression_yolo, non_max_suppression_ssd, non_max_suppression_retinaNet

from tqdm import tqdm
from glob import glob
colors = pickle.load(open("dataset//pallete", "rb")) 


def evaluate_yolo(save_image_path, model, dataset, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    orign_list = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        if targets.shape[0] == 0: continue
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)  
            suppress_output = non_max_suppression_yolo(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        
        for idx, img in enumerate(imgs):
            img = np.array(img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply (cv2 mode)
            pred_img = copy.deepcopy(img)
            suppress_o = suppress_output[idx] # [xmin, ymin, xmax, ymax, conf, cls]
            
            target_img = mark_target(img, targets, dataset, idx)
            pred_img = mark_pred(pred_img, suppress_o, dataset)
            vis = np.concatenate((target_img, pred_img), axis=1)
            # cv2.imshow('win', vis)
            # cv2.waitKey()
            cv2.imwrite(save_image_path + 'val{}_jpg.png'.format(idx + batch_i), vis)

        sample_metrics += get_batch_statistics(suppress_output, targets.cuda(), iou_threshold=iou_thres)
        if batch_i > 10: break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def evaluate_ssd(save_image_path, model, dataset, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # default_boxes for SSD
    default_boxes = get_dboxes()
    default_boxes = default_boxes.cuda()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    orign_list = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        if targets.shape[0] == 0: continue
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)  
            suppress_output = non_max_suppression_ssd(outputs, default_boxes) 
            suppress_output = suppress_output.cuda()
        
        for idx, img in enumerate(imgs):
            img = np.array(img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply (cv2 mode)
            pred_img = copy.deepcopy(img)
            suppress_o = suppress_output[idx] # [xmin, ymin, xmax, ymax, 0, class_score, class_pred] # [66, 7]
            
            target_img = mark_target(img, targets, dataset, idx)
            pred_img = mark_pred(pred_img, suppress_o, dataset)
            vis = np.concatenate((target_img, pred_img), axis=1)
            # cv2.imshow('win', vis)
            # cv2.waitKey()
            cv2.imwrite(save_image_path + 'val{}_jpg.png'.format(idx + batch_i), vis)

        sample_metrics += get_batch_statistics(suppress_output, targets.cuda(), iou_threshold=iou_thres)
        if batch_i > 10: break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def evaluate_retinaNet(save_image_path, model, dataset, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # anchor_boxes for RetinaNet
    input_size = torch.Tensor([opt.img_size,opt.img_size])
    anchor_boxes = get_anchor_boxes(input_size).to(device)

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    orign_list = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        if targets.shape[0] == 0: continue
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)  
            suppress_output = non_max_suppression_retinaNet(outputs, anchor_boxes) 
            suppress_output = suppress_output.cuda()
        
        for idx, img in enumerate(imgs):
            img = np.array(img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply (cv2 mode)
            pred_img = copy.deepcopy(img)
            suppress_o = suppress_output[idx] # [xmin, ymin, xmax, ymax, 0, class_score, class_pred] # [66, 7]
            
            target_img = mark_target(img, targets, dataset, idx)
            pred_img = mark_pred(pred_img, suppress_o, dataset)
            vis = np.concatenate((target_img, pred_img), axis=1)
            # cv2.imshow('win', vis)
            # cv2.waitKey()
            cv2.imwrite(save_image_path + 'val{}_jpg.png'.format(idx + batch_i), vis)

        sample_metrics += get_batch_statistics(suppress_output, targets.cuda(), iou_threshold=iou_thres)
        if batch_i > 10: break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--img_size", type=int, default=300, help="YOLO type(416), SSD(300), RetinaNet(600), size of each image dimension")
    parser.add_argument("--model", type=str, default="RetinaNet", help="Yolo_v1/Yolo_v2/Yolo_v3/Yolo_v4/SSD/RetinaNet")
    parser.add_argument('--dataset', default='AsiaTrafficDataset', type=str, help='training dataset, CoCo5K, CoCo, Container, VOCDataset, AsiaTrafficDataset')
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    
    # parser.add_argument("--num_class", type=int, default=80, help="CoCo5K")
    parser.add_argument("--num_class", type=int, default=14, help="MosquitoContainer")
    parser.add_argument("--reduction", type=int, default=32)
    # parser.add_argument("--saved_path", type=str, default="save")
    # parser.add_argument("--load_epoch", type=int, default=8)    
    parser.add_argument('--experiment_dir', help='dir of experiment', type=str, default = "run\AsiaTrafficDataset\RetinaNet\experiment_10")
    # parser.add_argument('--experiment_dir', help='dir of experiment', type=str, default = "run\AsiaTrafficDataset\SSD\experiment_0")
    # parser.add_argument('--experiment_dir', help='dir of experiment', type=str, default = "run\AsiaTrafficDataset\Yolo_v4\experiment_10")
    
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get val dataset (CoCo5K, CoCo, Container, VOCDataset)
    # 1. Padding h & w to each size
    # 2. calculate new padding coordinate targets (x, y, w, h)
    # 3. Selects new scale image size every tenth batch
    valid_dataset = None
    if opt.dataset == "CoCo5K":
        valid_dataset = CoCo5KDataset(img_size=opt.img_size, is_training = False)
    elif opt.dataset == "CoCo":
        valid_dataset = CoCoDataset(img_size=opt.img_size, is_training = False)
    elif opt.dataset == "Container":
        valid_dataset = ContainerDataset(img_size=opt.img_size, is_training = False)
    elif opt.dataset == "VOCDataset":
        valid_dataset = VOCDataset(img_size=opt.img_size, is_training = False)
    elif opt.dataset == "AsiaTrafficDataset":
        valid_dataset = AsiaTrafficDataset(img_size=opt.img_size, is_training = False)

    class_names = valid_dataset.classes
    opt.num_class = len(valid_dataset.classes)
    print("num_class:", opt.num_class)

    net = None
    save_image_path = './predict/{}/{}/'.format(opt.dataset, opt.model)
    if not os.path.exists(save_image_path): os.makedirs(save_image_path)

    if opt.model == "Yolo_v1":        
        # Yolo V1 define
        from model.yolo_v1 import YOLOv1, RegionLoss
        net = YOLOv1(dropout= opt.dropout, num_class = opt.num_class)
        # outputs_shape torch.Size([1, 7, 7, (80 + 5)])
    elif opt.model == "Yolo_v2":
        # Yolo V2 define
        from model.yolo_v2 import YOLOv2, RegionLoss
        net = YOLOv2(num_classes = opt.num_class)    
        # outputs_shape torch.Size([1, 5 * (80 + 5), 13, 13])   
    elif opt.model == "Yolo_v3":
        # Yolo V3 define
        from model.yolo_v3 import YOLOv3, MultiScaleRegionLoss
        net = YOLOv3(num_classes = opt.num_class)
        # outputs_0_shape torch.Size([1,  3 * (80 + 5), 13, 13])
        # outputs_1_shape torch.Size([1,  3 * (80 + 5), 26, 26])
        # outputs_2_shape torch.Size([1,  3 * (80 + 5), 52, 52])
    elif opt.model == "Yolo_v4":
        # Yolo V4 define
        from model.yolo_v4 import YOLOv4, MultiScaleRegionLoss
        net = YOLOv4(yolov4conv137weight=None, n_classes=opt.num_class, inference=False)
        # filters=(classes + 5)*<number of mask> mask = 3
        # outputs_0_shape torch.Size([1, 3 * (80 + 5), 52, 52])
        # outputs_1_shape torch.Size([1, 3 * (80 + 5), 26, 26])
        # outputs_2_shape torch.Size([1, 3 * (80 + 5), 13, 13])  
    elif opt.model == "SSD":
        from model.ssd import ssd, SSDLoss
        net = ssd(num_classes=opt.num_class)
        # outputs_shape torch.Size([2, 8732, (13 + 4)])
    elif opt.model == "RetinaNet":
        from model.retinaNet import RetinaNet, FocalLoss
        net = RetinaNet(fpn=101, num_classes = opt.num_class)
        # outputs_shape torch.Size([2, 76725, (13 + 4)])

    # Trained model path and name
    experiment_dir = opt.experiment_dir
    model_name = glob(os.path.join(opt.experiment_dir, "*.pkl"))[0]
    load_name = os.path.join(experiment_dir, 'checkpoint.pth.tar')

    # Load save/trained model
    if not os.path.isfile(model_name):
        raise RuntimeError("=> no model found at '{}'".format(model_name))
    print('====>loading trained model from ' + model_name)
    if not os.path.isfile(load_name):
        raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
    print('====>loading trained model from ' + load_name)

    net = torch.load(model_name)
    checkpoint = torch.load(load_name)

    # model_path = os.path.join(opt.saved_path, "{}_{}_params".format(opt.model, opt.load_epoch))
    # checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    if device.type == 'cpu':
        model = torch.nn.DataParallel(net)
    else:
        num_gpus = [i for i in range(opt.n_gpu)]
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    
    if opt.model == "SSD":
        precision, recall, AP, f1, ap_class = evaluate_ssd(
            save_image_path,
            model,
            dataset=valid_dataset,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=opt.batch_size,
        )
    elif opt.model == "RetinaNet":
        precision, recall, AP, f1, ap_class = evaluate_retinaNet(
            save_image_path,
            model,
            dataset=valid_dataset,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=opt.batch_size,
        )
    else:
        precision, recall, AP, f1, ap_class = evaluate_yolo(
            save_image_path,
            model,
            dataset=valid_dataset,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=opt.batch_size,
        )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")