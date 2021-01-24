from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import os
import sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import sys
import time
import datetime


import cv2
import copy
import json
from glob import glob
from dataset.augmentation import *
from torch.utils.data.dataloader import default_collate

import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET


if not os.path.isdir("videoTest"):    
    os.makedirs("videoTest")
colors = pickle.load(open("dataset//pallete", "rb"))   

class CoCo5KDataset(Dataset):
    def __init__(self, img_size=416, is_training = True) :
        train_path = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\CoCo5K\\coco\\trainvalno5k.txt'
        valid_path = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\CoCo5K\\coco\\5k.txt'
        list_path = train_path
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").strip()
            for path in self.img_files
        ]

        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"] 

        self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]

        self.img_size = img_size
        self.batch_count = 0       
        self.is_training = is_training
        print('label_files',len(self.label_files))
        print('img_files',len(self.img_files))
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        # ---------
        #  Label
        #  COCO Bounding box: (label_idx, x_center, y_center, width, height)
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            objects = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            objects = torch.cat([ objects[:, 1:], objects[:, 0].unsqueeze(1) ], dim = 1)

            # return size
            objects[:, 0] *= w
            objects[:, 1] *= h
            objects[:, 2] *= w
            objects[:, 3] *= h

            # calc (x1, y1, w, h)
            objects[:, 0] -=  0.5 * objects[:, 2]
            objects[:, 1] -=  0.5 * objects[:, 3]

            # calc (x1, y1, x2, y2)
            objects[:, 2] +=  objects[:, 0]
            objects[:, 3] +=  objects[:, 1]

            if self.is_training:
                # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
                transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
            else:
                transformations = Compose([Resize(self.img_size)])
            output_image, objects = transformations((img, objects.tolist())) # train: [x1, y1, w, h]
            padded_h, padded_w, _ = output_image.shape
            for idx in range(len(objects)):
                boxes = objects[idx][:-1]
                # Calculate train: [x1, y1, w, h] from transform, then normalized scale(padded_h, padded_w)
                x1 =  boxes[0] / padded_w
                y1 =  boxes[1] / padded_h
                w =  boxes[2] / padded_w
                h =  boxes[3] / padded_h 

                boxes = [x1, y1, w, h]
                objects[idx] = [0, objects[idx][-1]] + boxes
            '''Test Mark'''
            # for idx in range(len(objects)):
            #     boxes = objects[idx][1:]
            #     # orign_image = image
            #     xmin, ymin, xmax, ymax = int(boxes[1]*padded_w), int(boxes[2]*padded_h), int(boxes[3]*padded_w), int(boxes[4]*padded_h)
            #     xmax += xmin
            #     ymax += ymin
            #     cls_id = int(boxes[0])
            #     color = colors[cls_id]
            #     cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            #     text_size = cv2.getTextSize(self.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            #     cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            #     cv2.putText(
            #         output_image, self.classes[cls_id],
            #         (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            #         (255, 255, 255), 1)
            #     # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax))    

            # cv2.imshow("CoCo1", output_image)
            # cv2.waitKey()
            return np.transpose(np.array(output_image, dtype=np.float32), (2, 0, 1)), torch.Tensor(np.array(objects, dtype=np.float32))

    def collate_fn(self, batch):
        items = list(zip(*batch))
        items[0] = default_collate(items[0])  
        items[0] = items[0]/255     # Normalize 
        for i, data in enumerate(items[1]):
            if data.shape[0] == 0: 
                continue
            data[:,0] = i
        items[1] = torch.cat(items[1], dim = 0)
        return None, items[0], items[1]
    
    def __len__(self):
        return len(self.img_files)   

    def videoTest(self):
        out_path = "videoTest/CoCo5KDataset.mp4"       
        # input_image_folder = os.path.join(self.root_path, "VOC{}".format(self.year), "JPEGImages")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(out_path, fourcc, 2, (self.img_size, self.img_size))
        colors = pickle.load(open("dataset//pallete", "rb"))
        with tqdm(total = len(self.label_files)) as pbar:
            for index, img_path in enumerate(self.img_files):
                # ---------
                #  Image
                # ---------
                # img_path = os.path.join(self.image_path, img_fn)
                img = cv2.imread(img_path.strip())
                h, w, _ = img.shape
                # ---------
                #  Label
                # COCO Bounding box: (label_idx, x_center, y_center, width, height)
                # ---------
                label_path = self.label_files[index].rstrip()
                if os.path.exists(label_path):
                    objects = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                    objects = torch.cat([ objects[:, 1:], objects[:, 0].unsqueeze(1) ], dim = 1)

                    # return size
                    objects[:, 0] *= w
                    objects[:, 1] *= h
                    objects[:, 2] *= w
                    objects[:, 3] *= h

                    # calc (x1, y1, w, h)
                    objects[:, 0] -=  0.5 * objects[:, 2]
                    objects[:, 1] -=  0.5 * objects[:, 3]

                    # calc (x1, y1, x2, y2)
                    objects[:, 2] +=  objects[:, 0]
                    objects[:, 3] +=  objects[:, 1]

                if self.is_training:
                    # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
                    transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
                else:
                    transformations = Compose([Resize(self.img_size)])
                output_image, objects = transformations((img, objects.tolist()))
                padded_h, padded_w, _ = output_image.shape
                for idx in range(len(objects)):
                    boxes = objects[idx][:-1]
                    # Calculate drawing coord (x1,y1), (x2,y2)
                    x1 =  boxes[0] 
                    y1 =  boxes[1] 
                    w =  boxes[2] 
                    h =  boxes[3] 

                    x2 = x1 + w
                    y2 = y1 + h

                    boxes = [x1, y1, x2, y2]    
                                    
                    # boxes -----------> mark annotation
                    cls_id = int(objects[idx][-1])
                    xmin, ymin, xmax, ymax = int(x1 ) , int(y1 ) , int(x2) , int(y2)
                    color = colors[cls_id]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)

                    # Text
                    text_size = cv2.getTextSize(self.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                    cv2.putText(
                        output_image, self.classes[cls_id],
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
                    # print("Object: {}, Bounding box: ({},{}) ({},{})".format(self.classes[cls_id], xmin, xmax, ymin, ymax))
                video.write(output_image)
                # cv2.imshow("CoCo5K", output_image)
                # cv2.waitKey()
                pbar.update(1)
                if idx >= 30: break
            video.release()
 
