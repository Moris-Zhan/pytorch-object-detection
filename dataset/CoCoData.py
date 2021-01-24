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


class CoCoDataset(Dataset):
    def __init__(self, img_size=416, is_training = True):
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()

        # self.label_files = [
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #     for path in self.img_files
        # ]
        mode = "train"
        year = "2017"
        root_path = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO"
        self.image_path = os.path.join(root_path, "images", "{}{}".format(mode, year))
        self.img_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO\\images\\{}{}\\*.jpg".format(mode, year))
        self.ann_file = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO\\annotations\\instances_{}{}.json".format(mode, year)

        # ---------
        #  Parse Json to Label
        # COCO Bounding box: (x-top left, y-top left, width, height)
        # ---------

        dataset = json.load(open(self.ann_file, 'r'))
        # Parse Json
        image_data = {}
        invalid_anno = 0

        for image in dataset["images"]:
            if image["id"] not in image_data.keys():
                image_data[image["id"]] = {"file_name": image["file_name"], "objects": []}

        for ann in dataset["annotations"]:
            if ann["image_id"] not in image_data.keys():
                invalid_anno += 1
                continue
            # COCO Bounding box: (x-min, y-min, x-max, y-max)
            image_data[ann["image_id"]]["objects"].append(
                [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][0] + ann["bbox"][2]),
                int(ann["bbox"][1] + ann["bbox"][3]), ann["category_id"]])

            # COCO Bounding box: (x-top left, y-top left, width, height)
            # image_data[ann["image_id"]]["objects"].append(
            #     [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][2]), int(ann["bbox"][3]), ann["category_id"]])

        self.label_files = image_data  

        self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]  

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

        self.img_size = img_size
        self.batch_count = 0
        self.is_training = is_training
        print('label_files', len(self.label_files))
        print('img_files',len(self.img_files))

    def __len__(self):
        return len(self.img_files)    

    def __getitem__(self, index):
        # if index in self.label_files:
        _id = list(self.label_files.keys())[index]
        img_fn = self.label_files[_id]["file_name"]        
        # ---------
        #  Image
        # ---------
        img_path = os.path.join(self.image_path, img_fn)
        img = cv2.imread(img_path)
        # ---------
        #  Label
        # COCO Bounding box: (x-top left, y-top left, width, height)
        # ---------
        image_dict = self.label_files[_id]       
        objects = copy.deepcopy(image_dict["objects"])
        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])     
                    
        if self.is_training:
            # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
            transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
        else:
            transformations = Compose([Resize(self.img_size)])
        output_image, objects = transformations((img, objects)) # train: [x1, y1, w, h]
        padded_h, padded_w, _ = output_image.shape
        for idx in range(len(objects)):
            boxes = objects[idx][:-1]
            # Calculate train: [x1, y1, w, h]
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
        # cv2.imshow("CoCo1", items[0][0])
        # cv2.waitKey()
        return None, items[0], items[1]

    def videoTest(self):
        out_path = "videoTest/CoCoDataset.mp4"       
        # input_image_folder = os.path.join(self.root_path, "VOC{}".format(self.year), "JPEGImages")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(out_path, fourcc, 2, (self.img_size, self.img_size))
        colors = pickle.load(open("dataset//pallete", "rb"))
        with tqdm(total = len(self.label_files.keys())) as pbar:
            for index, _ in enumerate(self.img_files):
                _id = list(self.label_files.keys())[index]
                img_fn = self.label_files[_id]["file_name"] 
                # ---------
                #  Image
                # ---------
                img_path = os.path.join(self.image_path, img_fn)
                img = cv2.imread(img_path)
                # ---------
                #  Label
                # COCO Bounding box: (x-top left, y-top left, width, height)
                # ---------
                image_dict = self.label_files[_id]       
                objects = copy.deepcopy(image_dict["objects"])
                for idx in range(len(objects)):
                    objects[idx][4] = self.class_ids.index(objects[idx][4])     
                          
                if self.is_training:
                    # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
                    transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
                else:
                    transformations = Compose([Resize(self.img_size)])
                output_image, objects = transformations((img, objects))
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
                    cls_id = objects[idx][-1]
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
                # cv2.imshow("CoCo", output_image)
                # cv2.waitKey()
                pbar.update(1)
                if idx >= 30: break
            video.release()
 