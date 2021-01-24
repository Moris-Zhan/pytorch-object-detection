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

class ContainerDataset(Dataset):
    def __init__(self, img_size=416, is_training = True):
        self.img_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\MosquitoContainer\\train_cdc\\train_images\\*.jpg")
        self.label_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\MosquitoContainer\\train_cdc\\train_annotations\\*.xml")

        self.classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet',
                        'tub', 'washing_machine', 'water_tower'] 
        self.img_size = img_size
        self.batch_count = 0
        self.is_training = True
        print('label_files',len(self.label_files))
        print('img_files',len(self.img_files))
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        image_path = self.img_files[index % len(self.img_files)].rstrip()
        image = cv2.imread(image_path)

        # # ---------
        # #  Label
        # # ---------
        image_xml_path = self.label_files[index % len(self.img_files)]
        if os.path.exists(image_xml_path):
            annot = ET.parse(image_xml_path)
            objects = []
            for obj in annot.findall('object'):
                xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                        ["xmin", "xmax", "ymin", "ymax"]]
                label = self.classes.index(obj.find('name').text.lower().strip())
                objects.append([xmin, ymin, xmax, ymax, label])  
            if self.is_training:
                # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
                transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
            else:
                transformations = Compose([Resize(self.img_size)])
            output_image, objects = transformations((image, objects))  # train: [x1, y1, w, h]
            padded_h, padded_w, _ = output_image.shape    
            for idx in range(len(objects)):
                boxes = objects[idx][:-1]
                # Calculate drawing coord (x1,y1), (x2,y2)
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
        out_path = "videoTest/ContainerDataset.mp4"       
        # input_image_folder = os.path.join(self.root_path, "VOC{}".format(self.year), "JPEGImages")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(out_path, fourcc, 2, (self.img_size, self.img_size))
        colors = pickle.load(open("dataset//pallete", "rb"))
        with tqdm(total = len(self.img_files)) as pbar:
            for idx, image_path in enumerate(self.img_files):
                image = cv2.imread(image_path)

                image_xml_path = self.label_files[idx]
                if os.path.exists(image_xml_path):
                    annot = ET.parse(image_xml_path)
                    objects = []
                    for obj in annot.findall('object'):
                        xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                                ["xmin", "xmax", "ymin", "ymax"]]
                        label = self.classes.index(obj.find('name').text.lower().strip())
                        objects.append([xmin, ymin, xmax, ymax, label])  
                    if self.is_training:
                        # transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
                        transformations = Compose([VerticalFlip(), Crop(), Resize(self.img_size)])
                    else:
                        transformations = Compose([Resize(self.img_size)])
                    output_image, objects = transformations((image, objects))
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
                # cv2.imshow("Container", output_image)
                # cv2.waitKey()
                pbar.update(1)
                # if idx >= 30: break
            video.release()
 