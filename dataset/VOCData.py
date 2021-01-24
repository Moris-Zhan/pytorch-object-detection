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
import xml.etree.ElementTree as ET
import pickle
from tqdm import tqdm

import scipy.misc

if not os.path.isdir("videoTest"):    
    os.makedirs("videoTest")
# ----------------------------------------------
class VOCDataset(Dataset):
    def __init__(self, img_size=416, batch_size = 2, is_training = True):
        year = "2012"
        mode = "trainval"
        root_path = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\VOC\\VOCdevkit"
        self.data_path = os.path.join(root_path, "VOC{}".format(year))  
        id_list_path = os.path.join(self.data_path, "ImageSets\\Main\\{}.txt".format(mode))
        self.ids = [id.strip() for id in open(id_list_path)]

        self.image_path = os.path.join(root_path, "images", "{}{}".format(mode, year))
        self.img_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\VOC\\VOCdevkit\\VOC{}\\JPEGImages\\*.jpg".format(year))
        self.ann_file = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\VOC\\VOCdevkit\\VOC{}\\JPEGImages\\Annotations\\*.xml".format(year)

        # ---------
        #  Parse Json to Label
        # Pascal VOC Bounding box :(x-top left, y-top left,x-bottom right, y-bottom right)
        # https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
        # ---------      
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']    

        self.img_size = img_size
        self.batch_count = 0
        self.is_training = is_training
        self.root_path = root_path
        self.year = year
        # self.id_list_path = id_list_path
        print('label_files', len(self.ids))
        print('img_files',len(self.ids))

    def __len__(self):
        return len(self.ids)    

    def __getitem__(self, index):
        _id = self.ids[index]
        # ---------
        #  Image
        # ---------
        img_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(_id))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        # ---------
        #  Label
        # Pascal VOC Bounding box :(x-top left, y-top left,x-bottom right, y-bottom right)
        # ---------
        image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(_id))
        annot = ET.parse(image_xml_path)
        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])          
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.img_size)])
        else:
            transformations = Compose([Resize(self.img_size)])
        image, objects = transformations((img, objects)) # train: [x1, y1, w, h]
        padded_w, padded_h, _ = image.shape

        for idx in range(len(objects)):
            boxes = objects[idx][:-1]
            # Calculate train: [x1, y1, w, h] from transform, then normalized scale(padded_h, padded_w)
            x1 =  boxes[0] / padded_w
            y1 =  boxes[1] / padded_h
            x2 =  boxes[2] / padded_w
            y2 =  boxes[3] / padded_h            

            boxes = [x1, y1, x2, y2]
            objects[idx] = [0, objects[idx][-1]] + boxes
        
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), torch.Tensor(np.array(objects, dtype=np.float32))

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

    def videoTest(self):
        out_path = "videoTest/VOCDataset.mp4"       
        input_image_folder = os.path.join(self.root_path, "VOC{}".format(self.year), "JPEGImages")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(out_path, fourcc, 2, (self.img_size, self.img_size))
        colors = pickle.load(open("dataset//pallete", "rb"))
        with tqdm(total = len(self.ids)) as pbar:
            for idx, id in enumerate(self.ids):
                image_path = os.path.join(input_image_folder, "{}.jpg".format(id))
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow("Voc", image)
                # cv2.waitKey()

                image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
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
                # cv2.imshow("Voc", output_image)
                # cv2.waitKey()
                pbar.update(1)
                if idx >= 30: break
        video.release()
