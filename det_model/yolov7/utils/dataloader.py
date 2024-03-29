from random import sample, shuffle, random


import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from det_model.yolov7.utils.utils import cvtColor, preprocess_input
from det_model.yolov7.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.albumentations = Albumentations(self.input_shape, self.train) 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        # if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
        #     lines = sample(self.annotation_lines, 3)
        #     lines.append(self.annotation_lines[index])
        #     shuffle(lines)
        #     image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
        #     if self.mixup and self.rand() < self.mixup_prob:
        #         lines           = sample(self.annotation_lines, 1)
        #         image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
        #         image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        # else:
        #     image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # lines = sample(self.annotation_lines, 3)
            # lines.append(self.annotation_lines[index])
            # shuffle(lines)
            image, box  = self.get_random_data_with_Mosaic(index) 
            
            if self.mixup and self.rand() < self.mixup_prob:
                # lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(index) 
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box      = self.get_random_data(index)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        
        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL          = len(box)
        labels_out  = np.zeros((nL, 6))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            
            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            #---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
            
        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, index):
        annotation_line = self.annotation_lines[index]
        image = annotation_line.split()[0]
        image = cv2.imread(image)
        new_anno     = np.array([np.array(list(map(int,b.split(','))),dtype=np.float32) for b in annotation_line.split()[1:]])     

        # orign view
        for idx in range(len(new_anno)):
            b = new_anno[idx, :-1]         
            # cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,255,0), 2) # 邊框
        # cv2.imshow("image", image)
        # cv2.waitKey(0)   
        
        # xywh trans view
        new_anno[:, :-1] = xyxy2xywhn(new_anno[:, :-1], w=image.shape[1], h=image.shape[0], clip=True, eps=1E-3) # target is [x,y,w,h]
        box3 = xywhn2xyxy(new_anno[:, :-1],w=image.shape[1], h=image.shape[0], padw=0, padh=0)
        h, w , _ = image.shape

        for idx in range(len(box3)):
            b = box3[idx, :]               
        #     cv2.rectangle(image, (b[0], b[1]), (  b[2], b[3]  ), (255,255,0), 2) # 邊框
        # cv2.imshow("image", image)
        # cv2.waitKey(0)        

        # Albumentations
        image, new_anno = self.albumentations(image, new_anno)
        new_anno[:, :-1] = xywhn2xyxy(new_anno[:, :-1],w=w, h=h, padw=0, padh=0)
        for idx in range(len(new_anno)):
            b = new_anno[idx, :-1]                  
        #     cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,255,0), 2) # 邊框
        # cv2.imshow("image", image)
        # cv2.waitKey(0) 
        return image, new_anno  

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, index):
        lines = sample(self.annotation_lines, 3)
        lines.append(self.annotation_lines[index])
        shuffle(lines)
        scale_range = (0.3, 0.7)
        output_size = self.input_shape

        output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
        scale_x = scale_range[0] + random() * (scale_range[1] - scale_range[0])
        scale_y = scale_range[0] + random() * (scale_range[1] - scale_range[0])
        divid_point_x = int(scale_x * output_size[1])
        divid_point_y = int(scale_y * output_size[0])

        new_anno = []
        for i, annotation_line in enumerate(lines):
            path = annotation_line.split(" ")[0]
            all_annos = annotation_line.split(" ")[1:]
            img_annos     = np.array([np.array(list(map(int,b.split(','))),dtype=np.float32) for b in all_annos])   

            img = cv2.imread(path)
            h, w, _ = img.shape

            if i == 0:  # top-left                
                mosaic_albumentations = Albumentations((divid_point_y, divid_point_x), True) 
                img_annos[:, :-1] = xyxy2xywhn(img_annos[:, :-1], w=w, h=h, clip=True, eps=1E-3) # target is [x,y,w,h]
                img, img_annos = mosaic_albumentations(img, img_annos)
                img_annos[:, :-1] = xywhn2xyxy(img_annos[:, :-1],w=w, h=h, padw=0, padh=0)

                output_img[:divid_point_y, :divid_point_x, :] = img
                for bbox in img_annos:         
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[2]
                    ymax = bbox[3]           
                    new_anno.append([xmin, ymin, xmax, ymax, bbox[4]])
                #     cv2.rectangle(output_img, (int(xmin), int(ymin)), ( int(xmax), int(ymax)  ), (255,255,0), 2) # 邊框
                # cv2.imshow("output_img", output_img)
                # cv2.waitKey(0)
            elif i == 1:  # top-right
                mosaic_albumentations = Albumentations((divid_point_y, output_size[1] - divid_point_x), True) 
                img_annos[:, :-1] = xyxy2xywhn(img_annos[:, :-1], w=w, h=h, clip=True, eps=1E-3) # target is [x,y,w,h]
                img, img_annos = mosaic_albumentations(img, img_annos)
                img_annos[:, :-1] = xywhn2xyxy(img_annos[:, :-1],w=w, h=h, padw=0, padh=0)

                output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
                for bbox in img_annos:            
                    xmin = bbox[0] + divid_point_x
                    ymin = bbox[1]
                    xmax = bbox[2] + divid_point_x
                    ymax = bbox[3]
                    new_anno.append([xmin, ymin, xmax, ymax, bbox[4]])
                #     cv2.rectangle(output_img, (int(xmin), int(ymin)), ( int(xmax), int(ymax)  ), (255,255,0), 2) # 邊框
                # cv2.imshow("output_img", output_img)
                # cv2.waitKey(0)
            elif i == 2:  # bottom-left
                mosaic_albumentations = Albumentations((output_size[0] - divid_point_y, divid_point_x), True) 
                img_annos[:, :-1] = xyxy2xywhn(img_annos[:, :-1], w=w, h=h, clip=True, eps=1E-3) # target is [x,y,w,h]
                img, img_annos = mosaic_albumentations(img, img_annos)
                img_annos[:, :-1] = xywhn2xyxy(img_annos[:, :-1],w=w, h=h, padw=0, padh=0)

                output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
                for bbox in img_annos:          
                    xmin = bbox[0]
                    ymin = bbox[1] + divid_point_y
                    xmax = bbox[2]
                    ymax = bbox[3] + divid_point_y
                    new_anno.append([xmin, ymin, xmax, ymax, bbox[4]])
                #     cv2.rectangle(output_img, (int(xmin), int(ymin)), ( int(xmax), int(ymax)  ), (255,255,0), 2) # 邊框
                # cv2.imshow("output_img", output_img)
                # cv2.waitKey(0)
            else:  # bottom-right
                mosaic_albumentations = Albumentations((output_size[0] - divid_point_y, output_size[1] - divid_point_x), True) 
                img_annos[:, :-1] = xyxy2xywhn(img_annos[:, :-1], w=w, h=h, clip=True, eps=1E-3) # target is [x,y,w,h]
                img, img_annos = mosaic_albumentations(img, img_annos)
                img_annos[:, :-1] = xywhn2xyxy(img_annos[:, :-1],w=w, h=h, padw=0, padh=0)

                # img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
                output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
                for bbox in img_annos:        
                    xmin = bbox[0] + divid_point_x
                    ymin = bbox[1] + divid_point_y
                    xmax = bbox[2] + divid_point_x
                    ymax = bbox[3] + divid_point_y

                    new_anno.append([xmin, ymin, xmax, ymax, bbox[4]])
                #     cv2.rectangle(output_img, (int(xmin), int(ymin)), ( int(xmax), int(ymax)  ), (255,255,0), 2) # 邊框
                # cv2.imshow("output_img", output_img)
                # cv2.waitKey(0)
        
        # filter_scale = 1 / 100 
        # if 0 < filter_scale:
        #     new_anno = [anno for anno in new_anno if
        #             filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

        new_anno = np.array(new_anno)
        return output_img, new_anno  

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes
