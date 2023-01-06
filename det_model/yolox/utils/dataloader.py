from random import sample, shuffle, random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from det_model.yolox.utils.utils import cvtColor, preprocess_input
from det_model.yolox.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

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
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

        self.albumentations = Albumentations(self.input_shape, self.train) 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        # if self.mosaic:
        #     if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
        #         image, box = self.get_random_data_with_Mosaic(index)              
        #     else:
        #         image, box = self.get_random_data(index)   
        # else:                
        #     image, box = self.get_random_data(index)  

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
        box[:, :-1] = xyxy2xywhn(box[:, :-1], w=self.input_shape[1], h=self.input_shape[0], clip=True, eps=1E-3) # target is [x,y,w,h]

        return image, box  

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
    
    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w      = grid_shapes[l]
            anchors         = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]
            
            batch_target = np.zeros_like(targets)
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            max_ratios           = np.max(ratios, axis = -1)
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
                        
        return y_true

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
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes