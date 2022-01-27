import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from retinanet.utils.utils import cvtColor, preprocess_input
from retinanet.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

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


class RetinanetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(RetinanetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.albumentations = Albumentations(self.input_shape, self.train) 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        image, box = self.get_random_data(index)  

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)),(2,0,1))
        box         = np.array(box, dtype=np.float32)
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


    # def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    #     line    = annotation_line.split()
    #     #------------------------------#
    #     #   读取图像并转换成RGB图像
    #     #------------------------------#
    #     image   = Image.open(line[0])
    #     image   = cvtColor(image)
    #     #------------------------------#
    #     #   获得图像的高宽与目标高宽
    #     #------------------------------#
    #     iw, ih  = image.size
    #     h, w    = input_shape
    #     #------------------------------#
    #     #   获得预测框
    #     #------------------------------#
    #     box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    #     if not random:
    #         scale = min(w/iw, h/ih)
    #         nw = int(iw*scale)
    #         nh = int(ih*scale)
    #         dx = (w-nw)//2
    #         dy = (h-nh)//2

    #         #---------------------------------#
    #         #   将图像多余的部分加上灰条
    #         #---------------------------------#
    #         image       = image.resize((nw,nh), Image.BICUBIC)
    #         new_image   = Image.new('RGB', (w,h), (128,128,128))
    #         new_image.paste(image, (dx, dy))
    #         image_data  = np.array(new_image, np.float32)

    #         #---------------------------------#
    #         #   对真实框进行调整
    #         #---------------------------------#
    #         if len(box)>0:
    #             np.random.shuffle(box)
    #             box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
    #             box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
    #             box[:, 0:2][box[:, 0:2]<0] = 0
    #             box[:, 2][box[:, 2]>w] = w
    #             box[:, 3][box[:, 3]>h] = h
    #             box_w = box[:, 2] - box[:, 0]
    #             box_h = box[:, 3] - box[:, 1]
    #             box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

    #         return image_data, box
                
    #     #------------------------------------------#
    #     #   对图像进行缩放并且进行长和宽的扭曲
    #     #------------------------------------------#
    #     new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
    #     scale = self.rand(.25, 2)
    #     if new_ar < 1:
    #         nh = int(scale*h)
    #         nw = int(nh*new_ar)
    #     else:
    #         nw = int(scale*w)
    #         nh = int(nw/new_ar)
    #     image = image.resize((nw,nh), Image.BICUBIC)

    #     #------------------------------------------#
    #     #   将图像多余的部分加上灰条
    #     #------------------------------------------#
    #     dx = int(self.rand(0, w-nw))
    #     dy = int(self.rand(0, h-nh))
    #     new_image = Image.new('RGB', (w,h), (128,128,128))
    #     new_image.paste(image, (dx, dy))
    #     image = new_image

    #     #------------------------------------------#
    #     #   翻转图像
    #     #------------------------------------------#
    #     flip = self.rand()<.5
    #     if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    #     #------------------------------------------#
    #     #   色域扭曲
    #     #------------------------------------------#
    #     hue = self.rand(-hue, hue)
    #     sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
    #     val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
    #     x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    #     x[..., 0] += hue*360
    #     x[..., 0][x[..., 0]>1] -= 1
    #     x[..., 0][x[..., 0]<0] += 1
    #     x[..., 1] *= sat
    #     x[..., 2] *= val
    #     x[x[:,:, 0]>360, 0] = 360
    #     x[:, :, 1:][x[:, :, 1:]>1] = 1
    #     x[x<0] = 0
    #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

    #     #---------------------------------#
    #     #   对真实框进行调整
    #     #---------------------------------#
    #     if len(box)>0:
    #         np.random.shuffle(box)
    #         box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
    #         box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
    #         if flip: box[:, [0,2]] = w - box[:, [2,0]]
    #         box[:, 0:2][box[:, 0:2]<0] = 0
    #         box[:, 2][box[:, 2]>w] = w
    #         box[:, 3][box[:, 3]>h] = h
    #         box_w = box[:, 2] - box[:, 0]
    #         box_h = box[:, 3] - box[:, 1]
    #         box = box[np.logical_and(box_w>1, box_h>1)] 
        
    #     return image_data, box

# DataLoader中collate_fn使用
def retinanet_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

