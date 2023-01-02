import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from det_model.retinanet.utils.utils import cvtColor, preprocess_input
from det_model.retinanet.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

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

# DataLoader中collate_fn使用
def retinanet_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

