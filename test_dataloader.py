#-------------------------------------#
#       對數據集進行訓練
#-------------------------------------#
import numpy as np
from pytorch_lightning import os
import torch
from torch.utils.data import DataLoader

from yolov4.utils.dataloader import YoloDataset as Dataset , yolo_dataset_collate as dataset_collate
from yolov4.utils.utils import get_classes

from tqdm import tqdm

class DataType:
    VOC   = 0
    LANE   = 1
    BDD    = 2 
    COCO   = 3
    WIDERPERSON   = 4
    MosquitoContainer = 5
    AsianTraffic = 6

if __name__ == "__main__":
    # root_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//bdd100k'
    dataType = DataType.BDD

    if dataType == DataType.LANE:
        data_path = "D://WorkSpace//JupyterWorkSpace//DataSet//LANEdevkit"
        classes_path    = 'model_data/lane_classes.txt' 
    elif dataType == DataType.BDD:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//bdd100k'    
        classes_path    = 'model_data/bdd_classes.txt'    
    elif dataType == DataType.VOC:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//VOCdevkit'    
        classes_path    = 'model_data/voc_classes.txt'
    elif dataType == DataType.COCO:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//COCO'    
        classes_path    = 'model_data/coco_classes.txt'    
    elif dataType == DataType.WIDERPERSON:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//WiderPerson'    
        classes_path    = 'model_data/widerperson_classes.txt'   
    elif dataType == DataType.MosquitoContainer:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//MosquitoContainer'    
        classes_path    = 'model_data/MosquitoContainer_classes.txt'
    elif dataType == DataType.AsianTraffic:
        data_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//Asian-Traffic'    
        classes_path    = 'model_data/AsianTraffic_classes.txt'
    #-------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #-------------------------------#
    Cuda = True
    #--------------------------------------------------------#
    #   訓練前一定要修改classes_path，使其對應自己的數據集
    #--------------------------------------------------------#
    # classes_path    = 'model_data/lane_classes.txt'
    # classes_path    = 'model_data/bdd_classes.txt'       
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   獲得圖片路徑和標簽
    #----------------------------------------------------#
    train_annotation_path   = os.path.join(data_path, "Detection//train.txt")
    val_annotation_path   = os.path.join(data_path, "Detection//val.txt") 
    #----------------------------------------------------#
    #   獲取classes和anchor
    #----------------------------------------------------
    class_names, num_classes = get_classes(classes_path)   
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    batch_size  = 8
                    
    epoch_step      = num_train // batch_size

    input_shape     = [300, 300]

    # SSD
    train_dataset   = Dataset(train_lines, input_shape, num_classes, mosaic=False, train = True)
    val_dataset     = Dataset(val_lines, input_shape, num_classes, mosaic=False, train = False)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=dataset_collate)

    with tqdm(total=epoch_step,postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            images, targets = batch[0], batch[1]
            images  = torch.from_numpy(images).type(torch.FloatTensor)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            pbar.update(1)


    epoch_step      = num_val // batch_size

    with tqdm(total=epoch_step,postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            images, targets = batch[0], batch[1]
            images  = torch.from_numpy(images).type(torch.FloatTensor)
            targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            pbar.update(1)