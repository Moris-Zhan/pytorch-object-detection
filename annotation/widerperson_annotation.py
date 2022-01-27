import os
import random
import xml.etree.ElementTree as ET

from glob import glob
import json
from tqdm import tqdm
import cv2
import pickle

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'D://WorkSpace/JupyterWorkSpace/DataSet/WiderPerson'
#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的train.txt、val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的train.txt、val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 2
#-------------------------------------------------------------------#
#   必须要修改，用于生成train.txt、val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
#-------------------------------------------------------------------#
classes_path        = 'model_data//widerperson_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1 
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1 
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9


save_base = os.path.join(VOCdevkit_path,"Detection")
if not os.path.exists(save_base):
    os.makedirs(save_base)

VOCdevkit_sets  = [('train'), ('val')]
classes, _      = get_classes(classes_path)

def convert_annotation(image_id, image_set, list_file):
    image_label_path = os.path.join(VOCdevkit_path, 'Annotations/%s.jpg.txt'%(image_id))

    objects = []
    with open(image_label_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line.split(" ")) == 1: continue
            label, xmin, ymin , xmax, ymax = [int(item) for item in line.split(" ")]
            b = (xmin, ymin , xmax, ymax)

            if (((xmax - xmin) > 0) and ((ymax - ymin)> 0)) :
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(label))
         
        
if __name__ == "__main__":
    random.seed(0)
    colors = pickle.load(open("annotation//pallete", "rb"))
    
    print("Generate train.txt and val.txt for train.")   

    for image_set in VOCdevkit_sets:  
        image_dir = os.path.join(VOCdevkit_path, "Images")       
        image_ids = open('%s/%s.txt'%(VOCdevkit_path, image_set), 'r', encoding='utf-8').readlines()
        list_file = open('%s/%s/%s.txt'%(VOCdevkit_path, "Detection", image_set), 'w', encoding='utf-8')

        for image_id in tqdm(image_ids):
            image_id = image_id.strip()
            img_path = os.path.join(image_dir, "%s.jpg"%image_id)
            list_file.write(img_path)
            convert_annotation(image_id, image_set, list_file)
            list_file.write("\n")  

        list_file.close()           
    print("Generate train.txt and val.txt for train done.")
