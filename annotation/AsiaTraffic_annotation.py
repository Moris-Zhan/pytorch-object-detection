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
VOCdevkit_path  = 'D://WorkSpace/JupyterWorkSpace/DataSet/Asian-Traffic'
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
classes_path        = 'model_data//AsianTraffic_classes.txt'
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

def convert_annotation(image_id, list_file):
    image_label_path = os.path.join(VOCdevkit_path, 'Stage1//ivslab_train//Annotations//All//%s.xml'%(image_id))

    tree=ET.parse(image_label_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin, ymin , xmax, ymax = int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))
        b = (xmin, ymin , xmax, ymax)
        if (((xmax - xmin) > 0) and ((ymax - ymin)> 0)) :
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
         
        
if __name__ == "__main__":
    random.seed(0)
    colors = pickle.load(open("annotation//pallete", "rb"))
    
    print("Generate train.txt and val.txt for train.")   

    root_path = os.path.join(VOCdevkit_path,"Stage1\\ivslab_train")
    image_path = os.path.join(root_path, "JPEGImages")  
    data_path = os.path.join(root_path, "Annotations")  
    id_list_path = os.path.join(root_path, "ImageSets\\All.txt")
    ids = [id.strip() for id in open(id_list_path)]

    detfilepath = glob(os.path.join(root_path, "JPEGImages\\All\\*\\*.jpg"))
   
    num     = len(detfilepath)  
    list    = range(num) 
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr) 

    print("train and val size",tv)
    print("train size",tr)
    print("val size",tv-tr)

    # if not os.path.exists(os.path.join(VOCdevkit_path, "Detection")): os.makedirs(os.path.join(VOCdevkit_path, "Detection"))
    ftrain      = open(os.path.join(VOCdevkit_path, "Detection",'train.txt'), 'w')  
    fval        = open(os.path.join(VOCdevkit_path, "Detection",'val.txt'), 'w')

    for i in tqdm(list):  
        img=detfilepath[i][:-4] 
        # image_id = os.path.basename(img)
        image_id = "//".join(img.split("\\")[-2:])
        if i in trainval:  
            if i in train:                 
                ftrain.write(img + ".jpg")                 
                convert_annotation(image_id, ftrain)
                ftrain.write("\n") 
            else:                  
                fval.write(img + ".jpg") 
                convert_annotation(image_id, fval)
                fval.write("\n")   


    ftrain.close()           
    fval.close()  
    print("Generate train.txt and val.txt for train done.")

    for image_set in ["train", "val"]:
        # if not os.path.exists(os.path.join(VOCdevkit_path, "Detection")): os.makedirs(os.path.join(VOCdevkit_path, "Detection"))
        data = open(os.path.join(VOCdevkit_path, "Detection", "%s.txt"%(image_set)), 'r').read()

        with open(os.path.join(VOCdevkit_path, "Detection", "%s.txt"%(image_set)), 'w') as fp:
            lines = data.split("\n")
            lines = [fp.write(line + "\n") for line in lines if len(line.split()) > 1]
            print()
