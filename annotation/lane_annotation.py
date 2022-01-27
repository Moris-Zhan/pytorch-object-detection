import os
import random
import xml.etree.ElementTree as ET

from glob import glob

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

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
classes_path        = 'model_data//lane_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1 
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1 
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'D://WorkSpace//JupyterWorkSpace//DataSet//LANEdevkit'

save_base = os.path.join(VOCdevkit_path,"Detection")
if not os.path.exists(save_base):
    os.makedirs(save_base)

VOCdevkit_sets  = [('train'), ('test')]
classes, _      = get_classes(classes_path)

def convert_annotation(image_id, image_set, list_file):
    in_file = open(os.path.join(VOCdevkit_path, '%s/bbox_annotations/%s.xml'%(image_set, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    # SA(Straight Arrow)
    # LA(Left Arrow)
    # RA(Right Arrow)
    # SLA(Straight-Left Arrow)
    # SRA(Straight-Right Arrow)
    # DM(Diamond)
    # PC(Pedestrian Crossing)
    # JB(Junction Box)
    # SL(Slow)
    # BL(Bus Lane)
    # CL(Cycle Lane)

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    random.seed(0)
    
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate train.txt and val.txt for train.")
        for image_set in VOCdevkit_sets:            
            image_ids = glob(os.path.join(VOCdevkit_path, image_set, "images/*.jpg"))
            if image_set == "test":
                list_file = open('%s/Detection/%s.txt'%(VOCdevkit_path, "val"), 'w', encoding='utf-8')
            else:
                list_file = open('%s/Detection/%s.txt'%(VOCdevkit_path, image_set), 'w', encoding='utf-8')

            for image_id in image_ids:
                image_id = os.path.basename(image_id).split(".")[0]
                list_file.write('%s/%s/images/%s.jpg'%(os.path.abspath(VOCdevkit_path),image_set, image_id))
                convert_annotation(image_id, image_set, list_file)
                list_file.write('\n')
            list_file.close()

        print("Generate train.txt and val.txt for train done.")
