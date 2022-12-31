import os
import random
import xml.etree.ElementTree as ET

from glob import glob
import json
from tqdm import tqdm

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
classes_path        = 'model_data//bdd_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1 
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1 
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
     

def get_annotation(data_root):    
    random.seed(0)

    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = os.path.join(data_root, "bdd100k")

    save_base = os.path.join(data_root, "bdd100k", "Detection")
    if not os.path.exists(save_base):
        os.makedirs(save_base)

    VOCdevkit_sets  = [('train'), ('val')]
    # VOCdevkit_sets  = [('val')]
    classes, _      = get_classes(classes_path)
    
    print("Generate train.txt and val.txt for train.")
    for image_set in VOCdevkit_sets:   
        img_folders = os.listdir(os.path.join(VOCdevkit_path, "images", "track", image_set))    
        anno_groups = glob(os.path.join(VOCdevkit_path, "labels", "box_track_20", image_set, "*.json") ) 
           
        list_file = open('%s/Detection/%s.txt'%(VOCdevkit_path, image_set), 'w', encoding='utf-8')

        for folder in tqdm(img_folders):
            anno_group = os.path.join(VOCdevkit_path, "labels", "box_track_20", image_set, "%s.json"%(folder))
            f = open(anno_group, 'r')
            data = json.load(f)
            # Iterating through the json
            # list
            for item in data:
                name = item["name"]                
                img_path = os.path.join(VOCdevkit_path, "images", "track", image_set, folder, "%s"%(name))

                image_id = os.path.basename(img_path).split(".")[0]
                list_file.write('%s'%(os.path.abspath(img_path)))

                labels = item["labels"]
                # objects = []
                for label in labels:
                    objects = []
                    category = label["category"] # class
                    # ---------
                    #  Label
                    # one object instance per row is [class_label, x1, y1, x2, y2]
                    # ---------
                    box2d = label["box2d"]
                    xmin, xmax, ymin, ymax = int(box2d["x1"]), int(box2d["x2"]), int(box2d["y1"]), int(box2d["y2"])
                    # if category == 'other vehicle': continue
                    if category in ['pedestrian','other person']: category = "person"
                    # if category == 'bicycle': category = "bike"
                    # if category == 'motorcycle': category = "motor"                    
                    # if category == 'trailer': category = "truck"  
                    # objects.append([xmin, ymin, xmax, ymax])
                    if xmin >= 0 and ymin >= 0 and (xmax-xmin)>=0 and (ymax-ymin)>=0 :
                        objects = [xmin, ymin, xmax, ymax]
                        list_file.write(" " + ",".join(str(a) for a in objects) + ',' + str(classes.index(category)))
                list_file.write('\n')

        list_file.close()

        # remove empty annotation
        lines = []
        with open('%s/Detection/%s.txt'%(VOCdevkit_path, image_set)) as f:
            lines   = f.readlines()
            lines   = [line for line in lines if len(line.split()) > 1]

        with open('%s/Detection/%s.txt'%(VOCdevkit_path, image_set), 'w', encoding='utf-8') as f:
            f.writelines(lines)

    print("Generate train.txt and val.txt for train done.")

if __name__ == "__main__":
    data_root = '/home/leyan/DataSet/'
    get_annotation(data_root)    