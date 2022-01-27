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
VOCdevkit_path  = 'D://WorkSpace/JupyterWorkSpace/DataSet/COCO'
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
classes_path        = 'model_data//coco_classes.txt'
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
# VOCdevkit_sets  = [('val')]
classes, _      = get_classes(classes_path)

def convert_annotation(image_id, image_set, list_file):
    in_file = open(os.path.join(VOCdevkit_path, '%s/bbox_annotations/%s.xml'%(image_set, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
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
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    random.seed(0)
    colors = pickle.load(open("annotation//pallete", "rb"))
    
    print("Generate train.txt and val.txt for train.")

    class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]  
    
    classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]    

    for year in [2014, 2017]:
        for image_set in VOCdevkit_sets:  
            image_dir = os.path.join(VOCdevkit_path, "images", "{}{}".format(image_set, year))
            img_files = glob("{}\\images\\{}{}\\*.jpg".format(VOCdevkit_path, image_set, year))
            ann_file = "{}\\annotations\\instances_{}{}.json".format(VOCdevkit_path, image_set, year)

            list_file = open('%s/%s/%s_%s.txt'%(VOCdevkit_path, "Detection", image_set, year), 'w', encoding='utf-8')

            # ---------
            #  Parse Json to Label
            # COCO Bounding box: (x-top left, y-top left, width, height)
            # ---------

            dataset = json.load(open(ann_file, 'r'))
            # Parse Json
            image_data = {}
            invalid_anno = 0

            for image in tqdm(dataset["images"]):
                if image["id"] not in image_data.keys():
                    image_data[image["id"]] = {"file_name": image["file_name"], "objects": []}

            for ann in tqdm(dataset["annotations"]):
                if ann["image_id"] not in image_data.keys():
                    invalid_anno += 1
                    continue

                # COCO Bounding box: (x-min, y-min, x-max, y-max)
                image_data[ann["image_id"]]["objects"].append(
                [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][0] + ann["bbox"][2]),
                int(ann["bbox"][1] + ann["bbox"][3]), ann["category_id"]])

            label_files = image_data  

            for name in tqdm(list(label_files.keys())):
                img_id = label_files[name]["file_name"]   

                # objects = copy.deepcopy(image_dict["objects"])                
                image_dict = label_files[name]   
                filename = image_dict["file_name"] 
                image_path = os.path.join(image_dir, filename)
                # img = cv2.imread(image_path)
                list_file.write(image_path)

                objects = image_dict["objects"]
                for idx in range(len(objects)):
                    cls_id = objects[idx][4] = class_ids.index(objects[idx][4])   
                    # objects[idx][2] = objects[idx][2] - objects[idx][0]
                    # objects[idx][3] = objects[idx][3] - objects[idx][1]
                    color = colors[cls_id]
                    xmin = objects[idx][0]
                    ymin = objects[idx][1]
                    xmax = objects[idx][2]
                    ymax = objects[idx][3]
                    box = [xmin, ymin, xmax, ymax, cls_id]
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2) # 邊框

                    text_size = cv2.getTextSize(classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    # cv2.putText(
                    #     img, classes[cls_id],
                    #     (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    #     (255, 255, 255), 1)
                    b = (int(xmin), int(ymin), int(xmax), int(ymax))
                    list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
                # cv2.imshow("CoCo2", img)
                # cv2.waitKey()
                list_file.write('\n')

            list_file.close()           
        print("Generate train.txt and val.txt for train done.")

    # merge to txt
    for image_set in VOCdevkit_sets:  
        txts = glob('%s/%s/%s_*.txt'%(VOCdevkit_path, "Detection", image_set))
        # Reading data from file1
        with open(txts[0]) as fp:
            data = fp.read()
        
        # Reading data from file2
        with open(txts[1]) as fp:
            data2 = fp.read()

        # Merging 2 files
        # To add the data of file2
        # from next line
        data += "\n"
        data += data2
        
        with open(os.path.join(VOCdevkit_path, "Detection", "%s.txt"%(image_set)), 'w') as fp:
            fp.write(data)