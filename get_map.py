import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

# # from det_model.yolox.yolo import YOLO as Model
# # from det_model.yolov5.yolo import YOLO as Model
# # from det_model.yolov4.yolo import YOLO as Model
# # from det_model.yolov3.yolo import YOLO as Model
# # from det_model.ssd.ssd import SSD as Model
# # from det_model.retinanet.retinanet import Retinanet as Model
# # from det_model.faster_rcnn.frcnn import FRCNN as Model
# from det_model.centernet.centernet import CenterNet as Model

# from utils.choose_model import ModelType, check_model
# from utils.choose_data import DataType, get_data


import argparse, os
import importlib

from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.yolov8_base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')

    parser.add_argument("--map_mode", type=int, default=0 , help="miou mode")
    parser.add_argument("--MINOVERLAP", type=float, default=0.5 , help="MINOVERLAP用於指定想要獲得的mAP0.x")
    parser.add_argument("--map_vis", type=bool, default=False , help="map_vis")
    parser.add_argument("--confidence", type=float, default=0.001 , help="confidence")
    parser.add_argument("--nms_iou", type=float, default=0.5 , help="nms_iou")


    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)


    VOCdevkit_path, num_classes, name_classes = opt.data_path, opt.class_names, opt.num_classes    
    get_map = importlib.import_module("det_model.%s.utils.utils_map"%opt.net).get_map
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = opt.map_mode
    #-------------------------------------------------------#    
    #   MINOVERLAP用於指定想要獲得的mAP0.x
    #   比如計算mAP0.75，可以設定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = opt.MINOVERLAP
    #-------------------------------------------------------#
    #   map_vis用於指定是否開啟VOC_map計算的可視化
    #-------------------------------------------------------#
    map_vis         = opt.map_vis
    map_out_path    = os.path.join(opt.out_path, "map_out") 

    image_ids       = opt.val_lines
    image_ids = [os.path.basename(abs_path).split(".")[0] for abs_path in image_ids]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names = opt.class_names

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        model = opt.Model_Pred(confidence = opt.confidence, nms_iou = opt.nms_iou, classes_path = opt.classes_path, 
            model_path=f"work_dirs/{opt.exp_name}_{opt.net}/last_epoch_weights.pth")
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "test/images/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            model.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "test/bbox_annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
