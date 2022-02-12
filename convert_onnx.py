import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

# from yolov4.yolo import YOLO as Model
# from yolov3.yolo import YOLO as Model
from ssd.ssd import SSD as Model
# from faster_rcnn.frcnn import FRCNN as Model
# from retinanet.retinanet import Retinanet as Model
# from centernet.centernet import CenterNet as Model

from helps.choose_model import ModelType, check_model
from helps.choose_data import DataType, get_data
import torch
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from torch.autograd import Variable
from glob import glob

if __name__ == "__main__":
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    #------------------------------#
    VOCdevkit_path, classes_path = get_data(root_path, DataType.LANE)
    modelType = check_model(Model.__module__)
    #-------------------------------#
    if modelType == ModelType.YOLOV4: 
        from yolov4.utils.utils import get_classes, get_anchors
        from yolov4.utils.utils_map import get_coco_map, get_map

    elif modelType == ModelType.YOLOV3: 
        from yolov3.utils.utils import get_classes
        from yolov3.utils.utils_map import get_coco_map, get_map

    elif modelType == ModelType.SSD: 
        from ssd.utils.utils import get_classes
        from ssd.utils.utils_map import get_coco_map, get_map

    elif modelType == ModelType.RETINANET: 
        from retinanet.utils.utils import get_classes
        from retinanet.utils.utils_map import get_coco_map, get_map

    elif modelType == ModelType.FASTER_RCNN: 
        from faster_rcnn.utils.utils import get_classes
        from faster_rcnn.utils.utils_map import get_coco_map, get_map

    elif modelType == ModelType.CENTERNET: 
        from centernet.utils.utils import get_classes
        from centernet.utils.utils_map import get_coco_map, get_map
    '''
    Recall和Precision不像AP是一個面積的概念，在門限值不同時，網絡的Recall和Precision值是不同的。
    map計算結果中的Recall和Precision代表的是當預測時，門限置信度為0.5時，所對應的Recall和Precision值。

    此處獲得的./map_out/detection-results/里面的txt的框的數量會比直接predict多一些，這是因為這里的門限低，
    目的是為了計算不同門限條件下的Recall和Precision值，從而實現map的計算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用於指定該文件運行時計算的內容
    #   map_mode為0代表整個map計算流程，包括獲得預測結果、獲得真實框、計算VOC_map。
    #   map_mode為1代表僅僅獲得預測結果。
    #   map_mode為2代表僅僅獲得真實框。
    #   map_mode為3代表僅僅計算VOC_map。
    #   map_mode為4代表利用COCO工具箱計算當前數據集的0.50:0.95map。需要獲得預測結果、獲得真實框後並安裝pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#    
    #   MINOVERLAP用於指定想要獲得的mAP0.x
    #   比如計算mAP0.75，可以設定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis用於指定是否開啟VOC_map計算的可視化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC數據集所在的文件夾
    #   默認指向根目錄下的VOC數據集
    #-------------------------------------------------------#
    # VOCdevkit_path  = os.path.join(root_path, "DataSet/LANEdevkit")
    #-------------------------------------------------------#
    #   結果輸出的文件夾，默認為map_out
    #-------------------------------------------------------#
    map_out_path    = os.path.join("map_out", Model.__module__)

    image_ids = glob(os.path.join(VOCdevkit_path, "test/bbox_annotations", "*.xml"))
    image_ids = [os.path.basename(abs_path).split(".")[0] for abs_path in image_ids]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, num_classes = get_classes(classes_path)
    onnx_model_path = "onnx_model/%s.onnx"%(str(Model.__module__).split(".")[0].lower())

    # Test image
    sample_image = os.path.join(VOCdevkit_path, "test/images/77.jpg") 
    image = cv2.imread(sample_image)
    image_shape = np.array(np.shape(image)[0:2])

    if modelType in [ModelType.YOLOV4, ModelType.YOLOV3]:
        print("Load model.")
        model = Model(confidence = 0.001, nms_iou = 0.5, classes_path = classes_path)
        print("Load model done.")   
                
        # generate model input
        generated_input = Variable(
            torch.randn(1, 3, 608, 608)
        ).cuda()   

        # model export into ONNX format
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output%d" % i for i in range(3) ]
        torch.onnx.export(
            model.net.module,
            generated_input,
            onnx_model_path,
            verbose=True,
            input_names=input_names, output_names=output_names,
            opset_version=10
        )

        # Load the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)
        # Print a human readable representation of the graph
        graph = onnx.helper.printable_graph(onnx_model.graph)
        onnx_graph_path = onnx_model_path.replace(".onnx", ".txt")
        with open(onnx_graph_path, "w", encoding="utf-8") as f:
            f.write(graph)


        # Test forward with onnx session (test image) 
        ort_session = ort.InferenceSession(onnx_model_path)        
        new_image       = cv2.resize(image, (608, 608), interpolation=cv2.INTER_CUBIC)
        new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)),0)

        # Test onnx predict
    
        outputs = ort_session.run(
            None,
            {"actual_input_1": new_image},
        )

        outputs = [torch.from_numpy(o) for o in outputs]
        outputs = model.bbox_util.decode_box(outputs)
        results = model.bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, [608, 608], 
                            image_shape, False, conf_thres = 0.5, nms_thres = 0.3, official=False)
                                                        
        if results[0] is not None: 
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            for i, c in list(enumerate(top_label)):
                predicted_class = class_names[int(c)]
                
                # flit classes
                if int(c) in [5,8,9,10]: continue

                box             = top_boxes[i]
                score           = top_conf[i]

                top, left, bottom, right = box

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image_shape[0], np.floor(bottom).astype('int32'))
                right   = min(image_shape[1], np.floor(right).astype('int32'))
                label = '{} {:.2f}'.format(predicted_class, score)  

                cv2.rectangle(image, (left, bottom), (right, top), (255,255,0), 2) # 邊框

                # cv2.rectangle(image, (left, top), (right, top-30), (0, 255, 0), -1)
                cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("image", image)
            cv2.waitKey(0)  
    elif modelType == ModelType.SSD:
        print("Load model.")
        model = Model(confidence = 0.001, nms_iou = 0.5, classes_path = classes_path)
        print("Load model done.")   

         # generate model input
        generated_input = Variable(
            torch.randn(1, 3, 300, 300)
        ).cuda()   

        # model export into ONNX format
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output" ]
        torch.onnx.export(
            model.net.module,
            generated_input,
            onnx_model_path,
            verbose=True,
            input_names=input_names, output_names=output_names,
            opset_version=10
        )

        # Load the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)
        # Print a human readable representation of the graph
        graph = onnx.helper.printable_graph(onnx_model.graph)
        onnx_graph_path = onnx_model_path.replace(".onnx", ".txt")
        with open(onnx_graph_path, "w", encoding="utf-8") as f:
            f.write(graph)

        # Test forward with onnx session (test image) 
        ort_session = ort.InferenceSession(onnx_model_path)        
        new_image       = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
        new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)),0)
        
        outputs = ort_session.run(
            None,
            {"actual_input_1": new_image},
        )