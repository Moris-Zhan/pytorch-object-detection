import argparse
import sys
import time
import warnings
import colorsys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import datetime
import logging

import models
import importlib
import onnxruntime as ort
import numpy as np
import cv2
from glob import glob
import os
from PIL import ImageDraw, ImageFont, Image

def select_device(net, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'{net.upper()} 🚀 torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')

class Post:
    def __init__(self, opt):
        super(Post, self).__init__()
        self.opt = opt
        if opt.net == "yolox":
            from det_model.yolox.utils.utils_bbox import decode_outputs, non_max_suppression
            self.decode_outputs = decode_outputs
            self.non_max_suppression = non_max_suppression
        elif opt.net == "yolov5":
            from det_model.yolov5.utils.utils_bbox import DecodeBox
            self.bbox_util = DecodeBox(opt.anchors, opt.num_classes, opt.input_shape, opt.anchors_mask)
        elif opt.net == "yolov4":
            from det_model.yolov4.utils.utils_bbox import DecodeBox
            self.bbox_util = DecodeBox(opt.anchors, opt.num_classes, opt.input_shape, opt.anchors_mask)

        elif opt.net == "yolov3":
            from det_model.yolov3.utils.utils_bbox import DecodeBox
            self.bbox_util = DecodeBox(opt.anchors, opt.num_classes, opt.input_shape, opt.anchors_mask)
        elif opt.net == "faster_rcnn":
            from det_model.faster_rcnn.utils.utils_bbox import DecodeBox
            self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(opt.num_classes + 1)[None]
            self.bbox_util  = DecodeBox(self.std, opt.num_classes)

        elif opt.net == "centernet":
            from det_model.centernet.utils.utils_bbox import decode_bbox, postprocess
            self.decode_bbox = decode_bbox
            self.postprocess = postprocess
        elif opt.net == "retinanet":
            from det_model.retinanet.utils.utils_bbox import decodebox, non_max_suppression
            self.decodebox = decodebox
            self.non_max_suppression = non_max_suppression
        elif opt.net == "ssd":
            from det_model.ssd.utils.utils_bbox import BBoxUtility
            self.bbox_util = BBoxUtility(opt.num_classes)

    def process(self, outputs):
        if opt.net == "yolox":
            #---------------------------------------------------------------------#
            #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
            #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
            #---------------------------------------------------------------------#
            letterbox_image = True
            outputs = [torch.from_numpy(o) for o in outputs]
            outputs = self.decode_outputs(outputs, self.opt.input_shape)
            results = self.non_max_suppression(outputs, self.opt.num_classes, self.opt.input_shape, 
                        image_shape, letterbox_image, 
                        conf_thres = self.opt.conf_thres, nms_thres = self.opt.iou_thres)
            
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        elif opt.net == "yolov5":
            outputs = [torch.from_numpy(o) for o in outputs]
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), opt.num_classes, self.opt.input_shape, 
                                image_shape, False, 
                                conf_thres = self.opt.conf_thres, nms_thres = self.opt.iou_thres)
            
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        elif opt.net == "yolov4":
            outputs = [torch.from_numpy(o) for o in outputs]
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), opt.num_classes, self.opt.input_shape, 
                                image_shape, False, 
                                conf_thres = self.opt.conf_thres, nms_thres = self.opt.iou_thres)
            
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        elif opt.net == "yolov3":
            outputs = [torch.from_numpy(o) for o in outputs]
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), opt.num_classes, self.opt.input_shape, 
                                image_shape, False, 
                                conf_thres = self.opt.conf_thres, nms_thres = self.opt.iou_thres)
            
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        elif opt.net == "faster_rcnn":
            outputs = [torch.from_numpy(o) for o in outputs]
            # roi_cls_locs, roi_scores, rois, _ = outputs
            roi_cls_locs, roi_scores, rois = outputs[:4]
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, self.opt.input_shape, self.opt.input_shape, 
                                                    nms_iou = self.opt.iou_thres, confidence = self.opt.conf_thres)
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        elif opt.net == "centernet":
            letterbox_image = True
            nms = True
            outputs = [torch.from_numpy(o) for o in outputs]
            outputs = self.decode_bbox(outputs[0], outputs[1], outputs[2], self.opt.conf_thres, False)
            results = self.postprocess(outputs, nms, image_shape, self.opt.input_shape, letterbox_image, self.opt.iou_thres)

            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]            

        elif opt.net == "retinanet":
            letterbox_image = True
            # _, regression, classification, anchors = outputs
            outputs = [torch.from_numpy(o) for o in outputs]
            regression, classification, anchors = outputs[-3:]
            outputs     = self.decodebox(regression, anchors, self.opt.input_shape)
            results     = self.non_max_suppression(torch.cat([outputs, classification], axis=-1), self.opt.input_shape, 
                                    image_shape, letterbox_image, conf_thres = self.opt.conf_thres, nms_thres = self.opt.iou_thres)
               
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        elif opt.net == "ssd":
            letterbox_image = True
            outputs = [torch.from_numpy(o) for o in outputs]
            results     = self.bbox_util.decode_box(outputs, self.opt.anchors, image_shape, self.opt.input_shape, letterbox_image, 
                                                    nms_iou = self.opt.iou_thres, confidence = self.opt.conf_thres)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if not type(results[0]) == np.ndarray: 
                print("Not detected!!!")
                return None, _, _

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]

        return top_label, top_conf, top_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # error
    parser.add_argument('--config', type=str, default="configs.yolov5_base" ,help = 'Path to config .opt file. ')
    # parser.add_argument('--config', type=str, default="configs.fasterRcnn_base" ,help = 'Path to config .opt file. ')
    
    
    parser.add_argument('--weights', type=str, default='best_epoch_weights.pth', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', default=True, action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', default=True, action='store_true', help='simplify onnx model')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')

    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    conf = importlib.import_module(opt.config).get_opts(Train=False)
    for key, value in vars(conf).items():
        setattr(opt, key, value)
    opt.weights = os.path.join(opt.out_path, opt.weights)

    # print(opt)
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.net, opt.device)

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.input_shape).to(device)  # image size(1,3,320,192) iDetection

    print("Load model.")
    model, _  = models.get_model(opt, pred=True)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    print("Load model done.") 

    y = model(img)  # dry run

    if True:
        # ONNX export
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pth', '.onnx')  # filename
            model.eval()
            output_names = ['classes', 'boxes'] if y is None else ['output']

            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                'output': {0: 'batch', 2: 'y', 3: 'x'}}            

            input_names = ['images']
            # torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=input_names,
            torch.onnx.export(model, img, f, verbose=False, opset_version=14, input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            graph = onnx.helper.printable_graph(onnx_model.graph)
            # print(graph)  # print a human readable model         
            # onnx_graph_path = opt.weights.replace(".pth", ".txt")
            # with open(onnx_graph_path, "w", encoding="utf-8") as f:
            #     f.write(graph)
            

            if opt.simplify:
                try:
                    import onnxsim

                    print('\nStarting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    print(f'Simplifier failure: {e}')

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            f = opt.weights.replace('.pth', '_simp.onnx')  # filename
            onnx.save(onnx_model, f)
            print('ONNX export success, saved as %s' % f)


        except Exception as e:
            print('ONNX export failure: %s' % e)
            exit(-1)

        # Finish
        print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # from det_model.yolov5.utils.utils_bbox import DecodeBox
    # f = os.path.join(opt.out_path, "best_epoch_weights_simp.onnx")
    f = os.path.join(opt.out_path, "best_epoch_weights.onnx")
    ort_session = ort.InferenceSession(f)  

    # Test forward with onnx session (test image) 
    video_path      = os.path.join(opt.data_path, "Drive-View-Kaohsiung-Taiwan.mp4")
    capture = cv2.VideoCapture(video_path)

    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ref, frame = capture.read()

    fps = 0.0
    drawline = False
    post = Post(opt)

    # for frame in glob("test_images/*.jpg") :
    #     frame = cv2.imread(frame)

    while(True):
        t1 = time.time()
        # 讀取某一幀
        ref, frame = capture.read()
        if not ref:
            break
        t1 = time.time()
        image_shape = np.array(np.shape(frame)[0:2])              
        new_image       = cv2.resize(frame, opt.input_shape, interpolation=cv2.INTER_CUBIC)
        new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)),0)

        outputs = ort_session.run(
            None, 
            {"images": new_image
             },
        )
       
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / opt.num_classes, 1., 1.) for x in range(opt.num_classes)]
        opt.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        opt.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), opt.colors))

        top_label, top_conf, top_boxes = post.process(outputs)
        if type(top_label) != np.ndarray: 
            print("Not detected!!!")
            # continue 
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * frame.shape[1] + 0.5).astype('int32'))
        thickness   = int(max((frame.shape[0] + frame.shape[1]) // np.mean(opt.input_shape), 1))       
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        frame = Image.fromarray(frame)
        h, w = frame.size[:2]

        if type(top_label) == np.ndarray:
            for i, c in list(enumerate(top_label)):
                predicted_class = opt.class_names[int(c)]
                box             = top_boxes[i]
                score           = top_conf[i]

                top, left, bottom, right = box

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                
                bottom  = min(h, np.floor(bottom).astype('int32'))
                right   = min(w, np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(frame)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=opt.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=opt.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            
        frame = np.array(frame)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c= cv2.waitKey(1) & 0xff 
        if c==27:
            capture.release()
            break