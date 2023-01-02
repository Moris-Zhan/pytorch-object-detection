class ModelType:
    YOLOV5   = 0
    YOLOV4   = 1
    YOLOV3   = 2
    SSD      = 3
    RETINANET      = 4
    FASTER_RCNN    = 5
    CENTERNET      = 6
    YOLOX    = 7
    

def check_model(o):
    str__ = str(o).split(".")[1].lower()
    if "yolov5" in str__:  return ModelType.YOLOV5
    elif "yolov4" in str__:  return ModelType.YOLOV4
    elif "yolov3" in str__:  return ModelType.YOLOV3
    elif "ssd" in str__:  return ModelType.SSD
    elif "retinanet" in str__:  return ModelType.RETINANET
    elif "faster_rcnn" in str__:  return ModelType.FASTER_RCNN
    elif "centernet" in str__:  return ModelType.CENTERNET
    elif "yolox" in str__:  return ModelType.YOLOX