class ModelType:
    YOLOV4   = 0
    YOLOV3   = 1
    SSD      = 2
    RETINANET      = 3
    FASTER_RCNN    = 4
    CENTERNET      = 5
    

def check_model(o):
    str__ = str(o).split(".")[0].lower()
    if "yolov4" in str__:  return ModelType.YOLOV4
    elif "yolov3" in str__:  return ModelType.YOLOV3
    elif "ssd" in str__:  return ModelType.SSD
    elif "retinanet" in str__:  return ModelType.RETINANET
    elif "faster_rcnn" in str__:  return ModelType.FASTER_RCNN
    elif "centernet" in str__:  return ModelType.CENTERNET