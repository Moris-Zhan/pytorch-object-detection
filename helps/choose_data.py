import os

class DataType:
    VOC   = 0
    LANE   = 1
    BDD    = 2 
    COCO   = 3
    WIDERPERSON   = 4
    MosquitoContainer = 5
    AsianTraffic = 6

def get_data(root_path, dataType):
    #------------------------------#  
    #   數據集路徑
    #   訓練自己的數據集必須要修改的
    #------------------------------#  
    if dataType == DataType.LANE:
        data_path = os.path.join(root_path, "LANEdevkit")
        classes_path    = 'model_data/lane_classes.txt' 
        map_dict = {
            "SA":"Straight Arrow",
            "LA":"Left Arrow",
            "RA":"Right Arrow",
            "SLA":"Straight-Left Arrow",
            "SRA":"Straight-Right Arrow",
            "DM":"Diamond",
            "PC":"Pedestrian Crossing",
            "JB":"Junction Box",
            "SL":"Slow",
            "BL":"Bus Lane",
            "CL":"Cycle Lane"
        }
    elif dataType == DataType.BDD:
        data_path = os.path.join(root_path, 'bdd100k')    
        classes_path    = 'model_data/bdd_classes.txt'    
    elif dataType == DataType.VOC:
        data_path = os.path.join(root_path, 'VOCdevkit')    
        classes_path    = 'model_data/voc_classes.txt'
    elif dataType == DataType.COCO:
        data_path = os.path.join(root_path, 'COCO')    
        classes_path    = 'model_data/coco_classes.txt'    
    elif dataType == DataType.WIDERPERSON:
        data_path = os.path.join(root_path, 'WiderPerson')    
        classes_path    = 'model_data/widerperson_classes.txt'   
    elif dataType == DataType.MosquitoContainer:
        data_path = os.path.join(root_path, 'MosquitoContainer')    
        classes_path    = 'model_data/MosquitoContainer_classes.txt'
    elif dataType == DataType.AsianTraffic:
        data_path = os.path.join(root_path, 'Asian-Traffic')    
        classes_path    = 'model_data/AsianTraffic_classes.txt'

    return data_path, classes_path