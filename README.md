## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`COCO`, `VOC`, `water containers` , `lane` and `Asia Traffic`** Dataset.
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 5 different datases: **`VOC`, `COCO`, `MosquitoContainer`, `LANEdevkit` and `Asian-Traffic`** . Statistics of datasets I used for experiments is shown below

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:
  
  | Dataset                | Classes | #Train images/objects | #Validation images/objects |
  |------------------------|:---------:|:-----------------------:|:----------------------------:|
  | VOC2007                |    20   |      5011/12608       |           4952/-           |
  | VOC2012                |    20   |      5717/13609       |           5823/13841       |

  ```
  VOCDevkit
  ├── VOC2007
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   └── ...
  └── VOC2012
      ├── Annotations  
      ├── ImageSets
      ├── JPEGImages
      └── ...
  ```
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhk51a2a6DLg_Hgub?e=PhUN2s)
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:

  | Dataset                | Classes | #Train images/objects | #Validation images/objects |
  |------------------------|:---------:|:-----------------------:|:----------------------------:|
  | COCO2014               |    80   |         83k/-         |            41k/-           |
  | COCO2017               |    80   |         118k/-        |             5k/-           |
  ```
    COCO
    ├── annotations
    │   ├── instances_train2014.json
    │   ├── instances_train2017.json
    │   ├── instances_val2014.json
    │   └── instances_val2017.json
    │── images
    │   ├── train2014
    │   ├── train2017
    │   ├── val2014
    │   └── val2017
    └── anno_pickle
        ├── COCO_train2014.pkl
        ├── COCO_val2014.pkl
        ├── COCO_train2017.pkl
        └── COCO_val2017.pkl
  ```
  Processed File: [download link](https://1drv.ms/f/s!AvbkzP-JBXPAhlDiyVy9wsDGCCj8?e=nN58aZ)

- **MosquitoContainer**:
This challenge was provided by the Center for Disease Control (CDC) of the Ministry of Health and Welfare of the Republic of China. The following table is a description table of the object category and corresponding ID.
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhk0aKiCel_XGdMPi?e=xN4jLD)

  |  ID    |   Category       | 
  | ------ | -----------------| 
  |   1    | aquarium         | 
  |   2    | bottle           | 
  |   3    |   bowl           | 
  |   4    |   box            | 
  |   5    |   bucket         | 
  |   6    | plastic_bag      | 
  |   7    |   plate          | 
  |   8    |   styrofoam      | 
  |   9    |   tire           | 
  |   10   |   toilet         | 
  |   11   |   tub            | 
  |   12   | washing_machine  | 
  |   13   |   water_tower    | 


  Download the container images and annotations from [MosquitoContainer contest](https://aidea-web.tw/topic/47b8aaa7-f0fc-4fee-af28-e0f077b856ae?focus=intro). Make sure to put the files as the following structure:

  ```
    MosquitoContainer
    ├── train_cdc
    │   ├── train_annotations
    │   ├── train_images
    │   ├── train_labels  
    │     
    │── test_pub_cdc
    │   ├── test_pub_images
    │   
    └── test_cdc
        ├── test_images
  ```

- **Asian-Traffic**:
Object detection in the field of computer vision has been extensively studied, and the use of deep learning methods has made great progress in recent years.
In addition, existing open data sets for object detection in ADAS applications usually include pedestrians, vehicles, cyclists, and motorcyclists in Western countries, which is different from Taiwan and other crowded Asian countries (speeding on urban roads).

  Download the container images and annotations from [Asian Countries contest](https://aidea-web.tw/topic/35e0ddb9-d54b-40b7-b445-67d627890454?focus=intro&fbclid=IwAR3oSJ8ESSTjPmf0nyJtggacp0zjEf77E_H4JC_qMtPPx8xrG4ips9qp6tE). Make sure to put the files as the following structure:
  
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhjFOQXTUrYGpS39Z?e=c47FGc)
  ```
    Asian-Traffic(Stage1)
    ├── ivslab_train
    │   ├── Annotations
    │   ├── ImageSets
    │   ├── JPEGImages  
    │     
    │── ivslab_test_public
       ├── JPEGImages
  ```
  |  ID    |   Category       | 
  | ------ | -----------------| 
  |   1    |   pedestrian     | 
  |   2    |   vehicle        | 
  |   3    |   scooter        | 
  |   4    |   bicycle'       | 


- **Wider-Person**: Wider Person dataset is a public dataset for object detection, which includes over 40,000 human images captured in various scenes with diverse poses, viewpoints, and backgrounds. Each image in this dataset is annotated with bounding boxes that indicate the position and size of each person. Moreover, the dataset provides body part labels such as head, arm, leg, etc., and action labels such as walking, running, standing, etc. The Wider Person dataset can be used for human pose estimation, action recognition, behavior analysis, and other related research.

  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhXLOUZWx0BZogoXF?e=RfZp1m)

  |  ID    |           Category            | 
  | ------ | ------------------------------| 
  |   1    |   pedestrians                 | 
  |   2    |   riders                      | 
  |   3    |   partially-visible persons   | 
  |   4    |   ignore regions              | 

## Methods
- YOLOv3
- YOLOv4
- YOLOv5
- YOLOv7
- YOLOv8
- YOLOX
- SSD
- RetinaNet
- CenterNet
- FasterRCNN
#### Pretrain-Weights [download link](https://1drv.ms/f/s!AvbkzP-JBXPAhgInYgN0VAb0Qcfy?e=ewrLAj)

## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**

## Requirenents

```python
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
### 1. Prepare the dataset
* **Create your own `annotation.{dataset}.py` then create `Detection/train.txt` , `Detection/val.txt` and `model_data/dataset_classes.txt` let data to load.** 
* **Prepare pretrain download weight to `model_data/weight` .** 
* **Add new data in `helps/choose_data.py`.**

### 2. Create own model
* **Copy `det_model` directory and write self required function, like `dataset_collate, Dataset, freeze_backbone, unfreeze_backbone` ...etc.** 
* **Maintaion self directory like `nets, utils`.** 
* **Maintaion self detection configuration file like `model.py`.** 
* **Add new data in `helps/choose_model.py`.**

### 3. Train (Freeze backbone + UnFreeze backbone) 
* setup your `root_path` , choose `DataType` and switch detection model library import.
```python
python train.py
```

* setup distribute training
```python
python -m torch.distributed.launch --nproc_per_node=8 train.py
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 4. Evaluate  (get_map) 
* setup your `root_path` , choose `DataType` and switch detection model library import.
* setup your `model_path` and `classes_path` in `model/model.py`
```python
python get_map.py
```

### 5. predict
* Can switch **`predict mode` to detection image** or **`viedo` mode to detection video**
* setup your `model_path` and `classes_path` in `model/model.py`
```python
python predict.py
```

### 6. export
* Can switch your saved model export to ONNX format
```python
python export.py --config "configs.yolact_base"
```
## Demo
![40c](https://user-images.githubusercontent.com/24097516/213660395-343337a6-dfeb-4199-971e-a7ca0c22b5d6.jpg)

## Reference
- PyTorch-YOLOv3 : https://github.com/bubbliiiing/yolo3-pytorch
- PyTorch_YOLOv4 : https://github.com/bubbliiiing/yolov4-pytorch
- PyTorch_YOLOv5 : https://github.com/bubbliiiing/yolov5-pytorch
- PyTorch_YOLOv7 : https://github.com/bubbliiiing/yolov7-pytorch
- PyTorch_YOLOv8 : https://github.com/bubbliiiing/yolov8-pytorch
- PyTorch_YOLOX : https://github.com/bubbliiiing/yolox-pytorch
- SSD: https://github.com/bubbliiiing/ssd-pytorch
- RetinaNet: https://github.com/bubbliiiing/retinanet-pytorch
- CenterNet: https://github.com/bubbliiiing/centernet-pytorch
- FasterRCNN: https://github.com/bubbliiiing/faster-rcnn-pytorch
- Onnx convert reference: https://github.com/WongKinYiu/yolov7/blob/main/export.py

