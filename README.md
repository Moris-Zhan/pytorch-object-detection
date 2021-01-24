# ObjectDetection

## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`COCO`, `VOC`, `water containers` and `Asia Traffic`** Dataset.
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 4 different datases: **`VOC`, `COCO`, `MosquitoContainer` and `Asian-Traffic`** . Statistics of datasets I used for experiments is shown below

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

- **MosquitoContainer**:
This challenge was provided by the Center for Disease Control (CDC) of the Ministry of Health and Welfare of the Republic of China. The following table is a description table of the object category and corresponding ID.

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

## Methods
- Yolo V1
- Yolo V2
- Yolo V3
- Yolo V4

## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**

## Usage
### 0. Prepare the dataset
* **Download custom dataset in the  `data_paths`.** 
* **And create custom dataset `custom_dataset.py` in the `dataset`.**
* **Test dataset mark to video (need to change `custom_dataset` in code)**
```python
python videoTest.py 
```

### 1. Train
```python
python train.py --model Yolo_v4 --batch_size 4 --n_gpu 1
```

### 2. Evaluate + Predict
```python
python predict.py --model Yolo_v4  --experiment_dir "run\AsiaTrafficDataset\Yolo_v4\experiment_10"
```

## Reference
- PyTorch-YOLOv1 : https://github.com/ssaru/You_Only_Look_Once/blob/master/yolov1.py
- PyTorch-YOLOv2 : https://github.com/uvipen/Yolo-v2-pytorch/blob/9589413b5dce0476eb9cccc41945cf30cf131b34/src/yolo_net.py
- tensorflow-yolov3 : https://github.com/Leyan529/tensorflow-yolov3
- PyTorch-YOLOv3 : https://github.com/eavise-kul/lightnet/blob/b54f771a597d23863979d9274eb2ba90c05938e4/lightnet/models/_network_yolo_v3.py
- PyTorch_YOLOv4 : https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/4ccef0ec8fe984e059378813e33b3740929e0c19/models.py
- SSD: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/model.py#L323
