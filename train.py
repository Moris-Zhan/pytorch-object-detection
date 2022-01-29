#-------------------------------------#
#       對數據集進行訓練
#-------------------------------------#
import numpy as np
from pytorch_lightning import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from yolov4.nets.yolo import YoloBody as Model
from yolov3.nets.yolo import YoloBody as Model
from ssd.nets.ssd import SSD300  as Model
from retinanet.nets.retinanet import retinanet as Model
from faster_rcnn.nets.frcnn import FasterRCNN as Model
from centernet.nets.centernet import CenterNet_Resnet50 as Model

from helps.choose_data import DataType, get_data
from helps.choose_model import ModelType, check_model


'''
訓練自己的目標檢測模型一定需要注意以下幾點：
1、訓練前仔細檢查自己的格式是否滿足要求，該庫要求數據集格式為VOC格式，需要準備好的內容有輸入圖片和標簽
   輸入圖片為.jpg圖片，無需固定大小，傳入訓練前會自動進行resize。
   灰度圖會自動轉成RGB圖片進行訓練，無需自己修改。
   輸入圖片如果後綴非jpg，需要自己批量轉成jpg後再開始訓練。

   標簽為.xml格式，文件中會有需要檢測的目標信息，標簽文件和輸入圖片文件相對應。

2、訓練好的權值文件保存在logs文件夾中，每個epoch都會保存一次，如果只是訓練了幾個step是不會保存的，epoch和step的概念要捋清楚一下。
   在訓練過程中，該代碼並沒有設定只保存最低損失的，因此按默認參數訓練完會有100個權值，如果空間不夠可以自行刪除。
   這個並不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一點，為了滿足大多數的需求，還是都保存可選擇性高。

3、損失值的大小用於判斷是否收斂，比較重要的是有收斂的趨勢，即驗證集損失不斷下降，如果驗證集損失基本上不改變的話，模型基本上就收斂了。
   損失值的具體大小並沒有什麽意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數里面除上10000。
   訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中

4、調參是一門蠻重要的學問，沒有什麽參數是一定好的，現有的參數是我測試過可以正常訓練的參數，因此我會建議用現有的參數。
   但是參數本身並不是絕對的，比如隨著batch的增大學習率也可以增大，效果也會好一些；過深的網絡不要用太大的學習率等等。
   這些都是經驗上，只能靠各位同學多查詢資料和自己試試了。
'''  
if __name__ == "__main__":   
    #--------------------------------------------------------#
    #   訓練前一定要修改classes_path，使其對應自己的數據集
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    data_path, classes_path = get_data(root_path, DataType.LANE)

    modelType = check_model(Model.__module__)
    #-------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #-------------------------------#
    Cuda = True
    #-------------------------------#      
    if modelType == ModelType.YOLOV4: 
        from yolov4.nets.yolo_training import YOLOLoss, weights_init
        from yolov4.utils.callbacks import LossHistory
        from yolov4.utils.dataloader import YoloDataset as Dataset , yolo_dataset_collate as dataset_collate
        from yolov4.utils.utils import get_anchors, get_classes
        from yolov4.utils.utils_fit import fit_one_epoch

    elif modelType == ModelType.YOLOV3: 
        from yolov3.nets.yolo_training import YOLOLoss, weights_init
        from yolov3.utils.callbacks import LossHistory
        from yolov3.utils.dataloader import YoloDataset, yolo_dataset_collate
        from yolov3.utils.utils import get_anchors, get_classes
        from yolov3.utils.utils_fit import fit_one_epoch

    elif modelType == ModelType.SSD: 
        from ssd.nets.ssd_training import MultiboxLoss, weights_init
        from ssd.utils.anchors import get_anchors
        from ssd.utils.callbacks import LossHistory
        from ssd.utils.dataloader import SSDDataset, ssd_dataset_collate
        from ssd.utils.utils import get_classes
        from ssd.utils.utils_fit import fit_one_epoch

    elif modelType == ModelType.RETINANET: 
        from retinanet.nets.retinanet_training import FocalLoss
        from retinanet.utils.callbacks import LossHistory
        from retinanet.utils.dataloader import RetinanetDataset, retinanet_dataset_collate
        from retinanet.utils.utils import get_classes
        from retinanet.utils.utils_fit import fit_one_epoch

    elif modelType == ModelType.FASTER_RCNN: 
        from faster_rcnn.nets.frcnn_training import FasterRCNNTrainer, weights_init
        from faster_rcnn.utils.callbacks import LossHistory
        from faster_rcnn.utils.dataloader import FRCNNDataset, frcnn_dataset_collate
        from faster_rcnn.utils.utils import get_classes
        from faster_rcnn.utils.utils_fit import fit_one_epoch

    elif modelType == ModelType.CENTERNET: 
        from centernet.utils.callbacks import LossHistory
        from centernet.utils.dataloader import CenternetDataset, centernet_dataset_collate
        from centernet.utils.utils import get_classes
        from centernet.utils.utils_fit import fit_one_epoch
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網盤下載。模型的 預訓練權重 對不同數據集是通用的，因為特征是通用的。
    #   模型的 預訓練權重 比較重要的部分是 主幹特征提取網絡的權值部分，用於進行特征提取。
    #   預訓練權重對於99%的情況都必須要用，不用的話主幹部分的權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好
    #
    #   如果訓練過程中存在中斷訓練的操作，可以將model_path設置成logs文件夾下的權值文件，將已經訓練了一部分的權值再次載入。
    #   同時修改下方的 凍結階段 或者 解凍階段 的參數，來保證模型epoch的連續性。
    #   
    #   當model_path = ''的時候不加載整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此是在train.py進行加載的。
    #   如果想要讓模型從0開始訓練，則設置model_path = ''，下面的Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #   一般來講，從0開始訓練效果會很差，因為權值太過隨機，特征提取效果不明顯。
    #
    #   網絡一般不從0開始訓練，至少會使用主幹部分的權值，有些論文提到可以不用預訓練，主要原因是他們 數據集較大 且 調參能力優秀。
    #   如果一定要訓練網絡的主幹部分，可以了解imagenet數據集，首先訓練分類模型，分類模型的 主幹部分 和該模型通用，基於此進行訓練。

    #   anchors_path代表先驗框對應的txt文件，一般不修改。
    #   anchors_mask用於幫助代碼找到對應的先驗框，一般不修改。

    #   輸入的shape大小，一定要是32的倍數    
    #----------------------------------------------------------------------------------------------------------------------------#
    if modelType == ModelType.YOLOV4:
        model_path      = 'model_data/weight/yolo4_weights.pth' #coco
        anchors_path    = 'yolov4/yolo_anchors.txt'
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape     = [608, 608]  
        weight_decay    = 5e-4
        gamma           = 0.94

    elif modelType == ModelType.YOLOV3:
        model_path      = 'model_data/weight/yolo3_weights.pth' #coco
        anchors_path    = 'yolov3/yolo_anchors.txt'
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape     = [416, 416]  
        weight_decay    = 5e-4
        gamma           = 0.94
    
    elif modelType == ModelType.SSD:
        model_path      = 'model_data/weight/ssd_weights.pth'
        input_shape     = [300, 300]
        backbone        = "vgg"     #   vgg或者mobilenetv2
        weight_decay    = 5e-4
        gamma           = 0.94

    elif modelType == ModelType.RETINANET:
        model_path      = 'model_data/weight/retinanet_resnet50.pth'
        input_shape     = [600, 600]
        phi             = 2
        weight_decay    = 0
        gamma           = 0.96

    elif modelType == ModelType.FASTER_RCNN:
        model_path      = 'model_data/weight/voc_weights_resnet.pth'
        input_shape     = [600, 600]
        backbone        = "resnet50"  #   vgg或者resnet50
        weight_decay    = 5e-4
        gamma           = 0.96

    elif modelType == ModelType.CENTERNET:
        model_path      = 'model_data/weight/centernet_resnet50_voc.pth'
        input_shape     = [512, 512]
        backbone        = "resnet50"
        weight_decay    = 5e-4
        gamma           = 0.94
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #----------------------------------------------------#
    #   可用于设定先验框的大小，默认的anchors_size
    #   是根据voc数据集设定的，大多数情况下都是通用的！
    #   如果想要检测小物体，可以修改anchors_size
    #   一般调小浅层先验框的大小就行了！因为浅层负责小物体检测！
    #   比如anchors_size = [21, 45, 99, 153, 207, 261, 315]
    #----------------------------------------------------#
    # anchors_size    = [30, 60, 111, 162, 213, 264, 315]  # SSD
    # anchors_size    = [8, 16, 32]                        # FasterRCNN
    #------------------------------------------------------#
    #   Yolov4的tricks應用
    #   mosaic 馬賽克數據增強 True or False 
    #   實際測試時mosaic數據增強並不穩定，所以默認為False
    #   Cosine_lr 余弦退火學習率 True or False
    #   label_smoothing 標簽平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic              = False
    # mosaic              = True
    Cosine_lr           = False
    label_smoothing     = 0
    #----------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。
    #   顯存不足與數據集大小無關，提示顯存不足請調小batch_size。
    #   受到BatchNorm層影響，batch_size最小為2，不能為1。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特征提取網絡不發生改變
    #   占用的顯存較小，僅對網絡進行微調
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50 #50
    Freeze_batch_size   = int(8/4)
    Freeze_lr           = 1e-3
    #----------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100 #100
    Unfreeze_batch_size = int(4/2)
    Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   是否進行凍結訓練，默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   獲得圖片路徑和標簽
    #----------------------------------------------------#
    train_annotation_path   = os.path.join(data_path, "Detection//train.txt")
    val_annotation_path   = os.path.join(data_path, "Detection//val.txt") 
    #----------------------------------------------------#
    #   獲取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if modelType in [ModelType.YOLOV4, ModelType.YOLOV3]:
        # Yolov3 / Yolov4
        anchors, num_anchors     = get_anchors(anchors_path)
    elif modelType == ModelType.SSD:
        # SSD
        anchors_size    = [30, 60, 111, 162, 213, 264, 315]  # SSD
        num_classes += 1
        anchors = get_anchors(input_shape, anchors_size, backbone)
    #------------------------------------------------------#
    #   創建模型
    #------------------------------------------------------#
    if modelType in [ModelType.YOLOV4, ModelType.YOLOV3]:
        # Yolov3 / Yolov4
        model = Model(anchors_mask, num_classes)
        weights_init(model)

    elif modelType == ModelType.SSD:
        # SSD        
        model = Model(num_classes, backbone, pretrained)

    elif modelType == ModelType.RETINANET:
        # Retinanet
        model = Model(num_classes, phi, pretrained)

    elif modelType == ModelType.FASTER_RCNN:
        # FasterRCNN 
        anchors_size    = [8, 16, 32]                        # FasterRCNN
        model = Model(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)

    elif modelType == ModelType.CENTERNET:
        # CenterNet
        model = Model(num_classes, pretrained = pretrained)


    if model_path != '':
        #------------------------------------------------------#
        #   權值文件請看README，百度網盤下載
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

        if modelType == ModelType.FASTER_RCNN:
            # FasterRCNN  
            model.freeze_bn() 
    
    loss_history = LossHistory(model_train)

    if modelType == ModelType.YOLOV4:
        # Yolov4
        criterion    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)

    elif modelType == ModelType.YOLOV3:
        # Yolov3
        criterion    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

    elif modelType == ModelType.SSD:
        # SSD
        criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)

    elif modelType == ModelType.RETINANET:
        # Retinanet
        criterion      = FocalLoss()    
    
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   UnFreeze_Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        if modelType == ModelType.YOLOV4:    
            # Yolov4
            train_dataset   = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train = True)
            val_dataset     = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
        elif modelType == ModelType.YOLOV3:  
            # Yolov3
            train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
        elif modelType == ModelType.SSD:  
            # SSD
            train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)
            val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=ssd_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=ssd_dataset_collate)
        elif modelType == ModelType.RETINANET: 
            # Retinanet
            train_dataset   = RetinanetDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = RetinanetDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=retinanet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=retinanet_dataset_collate)
        elif modelType == ModelType.FASTER_RCNN: 
            # FasterRCNN
            train_dataset   = FRCNNDataset(train_lines, input_shape, train = True)
            val_dataset     = FRCNNDataset(val_lines, input_shape, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=frcnn_dataset_collate)
        elif modelType == ModelType.CENTERNET: 
            # CenterNet
            train_dataset   = CenternetDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = CenternetDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=centernet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=centernet_dataset_collate)

        #------------------------------------#
        #   凍結一定部分訓練
        #------------------------------------#
        if Freeze_Train:
            loss_history.set_status(freeze=True)
            model.freeze_backbone() 

            if modelType == ModelType.FASTER_RCNN:  
                # FasterRCNN                
                train_util      = FasterRCNNTrainer(model, optimizer)            

        for epoch in range(start_epoch, end_epoch):
            if modelType in [ModelType.YOLOV4, ModelType.YOLOV3, ModelType.SSD, ModelType.RETINANET]:  
                # Yolov3 / Yolov4 / SSD / Retinanet
                fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)

            elif modelType == ModelType.FASTER_RCNN:  
                # FasterRCNN
                fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)

            elif modelType == ModelType.CENTERNET:  
                # CenterNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, backbone)


            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = weight_decay)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma = gamma)

        if modelType == ModelType.YOLOV4:    
            # Yolov4
            train_dataset   = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train = True)
            val_dataset     = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
        elif modelType == ModelType.YOLOV3:  
            # Yolov3
            train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
        elif modelType == ModelType.SSD:  
            # SSD
            train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)
            val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=ssd_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=ssd_dataset_collate)
        elif modelType == ModelType.RETINANET: 
            # Retinanet
            train_dataset   = RetinanetDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = RetinanetDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=retinanet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=retinanet_dataset_collate)
        elif modelType == ModelType.FASTER_RCNN: 
            # FasterRCNN
            train_dataset   = FRCNNDataset(train_lines, input_shape, train = True)
            val_dataset     = FRCNNDataset(val_lines, input_shape, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=frcnn_dataset_collate)
        elif modelType == ModelType.CENTERNET: 
            # CenterNet
            train_dataset   = CenternetDataset(train_lines, input_shape, num_classes, train = True)
            val_dataset     = CenternetDataset(val_lines, input_shape, num_classes, train = False)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=centernet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=centernet_dataset_collate)

        #------------------------------------#
        #   不凍結一定部分訓練
        #------------------------------------#
        if Freeze_Train:
            loss_history.set_status(freeze=False)
            model.unfreeze_backbone()           
 
            if modelType == ModelType.FASTER_RCNN:                
                train_util      = FasterRCNNTrainer(model, optimizer)
            

        for epoch in range(start_epoch, end_epoch):
            if modelType in [ModelType.YOLOV4, ModelType.YOLOV3, ModelType.SSD, ModelType.RETINANET]:  
                # Yolov3 / Yolov4 / SSD / Retinanet
                fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)

            elif modelType == ModelType.FASTER_RCNN:  
                # FasterRCNN
                fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)

            elif modelType == ModelType.CENTERNET:  
                # CenterNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, backbone)
            
            lr_scheduler.step()
