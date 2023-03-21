import torch
import torch.nn as nn

import importlib
import torch.optim as optim

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_model(opt, pred=False):  
    model = init_dt_model(opt, pred)
    if opt.net == 'yolov8':
        YOLOLoss = init_loss(opt) 
        criterion = YOLOLoss(model)   
    else:
        criterion = init_loss(opt)   
    return model, criterion

def init_dt_model(opt, pred=False):
    if opt.net == 'ssd':
        from det_model.ssd.nets.ssd import SSD300
        model = SSD300(opt.num_classes, opt.backbone, opt.pretrained)
    elif opt.net == 'retinanet':
        from det_model.retinanet.nets.retinanet import retinanet
        model = retinanet(opt.num_classes, opt.phi, opt.pretrained, opt.fp16)
    elif opt.net == 'centernet':
        from det_model.centernet.nets.centernet import CenterNet_Resnet50
        model = CenterNet_Resnet50(opt.num_classes, opt.pretrained)
    elif opt.net == 'faster_rcnn':
        from det_model.faster_rcnn.nets.frcnn import FasterRCNN 
        if pred:
            model = FasterRCNN(opt.num_classes, "predict", anchor_scales = opt.anchors_size, backbone = opt.backbone)
        else:
            model = FasterRCNN(opt.num_classes, anchor_scales = opt.anchors_size, backbone = opt.backbone, pretrained = opt.pretrained)

    elif opt.net == 'yolov3':
        from det_model.yolov3.nets.yolo import YoloBody
        model = YoloBody(opt.anchors_mask, opt.num_classes)
        weights_init(model)
    elif opt.net == 'yolov4':
        from det_model.yolov4.nets.yolo import YoloBody
        model = YoloBody(opt.anchors_mask, opt.num_classes)
        weights_init(model)
    elif opt.net == 'yolov5':
        from det_model.yolov5.nets.yolo import YoloBody
        model = YoloBody(opt.anchors_mask, opt.num_classes, opt.phi)
        weights_init(model)
    elif opt.net == 'rtmdet':
        from det_model.rtmdet.nets.yolo import YoloBody
        model = YoloBody(opt.anchors_mask, opt.num_classes, opt.phi)
        weights_init(model)
    elif opt.net == 'yolov7':
        from det_model.yolov7.nets.yolo import YoloBody
        model = YoloBody(opt.anchors_mask, opt.num_classes, opt.phi, pretrained=opt.pretrained)
        weights_init(model)
    elif opt.net == 'yolov8':
        from det_model.yolov8.nets.yolo import YoloBody
        model = YoloBody(opt.input_shape, opt.num_classes, opt.phi, pretrained=opt.pretrained)
        weights_init(model)
    elif opt.net == 'yolox':
        from det_model.yolox.nets.yolo import YoloBody
        model = YoloBody(opt.num_classes, opt.phi)
        weights_init(model)

    return model    

def init_loss(opt):
    if opt.net == 'ssd':
        from det_model.ssd.nets.ssd_training import MultiboxLoss
        criterion       = MultiboxLoss(opt.num_classes, neg_pos_ratio=3.0)
    elif opt.net == 'retinanet':
        from det_model.retinanet.nets.retinanet_training import FocalLoss
        criterion      = FocalLoss()  
    elif opt.net == 'centernet':
        from det_model.centernet.nets.centernet_training import focal_loss, reg_l1_loss
        criterion      = (focal_loss, reg_l1_loss)
    elif opt.net == 'faster_rcnn':
        from det_model.retinanet.nets.retinanet_training import FocalLoss
        criterion      = "rpn_roi" 
    elif opt.net == 'yolov3':
        from det_model.yolov3.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.anchors, opt.num_classes, opt.input_shape, opt.Cuda, opt.anchors_mask) 
    elif opt.net == 'yolov4':
        from det_model.yolov4.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.anchors, opt.num_classes, opt.input_shape, opt.Cuda, opt.anchors_mask, opt.label_smoothing)
    elif opt.net == 'yolov5':
        from det_model.yolov5.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.anchors, opt.num_classes, opt.input_shape, opt.Cuda, opt.anchors_mask, opt.label_smoothing)
    elif opt.net == 'rtmdet':
        from det_model.rtmdet.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.anchors, opt.num_classes, opt.input_shape, opt.Cuda, opt.anchors_mask, opt.label_smoothing)
    elif opt.net == 'yolov7':
        from det_model.yolov7.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.anchors, opt.num_classes, opt.input_shape, opt.anchors_mask, opt.label_smoothing)
    elif opt.net == 'yolov8':
        from det_model.yolov8.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss
    elif opt.net == 'yolox':
        from det_model.yolox.nets.yolo_training import YOLOLoss
        criterion    = YOLOLoss(opt.num_classes, opt.fp16)

    return criterion

def get_optimizer(model, opt, optimizer_type):    
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), opt.Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[optimizer_type]   
    return optimizer

def generate_loader(opt):      

    if opt.net == 'ssd':
        from det_model.ssd.utils.dataloader import SSDDataset, ssd_dataset_collate        
        train_dataset   = SSDDataset(opt.train_lines, opt.input_shape, opt.anchors, opt.batch_size, opt.num_classes, train = True)
        val_dataset     = SSDDataset(opt.val_lines, opt.input_shape, opt.anchors, opt.batch_size, opt.num_classes, train = False)
        dataset_collate = ssd_dataset_collate  
    elif opt.net == 'retinanet':
        from det_model.retinanet.utils.dataloader import RetinanetDataset, retinanet_dataset_collate     
        train_dataset   = RetinanetDataset(opt.train_lines, opt.input_shape,  opt.num_classes, train = True)
        val_dataset     = RetinanetDataset(opt.val_lines, opt.input_shape, opt.num_classes, train = False)
        dataset_collate = retinanet_dataset_collate 
    elif opt.net == 'centernet':
        from det_model.centernet.utils.dataloader import CenternetDataset, centernet_dataset_collate    
        train_dataset   = CenternetDataset(opt.train_lines, opt.input_shape,  opt.num_classes, train = True)
        val_dataset     = CenternetDataset(opt.val_lines, opt.input_shape, opt.num_classes, train = False)
        dataset_collate = centernet_dataset_collate    
    elif opt.net == 'faster_rcnn':
        from det_model.faster_rcnn.utils.dataloader import FRCNNDataset, frcnn_dataset_collate
        train_dataset   = FRCNNDataset(opt.train_lines, opt.input_shape, train = True)
        val_dataset     = FRCNNDataset(opt.val_lines, opt.input_shape, train = False)
        dataset_collate = frcnn_dataset_collate
    elif opt.net == 'yolov3':
        from det_model.yolov3.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, train = True)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, train = False)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'yolov4':
        from det_model.yolov4.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, epoch_length = opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, epoch_length = opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'yolov5':
        from det_model.yolov5.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'rtmdet':
        from det_model.rtmdet.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'yolov7':
        from det_model.yolov7.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, opt.anchors, opt.anchors_mask, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'yolov8':
        from det_model.yolov8.utils.dataloader import YoloDataset, yolo_dataset_collate
        # train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
        #                                 mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        # val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
        #                                 mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, epoch_length=opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate
    elif opt.net == 'yolox':
        from det_model.yolox.utils.dataloader import YoloDataset, yolo_dataset_collate
        train_dataset   = YoloDataset(opt.train_lines, opt.input_shape, opt.num_classes, epoch_length = opt.UnFreeze_Epoch, \
                                        mosaic=opt.mosaic, mixup=opt.mixup, mosaic_prob=opt.mosaic_prob, mixup_prob=opt.mixup_prob, train=True, special_aug_ratio=opt.special_aug_ratio)
        val_dataset     = YoloDataset(opt.val_lines, opt.input_shape, opt.num_classes, epoch_length = opt.UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        dataset_collate = yolo_dataset_collate

    batch_size      = opt.batch_size
    if opt.distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // opt.ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val