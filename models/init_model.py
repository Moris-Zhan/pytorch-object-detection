import torch
import torch.nn as nn

import importlib
import torch.optim as optim


def get_model(opt):
    if opt.net == 'ssd':
        model = init_dt_model(opt)
        criterion = init_loss(opt)

    return model, criterion

def init_dt_model(opt):
    if opt.net == 'ssd':
        from det_model.ssd.nets.ssd import SSD300
        model = SSD300(opt.num_classes, opt.backbone, opt.pretrained)

    return model    

def init_loss(opt):
    if opt.net == 'ssd':
        from det_model.ssd.nets.ssd_training import MultiboxLoss
        criterion       = MultiboxLoss(opt.num_classes, neg_pos_ratio=3.0)
    return criterion

def get_optimizer(model, opt):
    if opt.net == 'ssd':
        optimizer = {
                'adam'  : optim.Adam(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
                'sgd'   : optim.SGD(model.parameters(), opt.Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
            }[opt.optimizer_type]   
    return optimizer

def generate_loader(opt):      

    if opt.net == 'ssd':
        from det_model.ssd.utils.dataloader import SSDDataset, ssd_dataset_collate        
        train_dataset   = SSDDataset(opt.train_lines, opt.input_shape, opt.anchors, opt.batch_size, opt.num_classes, train = True)
        val_dataset     = SSDDataset(opt.val_lines, opt.input_shape, opt.anchors, opt.batch_size, opt.num_classes, train = False)
        dataset_collate = ssd_dataset_collate  

    if opt.distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = opt.batch_size // opt.ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        batch_size      = opt.batch_size
        shuffle         = True

    gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val

# def init_optimizer(opt):
#     if opt.net == "ssd":
#         from det_model.ssd.nets.ssd_training import set_optimizer_lr
#         optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = weight_decay)
#         set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
#     return 0

# def init_scheduler(opt):
#     if opt.net == "ssd":
#         from det_model.ssd.nets.ssd_training import (MultiboxLoss, get_lr_scheduler,
#                                set_optimizer_lr
#     return 0
