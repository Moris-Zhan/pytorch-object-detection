import sys
sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np
import time, os
import math
import logging
import torch.backends.cudnn as cudnn

# from visdom import Visdom

# import models, datasets, utils
import models

import io
from contextlib import redirect_stdout
from torchinfo import summary
from pynvml import *
from functools import partial
import importlib
import threading
from tqdm import tqdm
from helps.utils import *
from models.script import get_fit_func



class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        
        self.model, self.criterion  = models.get_model(opt)     
        # ------------------------------------------------------------------------------- 
        rndm_input = torch.autograd.Variable(
            torch.rand(1, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1]), 
            requires_grad = False).cpu()
        opt.writer.add_graph(self.model, rndm_input) 

        f = io.StringIO()
        with redirect_stdout(f):        
            summary(self.model, (opt.batch_size, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1]) )
        lines = f.getvalue()

        with open( os.path.join(opt.out_path, "model.txt") ,"w") as f:
            [f.write(line) for line in lines]
        print(lines) 
        # ------------------------------------------------------------------------------
        if opt.model_path != '':
            #------------------------------------------------------#
            #   權值文件請看README，百度網盤下載
            #------------------------------------------------------#
            print('Load weights {}.'.format(opt.model_path))
            device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict      = self.model.state_dict()
            pretrained_dict = torch.load(opt.model_path, map_location = device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        # ------------------------------------------------------------------------------
        # self.model = self.model.to(self.device)

        # if opt.ngpu>1:
        #     self.model = nn.DataParallel(self.model)   

        # if opt.distributed:
        #     #----------------------------#
        #     #   多卡平行运行
        #     #----------------------------#
        #     self.model = self.model.cuda(opt.local_rank)
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[opt.local_rank], find_unused_parameters=True)
        # else:
        #     self.model_train = torch.nn.DataParallel(self.model)
        #     cudnn.benchmark = True
        #     self.model_train = self.model_train.cuda()

        self.model_train = self.model.train()
        if opt.Cuda:
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()

        self.model.train()  
      
        #-------------------------------------------------------------------#
        #   判断当前batch_size与64的差别，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs         = 64
        opt.Init_lr_fit = max(opt.batch_size / nbs * opt.Init_lr, 1e-4)
        opt.Min_lr_fit  = max(opt.batch_size / nbs * opt.Min_lr, 1e-6)

        self.optimizer = models.get_optimizer(self.model, opt, opt.optimizer_type)
        self.lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, opt.Init_lr_fit, opt.Min_lr_fit, opt.UnFreeze_Epoch)

        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        self.epoch_step      = opt.num_train // opt.batch_size
        self.epoch_step_val  = opt.num_val // opt.batch_size
        
        if self.epoch_step == 0 or self.epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")        

        self.train_loader, self.test_loader = models.generate_loader(opt) 
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}

        self.loss_history = LossHistory(opt)
        self.fit_one_epoch = get_fit_func(opt)
        print()
        
    
    
    def train(self):
        # self.opt.Init_Epoch = 49
        if self.opt.net == 'faster_rcnn':
            from det_model.faster_rcnn.nets.frcnn_training import FasterRCNNTrainer
            train_util      = FasterRCNNTrainer(self.model, self.optimizer) 
        if self.opt.Freeze_Train:
            #------------------------------------#
            #   凍結一定部分訓練
            #------------------------------------#
            self.loss_history.set_status(freeze=True)
            self.model.freeze_backbone() 
            self.loss_history.reset_stop()
        else:
            #------------------------------------#
            #   解凍後訓練
            #------------------------------------#
            self.loss_history.set_status(freeze=False)
            self.model.unfreeze_backbone()   
            self.loss_history.reset_stop() 
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(self.opt.Init_Epoch, self.opt.UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= self.opt.Freeze_Epoch and not self.opt.UnFreeze_flag and self.opt.Freeze_Train:
                #-----------------------------------------------------------------------------------------#
                batch_size = self.opt.Unfreeze_batch_size   
                self.opt.end_epoch = self.opt.UnFreeze_Epoch
                #-----------------------------------------------------------------------------------------#
                self.optimizer = models.get_optimizer(self.model, self.opt, 'adam')                                          
                #-----------------------------------------------------------------------------------------#
                self.loss_history.set_status(freeze=False)
                self.model.unfreeze_backbone()   
                self.loss_history.reset_stop() 
                #-----------------------------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if self.opt.optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if self.opt.optimizer_type == 'adam' else 5e-5
                Init_lr_fit     = min(max(batch_size / nbs * self.opt.Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * self.opt.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                self.lr_scheduler_func = get_lr_scheduler(self.opt.lr_decay_type, Init_lr_fit, Min_lr_fit, self.opt.UnFreeze_Epoch)
                
                if self.opt.backbone == "vgg":
                    for param in self.model.vgg[:28].parameters():
                        param.requires_grad = True
                else:
                    for param in self.model.mobilenet.parameters():
                        param.requires_grad = True
                        
                epoch_step      = self.opt.num_train // batch_size
                epoch_step_val  = self.opt.num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if self.opt.distributed:
                    batch_size = batch_size // self.opt.ngpus_per_node     
                
                self.train_loader, self.test_loader = models.generate_loader(self.opt) 

                self.opt.UnFreeze_flag = True

            # only early stop when UnFreeze Training
            if (self.opt.UnFreeze_flag and self.opt.Early_Stopping and self.loss_history.stopping): break

            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, epoch)

            if self.opt.net == 'faster_rcnn':
                self.fit_one_epoch(self.model, train_util, self.loss_history, self.optimizer, epoch, self.epoch_step, self.epoch_step_val, 
                            self.train_loader, self.test_loader, self.opt)                
            else:
                self.fit_one_epoch(self.model_train, self.model, self.criterion, self.loss_history, self.optimizer, epoch, self.epoch_step, self.epoch_step_val, 
                            self.train_loader, self.test_loader, self.opt)
                

        print("End of UnFreeze Training")
                        
        