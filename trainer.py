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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

class LossHistory():
    def __init__(self, opt, patience = 10):        
        self.losses     = []
        self.val_loss   = []
        self.writer = opt.writer
        self.freeze = False
        self.log_dir = opt.out_path
       
        # launch tensorboard
        t = threading.Thread(target=self.launchTensorBoard, args=([opt.out_path]))
        t.start()     

        # initial EarlyStopping
        self.patience = patience
        self.reset_stop()          

    def launchTensorBoard(self, tensorBoardPath, port = 8888):
        os.system('tensorboard --logdir=%s --port=%s --load_fast=false'%(tensorBoardPath, port))
        url = "http://localhost:%s/"%(port)
        # webbrowser.open_new(url)
        return

    def reset_stop(self):
        self.best_epoch_loss = np.Inf 
        self.stopping = False
        self.counter  = 0

    def set_status(self, freeze):
        self.freeze = freeze

    def epoch_loss(self, loss, val_loss, epoch):
        self.losses.append(loss)
        self.val_loss.append(val_loss)  

        prefix = "Freeze_epoch/" if self.freeze else "UnFreeze_epoch/"     
        self.writer.add_scalar(prefix+'Loss/Train', loss, epoch)
        self.writer.add_scalar(prefix+'Loss/Val', val_loss, epoch)
        self.decide(val_loss)   

    def step(self, steploss, iteration):        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Loss', steploss, iteration)

    def decide(self, epoch_loss):
        if epoch_loss > self.best_epoch_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Best lower loss:{self.best_epoch_loss}')
                self.stopping = True
        else:
            self.best_epoch_loss = epoch_loss           
            self.counter = 0 
            self.stopping = False

def fit_one_epoch(model_train, model, ssd_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    total_loss  = 0
    val_loss    = 0 

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor) 
            #----------------------#
            #   前向传播
            #----------------------#
            out = model_train(images)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   计算损失
            #----------------------#
            loss = ssd_loss.forward(targets, out)
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
                
    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor) 

                out = model_train(images)
                optimizer.zero_grad()
                loss = ssd_loss.forward(targets, out)
                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss'    : val_loss / (iteration + 1), 
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))


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
        print()
        
    
    
    def train(self):
        # self.opt.Init_Epoch = 49
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

            fit_one_epoch(self.model_train, self.model, self.criterion, self.loss_history, self.optimizer, epoch, self.epoch_step, self.epoch_step_val, 
                            self.train_loader, self.test_loader, self.opt.end_epoch, self.opt.Cuda)

        print("End of UnFreeze Training")
                        
        