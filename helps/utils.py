import numpy as np
import os
import math



from pynvml import *
from functools import partial
import threading


def get_data_path(root_path, dataType):
    #------------------------------#  
    #   數據集路徑
    #   訓練自己的數據集必須要修改的
    #------------------------------#  
    map_dict = { "AsianTraffic":'Asian-Traffic', "bdd":'bdd100k', "coco":'COCO',
                 "voc":'VOCdevkit', "lane":"LANEdevkit", "widerperson":'WiderPerson', 
                 "MosquitoContainer":'MosquitoContainer' }
    return os.path.join(root_path, map_dict[dataType]) 

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names) 

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

    def step_c(self, steploss, iteration):  # for centernet      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Classification_Loss', steploss, iteration)

    def step_r(self, steploss, iteration):  # for centernet        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Regression_Loss', steploss, iteration)

    def step_rpn_loc(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/rpn_loc_Loss', steploss, iteration)

    def step_rpn_cls(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/rpn_cls_Loss', steploss, iteration)

    def step_roi_loc(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/roi_loc_Loss', steploss, iteration)

    def step_roi_cls(self, steploss, iteration):  # for fasterrcnn     
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/roi_cls_Loss', steploss, iteration)        

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