import importlib
import threading
from tqdm import tqdm
import torch
from utils.helpers import get_lr
import os

def fit_ssd(model_train, model, crietion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    total_loss  = 0
    val_loss    = 0 
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = targets.cuda(local_rank)             
            if not fp16:
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
                loss = crietion.forward(targets, out)
                #----------------------#
                #   反向传播
                #----------------------#
                loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
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
                    loss = crietion.forward(targets, out)

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

            if local_rank == 0:                
                pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
                
    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = targets.cuda(local_rank) 

                out = model_train_eval(images)
                optimizer.zero_grad()
                loss = crietion.forward(targets, out)
                val_loss += loss.item()

                if local_rank == 0:
                    pbar.set_postfix(**{'val_loss'    : val_loss / (iteration + 1), 
                                        'lr'            : get_lr(optimizer)})
                    pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_retina(model_train, model, crietion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    total_loss  = 0
    val_loss    = 0 

    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():              
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   获得预测结果
                #----------------------#
                _, regression, classification, anchors = model_train(images) 
                #----------------------#
                #   计算损失
                #----------------------#
                loss, _, _ = crietion.forward(classification, regression, anchors, targets, cuda=cuda)
                #----------------------#
                #   反向传播
                #----------------------#
                loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    #----------------------#
                    #   获得预测结果
                    #----------------------#
                    _, regression, classification, anchors = model_train(images) 
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss, _, _ = crietion.forward(classification, regression, anchors, targets, cuda=cuda)
                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            if local_rank == 0:
                pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
                
    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():                
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]

                optimizer.zero_grad()
                _, regression, classification, anchors = model_train_eval(images) 
                
                loss, _, _ = crietion.forward(classification, regression, anchors, targets, cuda=cuda)
                val_loss += loss.item()

                if local_rank == 0:
                    pbar.set_postfix(**{'val_loss'    : val_loss / (iteration + 1), 
                                        'lr'            : get_lr(optimizer)})
                    pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_centernet(model_train, model, crietion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0
    focal_loss, reg_l1_loss = crietion

    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            with torch.no_grad():
                if cuda:
                    batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                if opt.backbone=="resnet50":
                    hm, wh, offset  = model_train(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                    loss            = c_loss + wh_loss + off_loss

                    total_loss      += loss.item()
                    total_c_loss    += c_loss.item()
                    total_r_loss    += wh_loss.item() + off_loss.item()
                else:
                    outputs         = model_train(batch_images)
                    loss            = 0
                    c_loss_all      = 0
                    r_loss_all      = 0
                    index           = 0
                    for output in outputs:
                        hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                        c_loss      = focal_loss(hm, batch_hms)
                        wh_loss     = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                        off_loss    = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                        loss        += c_loss + wh_loss + off_loss
                        
                        c_loss_all  += c_loss
                        r_loss_all  += wh_loss + off_loss
                        index       += 1
                    total_loss      += loss.item() / index
                    total_c_loss    += c_loss_all.item() / index
                    total_r_loss    += r_loss_all.item() / index
                loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    if opt.backbone=="resnet50":
                        hm, wh, offset  = model_train(batch_images)
                        c_loss          = focal_loss(hm, batch_hms)
                        wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                        
                        loss            = c_loss + wh_loss + off_loss

                        total_loss      += loss.item()
                        total_c_loss    += c_loss.item()
                        total_r_loss    += wh_loss.item() + off_loss.item()
                    else:
                        outputs         = model_train(batch_images)
                        loss            = 0
                        c_loss_all      = 0
                        r_loss_all      = 0
                        index           = 0
                        for output in outputs:
                            hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                            c_loss      = focal_loss(hm, batch_hms)
                            wh_loss     = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                            off_loss    = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                            loss        += c_loss + wh_loss + off_loss
                            
                            c_loss_all  += c_loss
                            r_loss_all  += wh_loss + off_loss
                            index       += 1
                        total_loss      += loss.item() / index
                        total_c_loss    += c_loss_all.item() / index
                        total_r_loss    += r_loss_all.item() / index

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if local_rank == 0:
                pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                    'total_c_loss'  : total_c_loss / (iteration + 1),
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_c(total_c_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_r(total_r_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))


    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [ann.cuda(local_rank) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                if opt.backbone=="resnet50":
                    hm, wh, offset  = model_train_eval(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()
                else:
                    outputs = model_train_eval(batch_images)
                    index = 0
                    loss = 0
                    for output in outputs:
                        hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                        c_loss          = focal_loss(hm, batch_hms)
                        wh_loss         = 0.1*reg_l1_loss(wh, batch_whs, batch_reg_masks)
                        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                        loss            += c_loss + wh_loss + off_loss
                        index           += 1
                    val_loss            += loss.item() / index

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)
    print('Finish Validation')
    
    if local_rank == 0:
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_faster_rcnn(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0

    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler)
            total_loss      += total.item()
            rpn_loc_loss    += rpn_loc.item()
            rpn_cls_loss    += rpn_cls.item()
            roi_loc_loss    += roi_loc.item()
            roi_cls_loss    += roi_cls.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                    'rpn_loc'       : rpn_loc_loss / (iteration + 1),  
                                    'rpn_cls'       : rpn_cls_loss / (iteration + 1), 
                                    'roi_loc'       : roi_loc_loss / (iteration + 1), 
                                    'roi_cls'       : roi_cls_loss / (iteration + 1), 
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_rpn_loc(rpn_loc_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_rpn_cls(rpn_cls_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_roi_loc(roi_loc_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))
            loss_history.step_roi_cls(roi_cls_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')
        
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

                train_util.optimizer.zero_grad()
                _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)

                val_loss += val_total.item()
                if local_rank == 0:
                    pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
                    pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_yolov3(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    loss        = 0
    val_loss    = 0
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value = loss_value_all / num_pos_all            

                #----------------------#
                #   反向传播
                #----------------------#
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs         = model_train(images)

                    loss_value_all  = 0
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    for l in range(len(outputs)):
                        loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                        loss_value_all  += loss_item
                    loss_value = loss_value_all

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()

            loss += loss_value.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train_eval(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()   
            if local_rank == 0:         
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')    
        loss_history.epoch_loss(loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))    
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_yolov4(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    loss        = 0
    val_loss    = 0
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value = loss_value_all / num_pos_all            

                #----------------------#
                #   反向传播
                #----------------------#
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs         = model_train(images)

                    loss_value_all  = 0
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    for l in range(len(outputs)):
                        with torch.cuda.amp.autocast(enabled=False):
                            predication = outputs[l].float()
                        loss_item, num_pos = yolo_loss(l, predication, targets)
                        loss_value_all  += loss_item
                    loss_value = loss_value_all

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()

            loss += loss_value.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train_eval(images)

                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()  
            if local_rank == 0:          
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')    
        loss_history.epoch_loss(loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))    
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_yolov5(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    loss        = 0
    val_loss    = 0
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value = loss_value_all

                #----------------------#
                #   反向传播
                #----------------------#
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs         = model_train(images)

                    loss_value_all  = 0
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets)
                        loss_value_all  += loss_item
                    loss_value = loss_value_all

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()

            loss += loss_value.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train_eval(images)

                loss_value_all  = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value  = loss_value_all

            val_loss += loss_value.item()
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')    
        loss_history.epoch_loss(loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))  
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def fit_yolox(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    loss        = 0
    val_loss    = 0
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]              
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

                #----------------------#
                #   反向传播
                #----------------------#
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model_train(images)
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss_value = yolo_loss(outputs, targets)

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            
            if ema:
                ema.update(model_train)

            loss += loss_value.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
                pbar.update(1)
            loss_history.step(loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    targets = [ann.cuda(local_rank) for ann in targets]  
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train_eval(images)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

            val_loss += loss_value.item()
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        print('Finish Validation')    
        loss_history.epoch_loss(loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))    
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))

def get_fit_func(opt):
    if opt.net == 'ssd':
        return fit_ssd
    elif opt.net == 'retinanet':
        return fit_retina
    elif opt.net == 'centernet':
        return fit_centernet
    elif opt.net == 'faster_rcnn':
        return fit_faster_rcnn
    elif opt.net == 'yolov3':
        return fit_yolov3
    elif opt.net == 'yolov4':
        return fit_yolov4
    elif opt.net == 'yolov5':
        return fit_yolov5
    elif opt.net == 'yolox':
        return fit_yolox