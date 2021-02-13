from __future__ import division


import os
from os import path

import torch

import os
import numpy as np
import torch

from tqdm import tqdm
import itertools

# YOLO
# ---------------------------------------------------------------------------------------------------------------------------------------------
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu().data.numpy(), pred_labels.cpu().data.numpy()])
    return batch_metrics

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# SSD
# -----------------------------------------------------------------------------------------------------------------------------------------
def get_dboxes(smin=0.07, smax=0.9, ars=[1, 2, (1/2.0), 3, (1/3.0)], fks=[38, 19, 10, 5, 3, 1], num_boxes=[3, 5, 5, 5, 3, 3]):
    m = len(fks)
    sks = [round(smin + (((smax-smin)/(m-1)) * (k-1)), 2) for k in range(1, m + 1)]

    boxes = []
    for k, feat_k in enumerate(fks):
        for i, j in itertools.product(range(feat_k), range(feat_k)):

            cx = (i + 0.5)/feat_k
            cy = (j + 0.5)/feat_k

            w = h = np.sqrt(sks[k] * sks[min(k+1, len(sks) - 1)])

            boxes.append([cx, cy, w, h])

            sk = sks[k]
            for ar in ars[:num_boxes[k]]:
                w = sk * np.sqrt(ar)
                h = sk / np.sqrt(ar)
                boxes.append([cx, cy, w, h])

    boxes = torch.tensor(boxes).float()
    return torch.clamp(boxes, max=1.0)

def center_to_points(center_tens):

    if center_tens.size(0) == 0:
        return center_tens
    
    assert center_tens.dim() == 3 
    assert center_tens.size(2) == 4 

    lp = torch.clamp(center_tens[:,:,:2] - center_tens[:,:,2:]/2.0, min=0.0)
    rp = torch.clamp(center_tens[:,:,:2] + center_tens[:,:,2:]/2.0, max=1.0)

    points = torch.cat([lp, rp], 2)

    return points

def undo_offsets(default_boxes, predicted_offsets, use_variance=True):
    
    offset1_mult = default_boxes[:,2:]
    offset2_mult = 1
    if use_variance:
        offset1_mult = offset1_mult * 0.1
        offset2_mult = offset2_mult * 0.2

    cx = (offset1_mult * predicted_offsets[:,:,:2]) + default_boxes[:,:2]
    wh = torch.exp(predicted_offsets[:,:,:2] * offset2_mult) * default_boxes[:,:2]

    return torch.cat([cx, wh], 2)

def get_nonzero_classes(predicted_classes, norm=False):
    
    if norm:
        pred_exp = torch.exp(predicted_classes)
        predicted_classes = pred_exp/pred_exp.sum(dim=2, keepdim=True)

    scores, classes = torch.max(predicted_classes, 2)

    non_zero_pred_idxs = (classes != 0).nonzero()

    if non_zero_pred_idxs.dim() > 1:
        non_zero_pred_idxs = non_zero_pred_idxs.squeeze(1)
    
    return classes, non_zero_pred_idxs, scores

# Takes in two tensors with boxes in point form and computes iou between them.
def iou(tens1, tens2):

    assert tens1.size() == tens2.size()

    squeeze = False
    if tens1.dim() == 2 and tens2.dim() == 2:
        squeeze = True
        tens1 = tens1.unsqueeze(0)
        tens2 = tens2.unsqueeze(0)
    
    assert tens1.dim() == 3 
    assert tens1.size(-1) == 4 and tens2.size(-1) == 4

    maxs = torch.max(tens1[:,:,:2], tens2[:,:,:2])
    mins = torch.min(tens1[:,:,2:], tens2[:,:,2:])

    diff = torch.clamp(mins - maxs, min=0.0)

    intersection = diff[:,:,0] * diff[:,:,1]

    diff1 = torch.clamp(tens1[:,:,2:] - tens1[:,:,:2], min=0.0)
    area1 = diff1[:,:,0] * diff1[:,:,1]

    diff2 = torch.clamp(tens2[:,:,2:] - tens2[:,:,:2], min=0.0)
    area2 = diff2[:,:,0] * diff2[:,:,1]

    iou = intersection/(area1 + area2 - intersection)

    if squeeze:
        iou = iou.squeeze(0)
    
    return iou
