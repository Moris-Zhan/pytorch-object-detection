import torch
from utils.func import xywh2xyxy, bbox_iou
from utils.func import center_to_points_nms, undo_offsets, get_nonzero_classes, iou

def non_max_suppression_yolo(predictions, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    if type(predictions) != list: 
        predictions = [predictions]
        # Anchor = 3
        # 507     3*13*13
        # 2028   3*26*26
        # 8112   3*52*52
        # [bsz, 3*(13+6),52*52]
    predictions_list = []
    for prediction in predictions:
        num_samples = prediction.size(0)
        grid_size = prediction.size(2)
        answers = int(prediction.size(1) /3) # [x, y, w, h, conf, numclass...] = 5 + C

        prediction = (
            # x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            prediction.view(num_samples, 3, answers, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        prediction = prediction.view(num_samples, -1 , answers)
        predictions_list.append(prediction)

    prediction = torch.cat(predictions_list, dim=1)
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= -0.0151]
        # image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def non_max_suppression_ssd(predictions, default_boxes, topk = 100, nms_thresh = 0.5, class_thresh = 0.45):
    predicted_classes, predicted_boxes_bsz = predictions
    predicted_boxes_bsz = undo_offsets(default_boxes, predicted_boxes_bsz)
    predicted_boxes_bsz = center_to_points_nms(predicted_boxes_bsz)

    classes_bsz, _, scores_bsz = get_nonzero_classes(predicted_classes.squeeze(0), norm=True)
    classes_unique = torch.unique(classes_bsz)
    # classes_unique = classes_unique[classes_unique != 0]
    # nms_boxes, num_boxes = utils.nms_and_thresh(classes_unique, scores_bsz, classes_bsz, predicted_boxes_bsz, 0.5, 0.65)
    nms_boxes = torch.zeros((2, classes_unique.size(0) * topk, 7))
    for i in range(len(nms_boxes)):
        nms_box = nms_boxes[i]
        num_boxes = 0
        classes = classes_bsz[i]
        scores = scores_bsz[i]
        predicted_boxes = predicted_boxes_bsz[i]
        
        for cl in classes_unique:
            # print(cl.item())            
            current_class_idxs = (classes == cl).nonzero()

            if current_class_idxs.dim() > 1:
                current_class_idxs = current_class_idxs.squeeze(1)
            
            current_class_scores = scores[current_class_idxs]
            current_boxes = predicted_boxes[current_class_idxs]

            sorted_scores, class_idxs_by_score = torch.sort(current_class_scores, descending=True)
            class_idxs_by_score = class_idxs_by_score[sorted_scores >= class_thresh]
            class_idxs_by_score = class_idxs_by_score[:topk]

            while len(class_idxs_by_score) > 0:

                curr_bbx = current_boxes[class_idxs_by_score[0]]
                info = torch.tensor([0, sorted_scores[class_idxs_by_score[0]].item(), cl.item()]).cuda()
                nms_box[num_boxes] = torch.cat((curr_bbx, info))

                num_boxes += 1

                other_bbxs = current_boxes[class_idxs_by_score]
                curr_bbx = curr_bbx.expand_as(other_bbxs)

                ious = iou(curr_bbx, other_bbxs)
                
                class_idxs_by_score = class_idxs_by_score[ious <= nms_thresh]
        
    return nms_boxes[:,:num_boxes]

def non_max_suppression_retinaNet(predictions, anchor_boxes, topk = 100, nms_thresh = 0.5, class_thresh = 0.45, mode='union'):
    loc_preds, cls_preds = predictions
    nms_boxes = []
    for bid in range(loc_preds.size(0)):
        loc_xy = loc_preds[bid, :,:2]
        loc_wh = loc_preds[bid, :,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        score, labels = cls_preds[bid].sigmoid().max(1)          # [#anchors,]
        ids = score > class_thresh
        ids = ids.nonzero().squeeze()             # [#obj,]
        # keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        bboxes = boxes[ids]
        scores = score[ids]
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1+1) * (y2-y1+1)
        _, order = scores.sort(0, descending=True)
        order = order[:topk]

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                break
            i = order[0].item()
            keep.append(i)           

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1+1).clamp(min=0)
            h = (yy2-yy1+1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=nms_thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        keep = torch.LongTensor(keep)
        nms_boxes.append(torch.cat([
            boxes[keep], 
            torch.zeros((len(keep),1)).type(torch.float32).cuda(),
            scores[keep].unsqueeze(1), 
            labels[keep].unsqueeze(1).type(torch.float32)
            ], dim = 1).unsqueeze(0))
    return torch.cat(nms_boxes, dim=0)