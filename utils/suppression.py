import torch
from utils.func import xywh2xyxy, bbox_iou

def non_max_suppression(predictions, conf_thres=0.5, nms_thres=0.4):
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
        image_pred = image_pred[image_pred[:, 4] <= -1.216]
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
