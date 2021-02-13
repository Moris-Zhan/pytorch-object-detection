import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16
from utils.func import center_to_points, iou
import numpy as np


class ssd(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(ssd, self).__init__()

        # self.num_classes = num_classes
        self.num_classes = num_classes
        self.layers = []
        self.vgg_layers = []
        self.size = (300, 300)

        new_layers = list(vgg16(pretrained=True).features)

        # 將 VGG16 的 pool5 層從 size=2x2, stride=2 更改為 size=3x3, stride=1
        new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
        new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)
        self.f1 = nn.Sequential(*new_layers[:23])
        self.vgg_layers.append(self.f1)


        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4*self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl1)

        self.bbx1 = nn.Sequential(
            nn.Conv2d(512, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx1)

        self.base1 = nn.Sequential(*new_layers[23:])
        self.vgg_layers.append(self.base1)

        # The refrence code uses a dilation of 6 which requires a padding of 6
        self.f2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f2)

        self.cl2 = nn.Sequential(
            nn.Conv2d(1024, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl2)

        self.bbx2 = nn.Sequential(
            nn.Conv2d(1024, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx2)

        self.f3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f3)

        self.cl3 = nn.Sequential(
            nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl3)

        self.bbx3 = nn.Sequential(
            nn.Conv2d(512, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx3)

        self.f4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f4)

        self.cl4 = nn.Sequential(
            nn.Conv2d(256, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl4)

        self.bbx4 = nn.Sequential(
            nn.Conv2d(256, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx4)

        self.f5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f5)

        self.cl5 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl5)

        self.bbx5 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx5)

        self.f6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f6)

        self.cl6 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl6)

        self.bbx6 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx6)

        if init_weights:
            self._init_weights(vgg_16_init=(not init_weights))
        
    def forward(self, x):
        out_cl = []
        out_bbx = []
        
        x1 = self.f1(x)
        # x1 = self.bn1(x1)
        
        out_cl.append(self.cl1(x1))
        out_bbx.append(self.bbx1(x1))

        x1 = self.base1(x1)
        
        x2 = self.f2(x1)

        out_cl.append(self.cl2(x2))
        out_bbx.append(self.bbx2(x2))

        x3 = self.f3(x2)

        out_cl.append(self.cl3(x3))
        out_bbx.append(self.bbx3(x3))

        x4 = self.f4(x3)

        out_cl.append(self.cl4(x4))
        out_bbx.append(self.bbx4(x4))

        x5 = self.f5(x4)
        
        out_cl.append(self.cl5(x5))
        out_bbx.append(self.bbx5(x5))

        x6 = self.f6(x5)

        out_cl.append(self.cl6(x6))
        out_bbx.append(self.bbx6(x6))

        for i in range(len(out_cl)):
            out_cl[i] = out_cl[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, self.num_classes)
            out_bbx[i] = out_bbx[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, 4)

        out_cl = torch.cat(out_cl, 1)
        out_bbx = torch.cat(out_bbx, 1)
        # res = torch.cat([out_cl, out_bbx], dim = 2)
        return out_cl, out_bbx

    def _init_weights(self, vgg_16_init=False):
    
        for module in self.layers:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
            
            if vgg_16_init:
                for module in self.vgg_layers:
                    for layer in module:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.xavier_normal_(layer.weight)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0)
                        elif isinstance(layer, nn.BatchNorm2d):
                            nn.init.constant_(layer.weight, 1)
                            nn.init.constant_(layer.bias, 0)



def expand_defaults_and_annotations(default_boxes, annotations_boxes):
    
    num_annotations = annotations_boxes.size(0)

    default_boxes = default_boxes.unsqueeze(0)
    default_boxes = default_boxes.expand(num_annotations, -1, -1)

    annotations_boxes = annotations_boxes.unsqueeze(1)
    annotations_boxes = annotations_boxes.expand_as(default_boxes)

    return default_boxes, annotations_boxes

def match(default_boxes, annotations_boxes, match_thresh):
    
    num_annotations = annotations_boxes.size(0)

    default_boxes_pt = center_to_points(default_boxes)
    annotations_boxes_pt = center_to_points(annotations_boxes)

    default_boxes_pt, annotations_boxes_pt = expand_defaults_and_annotations(default_boxes_pt, annotations_boxes_pt)

    ious = iou(default_boxes_pt, annotations_boxes_pt)

    _, annotation_with_box = torch.max(ious, 1)
    annotation_inds = torch.arange(num_annotations, dtype=torch.long).to(annotation_with_box.device)
    
    ious_max, box_with_annotation = torch.max(ious, 0)
    matched_boxes_bin = (ious_max >= match_thresh)
    matched_boxes_bin[annotation_with_box] = 1
    box_with_annotation[annotation_with_box] = annotation_inds
    
    return box_with_annotation, matched_boxes_bin

def compute_offsets(default_boxes, annotations_boxes, box_with_annotation_idx, use_variance=True):
    
    matched_boxes = annotations_boxes[box_with_annotation_idx]

    offset_cx = (matched_boxes[:,:2] - default_boxes[:,:2])

    if use_variance:
        offset_cx = offset_cx / (default_boxes[:,2:] * 0.1)
    else:
        offset_cx = offset_cx / default_boxes[:,2:]

    offset_wh = torch.log(matched_boxes[:,2:]/default_boxes[:,2:])

    if use_variance:
        offset_wh = offset_wh / 0.2
    
    return torch.cat([offset_cx, offset_wh], 1)

def compute_loss(default_boxes, annotations_classes, annotations_boxes, predicted_classes, predicted_offsets, match_thresh=0.5, duplciate_checking=True, neg_ratio=3):
    
    if annotations_classes.size(0) > 0:
        annotations_classes = annotations_classes.long()
        box_with_annotation_idx, matched_box_bin = match(default_boxes, annotations_boxes, match_thresh)

        matched_box_idxs = (matched_box_bin.nonzero()).squeeze(1)
        non_matched_idxs = (matched_box_bin == 0).nonzero().squeeze(1)
        N = matched_box_idxs.size(0)

        true_offsets = compute_offsets(default_boxes, annotations_boxes, box_with_annotation_idx)

        regression_loss_criterion = nn.SmoothL1Loss(reduction='none')
        regression_loss = regression_loss_criterion(predicted_offsets[matched_box_idxs], true_offsets[matched_box_idxs])

        true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
        true_classifications[matched_box_idxs] = annotations_classes[box_with_annotation_idx[matched_box_idxs]]
    
    else:
        matched_box_idxs = torch.LongTensor([])
        non_matched_idxs = torch.arange(default_boxes.size(0))
        N = 1

        regression_loss = torch.tensor([0.0]).to(predicted_classes.device)

        true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
            
    classifications_loss_criterion = nn.CrossEntropyLoss(reduction='none')
    classifications_loss_total = classifications_loss_criterion(predicted_classes, true_classifications)

    positive_classifications = classifications_loss_total[matched_box_idxs]
    negative_classifications = classifications_loss_total[non_matched_idxs]

    _, hard_negative_idxs = torch.sort(classifications_loss_total[non_matched_idxs], descending=True)
    hard_negative_idxs = hard_negative_idxs.squeeze()[:N * neg_ratio]

    classifications_loss = (positive_classifications.sum() + negative_classifications[hard_negative_idxs].sum())/N
    regression_loss = regression_loss.sum()/N

    return classifications_loss, regression_loss, matched_box_idxs

def SSDLoss(outputs, targets, default_boxes):
    predicted_classes, predicted_offsets = outputs
    assert predicted_classes.size(0) == predicted_offsets.size(0)
    batch_size = predicted_classes.size(0)

    classification_loss = 0
    localization_loss = 0
    match_idx_viz = None

    x = targets[:, 0]
    x_unique = x.unique(sorted=True)
    lens = torch.stack([(x==x_u).sum() for x_u in x_unique])
    for j in range(batch_size):
        current_classes = predicted_classes[j]
        current_offsets = predicted_offsets[j]
        
        # annotations_classes = targets[j][:lens[j]][:, 0] if lens[j].item() != 0 else torch.Tensor([])
        # annotations_boxes = targets[j][:lens[j]][:, 1:5] if lens[j].item() != 0 else torch.Tensor([])

        annotations_classes = targets[:lens[j]][:, 1]
        annotations_boxes = targets[:lens[j]][:, 2:]


        curr_cl_loss, curr_loc_loss, _mi = compute_loss(
            default_boxes, annotations_classes, annotations_boxes, current_classes, current_offsets)

        classification_loss += curr_cl_loss
        localization_loss += curr_loc_loss

        if j == 0:
            match_idx_viz = _mi

    localization_loss = localization_loss / batch_size
    classification_loss = classification_loss / batch_size
    total_loss = localization_loss + classification_loss
    return total_loss, localization_loss, classification_loss


if __name__ == "__main__":  
    num_classes = 10
    net = ssd(num_classes = num_classes)
    print(net)